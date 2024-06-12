#include "sources/kitti.hpp"

#include <sys/stat.h>

#include <fstream>
#include <iostream>
#include <opencv2/cudawarping.hpp>

#include "utils/path.hpp"

#define LEFT_CAM_ID 2
#define RIGHT_CAM_ID 3

#define PATH(ch, frame) (this->path + "/image_" + std::to_string(ch) + "/" + addLeadingZeros(frame, 6) + ".png")
#define LEFT_PATH(frame) PATH(LEFT_CAM_ID, frame)
#define RIGHT_PATH(frame) PATH(RIGHT_CAM_ID, frame)

std::string addLeadingZeros(int number, std::size_t length) {
    std::string numberAsString = std::to_string(number);
    return std::string(length - std::min(length, numberAsString.length()), '0') + numberAsString;
}

struct KITTICameraCalibration {
    uint8_t cameraId;
    float fx;
    float fy;
    float cx;
    float cy;
    float baseline;
};

bool readLine(std::string line, KITTICameraCalibration& calibration) {
    size_t pos = line.find(": ");
    if (pos == std::string::npos) {
        return false;
    }

    // Layout from https://www.cvlibs.net/publications/Geiger2013IJRR.pdf
    std::string token = line.substr(0, pos);
    line.erase(0, pos + 2);

    // Check if token starts with P
    if (token[0] != 'P') {
        return false;
    }

    KITTICameraCalibration local;  // Write locally first in case it fails
    local.cameraId = std::stoi(&token[1]);

    float fubx;

    int i = 0;
    while ((pos = line.find(" ")) != std::string::npos) {
        token = line.substr(0, pos);
        line.erase(0, pos + 1);

        switch (i) {
            case 0:
                local.fx = std::stof(token);
                break;
            case 5:
                local.fy = std::stof(token);
                break;
            case 3:
                fubx = std::stof(token);
                break;
            case 2:
                local.cx = std::stof(token);
                break;
            case 6:
                local.cy = std::stof(token);
                break;
        }

        i++;
    }

    if (i != 11) {
        return false;
    }

    local.baseline = -fubx / local.fx;
    calibration = local;
    return true;
}

namespace cart::sources {

KITTIDataSource::KITTIDataSource(std::string basePath, int sequence, cv::Size imageSize) : KITTIDataSource(basePath + "/sequences/" + addLeadingZeros(sequence, 2), imageSize) {
}

KITTIDataSource::KITTIDataSource(std::string path, cv::Size imageSize) : DataSource(imageSize) {
    this->path = cart::util::resolvePath(path);
    this->currentFrame = 0;

    std::string calibPath = this->path + "/calib.txt";
    std::ifstream inputFile(calibPath, std::ios::in);

    // Check if the file is successfully opened
    if (!inputFile.is_open()) {
        throw std::runtime_error("Failed to open calibration file at " + calibPath + ": " + strerror(errno));
    }

    KITTICameraCalibration leftCalibration;
    KITTICameraCalibration rightCalibration;
    bool left, right;

    std::string line;
    while (getline(inputFile, line)) {
        KITTICameraCalibration calibration;

        if (readLine(line, calibration)) {
            if (calibration.cameraId == LEFT_CAM_ID) {
                leftCalibration = calibration;
                left = true;
            } else if (calibration.cameraId == RIGHT_CAM_ID) {
                rightCalibration = calibration;
                right = true;
            }
        }
    }

    // Close the file
    inputFile.close();

    if (!left || !right) {
        throw std::runtime_error("Failed to read calibration file");
    }

    // A bit hacky, but we need to read the first image to get the image size
    cv::Mat tempImage = cv::imread(LEFT_PATH(this->currentFrame));
    cv::Size actualImageSize = tempImage.size();

    if (this->imageSize.width == 0 || this->imageSize.height == 0) {
        this->imageSize = actualImageSize;
    }

    float scaleWidth = static_cast<float>(this->imageSize.width) / actualImageSize.width;
    float scaleHeight = static_cast<float>(this->imageSize.height) / actualImageSize.height;

    this->intrinsics.Q = cv::Mat::eye(4, 4, CV_32F);

    this->intrinsics.Q.at<float>(0, 3) = -leftCalibration.cx * scaleWidth;
    this->intrinsics.Q.at<float>(1, 3) = -leftCalibration.cy * scaleHeight;
    this->intrinsics.Q.at<float>(2, 2) = 0;
    this->intrinsics.Q.at<float>(2, 3) = leftCalibration.fx * scaleWidth;
    this->intrinsics.Q.at<float>(3, 2) = -1.0 / leftCalibration.baseline;
    this->intrinsics.Q.at<float>(3, 3) = ((leftCalibration.cx - rightCalibration.cx) * scaleWidth / leftCalibration.baseline);
}

DataElementType KITTIDataSource::getProvidedType() {
    return DataElementType::STEREO;
}

boost::shared_ptr<DataElement> KITTIDataSource::getNextInternal(log4cxx::LoggerPtr logger, cv::cuda::Stream& stream) {
    cv::Mat left = cv::imread(LEFT_PATH(this->currentFrame));
    cv::Mat right = cv::imread(RIGHT_PATH(this->currentFrame));

    this->currentFrame++;

    auto element = boost::make_shared<StereoDataElement>();

    element->left.upload(left, stream);
    element->right.upload(right, stream);

    if (this->imageSize.width != left.cols || this->imageSize.height != left.rows) {
        cv::cuda::resize(element->left, element->left, this->imageSize, 0, 0, cv::INTER_LINEAR, stream);
        cv::cuda::resize(element->right, element->right, this->imageSize, 0, 0, cv::INTER_LINEAR, stream);
    }

    return element;
}

bool KITTIDataSource::isNextReady() {
    std::string leftPath = LEFT_PATH(this->currentFrame);

    // Check if the file exists
    struct stat buffer;
    return stat(leftPath.c_str(), &buffer) == 0;
}

bool KITTIDataSource::isFinished() {
    // This source does not implement the realtime simulation. This is possible to do, since the KITTI dataset does
    // provide the frame times. Future work could include this feature.
    return !this->isNextReady();
}
}  // namespace cart::sources