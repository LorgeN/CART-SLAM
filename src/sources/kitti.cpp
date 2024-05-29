#include "sources/kitti.hpp"

#include <sys/stat.h>

#include <fstream>
#include <iostream>

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

KITTIDataSource::KITTIDataSource(std::string basePath, int sequence) {
    this->path = basePath + "/sequences/" + addLeadingZeros(sequence, 2);
    this->currentFrame = 0;
}

KITTIDataSource::KITTIDataSource(std::string path) {
    this->path = path;
    this->currentFrame = 0;
}

const CameraIntrinsics KITTIDataSource::getCameraIntrinsics() const {
    std::string calibPath = this->path + "/calib.txt";
    std::ifstream inputFile(calibPath);

    // Check if the file is successfully opened
    if (!inputFile.is_open()) {
        throw std::runtime_error("Failed to open calibration file");
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

    CameraIntrinsics intrinsics;
    intrinsics.Q = cv::Mat::eye(4, 4, CV_32F);

    intrinsics.Q.at<float>(0, 3) = -1 * leftCalibration.cx;
    intrinsics.Q.at<float>(1, 3) = -1 * leftCalibration.cy;
    intrinsics.Q.at<float>(2, 2) = 0;
    intrinsics.Q.at<float>(2, 3) = leftCalibration.fx;
    intrinsics.Q.at<float>(3, 2) = -1.0 / leftCalibration.baseline;
    intrinsics.Q.at<float>(3, 3) = ((leftCalibration.cx - rightCalibration.cx) / leftCalibration.baseline);

    return intrinsics;
};

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

    return element;
}

bool KITTIDataSource::hasNext() {
    std::string leftPath = LEFT_PATH(this->currentFrame);

    // Check if the file exists
    struct stat buffer;
    return stat(leftPath.c_str(), &buffer) == 0;
}
}  // namespace cart::sources