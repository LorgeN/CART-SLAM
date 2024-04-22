#include "modules/superpixels/visualization.cuh"
#include "utils/cuda.cuh"

#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 32
#define X_BATCH 4
#define Y_BATCH 4

#define PADDING 1
#define SHARED_SIZE (X_BATCH * (2 + THREADS_PER_BLOCK_X)) * (Y_BATCH * (2 + THREADS_PER_BLOCK_Y))
#define LOCAL_INDEX(x, y) SHARED_INDEX(sharedPixelX + x, sharedPixelY + y, 0, PADDING, sharedRowStep)

__global__ void overlayBoundaryVisualization(cv::cuda::PtrStepSz<uint8_t> bgrImage, cv::cuda::PtrStepSz<cart::contour::label_t> labels, cv::cuda::PtrStepSz<uint8_t> out) {
    int x = (blockIdx.x * blockDim.x + threadIdx.x) * X_BATCH;
    int y = (blockIdx.y * blockDim.y + threadIdx.y) * Y_BATCH;

    size_t bgrStep = bgrImage.step / sizeof(uint8_t);
    size_t labelsStep = labels.step / sizeof(cart::contour::label_t);
    size_t outStep = out.step / sizeof(uint8_t);

    for (int i = 0; i < Y_BATCH; i++) {
        for (int j = 0; j < X_BATCH; j++) {
            if (x + j >= labels.cols - 1 || y + i >= labels.rows - 1) {
                continue;
            }

            uint8_t b = bgrImage[INDEX_BGR(x + j, y + i, 0, bgrStep)];
            uint8_t g = bgrImage[INDEX_BGR(x + j, y + i, 0, bgrStep)];
            uint8_t r = bgrImage[INDEX_BGR(x + j, y + i, 0, bgrStep)];

            cart::contour::label_t label = labels[INDEX(x + j, y + i, labelsStep)];
            cart::contour::label_t right = labels[INDEX(x + j + 1, y + i, labelsStep)];
            cart::contour::label_t down = labels[INDEX(x + j, y + i + 1, labelsStep)];

            if (label != right || label != down) {
                b = 0;
                g = 0;
                r = 255;
            }

            out[INDEX_BGR(x + j, y + i, 0, outStep)] = b;
            out[INDEX_BGR(x + j, y + i, 1, outStep)] = g;
            out[INDEX_BGR(x + j, y + i, 2, outStep)] = r;
        }
    }
}

__global__ void computeBoundaries(cv::cuda::PtrStepSz<cart::contour::label_t> labels, cv::cuda::PtrStepSz<uint8_t> out) {
    __shared__ cart::contour::label_t sharedLabels[SHARED_SIZE];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int sharedPixelX = threadIdx.x * X_BATCH;
    int sharedPixelY = threadIdx.y * Y_BATCH;

    int pixelX = x * X_BATCH;
    int pixelY = y * Y_BATCH;

    size_t sharedRowStep = X_BATCH * blockDim.x;
    size_t outStep = out.step / sizeof(uint8_t);

    copyToShared<cart::contour::label_t>(sharedLabels, labels, X_BATCH, Y_BATCH, 1, 1, labels.cols, labels.rows);

    __syncthreads();

    for (int i = 0; i < Y_BATCH; i++) {
        for (int j = 0; j < X_BATCH; j++) {
            if (pixelX + j >= labels.cols || pixelY + i >= labels.rows) {
                continue;
            }

            cart::contour::label_t label = sharedLabels[LOCAL_INDEX(j, i)];

            uint8_t border = false;

            for (int k = -1; k <= 1; k++) {
                for (int l = -1; l <= 1; l++) {
                    if (k == 0 && l == 0) {
                        continue;
                    }

                    cart::contour::label_t neighbor = sharedLabels[LOCAL_INDEX(j + k, i + l)];

                    if (label != neighbor) {
                        border = true;
                        break;
                    }
                }
            }

            out[INDEX(pixelX + j, pixelY + i, outStep)] = border;
        }
    }
}

namespace cart::contour {

void computeBoundaryOverlay(cv::cuda::GpuMat bgrImage, cv::cuda::GpuMat labelImage, cv::cuda::GpuMat &out_boundaryOverlay) {
    assert(bgrImage.type() == CV_8UC3);
    assert(labelImage.type() == cv::DataType<label_t>::type);

    if (out_boundaryOverlay.empty() || out_boundaryOverlay.size() != bgrImage.size()) {
        out_boundaryOverlay.create(bgrImage.size(), CV_8UC3);
    }

    dim3 threadsPerBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    dim3 numBlocks((labelImage.cols + threadsPerBlock.x * X_BATCH - 1) / (threadsPerBlock.x * X_BATCH), (labelImage.rows + threadsPerBlock.y * Y_BATCH - 1) / (threadsPerBlock.y * Y_BATCH));

    cudaStream_t stream;
    CUDA_SAFE_CALL(logger, cudaStreamCreate(&stream));

    overlayBoundaryVisualization<<<numBlocks, threadsPerBlock, 0, stream>>>(bgrImage, labelImage, out_boundaryOverlay);

    CUDA_SAFE_CALL(logger, cudaGetLastError());
    CUDA_SAFE_CALL(logger, cudaStreamSynchronize(stream));
}

void computeBoundaryImage(cv::cuda::GpuMat labelImage, cv::cuda::GpuMat &out_boundaryImage) {
    assert(labelImage.type() == cv::DataType<label_t>::type);

    if (out_boundaryImage.empty() || out_boundaryImage.size() != out_boundaryImage.size()) {
        out_boundaryImage.create(labelImage.size(), CV_8UC1);
    }

    dim3 threadsPerBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    dim3 numBlocks((labelImage.cols + threadsPerBlock.x * X_BATCH - 1) / (threadsPerBlock.x * X_BATCH), (labelImage.rows + threadsPerBlock.y * Y_BATCH - 1) / (threadsPerBlock.y * Y_BATCH));

    cudaStream_t stream;
    CUDA_SAFE_CALL(logger, cudaStreamCreate(&stream));

    computeBoundaries<<<numBlocks, threadsPerBlock, 0, stream>>>(labelImage, out_boundaryImage);

    CUDA_SAFE_CALL(logger, cudaGetLastError());
    CUDA_SAFE_CALL(logger, cudaStreamSynchronize(stream));
}
}  // namespace cart::contour