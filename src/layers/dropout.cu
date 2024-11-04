// File: src/layers/dropout.cu

#include "layers/dropout.cuh"

namespace unet {

namespace {
__global__ void scaleKernel(float* output, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] *= scale;
    }
}
} // anonymous namespace

Tensor<float> Dropout::forward(const Tensor<float>& input) {
    Tensor<float> output(input.dims());
    const int n = input.elementsCount();

    // In inference mode, dropout just scales the input by (1 - p)
    const float scale = 1.0f - p_;

    // First, copy input to output
    CUDA_CHECK(cudaMemcpy(
        output.data(),
        input.data(),
        n * sizeof(float),
        cudaMemcpyDeviceToDevice
    ));

    // Then scale if dropout probability is non-zero
    if (p_ > 0.0f) {
        const int blockSize = 256;
        const int numBlocks = (n + blockSize - 1) / blockSize;

        scaleKernel<<<numBlocks, blockSize>>>(output.data(), scale, n);
        CUDA_CHECK(cudaGetLastError());
    }

    return output;
}

} // namespace unet