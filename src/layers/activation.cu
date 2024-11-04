// File: src/layers/activation.cu

#include "layers/activation.cuh"

namespace unet {

namespace {
// CUDA kernels
__global__ void reluKernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = max(0.0f, input[idx]);
    }
}

__global__ void softplusKernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // softplus(x) = log(1 + exp(x))
        // Using a numerically stable version to avoid overflow
        const float x = input[idx];
        if (x > 20.0f) {
            // For large x, softplus(x) ≈ x
            output[idx] = x;
        } else if (x < -20.0f) {
            // For very negative x, softplus(x) ≈ 0
            output[idx] = 0.0f;
        } else {
            output[idx] = log1pf(expf(x));
        }
    }
}
} // anonymous namespace

Tensor<float> ReLU::forward(const Tensor<float>& input) {
    Tensor<float> output(input.dims());
    
    const int n = input.elementsCount();
    const int blockSize = 256;
    const int numBlocks = (n + blockSize - 1) / blockSize;
    
    reluKernel<<<numBlocks, blockSize>>>(
        input.data(),
        output.data(),
        n);
    
    CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor<float> Softplus::forward(const Tensor<float>& input) {
    Tensor<float> output(input.dims());
    
    const int n = input.elementsCount();
    const int blockSize = 256;
    const int numBlocks = (n + blockSize - 1) / blockSize;
    
    softplusKernel<<<numBlocks, blockSize>>>(
        input.data(),
        output.data(),
        n);
    
    CUDA_CHECK(cudaGetLastError());
    return output;
}

} // namespace unet