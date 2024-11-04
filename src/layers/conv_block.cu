// File: src/layers/conv_block.cu

#include "layers/conv_block.cuh"
#include <iostream>
#include <stdexcept>

namespace unet {

namespace {
// Define the concatenation kernel at the start of the file
__global__ void concatenateChannelsKernel(
    float* __restrict__ output,
    const float* __restrict__ x,
    const float* __restrict__ y,
    const int batch_size,
    const int x_channels,
    const int y_channels,
    const int seq_length) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * (x_channels + y_channels) * seq_length;
    
    if (idx < total_elements) {
        const int c = (idx / seq_length) % (x_channels + y_channels);
        const int b = idx / ((x_channels + y_channels) * seq_length);
        const int s = idx % seq_length;
        
        if (c < x_channels) {
            output[idx] = x[b * x_channels * seq_length + c * seq_length + s];
        } else {
            const int y_c = c - x_channels;
            output[idx] = y[b * y_channels * seq_length + y_c * seq_length + s];
        }
    }
}
} // anonymous namespace

ConvBNRelu::ConvBNRelu(int in_channels, int out_channels, 
                       int kernel_size, float dropout_p)
    : conv_(in_channels, out_channels, kernel_size, 1, (kernel_size - 1) / 2)
    , bn_(out_channels)
    , relu_()
    , dropout_(dropout_p) {
    
    std::cout << "\nInitializing ConvBNRelu with parameters:" << std::endl;
    std::cout << "  in_channels: " << in_channels << std::endl;
    std::cout << "  out_channels: " << out_channels << std::endl;
    std::cout << "  kernel_size: " << kernel_size << std::endl;
    std::cout << "  dropout_p: " << dropout_p << std::endl;
    
    // Validate parameters
    if (in_channels <= 0) {
        throw std::runtime_error("in_channels must be positive, got: " + 
                               std::to_string(in_channels));
    }
    if (out_channels <= 0) {
        throw std::runtime_error("out_channels must be positive, got: " + 
                               std::to_string(out_channels));
    }
    if (kernel_size <= 0) {
        throw std::runtime_error("kernel_size must be positive, got: " + 
                               std::to_string(kernel_size));
    }
    if (dropout_p < 0.0f || dropout_p > 1.0f) {
        throw std::runtime_error("dropout_p must be between 0 and 1, got: " + 
                               std::to_string(dropout_p));
    }
}

Tensor<float> ConvBNRelu::forward(const Tensor<float>& input, cudnnHandle_t cudnn_handle) {
    auto x = conv_.forward(input, cudnn_handle);
    x = bn_.forward(x, cudnn_handle);
    x = relu_.forward(x);
    x = dropout_.forward(x);
    return x;
}

void ConvBNRelu::loadWeights(const float* conv_weights, const float* conv_bias,
                            const float* bn_scale, const float* bn_bias,
                            const float* bn_mean, const float* bn_var) {
    conv_.loadWeights(conv_weights, conv_bias);
    bn_.loadWeights(bn_scale, bn_bias, bn_mean, bn_var);
}

Tensor<float> combine(const Tensor<float>& x, const Tensor<float>& y, const std::string& mode) {
    const auto& x_dims = x.dims();
    const auto& y_dims = y.dims();
    
    // Check that dimensions match except for channels
    if (x_dims[0] != y_dims[0] || x_dims[2] != y_dims[2]) {
        throw std::runtime_error("Incompatible tensor dimensions for combination");
    }
    
    if (mode == "concat") {
        // Create output tensor with combined channels
        std::vector<int> out_dims = x_dims;
        out_dims[1] += y_dims[1];  // Combine channels
        Tensor<float> output(out_dims);
        
        // Launch kernel to concatenate along channel dimension
        const int batch_size = x_dims[0];
        const int x_channels = x_dims[1];
        const int y_channels = y_dims[1];
        const int seq_length = x_dims[2];
        
        const int blockSize = 256;
        const int total_elements = batch_size * (x_channels + y_channels) * seq_length;
        const int numBlocks = (total_elements + blockSize - 1) / blockSize;
        
        concatenateChannelsKernel<<<numBlocks, blockSize>>>(
            output.data(),
            x.data(),
            y.data(),
            batch_size,
            x_channels,
            y_channels,
            seq_length
        );
        
        CUDA_CHECK(cudaGetLastError());
        return output;
    } 
    else if (mode == "add") {
        // Check that channel dimensions match for addition
        if (x_dims[1] != y_dims[1]) {
            throw std::runtime_error("Channel dimensions must match for addition");
        }
        
        Tensor<float> output(x_dims);
        
        // First copy x to output
        CUDA_CHECK(cudaMemcpy(
            output.data(),
            x.data(),
            x.elementsCount() * sizeof(float),
            cudaMemcpyDeviceToDevice));

        // Then add y using cublasSaxpy
        float alpha = 1.0f;
        CUBLAS_CHECK(cublasSaxpy(
            cublas_handle,
            x.elementsCount(),
            &alpha,
            y.data(), 1,
            output.data(), 1));
        
        return output;
    } 
    else {
        throw std::runtime_error("Invalid combination mode: " + mode);
    }
}

} // namespace unet