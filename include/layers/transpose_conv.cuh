// File: include/layers/transpose_conv.cuh

#pragma once

#include </usr/include/cudnn.h>
#include <memory>
#include "common/tensor.cuh"

namespace unet {

class ConvTranspose1d {
public:
    ConvTranspose1d(int in_channels, int out_channels, 
                    int kernel_size, int stride = 1, int padding = 0);
    ~ConvTranspose1d();

    // Prevent copying
    ConvTranspose1d(const ConvTranspose1d&) = delete;
    ConvTranspose1d& operator=(const ConvTranspose1d&) = delete;

    // Allow moving
    ConvTranspose1d(ConvTranspose1d&&) noexcept = default;
    ConvTranspose1d& operator=(ConvTranspose1d&&) noexcept = default;

    // Forward pass
    Tensor<float> forward(const Tensor<float>& input, cudnnHandle_t cudnn_handle);

    // Load weights
    void loadWeights(const float* weights_data, const float* bias_data);

private:
    void cleanup();

    int in_channels_;
    int out_channels_;
    int kernel_size_;
    int stride_;
    int padding_;

    std::unique_ptr<Tensor<float>> weights_;
    std::unique_ptr<Tensor<float>> bias_;
    
    cudnnConvolutionDescriptor_t conv_desc_ = nullptr;
    cudnnFilterDescriptor_t filter_desc_ = nullptr;
    cudnnTensorDescriptor_t input_desc_ = nullptr;
    cudnnTensorDescriptor_t output_desc_ = nullptr;
};

} // namespace unet
