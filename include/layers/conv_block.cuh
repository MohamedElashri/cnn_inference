// File: include/layers/conv_block.cuh

#pragma once

#include "layers/conv.cuh"
#include "layers/batchnorm.cuh"
#include "layers/activation.cuh"
#include "layers/dropout.cuh"

namespace unet {

class ConvBNRelu {
public:
    ConvBNRelu(int in_channels, int out_channels, 
               int kernel_size = 3, float dropout_p = 0.0f);

    // Delete default constructor since members need initialization
    ConvBNRelu() = delete;
    
    // Prevent copying
    ConvBNRelu(const ConvBNRelu&) = delete;
    ConvBNRelu& operator=(const ConvBNRelu&) = delete;

    // Allow moving
    ConvBNRelu(ConvBNRelu&&) = default;
    ConvBNRelu& operator=(ConvBNRelu&&) = default;

    // Forward pass
    Tensor<float> forward(const Tensor<float>& input, cudnnHandle_t cudnn_handle);

    // Load weights
    void loadWeights(const float* conv_weights, const float* conv_bias,
                    const float* bn_scale, const float* bn_bias,
                    const float* bn_mean, const float* bn_var);

private:
    Conv1d conv_;
    BatchNorm1d bn_;
    ReLU relu_;
    Dropout dropout_;
};

// Helper function to combine tensors (for skip connections)
Tensor<float> combine(const Tensor<float>& x, const Tensor<float>& y, const std::string& mode="concat");

} // namespace unet