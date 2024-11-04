#pragma once

#include <string>
#include "common/tensor.cuh"
#include "common/cuda_common.cuh"
#include "layers/conv.cuh"
#include "layers/conv_block.cuh"
#include "layers/pooling.cuh"
#include "layers/activation.cuh"
#include "layers/transpose_conv.cuh"
#include "io/npy_loader.cuh"

namespace unet {

class UNet {
public:
    UNet(int latent_channels = 8, int n = 64, 
         const std::string& sc_mode = "concat");
    ~UNet();

    // Delete copy constructor and assignment operator
    UNet(const UNet&) = delete;
    UNet& operator=(const UNet&) = delete;

    // Allow moving
    UNet(UNet&&) = default;
    UNet& operator=(UNet&&) = default;

    // Main operations
    Tensor<float> forward(const Tensor<float>& raw_input);
    void loadWeights(const std::string& weights_path);

private:
    // Helper functions
    Tensor<float> combine_features(const Tensor<float>& x, const Tensor<float>& skip);
    Tensor<float> generateMockFCOutput(const Tensor<float>& input);

    // Model parameters (initialized first)
    int latent_channels_;
    int n_;  // base number of channels
    std::string sc_mode_;  // skip connection mode ("concat" or "add")

    // CUDA handles (initialized next)
    cudnnHandle_t cudnn_handle_;
    cublasHandle_t cublas_handle_;

    // Model components (in initialization order)
    ConvBNRelu rcbn1_;         // input -> n channels
    ConvBNRelu rcbn2_;         // n -> n channels
    ConvBNRelu rcbn3_;         // n -> n channels
    MaxPool1d pool_;           // pooling layer (shared)
    ConvTranspose1d up1_;      // n -> n channels
    ConvBNRelu up1_conv_;      // n -> n channels
    ConvTranspose1d up2_;      // 2n -> n or n -> n channels
    ConvBNRelu up2_conv_;      // n -> n channels
    Conv1d out_intermediate_;   // 2n -> n or n -> n channels
    Conv1d outc_;              // n -> 1 channel
    Softplus softplus_;        // activation
};

} // namespace unet