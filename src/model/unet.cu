// File: src/model/unet.cu

#include "model/unet.cuh"
#include "io/npy_loader.cuh"
#include <stdexcept>
#include <iostream>

namespace unet {

namespace {
__global__ void generateMockFCOutputKernel(
    float* output,
    const float* input_data,
    int batch_size,
    int latent_channels,
    int output_size,
    int input_features) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * latent_channels * output_size;
    
    if (idx < total_elements) {
        // Calculate indices
        const int b = idx / (latent_channels * output_size);        // batch index
        const int c = (idx / output_size) % latent_channels;        // channel index
        const int s = idx % output_size;                            // sequence index
        
        // Access input data
        // Ensure that s % input_features is within the range of input features
        const int input_idx = b * input_features + (s % input_features);
        const float input_value = input_data[input_idx];
        
        // Generate output based on input data and indices
        output[idx] = input_value * (c + 1) * (b + 1); // simple multiplication that depends on input (batch b )
    }
}
} // anonymous namespace    

UNet::UNet(int latent_channels, int n, const std::string& sc_mode)
    : latent_channels_(latent_channels)
    , n_(n)
    , sc_mode_(sc_mode)
    , cudnn_handle_(nullptr)
    , cublas_handle_(nullptr)
    // Initialize downsampling path
    , rcbn1_(latent_channels_, n_, 25)
    , rcbn2_(n_, n_, 7)
    , rcbn3_(n_, n_, 5)
    // Initialize pooling layer
    , pool_(2, 2)
    // Initialize first upsampling path
    , up1_(n_, n_, 2, 2)
    , up1_conv_(n_, n_, 5)
    // Initialize second upsampling path (account for skip connections)
    , up2_((sc_mode == "concat") ? 2 * n_ : n_, n_, 2, 2)
    , up2_conv_(n_, n_, 5)
    // Initialize output layers
    , out_intermediate_((sc_mode == "concat") ? 2 * n_ : n_, n_, 5, 1, 2)
    , outc_(n_, 1, 5, 1, 2)
    , softplus_() {

    std::cout << "\nStarting UNet initialization with parameters:" << std::endl;
    std::cout << "  latent_channels: " << latent_channels_ << std::endl;
    std::cout << "  base channels (n): " << n_ << std::endl;
    std::cout << "  skip connection mode: " << sc_mode_ << std::endl;

    try {
        std::cout << "\nInitializing CUDA handles..." << std::endl;
        CUDNN_CHECK(cudnnCreate(&cudnn_handle_));
        CUBLAS_CHECK(cublasCreate(&cublas_handle_));
        std::cout << "CUDA handles initialized successfully" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error during UNet initialization: " << e.what() << std::endl;

        // Clean up CUDA handles if they were created
        if (cudnn_handle_) {
            cudnnDestroy(cudnn_handle_);
            cudnn_handle_ = nullptr;
        }
        if (cublas_handle_) {
            cublasDestroy(cublas_handle_);
            cublas_handle_ = nullptr;
        }

        throw;
    }
}

UNet::~UNet() {
    if (cudnn_handle_) {
        cudnnDestroy(cudnn_handle_);
    }
    if (cublas_handle_) {
        cublasDestroy(cublas_handle_);
    }
}

Tensor<float> UNet::generateMockFCOutput(const Tensor<float>& input) {
    const auto& in_dims = input.dims();
    const int batch_size = in_dims[0];
    const int input_features = in_dims[1];  // Assuming input is [batch_size, input_features]
    const int output_size = 100;  // Fixed size from the original model
    
    // Create output tensor with shape [batch_size, latent_channels_, output_size]
    std::vector<int> out_dims{batch_size, latent_channels_, output_size};
    Tensor<float> output(out_dims);
    
    // Get pointer to input data on device
    const float* input_data = input.data();
    
    // Launch kernel to generate mock data
    const int blockSize = 256;
    const int total_elements = batch_size * latent_channels_ * output_size;
    const int numBlocks = (total_elements + blockSize - 1) / blockSize;
    
    // Generate data that depends on input
    generateMockFCOutputKernel<<<numBlocks, blockSize>>>(
        output.data(),
        input_data,
        batch_size,
        latent_channels_,
        output_size,
        input_features
    );
    
    CUDA_CHECK(cudaGetLastError());
    return output;
}


Tensor<float> UNet::forward(const Tensor<float>& raw_input) {
    // Generate mock FC output first
    auto input = generateMockFCOutput(raw_input);
    
    std::cout << "Generated mock FC output with shape: [" 
              << input.dims()[0] << ", " 
              << input.dims()[1] << ", " 
              << input.dims()[2] << "]" << std::endl;

    // Downsampling path
    auto x1 = rcbn1_.forward(input, cudnn_handle_);                // Size: 100
    auto x2 = pool_.forward(rcbn2_.forward(x1, cudnn_handle_), cudnn_handle_);   // Size: 50
    auto x = pool_.forward(rcbn3_.forward(x2, cudnn_handle_), cudnn_handle_);    // Size: 25

    // Upsampling path with skip connections
    x = up1_.forward(x, cudnn_handle_);                           // Size: 50
    x = up1_conv_.forward(x, cudnn_handle_);
    x = combine_features(x, x2);                                  // Skip connection 1
    
    x = up2_.forward(x, cudnn_handle_);                          // Size: 100
    x = up2_conv_.forward(x, cudnn_handle_);
    x = combine_features(x, x1);                                  // Skip connection 2

    // Output layers
    x = out_intermediate_.forward(x, cudnn_handle_);
    x = outc_.forward(x, cudnn_handle_);
    x = softplus_.forward(x);

    return x;
}


void UNet::loadWeights(const std::string& weights_path) {
    WeightLoader loader(weights_path);

    try {
        // Load downsampling path weights
        rcbn1_.loadWeights(
            loader.loadTensor<float>("rcbn1.0.weight").data(),
            loader.loadTensor<float>("rcbn1.0.bias").data(),
            loader.loadTensor<float>("rcbn1.1.weight").data(),
            loader.loadTensor<float>("rcbn1.1.bias").data(),
            loader.loadTensor<float>("rcbn1.1.running_mean").data(),
            loader.loadTensor<float>("rcbn1.1.running_var").data()
        );

        rcbn2_.loadWeights(
            loader.loadTensor<float>("rcbn2.0.weight").data(),
            loader.loadTensor<float>("rcbn2.0.bias").data(),
            loader.loadTensor<float>("rcbn2.1.weight").data(),
            loader.loadTensor<float>("rcbn2.1.bias").data(),
            loader.loadTensor<float>("rcbn2.1.running_mean").data(),
            loader.loadTensor<float>("rcbn2.1.running_var").data()
        );

        rcbn3_.loadWeights(
            loader.loadTensor<float>("rcbn3.0.weight").data(),
            loader.loadTensor<float>("rcbn3.0.bias").data(),
            loader.loadTensor<float>("rcbn3.1.weight").data(),
            loader.loadTensor<float>("rcbn3.1.bias").data(),
            loader.loadTensor<float>("rcbn3.1.running_mean").data(),
            loader.loadTensor<float>("rcbn3.1.running_var").data()
        );

        // Load upsampling path weights
        up1_.loadWeights(
            loader.loadTensor<float>("up1.0.weight").data(),
            loader.loadTensor<float>("up1.0.bias").data()
        );

        up1_conv_.loadWeights(
            loader.loadTensor<float>("up1.1.0.weight").data(),
            loader.loadTensor<float>("up1.1.0.bias").data(),
            loader.loadTensor<float>("up1.1.1.weight").data(),
            loader.loadTensor<float>("up1.1.1.bias").data(),
            loader.loadTensor<float>("up1.1.1.running_mean").data(),
            loader.loadTensor<float>("up1.1.1.running_var").data()
        );

        up2_.loadWeights(
            loader.loadTensor<float>("up2.0.weight").data(),
            loader.loadTensor<float>("up2.0.bias").data()
        );

        up2_conv_.loadWeights(
            loader.loadTensor<float>("up2.1.0.weight").data(),
            loader.loadTensor<float>("up2.1.0.bias").data(),
            loader.loadTensor<float>("up2.1.1.weight").data(),
            loader.loadTensor<float>("up2.1.1.bias").data(),
            loader.loadTensor<float>("up2.1.1.running_mean").data(),
            loader.loadTensor<float>("up2.1.1.running_var").data()
        );

        // Load output layer weights
        out_intermediate_.loadWeights(
            loader.loadTensor<float>("out_intermediate.weight").data(),
            loader.loadTensor<float>("out_intermediate.bias").data()
        );

        outc_.loadWeights(
            loader.loadTensor<float>("outc.weight").data(),
            loader.loadTensor<float>("outc.bias").data()
        );

    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load weights: " + std::string(e.what()));
    }
}

Tensor<float> UNet::combine_features(const Tensor<float>& x, const Tensor<float>& skip) {
    return combine(x, skip, sc_mode_);
}

} // namespace unet