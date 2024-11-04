// File: src/layers/batchnorm.cu

#include "layers/batchnorm.cuh"
#include <iostream>
#include <sstream>

namespace unet {

namespace {
std::string tensorShapeStr(const Tensor<float>& t) {
    const auto& dims = t.dims();
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << dims[i];
    }
    ss << "]";
    return ss.str();
}
} // anonymous namespace

BatchNorm1d::BatchNorm1d(int num_features, float eps, float momentum)
    : num_features_(num_features)
    , eps_(eps)
    , momentum_(momentum) {
    
    std::cout << "Initializing BatchNorm1d with parameters:" << std::endl;
    std::cout << "  num_features: " << num_features_ << std::endl;
    std::cout << "  eps: " << eps_ << std::endl;
    std::cout << "  momentum: " << momentum_ << std::endl;

    try {
        // Create descriptor for BatchNorm parameters
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&bn_tensor_desc_));
        
        // Initialize tensors for BatchNorm parameters
        // For BatchNorm, these are 1D tensors of size [num_features]
        scale_ = std::make_unique<Tensor<float>>(std::vector<int>{num_features_});
        bias_ = std::make_unique<Tensor<float>>(std::vector<int>{num_features_});
        running_mean_ = std::make_unique<Tensor<float>>(std::vector<int>{num_features_});
        running_var_ = std::make_unique<Tensor<float>>(std::vector<int>{num_features_});

        // Initialize BatchNorm descriptor for 1D input
        // cuDNN expects 4D tensor in NCHW format, where H=1 and W=1 for 1D BatchNorm
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            bn_tensor_desc_,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            1,                  // N (batch size) - will be updated during forward
            num_features_,      // C (number of features)
            1,                  // H (height = 1 for 1D)
            1                  // W (width = 1 for 1D)
        ));

        std::cout << "  Successfully initialized BatchNorm1d" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error initializing BatchNorm1d: " << e.what() << std::endl;
        cleanup();
        throw;
    }
}

BatchNorm1d::~BatchNorm1d() {
    cleanup();
}

void BatchNorm1d::cleanup() {
    if (bn_tensor_desc_) {
        cudnnDestroyTensorDescriptor(bn_tensor_desc_);
        bn_tensor_desc_ = nullptr;
    }
}

Tensor<float> BatchNorm1d::forward(const Tensor<float>& input, cudnnHandle_t cudnn_handle) {
    const auto& in_dims = input.dims();
    std::cout << "BatchNorm1d forward input shape: " << tensorShapeStr(input) << std::endl;

    try {
        // Input is [N, C, L] - we need to handle it as [N, C, 1, L] for cuDNN
        int batch_size = in_dims[0];
        int channels = in_dims[1];
        int length = in_dims[2];

        if (channels != num_features_) {
            throw std::runtime_error("Input channel dimension (" + 
                std::to_string(channels) + ") doesn't match num_features (" + 
                std::to_string(num_features_) + ")");
        }

        // Create output tensor with same shape as input
        Tensor<float> output(in_dims);

        // Create temporary 4D tensor descriptors for input/output
        cudnnTensorDescriptor_t input_desc;
        cudnnTensorDescriptor_t output_desc;
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));

        // Set up 4D tensor descriptors
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            input_desc,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            batch_size,     // N
            channels,       // C
            1,             // H
            length         // W (sequence length)
        ));

        CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            output_desc,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            batch_size,     // N
            channels,       // C
            1,             // H
            length         // W (sequence length)
        ));

        const float alpha = 1.0f;
        const float beta = 0.0f;

        // Perform BatchNorm inference
        CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
            cudnn_handle,
            CUDNN_BATCHNORM_SPATIAL,  // mode for CNN
            &alpha,
            &beta,
            input_desc,
            input.data(),
            output_desc,
            output.data(),
            bn_tensor_desc_,
            scale_->data(),
            bias_->data(),
            running_mean_->data(),
            running_var_->data(),
            eps_
        ));

        // Clean up temporary descriptors
        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroyTensorDescriptor(output_desc);

        return output;

    } catch (const std::exception& e) {
        std::cerr << "Error in BatchNorm1d forward pass: " << e.what() << std::endl;
        throw;
    }
}

void BatchNorm1d::loadWeights(const float* scale, const float* bias,
                             const float* mean, const float* var) {
    try {
        std::cout << "Loading BatchNorm1d weights..." << std::endl;
        scale_->loadFromHost(scale);
        bias_->loadFromHost(bias);
        running_mean_->loadFromHost(mean);
        running_var_->loadFromHost(var);
        std::cout << "Successfully loaded BatchNorm1d weights" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error loading BatchNorm1d weights: " << e.what() << std::endl;
        throw;
    }
}

} // namespace unet