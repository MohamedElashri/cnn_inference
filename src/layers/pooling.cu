// File: src/layers/pooling.cu

#include "layers/pooling.cuh"
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

MaxPool1d::MaxPool1d(int kernel_size, int stride)
    : kernel_size_(kernel_size)
    , stride_(stride) {
    
    std::cout << "Initializing MaxPool1d with parameters:" << std::endl;
    std::cout << "  kernel_size: " << kernel_size_ << std::endl;
    std::cout << "  stride: " << stride_ << std::endl;

    try {
        // Create pooling descriptor
        CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pooling_desc_));
        
        // For 1D pooling, we use 2D pooling with height=1
        int window_size[] = {1, kernel_size_};  // [H, W]
        int padding[] = {0, 0};                 // No padding
        int stride[] = {1, stride_};            // [H, W]

        // Set up pooling descriptor
        CUDNN_CHECK(cudnnSetPoolingNdDescriptor(
            pooling_desc_,
            CUDNN_POOLING_MAX,           // mode
            CUDNN_NOT_PROPAGATE_NAN,     // maxpooling nan opt
            2,                           // nbDims (2D pooling)
            window_size,                 // windowDimA
            padding,                     // paddingA
            stride                       // strideA
        ));

        std::cout << "  Successfully initialized MaxPool1d" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error in MaxPool1d initialization: " << e.what() << std::endl;
        if (pooling_desc_) {
            cudnnDestroyPoolingDescriptor(pooling_desc_);
            pooling_desc_ = nullptr;
        }
        throw;
    }
}

MaxPool1d::~MaxPool1d() {
    if (pooling_desc_) {
        cudnnDestroyPoolingDescriptor(pooling_desc_);
    }
}

MaxPool1d::MaxPool1d(MaxPool1d&& other) noexcept
    : kernel_size_(other.kernel_size_)
    , stride_(other.stride_)
    , pooling_desc_(other.pooling_desc_) {
    other.pooling_desc_ = nullptr;
}

MaxPool1d& MaxPool1d::operator=(MaxPool1d&& other) noexcept {
    if (this != &other) {
        kernel_size_ = other.kernel_size_;
        stride_ = other.stride_;
        
        if (pooling_desc_) {
            cudnnDestroyPoolingDescriptor(pooling_desc_);
        }
        pooling_desc_ = other.pooling_desc_;
        other.pooling_desc_ = nullptr;
    }
    return *this;
}

Tensor<float> MaxPool1d::forward(const Tensor<float>& input, cudnnHandle_t cudnn_handle) {
    const auto& in_dims = input.dims();
    std::cout << "MaxPool1d forward input shape: " << tensorShapeStr(input) << std::endl;

    try {
        // Handle 1D input as 4D NCHW tensor with H=1
        int batch_size = in_dims[0];
        int channels = in_dims[1];
        int length = in_dims[2];
        
        // Calculate output length
        int output_length = (length - kernel_size_) / stride_ + 1;
        std::cout << "  Expected output length: " << output_length << std::endl;

        // Create temporary descriptors for input/output
        cudnnTensorDescriptor_t input_desc;
        cudnnTensorDescriptor_t output_desc;
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));

        // Set up 4D tensor descriptors (NCHW format)
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
            batch_size,      // N
            channels,        // C
            1,              // H
            output_length   // W
        ));

        // Create output tensor
        Tensor<float> output({batch_size, channels, output_length});

        // Perform pooling
        const float alpha = 1.0f;
        const float beta = 0.0f;

        CUDNN_CHECK(cudnnPoolingForward(
            cudnn_handle,
            pooling_desc_,
            &alpha,
            input_desc,
            input.data(),
            &beta,
            output_desc,
            output.data()
        ));

        // Clean up temporary descriptors
        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroyTensorDescriptor(output_desc);

        std::cout << "  Output shape: " << tensorShapeStr(output) << std::endl;
        return output;

    } catch (const std::exception& e) {
        std::cerr << "Error in MaxPool1d forward pass: " << e.what() << std::endl;
        throw;
    }
}

} // namespace unet