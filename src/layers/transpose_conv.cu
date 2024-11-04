// File: src/layers/transpose_conv.cu

#include "layers/transpose_conv.cuh"
#include <iostream>
#include <sstream>

namespace unet {

namespace {
// Helper function to create a 4D tensor descriptor for cuDNN
void createTensor4dDesc(cudnnTensorDescriptor_t& desc, 
                       int n, int c, int h, int w) {
    if (desc == nullptr) {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
    }
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        n, c, h, w
    ));
}

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

ConvTranspose1d::ConvTranspose1d(int in_channels, int out_channels, 
                                int kernel_size, int stride, int padding)
    : in_channels_(in_channels)
    , out_channels_(out_channels)
    , kernel_size_(kernel_size)
    , stride_(stride)
    , padding_(padding)
    , input_desc_(nullptr)
    , output_desc_(nullptr) {
    
    std::cout << "ConvTranspose1d Parameters:" << std::endl;
    std::cout << "  in_channels: " << in_channels_ << std::endl;
    std::cout << "  out_channels: " << out_channels_ << std::endl;
    std::cout << "  kernel_size: " << kernel_size_ << std::endl;
    std::cout << "  stride: " << stride_ << std::endl;
    std::cout << "  padding: " << padding_ << std::endl;
    
    try {
        // Create descriptors
        CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc_));
        CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc_));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc_));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc_));
        
        // Set convolution descriptor for transposed convolution
        int pad[] = {0, padding_};     // no pad in height, pad in width
        int stride[] = {1, stride_};    // stride=1 in height, specified stride in width
        int dilation[] = {1, 1};        // no dilation

        CUDNN_CHECK(cudnnSetConvolutionNdDescriptor(
            conv_desc_,
            2,                      // 2D convolution (simulate 1D)
            pad,                    // padA
            stride,                 // filterStrideA
            dilation,               // dilationA
            CUDNN_CROSS_CORRELATION,
            CUDNN_DATA_FLOAT));

        // Set filter descriptor [in_channels, out_channels, 1, kernel_size]
        int filter_dims[] = {in_channels_, out_channels_, 1, kernel_size_};
        CUDNN_CHECK(cudnnSetFilterNdDescriptor(
            filter_desc_,
            CUDNN_DATA_FLOAT,
            CUDNN_TENSOR_NCHW,
            4,
            filter_dims));

        // Allocate weights and bias
        weights_ = std::make_unique<Tensor<float>>(
            std::vector<int>{in_channels_, out_channels_, 1, kernel_size_});
        bias_ = std::make_unique<Tensor<float>>(std::vector<int>{1, out_channels_, 1, 1});

    } catch (const std::exception& e) {
        cleanup();
        throw;
    }
}

ConvTranspose1d::~ConvTranspose1d() {
    cleanup();
}

void ConvTranspose1d::cleanup() {
    if (conv_desc_) {
        cudnnDestroyConvolutionDescriptor(conv_desc_);
        conv_desc_ = nullptr;
    }
    if (filter_desc_) {
        cudnnDestroyFilterDescriptor(filter_desc_);
        filter_desc_ = nullptr;
    }
    if (input_desc_) {
        cudnnDestroyTensorDescriptor(input_desc_);
        input_desc_ = nullptr;
    }
    if (output_desc_) {
        cudnnDestroyTensorDescriptor(output_desc_);
        output_desc_ = nullptr;
    }
}

Tensor<float> ConvTranspose1d::forward(const Tensor<float>& input, cudnnHandle_t cudnn_handle) {
    const auto& in_dims = input.dims();
    std::cout << "\nConvTranspose1d forward:" << std::endl;
    std::cout << "  Input shape: " << tensorShapeStr(input) << std::endl;
    std::cout << "  Weight shape: " << tensorShapeStr(*weights_) << std::endl;

    try {
        // Set up 4D dimensions
        int batch_size = in_dims[0];
        int input_length = in_dims[2];
        int output_length = (input_length - 1) * stride_ - 2 * padding_ + kernel_size_;

        // Set up tensor descriptors
        createTensor4dDesc(input_desc_, batch_size, in_channels_, 1, input_length);
        createTensor4dDesc(output_desc_, batch_size, out_channels_, 1, output_length);

        std::cout << "  Expected output length: " << output_length << std::endl;

        // Create output tensor
        Tensor<float> output({batch_size, out_channels_, 1, output_length});

        // Get algorithm
        cudnnConvolutionBwdDataAlgoPerf_t algo_perf;
        int returnedAlgoCount;
        CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm_v7(
            cudnn_handle,
            filter_desc_,
            input_desc_,
            conv_desc_,
            output_desc_,
            1,  // requested algo count
            &returnedAlgoCount,
            &algo_perf));

        // Get and allocate workspace
        size_t workspace_size = 0;
        CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
            cudnn_handle,
            filter_desc_,
            input_desc_,
            conv_desc_,
            output_desc_,
            algo_perf.algo,
            &workspace_size));

        void* workspace = nullptr;
        if (workspace_size > 0) {
            CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
        }

        // Perform convolution backward data (transposed convolution)
        float alpha = 1.0f;
        float beta = 0.0f;

        CUDNN_CHECK(cudnnConvolutionBackwardData(
            cudnn_handle,
            &alpha,
            filter_desc_, weights_->data(),
            input_desc_, input.data(),
            conv_desc_,
            algo_perf.algo,
            workspace, workspace_size,
            &beta,
            output_desc_, output.data()));

        // Add bias
        CUDNN_CHECK(cudnnAddTensor(
            cudnn_handle,
            &alpha,
            bias_->tensorDesc(), bias_->data(),
            &alpha,
            output_desc_, output.data()));

        // Free workspace
        if (workspace) {
            CUDA_CHECK(cudaFree(workspace));
        }

        // Return output reshaped to 3D
        return Tensor<float>({batch_size, out_channels_, output_length});

    } catch (const std::exception& e) {
        std::cerr << "Error in ConvTranspose1d forward pass: " << e.what() << std::endl;
        throw;
    }
}

void ConvTranspose1d::loadWeights(const float* weights_data, const float* bias_data) {
    weights_->loadFromHost(weights_data);
    bias_->loadFromHost(bias_data);
}

} // namespace unet
