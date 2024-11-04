#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "/usr/include/cudnn.h"
#include <cublas_v2.h>
#include <string>
#include <sstream>
#include <stdexcept>

namespace unet {

// Global handles for cuDNN and cuBLAS
extern cudnnHandle_t cudnn_handle;
extern cublasHandle_t cublas_handle;

// cuBLAS error checking
inline void checkCublasError(cublasStatus_t status, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::string error = std::string("cuBLAS error at ") + file + ":" + std::to_string(line);
        throw std::runtime_error(error);
    }
}

#define CUBLAS_CHECK(err) checkCublasError(err, __FILE__, __LINE__)

// Custom 1D convolution descriptors
inline cudnnStatus_t cudnnSetConvolution1dDescriptor(
    cudnnConvolutionDescriptor_t conv_desc,
    int pad,
    int stride,
    int dilation,
    int groups,
    cudnnConvolutionMode_t mode,
    cudnnDataType_t data_type) {
    return cudnnSetConvolutionNdDescriptor(
        conv_desc,
        1,          // nbDims
        &pad,       // padA
        &stride,    // filterStrideA
        &dilation,  // dilationA
        mode,
        data_type);
}

inline cudnnStatus_t cudnnGetConvolution1dForwardOutputDim(
    const cudnnConvolutionDescriptor_t conv_desc,
    const cudnnTensorDescriptor_t input_desc,
    const cudnnFilterDescriptor_t filter_desc,
    int* n,
    int* c,
    int* h) {
    int out_dims[3];  // n, c, h
    cudnnStatus_t status = cudnnGetConvolutionNdForwardOutputDim(
        conv_desc,
        input_desc,
        filter_desc,
        3,          // nbDims
        out_dims);  // tensorOuputDimA
    if (status == CUDNN_STATUS_SUCCESS) {
        *n = out_dims[0];
        *c = out_dims[1];
        *h = out_dims[2];
    }
    return status;
}

inline cudnnStatus_t cudnnSetPooling1dDescriptor(
    cudnnPoolingDescriptor_t pooling_desc,
    cudnnPoolingMode_t mode,
    cudnnNanPropagation_t maxpooling_nan_opt,
    int window_size,
    int stride) {
    return cudnnSetPoolingNdDescriptor(
        pooling_desc,
        mode,
        maxpooling_nan_opt,
        1,              // nbDims
        &window_size,   // windowDimA
        nullptr,        // paddingA (no padding)
        &stride);       // strideA
}

// Initialize global handles
inline void initializeCudaHandles() {
    cudnnCreate(&cudnn_handle);
    cublasCreate(&cublas_handle);
}

// Destroy global handles
inline void destroyCudaHandles() {
    cudnnDestroy(cudnn_handle);
    cublasDestroy(cublas_handle);
}

} // namespace unet