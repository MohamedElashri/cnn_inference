// File: include/common/error_handling.cuh

#pragma once

#include <cuda_runtime.h>
#include "/usr/include/cudnn.h"
#include <stdexcept>
#include <string>
#include <sstream>

namespace unet {

// CUDA error checking
inline void checkCudaError(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        std::ostringstream ss;
        ss << "CUDA error at " << file << ":" << line 
           << ": " << cudaGetErrorString(error);
        throw std::runtime_error(ss.str());
    }
}

// cuDNN error checking
inline void checkCudnnError(cudnnStatus_t status, const char* file, int line) {
    if (status != CUDNN_STATUS_SUCCESS) {
        std::ostringstream ss;
        ss << "cuDNN error at " << file << ":" << line 
           << ": " << cudnnGetErrorString(status);
        throw std::runtime_error(ss.str());
    }
}

#define CUDA_CHECK(err) checkCudaError(err, __FILE__, __LINE__)
#define CUDNN_CHECK(err) checkCudnnError(err, __FILE__, __LINE__)

} // namespace unet