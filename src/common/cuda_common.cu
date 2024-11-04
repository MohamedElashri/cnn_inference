// File: src/common/cuda_common.cu

#include "common/cuda_common.cuh"

namespace unet {

// Define global handles
cudnnHandle_t cudnn_handle;
cublasHandle_t cublas_handle;

} // namespace unet