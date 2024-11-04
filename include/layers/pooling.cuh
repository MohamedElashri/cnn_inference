#pragma once

#include </usr/include/cudnn.h>
#include "common/tensor.cuh"

namespace unet {

class MaxPool1d {
public:
    MaxPool1d(int kernel_size = 2, int stride = 2);
    ~MaxPool1d();

    // Prevent copying
    MaxPool1d(const MaxPool1d&) = delete;
    MaxPool1d& operator=(const MaxPool1d&) = delete;

    // Allow moving
    MaxPool1d(MaxPool1d&&) noexcept;
    MaxPool1d& operator=(MaxPool1d&&) noexcept;

    // Forward pass
    Tensor<float> forward(const Tensor<float>& input, cudnnHandle_t cudnn_handle);

private:
    int kernel_size_;
    int stride_;
    cudnnPoolingDescriptor_t pooling_desc_ = nullptr;
};

} // namespace unet