#pragma once

#include </usr/include/cudnn.h>
#include "common/tensor.cuh"

namespace unet {

class BatchNorm1d {
public:
    explicit BatchNorm1d(int num_features, float eps = 1e-5, float momentum = 0.1);
    ~BatchNorm1d();

    // Prevent copying
    BatchNorm1d(const BatchNorm1d&) = delete;
    BatchNorm1d& operator=(const BatchNorm1d&) = delete;

    // Allow moving
    BatchNorm1d(BatchNorm1d&&) noexcept = default;
    BatchNorm1d& operator=(BatchNorm1d&&) noexcept = default;

    // Forward pass
    Tensor<float> forward(const Tensor<float>& input, cudnnHandle_t cudnn_handle);

    // Load weights
    void loadWeights(const float* scale, const float* bias,
                    const float* mean, const float* var);

private:
    void cleanup();

    int num_features_;
    float eps_;
    float momentum_;

    std::unique_ptr<Tensor<float>> scale_;      // gamma
    std::unique_ptr<Tensor<float>> bias_;       // beta
    std::unique_ptr<Tensor<float>> running_mean_;
    std::unique_ptr<Tensor<float>> running_var_;

    cudnnTensorDescriptor_t bn_tensor_desc_ = nullptr;
};

} // namespace unet