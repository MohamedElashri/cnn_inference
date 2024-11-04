// File: include/layers/dropout.cuh

#pragma once
#include "common/tensor.cuh"
#include "common/cuda_common.cuh"
#include "common/error_handling.cuh"

namespace unet {

class Dropout {
public:
    explicit Dropout(float p = 0.5) : p_(p) {}
    ~Dropout() = default;

    // Allow copying and moving (no resources to manage)
    Dropout(const Dropout&) = default;
    Dropout& operator=(const Dropout&) = default;
    Dropout(Dropout&&) = default;
    Dropout& operator=(Dropout&&) = default;

    // Forward pass (inference mode)
    Tensor<float> forward(const Tensor<float>& input);

private:
    float p_;  // dropout probability
};

} // namespace unet