// File: include/layers/activation.cuh

#pragma once

#include "common/tensor.cuh"

namespace unet {

class ReLU {
public:
    ReLU() = default;
    ~ReLU() = default;

    // No need for copy/move since we have no resources to manage
    ReLU(const ReLU&) = default;
    ReLU& operator=(const ReLU&) = default;
    ReLU(ReLU&&) = default;
    ReLU& operator=(ReLU&&) = default;

    // Forward pass implementation
    Tensor<float> forward(const Tensor<float>& input);
};

class Softplus {
public:
    Softplus() = default;
    ~Softplus() = default;

    // No need for copy/move since we have no resources to manage
    Softplus(const Softplus&) = default;
    Softplus& operator=(const Softplus&) = default;
    Softplus(Softplus&&) = default;
    Softplus& operator=(Softplus&&) = default;

    // Forward pass implementation
    Tensor<float> forward(const Tensor<float>& input);
};

} // namespace unet