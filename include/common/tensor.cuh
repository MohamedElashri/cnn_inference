#pragma once

#include "common/cuda_common.cuh"
#include "common/error_handling.cuh"

#include <memory>
#include <vector>

namespace unet {

template<typename T = float>
class Tensor {
public:
    struct Deleter {
        void operator()(T* ptr) const;
    };

    Tensor() = default;
    explicit Tensor(const std::vector<int>& dims);
    ~Tensor();

    // Copy/move operations
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;

    // Data transfer
    void loadFromHost(const T* host_data);
    void copyToHost(T* host_data) const;

    // Getters
    T* data() const { return data_.get(); }
    const std::vector<int>& dims() const { return dims_; }
    cudnnTensorDescriptor_t tensorDesc() const { return tensor_desc_; }
    size_t elementsCount() const;

private:
    std::unique_ptr<T, Deleter> data_;
    std::vector<int> dims_;
    cudnnTensorDescriptor_t tensor_desc_ = nullptr;
};

} // namespace unet