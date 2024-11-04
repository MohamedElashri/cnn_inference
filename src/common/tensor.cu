// File: src/common/tensor.cu

#include "common/tensor.cuh"

namespace unet {

template<typename T>
void Tensor<T>::Deleter::operator()(T* ptr) const {
    if (ptr) {
        CUDA_CHECK(cudaFree(ptr));
    }
}

template<typename T>
Tensor<T>::Tensor(const std::vector<int>& dims) : dims_(dims) {
    size_t total_size = elementsCount() * sizeof(T);
    T* raw_ptr;
    CUDA_CHECK(cudaMalloc(&raw_ptr, total_size));
    data_ = std::unique_ptr<T, Deleter>(raw_ptr);
    
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&tensor_desc_));
    if (dims.size() == 4) {
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            tensor_desc_,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            dims[0], dims[1], dims[2], dims[3]));
    }
}

template<typename T>
Tensor<T>::~Tensor() {
    if (tensor_desc_) {
        cudnnDestroyTensorDescriptor(tensor_desc_);
    }
}

template<typename T>
Tensor<T>::Tensor(const Tensor& other) : dims_(other.dims_) {
    size_t total_size = elementsCount() * sizeof(T);
    T* raw_ptr;
    CUDA_CHECK(cudaMalloc(&raw_ptr, total_size));
    data_ = std::unique_ptr<T, Deleter>(raw_ptr);
    CUDA_CHECK(cudaMemcpy(raw_ptr, other.data(), total_size, cudaMemcpyDeviceToDevice));
    
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&tensor_desc_));
    if (dims_.size() == 4) {
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            tensor_desc_,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            dims_[0], dims_[1], dims_[2], dims_[3]));
    }
}

template<typename T>
Tensor<T>::Tensor(Tensor&& other) noexcept 
    : data_(std::move(other.data_)), 
      dims_(std::move(other.dims_)),
      tensor_desc_(other.tensor_desc_) {
    other.tensor_desc_ = nullptr;
}

template<typename T>
Tensor<T>& Tensor<T>::operator=(const Tensor& other) {
    if (this != &other) {
        Tensor tmp(other);
        *this = std::move(tmp);
    }
    return *this;
}

template<typename T>
Tensor<T>& Tensor<T>::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        data_ = std::move(other.data_);
        dims_ = std::move(other.dims_);
        if (tensor_desc_) {
            cudnnDestroyTensorDescriptor(tensor_desc_);
        }
        tensor_desc_ = other.tensor_desc_;
        other.tensor_desc_ = nullptr;
    }
    return *this;
}

template<typename T>
void Tensor<T>::loadFromHost(const T* host_data) {
    CUDA_CHECK(cudaMemcpy(
        data_.get(), 
        host_data, 
        elementsCount() * sizeof(T), 
        cudaMemcpyHostToDevice));
}

template<typename T>
void Tensor<T>::copyToHost(T* host_data) const {
    CUDA_CHECK(cudaMemcpy(
        host_data, 
        data_.get(), 
        elementsCount() * sizeof(T), 
        cudaMemcpyDeviceToHost));
}

template<typename T>
size_t Tensor<T>::elementsCount() const {
    size_t count = 1;
    for (int dim : dims_) {
        count *= dim;
    }
    return count;
}

// Explicit template instantiation
template class Tensor<float>;
template class Tensor<half>;

} // namespace unet