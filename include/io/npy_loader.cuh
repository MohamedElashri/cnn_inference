// File: include/io/npy_loader.cuh

#pragma once

#include "common/tensor.cuh"
#include <cnpy.h>
#include <string>
#include <unordered_map>

namespace unet {

class WeightLoader {
public:
    explicit WeightLoader(const std::string& npz_path);
    
    template<typename T>
    Tensor<T> loadTensor(const std::string& name);
    
    template<typename T>
    std::unordered_map<std::string, Tensor<T>> loadAllTensors() {
        std::unordered_map<std::string, Tensor<T>> tensors;
        for (const auto& pair : weights_file_) {
            tensors.emplace(pair.first, loadTensor<T>(pair.first));
        }
        return tensors;
    }

private:
    cnpy::npz_t weights_file_;
};

class ValidationDataLoader {
public:
    ValidationDataLoader(const std::string& npy_path, int batch_size);
    
    template<typename T>
    Tensor<T> getNextBatch() {
        if (current_batch_ >= num_batches_) {
            current_batch_ = 0;
        }

        int start_idx = current_batch_ * batch_size_;
        int actual_batch_size = std::min(batch_size_, 
                                      static_cast<int>(total_samples_ - start_idx));

        std::vector<int> batch_dims{actual_batch_size, feature_dim_, seq_length_};
        Tensor<T> batch_tensor(batch_dims);

        const T* data_start = data_file_.data<T>() + start_idx * feature_dim_ * seq_length_;
        batch_tensor.loadFromHost(data_start);

        current_batch_++;
        return batch_tensor;
    }

    int getBatchSize() const { return batch_size_; }
    int getNumBatches() const { return num_batches_; }
    int getFeatureDim() const { return feature_dim_; }
    int getSeqLength() const { return seq_length_; }

private:
    cnpy::NpyArray data_file_;
    int batch_size_;
    int current_batch_ = 0;
    size_t total_samples_;
    int num_batches_;
    int feature_dim_;
    int seq_length_;
};

} // namespace unet