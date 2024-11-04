// File: src/io/npy_loader.cu

#include "io/npy_loader.cuh"
#include <sstream>

namespace unet {

WeightLoader::WeightLoader(const std::string& npz_path) {
    try {
        weights_file_ = cnpy::npz_load(npz_path);
    } catch (const std::runtime_error& e) {
        std::ostringstream ss;
        ss << "Failed to load weights from " << npz_path << ": " << e.what();
        throw std::runtime_error(ss.str());
    }
}

template<typename T>
Tensor<T> WeightLoader::loadTensor(const std::string& name) {
    if (!weights_file_.count(name)) {
        std::ostringstream ss;
        ss << "Weight '" << name << "' not found in weights file";
        throw std::runtime_error(ss.str());
    }

    cnpy::NpyArray arr = weights_file_[name];
    std::vector<int> dims;
    for (size_t i = 0; i < arr.shape.size(); ++i) {
        dims.push_back(static_cast<int>(arr.shape[i]));
    }

    Tensor<T> tensor(dims);
    tensor.loadFromHost(arr.data<T>());
    return tensor;
}

ValidationDataLoader::ValidationDataLoader(const std::string& npy_path, int batch_size) 
    : batch_size_(batch_size) {
    try {
        data_file_ = cnpy::npy_load(npy_path);
        
        total_samples_ = data_file_.shape[0];
        num_batches_ = (total_samples_ + batch_size - 1) / batch_size;
        feature_dim_ = data_file_.shape[1];
        seq_length_ = data_file_.shape[2];
        
    } catch (const std::runtime_error& e) {
        std::ostringstream ss;
        ss << "Failed to load validation data from " << npy_path << ": " << e.what();
        throw std::runtime_error(ss.str());
    }
}

// Explicit template instantiations
template Tensor<float> WeightLoader::loadTensor<float>(const std::string& name);
template std::unordered_map<std::string, Tensor<float>> WeightLoader::loadAllTensors<float>();
template Tensor<float> ValidationDataLoader::getNextBatch<float>();

} // namespace unet