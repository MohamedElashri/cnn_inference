// File: src/main.cu

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include "model/unet.cuh"
#include "io/npy_loader.cuh"
#include "common/cuda_common.cuh"

void printUsage(const char* programName) {
    std::cerr << "Usage: " << programName 
              << " -w weights.npz -d validation_data.npy -o output.csv" << std::endl;
}

int main(int argc, char* argv[]) {
    try {
        // Parse command line arguments
        std::string weights_path;
        std::string data_path;
        std::string output_path;

        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "-w" && i + 1 < argc) {
                weights_path = argv[++i];
            } else if (arg == "-d" && i + 1 < argc) {
                data_path = argv[++i];
            } else if (arg == "-o" && i + 1 < argc) {
                output_path = argv[++i];
            }
        }

        if (weights_path.empty() || data_path.empty() || output_path.empty()) {
            printUsage(argv[0]);
            return 1;
        }

        // Initialize CUDA handles
        unet::initializeCudaHandles();

        // Create model
        constexpr int LATENT_CHANNELS = 8;
        constexpr int BASE_CHANNELS = 64;
        unet::UNet model(LATENT_CHANNELS, BASE_CHANNELS);

        // Load weights
        std::cout << "Loading weights from: " << weights_path << std::endl;
        // Pass the path to loadWeights
        model.loadWeights(weights_path);

        // Load validation data
        std::cout << "Loading validation data from: " << data_path << std::endl;
        unet::ValidationDataLoader val_loader(data_path, 32);  // batch size = 32

        // Process data
        const size_t num_batches = val_loader.getNumBatches();
        std::cout << "Processing " << num_batches << " batches..." << std::endl;

        // Open output file
        std::ofstream output_file(output_path);
        if (!output_file) {
            throw std::runtime_error("Failed to open output file: " + output_path);
        }

        // Write header
        output_file << "event_id,bin_values\n";

        for (size_t i = 0; i < num_batches; ++i) {
            auto input_batch = val_loader.getNextBatch<float>();
            auto output_batch = model.forward(input_batch);

            // Save results
            std::vector<float> cpu_output(output_batch.elementsCount());
            output_batch.copyToHost(cpu_output.data());
            
            // Write to file
            output_file << i << ",\"";
            for (size_t j = 0; j < cpu_output.size(); ++j) {
                if (j > 0) output_file << " ";
                output_file << cpu_output[j];
            }
            output_file << "\"\n";

            if ((i + 1) % 10 == 0) {
                std::cout << "\rProcessed " << (i + 1) << "/" << num_batches 
                         << " batches" << std::flush;
            }
        }
        std::cout << "\nDone!" << std::endl;

        // Cleanup
        output_file.close();
        unet::destroyCudaHandles();
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}