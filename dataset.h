#ifndef IMAGE_SET_H
#define IMAGE_SET_H

#include <torch/torch.h>
#include <vector>
#include <filesystem>

#define STB_IMAGE_IMPLEMENTATION

#include "stb_image.h"

class ImageSet : public torch::data::datasets::Dataset<ImageSet> {
private:
    std::vector<torch::Tensor> images;
    std::vector<torch::Tensor> labels;
    bool predict;
    size_t sz = 0;
    // Function to load an image using stb_image
    torch::Tensor load_image(const std::string& path, bool normalize) {
        int width, height, channels;
        unsigned char* img = stbi_load(path.c_str(), &width, &height, &channels, 1);

        if (!img) {
            throw std::runtime_error("Failed to load image: " + path);
        }

        // Convert to torch tensor (float32 normalized to [0,1])

        torch::Tensor tensor = torch::from_blob(img, {1, height, width}, torch::kU8).to(torch::kFloat);

        if (normalize){
            tensor = tensor /255.0;
        }

        // Free stb_image memory
        stbi_image_free(img);

        return tensor;
    }

public:
    // Constructor
    ImageSet(const std::string& image_path, const std::string& label_path = "", bool predict = false)
        : predict(predict) {
        namespace fs = std::filesystem;
        std::vector<std::string> image_files;

        for (const auto& entry : fs::directory_iterator(image_path)) {
            image_files.push_back(entry.path().string());
        }

        // Sort filenames to ensure matching order for images and labels
        std::sort(image_files.begin(), image_files.end());

        for (const auto& image_name : image_files) {
            torch::Tensor image = load_image(image_name, true);

            // Get original dimensions
            int height = image.size(1);
            int width = image.size(2);

            // Calculate padding to the nearest multiple of 32
            int pad_h = (8 - (height % 8)) % 8;
            int pad_w = (8 - (width % 8)) % 8;

            // Split padding equally on both sides
            int pad_top = pad_h / 2, pad_bottom = pad_h - pad_top;
            int pad_left = pad_w / 2, pad_right = pad_w - pad_left;

            // Pad image
            image = torch::nn::functional::pad(image, 
                torch::nn::functional::PadFuncOptions({pad_left, pad_right, pad_top, pad_bottom}).value(0));

            images.push_back(image);

            // Load corresponding label if not in deploy mode
            if (!predict) {
                std::string label_name = label_path + "/" + fs::path(image_name).filename().string();
                torch::Tensor label = load_image(label_name, false);

                // Apply the same padding
                label = torch::nn::functional::pad(label, 
                    torch::nn::functional::PadFuncOptions({pad_left, pad_right, pad_top, pad_bottom}).value(0));

                labels.push_back(label);
            }
            sz = images.size();
           
        }
    }

    // Get dataset size
    std::optional<size_t> size() const override {
        return sz;
    }

    // Get an item (ensures consistent batch-like structure)
    torch::data::Example<> get(size_t index) override {
        
        if (predict) {
            torch::Tensor dummy = torch::empty(1);
            return {images[index], dummy};
            
        }

        return {images[index], labels[index].squeeze(0)};
    }

    
};

#endif
