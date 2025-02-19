#ifndef HELPERS_H
#define HELPERS_H

#include "torch/torch.h"
#include <iostream>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60


void print_progress(float percentage,std::string message){
    static int lastlpad = -1;
    int val = (int)(percentage * 100);
    int lpad = (int)(percentage*PBWIDTH);
    if(lastlpad==lpad){
        return;
    }
    lastlpad = lpad;
    int rpad = PBWIDTH - lpad;
    printf("\r%s   %3d%% [\033[32m%.*s%*s\033[0m]", message.c_str(), val, lpad, PBSTR, rpad, "");
    fflush(stdout);
    if(val == 100){
        printf("\n");
    }
}

void save_mask(torch::Tensor mask, const std::string& output_path) {
    mask = mask.detach().cpu().to(torch::kByte);
    
    int height = mask.size(0);
    int width = mask.size(1);

    auto mask_ptr = mask.data_ptr<uint8_t>();

    std::vector<uint8_t> pixels(width * height * 3);

    std::vector<std::array<uint8_t, 3>> colormap = {
        {0, 0, 0},       // Class 0 - Black
        {255, 0, 0},     // Class 1 - Red
        {0, 255, 0},     // Class 2 - Green
        {0, 0, 255},     // Class 3 - Blue
 
    };

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * 3;
            int class_id = mask_ptr[y * width + x];  // Get class index for this pixel
        
            // Use the colormap to set RGB values
            pixels[idx] = colormap[class_id][0];   // R
            pixels[idx + 1] = colormap[class_id][1]; // G
            pixels[idx + 2] = colormap[class_id][2]; // B
        }
    }

    stbi_write_png(output_path.c_str(), width, height, 3, pixels.data(), width * 3);
}



// void save_image(torch::Tensor image, const std::string& output_path) {
//     // Make sure the image is on the CPU and in the correct format (Byte)
//     image = image.detach().cpu();
//     image = image.clamp(0, 1) * 255;
//     image = image.to(torch::kByte);

//     // The image has the shape [batch_size, 1, height, width], so we need to squeeze it to [height, width]
//     image = image.squeeze(0);  // Remove the channel dimension (now [height, width])

//     // Convert the tensor into a vector of pixels (grayscale)
//     std::vector<uint8_t> pixels(image.size(0) * image.size(1));

//     for (int y = 0; y < image.size(0); ++y) {
//         for (int x = 0; x < image.size(1); ++x) {
//             pixels[y * image.size(1) + x] = static_cast<uint8_t>(image[y][x].item<int>());
//         }
//     }

//     // Save the image (using stb_image_write for simplicity)
//     stbi_write_png(output_path.c_str(), image.size(1), image.size(0), 1, pixels.data(), image.size(1));
// }


void save_overlay(torch::Tensor original_image, torch::Tensor mask, const std::string& output_path) {
    original_image = original_image.detach().cpu();
    original_image = original_image * 255;  // Convert to range [0, 255]
    original_image = original_image.clamp(0, 255).to(torch::kByte);  

    int height = original_image.size(1);  
    int width = original_image.size(2);   

    auto original_ptr = original_image.data_ptr<uint8_t>();
    auto mask_ptr = mask.detach().cpu().to(torch::kByte).data_ptr<uint8_t>();  

    std::vector<uint8_t> overlay_pixels(width * height * 3);  // RGB image

    // Colormap for the mask 
    std::vector<std::array<uint8_t, 3>> colormap = {
        {0, 0, 0},       // Class 0 - Black
        {255, 0, 0},     // Class 1 - Red
        {0, 255, 0},     // Class 2 - Green
        {0, 0, 255},     // Class 3 - Blue
    };

    float alpha = 1.0; // Opacity

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * 3;  // RGB index

            // Convert grayscale to RGB by copying the grayscale value to R, G, and B channels
            uint8_t gray_value = original_ptr[y * width + x];

            // Use the colormap for the mask to apply color
            int class_id = mask_ptr[y * width + x];
            auto mask_color = colormap[class_id];

            // Overlay the mask color onto the grayscale image
            // If class_id == 0 --> save original pixel, else apply colormap
            if (class_id == 0) {
                // No mask, just use the grayscale value
                overlay_pixels[idx] = gray_value;   // R
                overlay_pixels[idx + 1] = gray_value; // G
                overlay_pixels[idx + 2] = gray_value; // B
            } else {
                // Apply the colormap 
                overlay_pixels[idx] = std::min((1.0f - alpha) * gray_value + alpha * mask_color[0], 255.0f);   // R
                overlay_pixels[idx + 1] = std::min((1.0f - alpha) * gray_value + alpha * mask_color[1], 255.0f); // G
                overlay_pixels[idx + 2] = std::min((1.0f - alpha) * gray_value + alpha * mask_color[2], 255.0f); // B
            }
        }
    }

    stbi_write_png(output_path.c_str(), width, height, 3, overlay_pixels.data(), width * 3);
}



#endif