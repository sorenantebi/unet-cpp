#include "unet.h"
#include "dataset.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#include <iostream>


void save_mask(torch::Tensor mask, const std::string& output_path) {
    // Move tensor to CPU and ensure correct type
    mask = mask.detach().cpu().to(torch::kByte);
    
    int height = mask.size(0);
    int width = mask.size(1);

    auto mask_ptr = mask.data_ptr<uint8_t>();

    std::vector<uint8_t> pixels(width * height * 3);

    // Apply a color map similar to Matplotlib's "jet" or "viridis"
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
            pixels[idx] = colormap[class_id][0];   // Red
            pixels[idx + 1] = colormap[class_id][1]; // Green
            pixels[idx + 2] = colormap[class_id][2]; // Blue
        }
    }

    // Save as PNG using stb_image
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
    // Ensure the original image is de-normalized (assuming it's in the range [0, 1])
    original_image = original_image.detach().cpu();
    original_image = original_image * 255;  // Convert to range [0, 255]
    original_image = original_image.clamp(0, 255).to(torch::kByte);  // Convert to uint8

    int height = original_image.size(1);  // Height of the image
    int width = original_image.size(2);   // Width of the image

    auto original_ptr = original_image.data_ptr<uint8_t>();
    auto mask_ptr = mask.detach().cpu().to(torch::kByte).data_ptr<uint8_t>();  // Ensure mask is in uint8 and on CPU

    std::vector<uint8_t> overlay_pixels(width * height * 3);  // RGB image

    // Colormap for the mask (similar to jet/viridis)
    std::vector<std::array<uint8_t, 3>> colormap = {
        {0, 0, 0},       // Class 0 - Black
        {255, 0, 0},     // Class 1 - Red
        {0, 255, 0},     // Class 2 - Green
        {0, 0, 255},     // Class 3 - Blue
    };

    float alpha = 1.0; 

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * 3;  // RGB index

            // Convert grayscale to RGB by copying the grayscale value to R, G, and B channels
            uint8_t gray_value = original_ptr[y * width + x];

            // Use the colormap for the mask to apply color
            int class_id = mask_ptr[y * width + x];
            auto mask_color = colormap[class_id];

            // Overlay the mask color onto the grayscale image
            if (class_id == 0) {
                // No mask, just use the grayscale value
                overlay_pixels[idx] = gray_value;   // Red channel
                overlay_pixels[idx + 1] = gray_value; // Green channel
                overlay_pixels[idx + 2] = gray_value; // Blue channel
            } else {
                // Apply the colormap to the non-black mask pixels
                overlay_pixels[idx] = std::min((1.0f - alpha) * gray_value + alpha * mask_color[0], 255.0f);   // Red channel
                overlay_pixels[idx + 1] = std::min((1.0f - alpha) * gray_value + alpha * mask_color[1], 255.0f); // Green channel
                overlay_pixels[idx + 2] = std::min((1.0f - alpha) * gray_value + alpha * mask_color[2], 255.0f); // Blue channel
            }
        }
    }

    // Save the overlay as PNG using stb_image
    stbi_write_png(output_path.c_str(), width, height, 3, overlay_pixels.data(), width * 3);
}



int main() {
    torch::Device device(torch::kMPS);
    if (!torch::cuda::is_available() && !torch::hasMPS()) {
        std::cerr << "MPS is not available. Running on CPU." << std::endl;
        device = torch::kCPU;
    } else if (torch::cuda::is_available()) {
        device = torch::kCUDA;
    }

    float best_val_loss = std::numeric_limits<float>::max();
    std::string best_model_path = "best_model.pth";
    int num_class = 4;
    int num_epochs = 10;

    //Unet model(1, num_class, 16);
    auto model = std::make_shared<Unet>(1, num_class, 16);
    model->to(device); 
   
    auto optimizer = torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(1e-3));
    auto criterion = torch::nn::CrossEntropyLoss();

    auto train_dataset = ImageSet("/Users/sorenantebi/Desktop/coursework_02/Task01_BrainTumour_2D/training_images", "/Users/sorenantebi/Desktop/coursework_02/Task01_BrainTumour_2D/training_labels", false).map(torch::data::transforms::Stack<>());

    auto test_dataset = ImageSet("/Users/sorenantebi/Desktop/coursework_02/Task01_BrainTumour_2D/test_images", "/Users/sorenantebi/Desktop/coursework_02/Task01_BrainTumour_2D/test_labels", false).map(torch::data::transforms::Stack<>());

    auto train_data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset), torch::data::DataLoaderOptions().batch_size(4)
    );
    
    auto test_data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(test_dataset), torch::data::DataLoaderOptions().batch_size(4)
    );

    std::string masks_dir = "output_masks";
    if (!std::filesystem::exists(masks_dir)) {
        std::filesystem::create_directory(masks_dir);
    }

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        model->train();

        int train_batch_idx = 0;
        int test_batch_idx = 0;
        float train_total_loss = 0.0;
        float test_total_loss = 0.0;

        for (auto& batch : *train_data_loader) {
            
            auto images = batch.data.to(device);
            auto labels = batch.target.to(device);
            

        
            torch::Tensor output = model->forward(images);

            torch::Tensor loss = criterion(output, labels);
            
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            train_total_loss += loss.item<float>();

            train_batch_idx++;
        }

        std::cout << "Epoch [" << (epoch + 1) << "] - Train Loss: " << (train_total_loss / train_batch_idx) << "\n";

        model->eval(); 
        {
            torch::NoGradGuard no_grad;
            for (const auto& batch : *test_data_loader) {
                auto images = batch.data.to(device);
                auto labels = batch.target.to(device);
               
                torch::Tensor output = model->forward(images);
                torch::Tensor loss = criterion(output, labels);

                test_total_loss += loss.item<float>();

                test_batch_idx++;

                // auto output_mask = output.detach().cpu();  
                // auto predicted_class = output_mask.argmax(1);  // Shape: [batch_size, height, width]
                // for (int i = 0; i < images.size(0); i++) {
                //     auto mask = predicted_class[i];  // Shape: [height, width]
                //     std::string mask_filename = masks_dir + "/mask_" + std::to_string(test_batch_idx) + ".png";
            
                //     //save_mask(mask, mask_filename);
                //     save_overlay(images[i], mask, mask_filename);
                // }
            }
            float val_loss = test_total_loss / test_batch_idx;
            std::cout << "Epoch [" << (epoch + 1) << "] - Test Loss: " << (val_loss) << "\n";

            if (val_loss < best_val_loss) {
                best_val_loss = val_loss;
                torch::save(model, best_model_path);
                std::cout << "Best model updated at epoch " << (epoch + 1) << " with val_loss: " << val_loss << "\n";
            }
        }
        // {
        //     torch::NoGradGuard no_grad;
        //     for (const auto& batch : *test_data_loader) {
        //         auto images = batch.data.to(device);

        //         // Save the raw grayscale test image (before any processing for mask generation)
        //         save_image(images[0], "raw_test_image_" + std::to_string(test_batch_idx) + ".png");

        //         test_batch_idx++;
        //     }
        // }
    }

    auto mod = std::make_shared<Unet>(1, num_class, 16);
    
    torch::load(mod, best_model_path);

    mod->to(device);
    torch::NoGradGuard no_grad; 
    mod->eval();
    int test_batch_idx = 0;
    for (auto& batch : *test_data_loader) {
        
        auto images = batch.data.to(device);
        torch::Tensor output = mod->forward(images);
        auto output_mask = output.detach().cpu();  
        auto predicted_class = output_mask.argmax(1);  // Shape: [batch_size, height, width]
        test_batch_idx++;
        for (int i = 0; i < images.size(0); i++) {
            auto mask = predicted_class[i];  // Shape: [height, width]
            std::string mask_filename = masks_dir + "/mask_" + std::to_string(test_batch_idx) + ".png";
    
            //save_mask(mask, mask_filename);
            save_overlay(images[i], mask, mask_filename);
        }
        
    }

    return 0;
}