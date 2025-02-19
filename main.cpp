#include "unet.h"
#include "dataset.h"
#include "helpers.h"


#include <iostream>

int main() {

    // Device
    torch::Device device(torch::kMPS);
    if (!torch::cuda::is_available() && !torch::hasMPS()) {
        std::cerr << "MPS is not available. Running on CPU." << std::endl;
        device = torch::kCPU;
    } else if (torch::cuda::is_available()) {
        device = torch::kCUDA;
    }

    // Presets
    float best_val_loss = std::numeric_limits<float>::max();

    // Hyperparams
    std::string masks_dir = "output_masks";
    if (!std::filesystem::exists(masks_dir)) {
        std::filesystem::create_directory(masks_dir);
    }
    
    std::string best_model_path = "best_model.pth";
    int num_class = 4;
    int num_epochs = 10;
    size_t batch_size = 4;
    
    // Model
    auto model = std::make_shared<Unet>(1, num_class, 16);
    model->to(device); 
   
    // Optimizer + Criterion
    auto optimizer = torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(1e-3));
    auto criterion = torch::nn::CrossEntropyLoss();

    // Dataset
    auto train_dataset = ImageSet("/Users/sorenantebi/Desktop/coursework_02/Task01_BrainTumour_2D/training_images", "/Users/sorenantebi/Desktop/coursework_02/Task01_BrainTumour_2D/training_labels", false).map(torch::data::transforms::Stack<>());
    auto test_dataset = ImageSet("/Users/sorenantebi/Desktop/coursework_02/Task01_BrainTumour_2D/test_images", "/Users/sorenantebi/Desktop/coursework_02/Task01_BrainTumour_2D/test_labels", false).map(torch::data::transforms::Stack<>());

    // Dataloader
    auto train_data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset), torch::data::DataLoaderOptions().batch_size(batch_size)
    );
    
    auto test_data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(test_dataset), torch::data::DataLoaderOptions().batch_size(batch_size)
    );

    // Train size
    std::optional<size_t> dataset_size_opt = train_dataset.size();

    // Train loop
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
           
            // Progress bar
            std::string message = "Training: Epoch [" + std::to_string(epoch + 1) + "]";
            print_progress(static_cast<float>(train_batch_idx)/(dataset_size_opt.value()/batch_size), message);
        }

        std::cout << "Epoch [" << (epoch + 1) << "] - Train Loss: " << (train_total_loss / train_batch_idx) << "\n";

        // Val loop
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

    // Load best model + save masks on test images
    auto mod = std::make_shared<Unet>(1, num_class, 16);
    torch::load(mod, best_model_path);

    mod->to(device);
    mod->eval();
    {
        torch::NoGradGuard no_grad; 
        int test_batch_idx = 0;
        
        for (auto& batch : *test_data_loader) {
            
            auto images = batch.data.to(device);
            torch::Tensor output = mod->forward(images);

            auto output_mask = output.detach().cpu();  

            // Get top predicted class
            auto predicted_class = output_mask.argmax(1);  // Shape: [batch_size, height, width]

            test_batch_idx++;

            for (int i = 0; i < images.size(0); i++) {
                auto mask = predicted_class[i];  // Shape: [height, width]
                std::string mask_filename = masks_dir + "/mask_" + std::to_string(test_batch_idx) + ".png";
        
                //save_mask(mask, mask_filename);
                save_overlay(images[i], mask, mask_filename);
            }
            
        }
    }

    return 0;
}