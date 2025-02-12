#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>


struct Unet : torch::nn::Module {

    torch::nn::Sequential enc1, enc2, enc3, enc4;
    torch::nn::Sequential dec1, dec2, dec3, dec4;

    Unet(int input_channel = 1, int output_channel = 1, int num_filter = 16) {
        
        int n = num_filter;
        std::cout << input_channel << std::endl;

        enc1 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channel, n, 3).padding(1)),
            torch::nn::BatchNorm2d(n),
            torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(n, n, 3).padding(1)),
            torch::nn::BatchNorm2d(n),
            torch::nn::ReLU()
        );

        register_module("enc1", enc1);
    }

    torch::Tensor forward(torch::Tensor x) {
        return enc1->forward(x);
    }


};


void test_unet() {
    
    torch::Device device(torch::kMPS);
    if (!torch::cuda::is_available() && !torch::hasMPS()) {
        std::cerr << "MPS is not available. Running on CPU." << std::endl;
        device = torch::kCPU;
    }

    Unet model;
    model.to(device); 

    torch::Tensor input = torch::rand({1, 1, 64, 64}, device); 
    torch::Tensor output = model.forward(input);
    
    std::cout << "Output shape: " << output.sizes() << std::endl;
    std::cout << output.device() << std::endl;
}

int main() {
    test_unet();
    return 0;
}
