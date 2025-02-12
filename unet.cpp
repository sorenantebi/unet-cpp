#include "unet.h"

Unet::Unet(int64_t input_channel, int64_t output_channel, int64_t num_filter){
        
        int64_t n = num_filter; // 16
        // Encoder
        enc1 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channel, n, 3).padding(1)), // 64, 64
            torch::nn::BatchNorm2d(n),
            torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(n, n, 3).padding(1)),
            torch::nn::BatchNorm2d(n),
            torch::nn::ReLU()
        );

        n = n*2; // 32
        enc2 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions((int64_t)(n/2), n, 3).stride(2).padding(1)), // 32, 32
            torch::nn::BatchNorm2d(n),
            torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(n, n, 3).padding(1)),
            torch::nn::BatchNorm2d(n),
            torch::nn::ReLU()
        );

        n = n*2; // 64
        enc3 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions((int64_t)(n/2), n, 3).stride(2).padding(1)), // 16, 16
            torch::nn::BatchNorm2d(n),
            torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(n, n, 3).padding(1)),
            torch::nn::BatchNorm2d(n),
            torch::nn::ReLU()
        );

        n = n*2; // 128
        enc4 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions((int64_t)(n/2), n, 3).stride(2).padding(1)), // 8, 8
            torch::nn::BatchNorm2d(n),
            torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(n, n, 3).padding(1)),
            torch::nn::BatchNorm2d(n),
            torch::nn::ReLU()
        );

        // [bs, 128, 8, 8]
        register_module("enc1", enc1);
        register_module("enc2", enc2);
        register_module("enc3", enc3);
        register_module("enc4", enc4);
        std::cout << n << std::endl;
        // Decoder
        bottom = torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(n, (int64_t)(n/2), 2).stride(2).padding(0));

        // n = (int)(n/2); // 64
        
        // dec1 = torch::nn::Sequential(
        //     torch::nn::Conv2d(torch::nn::Conv2dOptions(n*2, n, 3).padding(1)), // 8, 8
        //     torch::nn::BatchNorm2d(n),
        //     torch::nn::ReLU(),
        //     torch::nn::Conv2d(torch::nn::Conv2dOptions(n, n, 3).padding(1)),
        //     torch::nn::BatchNorm2d(n),
        //     torch::nn::ReLU(),
        //     torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(n, (int64_t)(n/2), 2).stride(2).padding(0))
        // );
        

        register_module("bottom", bottom);
        //register_module("dec1", dec1);



}

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
