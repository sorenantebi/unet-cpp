#include "decoder.h"


Decoder::Decoder(int64_t output_channel, int64_t &n){
    b = torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(n, (int64_t)(n/2), 2).stride(2).padding(0));

    // Decoder
    n = (int64_t)(n/2); // 64
    
    dec1 = torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(n*2, n, 3).padding(1)), 
        torch::nn::BatchNorm2d(n),
        torch::nn::ReLU(),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(n, n, 3).padding(1)),
        torch::nn::BatchNorm2d(n),
        torch::nn::ReLU()
        
    );

    // [1, 32, 32, 32]
    dec1_t = torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(n, (int64_t)(n/2), 2).stride(2).padding(0));
    
    n = (int64_t)(n/2); //32

    dec2 = torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(n*2, n, 3).padding(1)), 
        torch::nn::BatchNorm2d(n),
        torch::nn::ReLU(),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(n, n, 3).padding(1)),
        torch::nn::BatchNorm2d(n),
        torch::nn::ReLU()
        
    );

    // [1, 16, 64, 64]
    dec2_t = torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(n, (int64_t)(n/2), 2).stride(2).padding(0));

    n = (int64_t)(n/2); // 16

    dec3 = torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(n*2, n, 3).padding(1)), 
        torch::nn::BatchNorm2d(n),
        torch::nn::ReLU(),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(n, n, 3).padding(1)),
        torch::nn::BatchNorm2d(n),
        torch::nn::ReLU(),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(n, output_channel, 1).stride(1))
    );

    // --> [1, 1, 64, 64]

    register_module("bottom", b);

    register_module("dec1", dec1);
    register_module("dec1_t", dec1_t);
    register_module("dec2", dec2);
    register_module("dec2_t", dec2_t);
    register_module("dec3", dec3);

}

torch::Tensor Decoder::forward(std::vector<torch::Tensor> connections){

    auto b1 = b->forward(connections[3]);

    auto y1 = dec1->forward(torch::cat({b1, connections[2]}, 1));
    auto t1 = dec1_t->forward(y1);
    auto y2 = dec2->forward(torch::cat({t1, connections[1]}, 1));
    auto t2 = dec2_t->forward(y2);
    auto y3 = dec3->forward(torch::cat({t2, connections[0]}, 1));
    

    return y3;
}