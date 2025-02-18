#include "encoder.h"

Encoder::Encoder(int64_t input_channel, int64_t &n) {
    
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

    register_module("enc1", enc1);
    register_module("enc2", enc2);
    register_module("enc3", enc3);
    register_module("enc4", enc4);
}

std::vector<torch::Tensor> Encoder::forward(torch::Tensor x){
    std::vector<torch::Tensor> connections;

    auto x1 = enc1->forward(x);
    connections.push_back(x1);
    auto x2 = enc2->forward(x1);
    connections.push_back(x2);
    auto x3 = enc3->forward(x2);
    connections.push_back(x3);
    auto x4 = enc4->forward(x3);
    connections.push_back(x4);

    return connections;
}