#ifndef ENCODER
#define ENCODER

#include <torch/torch.h>
#include <torch/script.h>


struct Encoder: torch::nn::Module{

    torch::nn::Sequential enc1, enc2, enc3, enc4;
    Encoder(int64_t input_channel, int64_t &n);

    std::vector<torch::Tensor> forward(torch::Tensor x);
};

#endif