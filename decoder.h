#ifndef DECODER
#define DECODER

#include <torch/torch.h>
#include <torch/script.h>


struct Decoder: torch::nn::Module{

    torch::nn::Sequential dec1, dec2, dec3, dec4;
    torch::nn::ConvTranspose2d b{nullptr}, dec1_t{nullptr}, dec2_t{nullptr};

    Decoder(int64_t output_channel, int64_t &n);

    torch::Tensor forward(std::vector<torch::Tensor> connections);
};

#endif