#include "unet.h"

Unet::Unet(int64_t input_channel, int64_t output_channel, int64_t num_filter){
        n = num_filter;
        encoder = register_module("encoder", std::make_shared<Encoder>(input_channel, n));
        decoder = register_module("decoder", std::make_shared<Decoder>(output_channel, n));

}

torch::Tensor Unet::forward(torch::Tensor x) {
      
    auto connections = encoder->forward(x);
    return decoder->forward(connections);
    
}