#ifndef U_NET
#define U_NET

#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>



struct Unet : torch::nn::Module {

    torch::nn::Sequential enc1, enc2, enc3, enc4;
    torch::nn::Sequential dec1, dec2, dec3, dec4;
    torch::nn::ConvTranspose2d bottom = nullptr;

    Unet(int64_t input_channel = 1, int64_t output_channel = 1, int64_t num_filter = 16); 

    torch::Tensor forward(torch::Tensor x) {
        auto x1 = enc1->forward(x);
        auto x2 = enc2->forward(x1);
        auto x3 = enc3->forward(x2);
        auto x4 = enc4->forward(x3);

        std::cout << x4.sizes() << std::endl;
        auto m1 = bottom->forward(x4);
        std::cout << "Here" << std::endl;
        //auto y1 = dec1->forward(m1);
        return m1;
    }


};




#endif