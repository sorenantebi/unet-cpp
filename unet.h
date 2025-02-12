#ifndef U_NET
#define U_NET

#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>



struct Unet : torch::nn::Module {

    torch::nn::Sequential enc1, enc2, enc3, enc4;
    torch::nn::Sequential dec1, dec2, dec3, dec4;
    torch::nn::ConvTranspose2d b{nullptr}, dec1_t{nullptr}, dec2_t{nullptr};
    

    Unet(int64_t input_channel = 1, int64_t output_channel = 1, int64_t num_filter = 16); 

    torch::Tensor forward(torch::Tensor x) {
        auto x1 = enc1->forward(x);
        auto x2 = enc2->forward(x1);
        auto x3 = enc3->forward(x2);
        auto x4 = enc4->forward(x3);

        auto b1 = b->forward(x4);

        auto y1 = dec1->forward(torch::cat({b1, x3}, 1));
        auto t1 = dec1_t->forward(y1);
        auto y2 = dec2->forward(torch::cat({t1, x2}, 1));
        auto t2 = dec2_t->forward(y2);
        auto y3 = dec3->forward(torch::cat({t2, x1}, 1));
      

        return y3;
    }


};




#endif