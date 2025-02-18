#ifndef U_NET
#define U_NET


#include "encoder.h"
#include "decoder.h"


//#include <pybind11/pybind11.h>

//namespace py = pybind11;


struct Unet : torch::nn::Module {

    torch::nn::ModuleHolder<Encoder> encoder{nullptr};
    torch::nn::ModuleHolder<Decoder> decoder{nullptr};

    int64_t n = 16;

    Unet(int64_t input_channel = 1, int64_t output_channel = 1, int64_t num_filter = 16); 

    torch::Tensor forward(torch::Tensor x);

    
};

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     py::class_<Unet, std::shared_ptr<Unet>>(m, "Unet")
//         .def(py::init<int64_t, int64_t, int64_t>(),
//              py::arg("input_channel") = 1,
//              py::arg("output_channel") = 1,
//              py::arg("num_filter") = 16)
//         .def("forward", &Unet::forward);
// }



#endif