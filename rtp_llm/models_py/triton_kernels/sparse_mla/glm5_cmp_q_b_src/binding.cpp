#include <torch/extension.h>

void q_b_proj_h8(const at::Tensor&,
                 const at::Tensor&,
                 const at::Tensor&,
                 const at::Tensor&,
                 const at::Tensor&,
                 const at::Tensor&,
                 const at::Tensor&,
                 const at::Tensor&,
                 bool,
                 bool);
void initialize_q_b_h8();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("q_b_proj_h8", &q_b_proj_h8);
    module.def("initialize", &initialize_q_b_h8);
}
