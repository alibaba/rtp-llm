#include <torch/library.h>

#include "rtp_llm/models_py/bindings/cuda/kernels/scaled_fp8_quant.h"

// Test-only registration of the production CUDA function. No quantization
// arithmetic or kernel implementation is duplicated in this binding.
TORCH_LIBRARY(ple_component_test, m) {
    m.def("per_tensor_quant_fp8(Tensor input, Tensor(a!) output_q, "
          "Tensor(b!) output_s, bool is_static) -> ()",
          &rtp_llm::per_tensor_quant_fp8);
}
