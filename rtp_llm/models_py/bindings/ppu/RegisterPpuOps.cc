#include "rtp_llm/models_py/bindings/RegisterOps.h"
#include "rtp_llm/models_py/bindings/ppu/PpuSiluMulMxfp4Op.h"
#include "rtp_llm/models_py/bindings/cuda/RegisterBaseBindings.hpp"
#include "rtp_llm/models_py/bindings/cuda/RegisterAttnOpBindings.hpp"
#ifdef ENABLE_FP8
#include "rtp_llm/models_py/bindings/cuda/kernels/scaled_fp8_quant.h"
#endif
#include "rtp_llm/models_py/bindings/common/kernels/moe/ep_utils.h"

namespace rtp_llm {

#ifndef ENABLE_FP8
// PPU stubs for FP8 quant functions (needed by fp8_kernel.py import chain)
static void
per_tensor_quant_fp8_stub(torch::Tensor input, torch::Tensor output_q, torch::Tensor output_s, bool is_static) {
    throw std::runtime_error("per_tensor_quant_fp8 not available on PPU (ENABLE_FP8 not defined)");
}
static void per_token_quant_fp8_stub(torch::Tensor input, torch::Tensor output_q, torch::Tensor output_s) {
    throw std::runtime_error("per_token_quant_fp8 not available on PPU (ENABLE_FP8 not defined)");
}
#endif

void registerPyModuleOps(py::module& rtp_ops_m) {
#ifdef USE_PPU
    rtp_ops_m.def("ppu_silu_and_mul_post_quant_mxfp4",
                  [](torch::Tensor gate_up, py::object swiglu_limit) {
                      const bool apply_swiglu_limit = !swiglu_limit.is_none();
                      const double limit = apply_swiglu_limit ? swiglu_limit.cast<double>() : 0.0;
                      return rtp_llm::PpuSiluAndMulPostQuantMxfp4(
                          gate_up, limit, apply_swiglu_limit);
                  },
                  "PPU fused SwiGLU and compact MXFP4 quantization",
                  py::arg("gate_up"),
                  py::arg("swiglu_limit") = py::none());
#endif

#ifdef ENABLE_FP8
    rtp_ops_m.def("per_tensor_quant_fp8",
                  &per_tensor_quant_fp8,
                  py::arg("input"),
                  py::arg("output_q"),
                  py::arg("output_s"),
                  py::arg("is_static"));

    rtp_ops_m.def(
        "per_token_quant_fp8", &per_token_quant_fp8, py::arg("input"), py::arg("output_q"), py::arg("output_s"));
#else
    rtp_ops_m.def("per_tensor_quant_fp8",
                  &per_tensor_quant_fp8_stub,
                  py::arg("input"),
                  py::arg("output_q"),
                  py::arg("output_s"),
                  py::arg("is_static"));

    rtp_ops_m.def(
        "per_token_quant_fp8", &per_token_quant_fp8_stub, py::arg("input"), py::arg("output_q"), py::arg("output_s"));
#endif

    rtp_ops_m.def("moe_pre_reorder",
                  &moe_pre_reorder,
                  "moe ep permute kernel",
                  py::arg("input"),
                  py::arg("topk_ids"),
                  py::arg("token_expert_indices"),
                  py::arg("expert_map") = py::none(),
                  py::arg("n_expert"),
                  py::arg("n_local_expert"),
                  py::arg("topk"),
                  py::arg("align_block_size") = py::none(),
                  py::arg("permuted_input"),
                  py::arg("expert_first_token_offset"),
                  py::arg("inv_permuted_idx"),
                  py::arg("permuted_idx"));

    rtp_ops_m.def("moe_post_reorder",
                  &moe_post_reorder,
                  "moe ep unpermute kernel",
                  py::arg("permuted_hidden_states"),
                  py::arg("topk_weights"),
                  py::arg("inv_permuted_idx"),
                  py::arg("expert_first_token_offset") = py::none(),
                  py::arg("topk"),
                  py::arg("hidden_states"));

    registerBaseCudaBindings(rtp_ops_m);
    registerAttnOpBindings(rtp_ops_m);
}

}  // namespace rtp_llm
