#include "rtp_llm/models_py/bindings/RegisterOps.h"
#include "rtp_llm/models_py/bindings/cuda/RegisterBaseBindings.hpp"
#include "rtp_llm/models_py/bindings/cuda/RegisterAttnOpBindings.hpp"

#if defined(ENABLE_FP4)
#include "rtp_llm/models_py/bindings/cuda/kernels/scaled_fp4_quant.h"
#include "rtp_llm/models_py/bindings/cuda/cutlass/cutlass_kernels/fp4_gemm/nvfp4_scaled_mm.h"
#endif

#include "rtp_llm/models_py/bindings/cuda/kernels/scaled_fp8_quant.h"
#include "rtp_llm/models_py/bindings/common/kernels/moe/ep_utils.h"

namespace rtp_llm {

void registerPyModuleOps(py::module& rtp_ops_m) {
    rtp_ops_m.def("per_tensor_quant_fp8",
                  &per_tensor_quant_fp8,
                  py::arg("input"),
                  py::arg("output_q"),
                  py::arg("output_s"),
                  py::arg("is_static"));

    rtp_ops_m.def(
        "per_token_quant_fp8", &per_token_quant_fp8, py::arg("input"), py::arg("output_q"), py::arg("output_s"));

    // Only available when compiling device code for >= sm100.
#if defined(ENABLE_FP4)
    rtp_ops_m.def("cutlass_scaled_fp4_mm",
                  &cutlass_scaled_fp4_mm_sm100a_sm120a,
                  py::arg("out"),
                  py::arg("a"),
                  py::arg("b"),
                  py::arg("a_sf"),
                  py::arg("b_sf"),
                  py::arg("alpha"));

    rtp_ops_m.def("scaled_fp4_quant",
                  &scaled_fp4_quant_sm100a_sm120a,
                  py::arg("output"),
                  py::arg("input"),
                  py::arg("output_sf"),
                  py::arg("input_sf"));

    rtp_ops_m.def("scaled_fp4_experts_quant",
                  &scaled_fp4_experts_quant_sm100a,
                  py::arg("output"),
                  py::arg("output_scale"),
                  py::arg("input"),
                  py::arg("input_global_scale"),
                  py::arg("input_offset_by_experts"),
                  py::arg("output_scale_offset_by_experts"));

    rtp_ops_m.def("silu_and_mul_scaled_fp4_experts_quant",
                  &silu_and_mul_scaled_fp4_experts_quant_sm100a,
                  py::arg("output"),
                  py::arg("output_scale"),
                  py::arg("input"),
                  py::arg("input_global_scale"),
                  py::arg("mask"),
                  py::arg("use_silu_and_mul"));
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
