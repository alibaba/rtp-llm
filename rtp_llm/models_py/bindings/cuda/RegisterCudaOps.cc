#include "rtp_llm/models_py/bindings/RegisterOps.h"
#include "rtp_llm/models_py/bindings/cuda/RegisterBaseBindings.hpp"
#include "rtp_llm/models_py/bindings/cuda/RegisterAttnOpBindings.hpp"
#include "rtp_llm/models_py/bindings/cuda/Bf16GemmOp.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/push_reduce_scatter.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/custom_all_gather.h"

#if defined(ENABLE_FP4)
#include "rtp_llm/models_py/bindings/cuda/kernels/scaled_fp4_quant.h"
#include "rtp_llm/models_py/bindings/cuda/cutlass/cutlass_kernels/fp4_gemm/nvfp4_scaled_mm.h"
#endif

#include "rtp_llm/models_py/bindings/cuda/kernels/scaled_fp8_quant.h"
#include "rtp_llm/models_py/bindings/common/kernels/moe/ep_utils.h"

namespace rtp_llm {

void registerPyModuleOps(py::module& rtp_ops_m) {
    rtp_ops_m.def("custom_all_gather_staging",
                  &custom_all_gather_staging,
                  py::arg("input"),
                  py::arg("output"),
                  py::arg("workspace"),
                  py::arg("counters"),
                  py::arg("workspace_mc_ptr"),
                  py::arg("rank"),
                  py::arg("blocks"),
                  py::arg("threads"));
    rtp_ops_m.def("custom_all_gather_direct",
                  &custom_all_gather_direct,
                  py::arg("input"),
                  py::arg("output"),
                  py::arg("semaphores"),
                  py::arg("output_mc_ptr"),
                  py::arg("semaphore_mc_ptr"),
                  py::arg("rank"),
                  py::arg("blocks"),
                  py::arg("threads"));
    rtp_ops_m.def("custom_all_gather_fp8_staging",
                  &custom_all_gather_fp8_staging,
                  py::arg("values"),
                  py::arg("scales"),
                  py::arg("output_values"),
                  py::arg("output_scales"),
                  py::arg("workspace"),
                  py::arg("counters"),
                  py::arg("workspace_mc_ptr"),
                  py::arg("rank"),
                  py::arg("blocks"),
                  py::arg("threads"));
    rtp_ops_m.def("custom_all_gather_fp8_direct",
                  &custom_all_gather_fp8_direct,
                  py::arg("values"),
                  py::arg("scales"),
                  py::arg("output_values"),
                  py::arg("output_scales"),
                  py::arg("semaphores"),
                  py::arg("output_values_mc_ptr"),
                  py::arg("output_scales_mc_ptr"),
                  py::arg("semaphore_mc_ptr"),
                  py::arg("rank"),
                  py::arg("blocks"),
                  py::arg("threads"));
    rtp_ops_m.def("push_reduce_scatter",
                  &push_reduce_scatter,
                  py::arg("input"),
                  py::arg("output"),
                  py::arg("peers"),
                  py::arg("counters"),
                  py::arg("rank"),
                  py::arg("blocks"),
                  py::arg("threads"));
    rtp_ops_m.def("cublas_gemm_bf16_bf16_fp32",
                  &torch_ext::cublas_gemm_bf16_bf16_fp32,
                  "cuBLAS BF16 x BF16 GEMM with FP32 accumulation and FP32 output",
                  py::arg("input"),
                  py::arg("weight"));

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
