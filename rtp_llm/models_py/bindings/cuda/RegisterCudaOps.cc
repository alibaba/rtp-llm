#include "rtp_llm/models_py/bindings/RegisterOps.h"
#include "rtp_llm/models_py/bindings/cuda/RegisterBaseBindings.hpp"
#include "rtp_llm/models_py/bindings/cuda/RegisterAttnOpBindings.hpp"
#include "rtp_llm/models_py/bindings/cuda/Bf16GemmOp.h"
#ifdef RTP_K3_NATIVE_ATTNRES
#include "rtp_llm/models_py/bindings/cuda/kernels/kimi_k3_attn_res.h"
#endif

#ifdef RTP_K3_NATIVE_RMS_NORM
#include "rtp_llm/models_py/bindings/cuda/kernels/kimi_k3_rms_norm.h"
#endif

#ifdef RTP_K3_NATIVE_ROUTING
#include "rtp_llm/models_py/bindings/cuda/kernels/kimi_k3_topk.h"
#endif

#if defined(ENABLE_FP4)
#include "rtp_llm/models_py/bindings/cuda/kernels/scaled_fp4_quant.h"
#include "rtp_llm/models_py/bindings/cuda/cutlass/cutlass_kernels/fp4_gemm/nvfp4_scaled_mm.h"
#endif

#include "rtp_llm/models_py/bindings/cuda/kernels/scaled_fp8_quant.h"
#include "rtp_llm/models_py/bindings/common/kernels/moe/ep_utils.h"

namespace rtp_llm {

void registerPyModuleOps(py::module& rtp_ops_m) {
#ifdef RTP_K3_NATIVE_RMS_NORM
    rtp_ops_m.def("kimi_k3_rms_norm", &kimi_k3_rms_norm,
                 py::arg("input"), py::arg("weight"), py::arg("epsilon"));
#endif
#ifdef RTP_K3_NATIVE_ROUTING
    rtp_ops_m.def("kimi_k3_grouped_topk", &kimi_k3_grouped_topk,
                 "Native K3 fused sigmoid, grouped top-k and routing normalization",
                 py::arg("scores"), py::arg("bias"), py::arg("n_group"),
                 py::arg("topk_group"), py::arg("topk"), py::arg("renormalize"),
                 py::arg("scale"));
#endif

#ifdef RTP_K3_NATIVE_ATTNRES
    rtp_ops_m.def("kimi_k3_attn_res", &kimi_k3_attn_res,
                 "Native Blackwell K3 AttnRes with optional fused RMSNorm",
                 py::arg("prefix"), py::arg("delta"), py::arg("blocks"),
                 py::arg("norm_weight"), py::arg("qk_weight"),
                 py::arg("output_norm_weight"), py::arg("output"),
                 py::arg("num_blocks"), py::arg("block_write_idx"),
                 py::arg("eps"), py::arg("output_norm_eps"));
#endif

    rtp_ops_m.def("cublas_gemm_bf16_fp32_accum_add",
                  &torch_ext::cublas_gemm_bf16_fp32_accum_add,
                  "BF16 GEMM with FP32 reduction policy and native addmm residual semantics",
                  py::arg("input"), py::arg("weight"), py::arg("residual"));

    rtp_ops_m.def("cublas_gemm_bf16_fp32_accum",
                  &torch_ext::cublas_gemm_bf16_fp32_accum,
                  "BF16 GEMM with FP32 intermediate reductions and BF16 output",
                  py::arg("input"),
                  py::arg("weight"));

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
