#include "rtp_llm/models_py/bindings/RegisterOps.h"
#include "rtp_llm/models_py/bindings/cuda/RegisterBaseBindings.hpp"
#include "rtp_llm/models_py/bindings/cuda/RegisterAttnOpBindings.hpp"
#include "rtp_llm/models_py/bindings/cuda/Bf16GemmOp.h"
#include "rtp_llm/models_py/bindings/cuda/Gemma4AddScaleOp.h"
#include "rtp_llm/models_py/bindings/cuda/Gemma4ExpandKvOp.h"
#include "rtp_llm/models_py/bindings/cuda/Gemma4GegluOp.h"
#include "rtp_llm/models_py/bindings/cuda/Gemma4KvCacheOp.h"
#include "rtp_llm/models_py/bindings/cuda/Gemma4MoeOp.h"
#include "rtp_llm/models_py/bindings/cuda/Gemma4RmsNormOp.h"
#include "rtp_llm/models_py/bindings/cuda/Gemma4RopeOp.h"
#include "rtp_llm/models_py/bindings/cuda/Gemma4SoftmaxOp.h"

#if defined(ENABLE_FP4)
#include "rtp_llm/models_py/bindings/cuda/kernels/scaled_fp4_quant.h"
#include "rtp_llm/models_py/bindings/cuda/cutlass/cutlass_kernels/fp4_gemm/nvfp4_scaled_mm.h"
#endif

#include "rtp_llm/models_py/bindings/cuda/kernels/scaled_fp8_quant.h"
#include "rtp_llm/models_py/bindings/common/kernels/moe/ep_utils.h"

namespace rtp_llm {

void registerPyModuleOps(py::module& rtp_ops_m) {
    rtp_ops_m.def("cublas_gemm_bf16_bf16_fp32",
                  &torch_ext::cublas_gemm_bf16_bf16_fp32,
                  "cuBLAS BF16 x BF16 GEMM with FP32 accumulation and FP32 output",
                  py::arg("input"),
                  py::arg("weight"));

    rtp_ops_m.def("gemma4_gather_rows_bf16",
                  &torch_ext::gemma4_gather_rows_bf16,
                  "Gemma4 BF16 row gather with int32 indices",
                  py::arg("input"),
                  py::arg("indices"));

    rtp_ops_m.def("gemma4_logit_softcap_fp32",
                  &torch_ext::gemma4_logit_softcap_fp32,
                  "Gemma4 FP32 final logit softcap",
                  py::arg("input"),
                  py::arg("cap"));

    rtp_ops_m.def(
        "gemma4_add_bf16", &torch_ext::gemma4_add_bf16, "Gemma4 vectorized BF16 add", py::arg("lhs"), py::arg("rhs"));

    rtp_ops_m.def("gemma4_scale_bf16",
                  &torch_ext::gemma4_scale_bf16,
                  "Gemma4 vectorized BF16 scalar multiply",
                  py::arg("input"),
                  py::arg("scale"));

    rtp_ops_m.def("gemma4_add_scale_bf16",
                  &torch_ext::gemma4_add_scale_bf16,
                  "Gemma4 BF16 residual add followed by scalar multiply",
                  py::arg("residual"),
                  py::arg("hidden"),
                  py::arg("scale"));

    rtp_ops_m.def("gemma4_router_scale_bf16",
                  &torch_ext::gemma4_router_scale_bf16,
                  "Gemma4 BF16 router channel and scalar scaling",
                  py::arg("input"),
                  py::arg("channel_scale"),
                  py::arg("scalar_scale"));

    rtp_ops_m.def("gemma4_expand_kv_heads_8_bf16",
                  &torch_ext::gemma4_expand_kv_heads_8_bf16,
                  "Gemma4 BF16 fused K/V expansion from 2 to 16 heads",
                  py::arg("k"),
                  py::arg("v"));

    rtp_ops_m.def("gemma4_expand_kv_heads_2_bf16",
                  &torch_ext::gemma4_expand_kv_heads_2_bf16,
                  "Gemma4 BF16 fused K/V expansion from 8 to 16 heads",
                  py::arg("k"),
                  py::arg("v"));

    rtp_ops_m.def("gemma4_geglu_tanh_bf16",
                  &torch_ext::gemma4_geglu_tanh_bf16,
                  "Gemma4 BF16 GeGLU tanh activation",
                  py::arg("gate_up"));

    rtp_ops_m.def("gemma4_gather_paged_kv_bf16",
                  &torch_ext::gemma4_gather_paged_kv_bf16,
                  "Gemma4 BF16 paged KV gather with sparse-page validity",
                  py::arg("k_cache"),
                  py::arg("v_cache"),
                  py::arg("page_indices"),
                  py::arg("first_offset"),
                  py::arg("token_count"),
                  py::arg("page_size"));

    rtp_ops_m.def("gemma4_append_swa_kv_cache_bf16",
                  &torch_ext::gemma4_append_swa_kv_cache_bf16,
                  "Gemma4 BF16 SWA paged KV append with NULL-page skip",
                  py::arg("key"),
                  py::arg("value"),
                  py::arg("batch_indices"),
                  py::arg("positions"),
                  py::arg("k_cache"),
                  py::arg("v_cache"),
                  py::arg("page_indices"),
                  py::arg("page_indptr"),
                  py::arg("page_size"));

    rtp_ops_m.def("gemma4_topk_8_bf16",
                  &torch_ext::gemma4_topk_8_bf16,
                  "Gemma4 PyTorch-compatible BF16 top-8 over 128 experts",
                  py::arg("input"));

    rtp_ops_m.def("gemma4_weighted_reorder_bf16",
                  &torch_ext::gemma4_weighted_reorder_bf16,
                  "Gemma4 BF16 expert weighting and inverse permutation",
                  py::arg("expert_output"),
                  py::arg("sorted_weight"),
                  py::arg("inverse_permutation"));

    rtp_ops_m.def("gemma4_gather_sorted_expert_input_bf16",
                  &torch_ext::gemma4_gather_sorted_expert_input_bf16,
                  "Gemma4 BF16 gather by sorted flattened top-k permutation",
                  py::arg("input"),
                  py::arg("permutation"),
                  py::arg("top_k"));

    rtp_ops_m.def("gemma4_prepare_grouped_moe",
                  &torch_ext::gemma4_prepare_grouped_moe,
                  "Gemma4 counting-based grouped MoE permutation and offsets",
                  py::arg("expert_ids"),
                  py::arg("weights"),
                  py::arg("num_experts"));

    rtp_ops_m.def("gemma4_top8_sum_bf16",
                  &torch_ext::gemma4_top8_sum_bf16,
                  "Gemma4 exact ATen-tree BF16 top-8 reduction",
                  py::arg("input"));

    rtp_ops_m.def("gemma4_finalize_router_weights_bf16",
                  &torch_ext::gemma4_finalize_router_weights_bf16,
                  "Gemma4 exact BF16 router top-8 normalization and expert scaling",
                  py::arg("top_weights"),
                  py::arg("top_indices"),
                  py::arg("expert_scales"));

    rtp_ops_m.def("gemma4_qk_bmm_8192_bf16",
                  &torch_ext::gemma4_qk_bmm_8192_bf16,
                  "Gemma4 8K full-attention QK batched GEMM",
                  py::arg("q"),
                  py::arg("k"));

    rtp_ops_m.def("gemma4_qk_bmm_8192_bf16_key_len",
                  &torch_ext::gemma4_qk_bmm_8192_bf16_key_len,
                  "Gemma4 8K full-attention QK batched GEMM with bounded key length",
                  py::arg("q"),
                  py::arg("k"),
                  py::arg("key_len"));

    rtp_ops_m.def("gemma4_pv_bmm_8192_bf16",
                  &torch_ext::gemma4_pv_bmm_8192_bf16,
                  "Gemma4 8K full-attention PV batched GEMM",
                  py::arg("probabilities"),
                  py::arg("v"));

    rtp_ops_m.def("gemma4_pv_bmm_8192_bf16_out",
                  &torch_ext::gemma4_pv_bmm_8192_bf16_out,
                  "Gemma4 8K full-attention PV batched GEMM into caller output",
                  py::arg("probabilities"),
                  py::arg("v"),
                  py::arg("output"));

    rtp_ops_m.def("gemma4_pv_bmm_8192_bf16_out_key_len",
                  &torch_ext::gemma4_pv_bmm_8192_bf16_out_key_len,
                  "Gemma4 8K full-attention PV batched GEMM with bounded key length",
                  py::arg("probabilities"),
                  py::arg("v"),
                  py::arg("output"),
                  py::arg("key_len"));

    rtp_ops_m.def("gemma4_swa_pv_bmm_8192_bf16",
                  &torch_ext::gemma4_swa_pv_bmm_8192_bf16,
                  "Gemma4 8K SWA PV batched GEMM with direct output layout",
                  py::arg("probabilities"),
                  py::arg("v"));

    rtp_ops_m.def("gemma4_swa_pv_bmm_8192_bf16_out",
                  &torch_ext::gemma4_swa_pv_bmm_8192_bf16_out,
                  "Gemma4 8K SWA PV batched GEMM into caller output",
                  py::arg("probabilities"),
                  py::arg("v"),
                  py::arg("output"));

    rtp_ops_m.def("gemma4_softmax_8192_bf16",
                  &torch_ext::gemma4_softmax_8192_bf16,
                  "Gemma4 BF16 softmax for rows of 8192 elements",
                  py::arg("input"),
                  py::arg("query_start") = -1,
                  py::arg("window_left") = -1);

    rtp_ops_m.def("gemma4_rms_square_bf16",
                  &torch_ext::gemma4_rms_square_bf16,
                  "Gemma4 BF16 RMSNorm square stage",
                  py::arg("input"));

    rtp_ops_m.def("gemma4_rms_mean_fp32",
                  &torch_ext::gemma4_rms_mean_fp32,
                  "Gemma4 exact FP32 RMSNorm mean stage",
                  py::arg("input"));

    rtp_ops_m.def("gemma4_rms_inv_fp32",
                  &torch_ext::gemma4_rms_inv_fp32,
                  "Gemma4 exact FP32 RMSNorm mean/add/rsqrt stage",
                  py::arg("input"),
                  py::arg("eps"));

    rtp_ops_m.def("gemma4_rms_apply_bf16",
                  &torch_ext::gemma4_rms_apply_bf16,
                  "Gemma4 BF16 weighted RMSNorm apply stage",
                  py::arg("input"),
                  py::arg("inv_rms"),
                  py::arg("weight"));

    rtp_ops_m.def("gemma4_rms_apply_unweighted_bf16",
                  &torch_ext::gemma4_rms_apply_unweighted_bf16,
                  "Gemma4 BF16 unweighted RMSNorm apply stage",
                  py::arg("input"),
                  py::arg("inv_rms"));

    rtp_ops_m.def("gemma4_rope_cos_sin_bf16",
                  &torch_ext::gemma4_rope_cos_sin_bf16,
                  "Gemma4 BF16 rotary table generation",
                  py::arg("positions"),
                  py::arg("inv_freq"));

    rtp_ops_m.def("gemma4_qk_rope_bf16",
                  &torch_ext::gemma4_qk_rope_bf16,
                  "Gemma4 paired BF16 Q/K rotary embedding",
                  py::arg("q"),
                  py::arg("k"),
                  py::arg("cos"),
                  py::arg("sin"));

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
