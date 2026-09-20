#include "rtp_llm/models_py/bindings/RegisterOps.h"
#include "rtp_llm/models_py/bindings/cuda/RegisterBaseBindings.hpp"
#include "rtp_llm/models_py/bindings/cuda/RegisterAttnOpBindings.hpp"
#include "rtp_llm/models_py/bindings/cuda/Bf16GemmOp.h"
#ifdef ENABLE_W4A16_SM120
#include "rtp_llm/models_py/bindings/cuda/W4A16GemmOp.h"
#endif

#if defined(ENABLE_FP4)
#include "rtp_llm/models_py/bindings/cuda/kernels/scaled_fp4_quant.h"
#include "rtp_llm/models_py/bindings/cuda/cutlass/cutlass_kernels/fp4_gemm/nvfp4_scaled_mm.h"
#endif

#include "rtp_llm/models_py/bindings/cuda/kernels/scaled_fp8_quant.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/fp8_kv_cache.h"
#ifndef USE_PPU
#include "rtp_llm/models_py/bindings/cuda/kernels/fast_bf16_int8.h"
#endif
#include "rtp_llm/models_py/bindings/common/kernels/moe/ep_utils.h"

namespace rtp_llm {

void registerPyModuleOps(py::module& rtp_ops_m) {
#ifdef ENABLE_W4A16_SM120
    rtp_ops_m.def("w4a16_sm120_hadamard", &w4a16Sm120Hadamard, py::arg("input"), py::arg("scale"));
    rtp_ops_m.def("w4a16_sm120_transform", &w4a16Sm120Transform);
    rtp_ops_m.def("w4a16_sm120_gemm",
                  &w4a16Sm120Gemm,
                  py::arg("input"),
                  py::arg("packed"),
                  py::arg("scales"),
                  py::arg("output"),
                  py::arg("output_size"),
                  py::arg("input_size"),
                  py::arg("split_k") = 0);
#endif
#ifndef USE_PPU
    rtp_ops_m.def("fast_bf16_int8_quantize",
                  &fastBf16Int8Quantize,
                  "Quantize BF16 groups to INT8 with BF16 scales",
                  py::arg("input"),
                  py::arg("output_q"),
                  py::arg("output_s"),
                  py::arg("group_size"));
    rtp_ops_m.def("fast_bf16_int8_dequantize_reduce",
                  &fastBf16Int8DequantizeReduce,
                  "Dequantize INT8 sources and sum in order with BF16 rounding",
                  py::arg("input_q"),
                  py::arg("input_s"),
                  py::arg("output"),
                  py::arg("group_size"));
#endif

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

    rtp_ops_m.def("fused_rope_quantize_and_write_fp8_kv_cache",
                  &fused_rope_quantize_and_write_fp8_kv_cache,
                  "Apply RoPE to packed QKV and dynamically quantize/write paged FP8 K/V",
                  py::arg("qkv"),
                  py::arg("kv_cache"),
                  py::arg("kv_scales"),
                  py::arg("batch_indices"),
                  py::arg("positions"),
                  py::arg("page_indptr"),
                  py::arg("page_indices"),
                  py::arg("num_q_heads"),
                  py::arg("num_kv_heads"),
                  py::arg("kernel_page_size"),
                  py::arg("rope_config"),
                  py::arg("cos_sin_cache") = std::nullopt);

    rtp_ops_m.def("quantize_and_write_fp8_kv_cache",
                  &quantize_and_write_fp8_kv_cache,
                  "Dynamically quantize post-RoPE K/V and write persistent paged FP8 cache rows",
                  py::arg("k"),
                  py::arg("v"),
                  py::arg("kv_cache"),
                  py::arg("kv_scales"),
                  py::arg("target_physical_page_ids"),
                  py::arg("token_offsets"),
                  py::arg("physical_page_size"),
                  py::arg("kernel_page_size"),
                  py::arg("subdivision"));

    rtp_ops_m.def("gather_and_dequantize_fp8_kv_cache",
                  &gather_and_dequantize_fp8_kv_cache,
                  "Gather kernel pages from persistent FP8 KV cache and dequantize with per-row scales",
                  py::arg("kv_cache"),
                  py::arg("kv_scales"),
                  py::arg("source_kernel_page_ids"),
                  py::arg("output"),
                  py::arg("physical_page_size"),
                  py::arg("kernel_page_size"),
                  py::arg("subdivision"));

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
