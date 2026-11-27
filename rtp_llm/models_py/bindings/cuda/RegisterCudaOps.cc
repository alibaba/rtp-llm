#include "rtp_llm/models_py/bindings/RegisterOps.h"
#include "rtp_llm/models_py/bindings/cuda/RegisterBaseBindings.hpp"
#include "rtp_llm/models_py/bindings/cuda/RegisterAttnOpBindings.hpp"
#include "rtp_llm/models_py/bindings/cuda/Bf16GemmOp.h"
#include "rtp_llm/models_py/bindings/common/CpuTpStatus.h"

#if defined(ENABLE_FP4)
#include "rtp_llm/models_py/bindings/cuda/kernels/scaled_fp4_quant.h"
#endif

#include "rtp_llm/models_py/bindings/cuda/kernels/dsv4_mhc_post.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/dsv4_prefill_rope_quant.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/scaled_fp8_quant.h"
#include "rtp_llm/models_py/bindings/common/kernels/moe/ep_utils.h"

namespace rtp_llm {

void registerPyModuleOps(py::module& rtp_ops_m) {
    rtp_ops_m.def("cpu_status_size", &cpu_status_size, py::arg("world_size") = 2);
    rtp_ops_m.def(
        "cpu_status_initialize",
        [](uintptr_t pointer, size_t bytes, int world_size) {
            return cpu_status_initialize(reinterpret_cast<void*>(pointer), bytes, world_size);
        },
        py::arg("pointer"),
        py::arg("bytes"),
        py::arg("world_size") = 2);
    rtp_ops_m.def("cpu_status_break",
                  [](uintptr_t pointer) { return cpu_status_break(reinterpret_cast<void*>(pointer)); });
    rtp_ops_m.def(
        "cpu_status_reduce",
        [](uintptr_t pointer, int rank, uint64_t generation, int local_ok, int timeout_ms) {
            return cpu_status_reduce(reinterpret_cast<void*>(pointer), rank, generation, local_ok, timeout_ms);
        },
        py::call_guard<py::gil_scoped_release>());

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

    // NVFP4 mm/quant live in rtp_kernel.nvfp4. Keep the grouped experts
    // quantizers here; they are not part of that port.
#if defined(ENABLE_FP4)
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

    rtp_ops_m.def("dsv4_fused_rmsnorm_rope",
                  &torch_ext::dsv4_fused_rmsnorm_rope,
                  "Fused RMSNorm + partial RoPE",
                  py::arg("x"),
                  py::arg("weight") = py::none(),
                  py::arg("freqs_ri"),
                  py::arg("position_ids") = py::none(),
                  py::arg("out"),
                  py::arg("freq_stride_n"),
                  py::arg("rope_head_dim"),
                  py::arg("eps"),
                  py::arg("inverse") = false);

    rtp_ops_m.def("dsv4_fused_inv_rope_fp8_quant",
                  &torch_ext::dsv4_fused_inv_rope_fp8_quant,
                  "Fused inverse-RoPE + block-128 UE8M0 FP8 quant",
                  py::arg("o"),
                  py::arg("freqs_ri"),
                  py::arg("position_ids") = py::none(),
                  py::arg("fp8_work"),
                  py::arg("scale_work"),
                  py::arg("q_len_per_b"),
                  py::arg("n_groups"),
                  py::arg("heads_per_group"),
                  py::arg("nope_dim"),
                  py::arg("rope_head_dim"),
                  py::arg("quant_group_size"),
                  py::arg("fp8_max"),
                  py::arg("eps"),
                  py::arg("scale_zero_rows"));

    rtp_ops_m.def("dsv4_indexer_q_rope_fp8_fold",
                  &torch_ext::dsv4_indexer_q_rope_fp8_fold,
                  "Fused indexer Q RoPE + per-head FP8 quant + weight fold",
                  py::arg("q_bf16"),
                  py::arg("weights"),
                  py::arg("freqs_ri"),
                  py::arg("q_fp8"),
                  py::arg("w_fold"),
                  py::arg("freq_stride_n"),
                  py::arg("rope_head_dim"),
                  py::arg("fp8_max"));

    rtp_ops_m.def("dsv4_mhc_post_fwd",
                  &torch_ext::dsv4_mhc_post_fwd,
                  "Vectorized DeepSeek-V4 mHC post-residual merge",
                  py::arg("x"),
                  py::arg("residual"),
                  py::arg("post"),
                  py::arg("comb"),
                  py::arg("out"));

    registerBaseCudaBindings(rtp_ops_m);
    registerAttnOpBindings(rtp_ops_m);
}

}  // namespace rtp_llm
