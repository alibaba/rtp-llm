#include "rtp_llm/models_py/bindings/RegisterOps.h"
#include "rtp_llm/models_py/bindings/cuda/RegisterBaseBindings.hpp"
#include "rtp_llm/models_py/bindings/cuda/RegisterAttnOpBindings.hpp"
#include "rtp_llm/models_py/bindings/cuda/Bf16GemmOp.h"
#ifdef RTP_V41_FRONT_GEMM
#include "rtp_llm/models_py/bindings/cuda/V41FrontGemmOp.h"
#include "rtp_llm/models_py/bindings/cuda/V41QBGemmOp.h"
#endif

#if defined(ENABLE_FP4)
#include "rtp_llm/models_py/bindings/cuda/kernels/scaled_fp4_quant.h"
#endif

#include "rtp_llm/models_py/bindings/cuda/kernels/scaled_fp8_quant.h"
#include "rtp_llm/models_py/bindings/cuda/Dsv41DecodeReplayMetadataOp.h"
#include "rtp_llm/models_py/bindings/common/kernels/moe/ep_utils.h"

namespace rtp_llm {

void registerPyModuleOps(py::module& rtp_ops_m) {
#ifdef RTP_V41_FRONT_GEMM
    rtp_ops_m.def("v41_front_gemm", &torch_ext::v41_front_gemm,
                  py::arg("x"), py::arg("a"), py::arg("sa"), py::arg("b"), py::arg("sb"), py::arg("head"));
    rtp_ops_m.def("v41_qb_gemm", &torch_ext::v41_qb_gemm,
                  py::arg("a"), py::arg("sa"), py::arg("b"), py::arg("sb"),
                  py::arg("weights"), py::arg("frequencies"), py::arg("positions"));
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

    rtp_ops_m.def("dsv41_decode_replay_metadata_plan",
                  &rtp_llm::buildDsv41DecodeReplayMetadataPlan,
                  "Build the capture-time fused DSv4.1 decode replay metadata plan",
                  py::arg("block_tables"),
                  py::arg("roles"),
                  py::arg("entries_per_block"),
                  py::arg("tokens_per_block"),
                  py::arg("raw_tokens_per_block"),
                  py::arg("valid_requests"),
                  py::arg("start_pos"),
                  py::arg("position_ids"),
                  py::arg("position_ids_long"),
                  py::arg("decode_seq_start_per_req"),
                  py::arg("slot_mapping_swa"),
                  py::arg("topk_window_idxs"),
                  py::arg("swa_abs_idx"),
                  py::arg("topk_buffer_compressed"),
                  py::arg("swa_topk_length"),
                  py::arg("pool_write_slots"),
                  py::arg("pool_state_slots"),
                  py::arg("pool_block_table_mirrors"),
                  py::arg("swa_slot_outputs"),
                  py::arg("swa_slot_pool_indices"),
                  py::arg("q_len"),
                  py::arg("batch_size"),
                  py::arg("window"),
                  py::arg("max_start"));
    registerBaseCudaBindings(rtp_ops_m);
    registerAttnOpBindings(rtp_ops_m);
}

}  // namespace rtp_llm
