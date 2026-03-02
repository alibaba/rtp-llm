#pragma once

#include "rtp_llm/models_py/bindings/common/RtpNorm.h"
#include "rtp_llm/models_py/bindings/common/RtpEmbeddingLookup.h"
#include "rtp_llm/models_py/bindings/common/FusedQKRmsNorm.h"
#include "rtp_llm/models_py/bindings/common/WriteCacheStoreOp.h"
#include "rtp_llm/models_py/bindings/common/CudaGraphPrefillCopy.h"
#include "rtp_llm/models_py/bindings/cuda/FlashInferOp.h"
#include "rtp_llm/models_py/bindings/cuda/FlashInferMlaParams.h"
#include "rtp_llm/models_py/bindings/cuda/FusedMoEOp.h"
#include "rtp_llm/models_py/bindings/cuda/SelectTopkOp.h"
#include "rtp_llm/models_py/bindings/cuda/GroupTopKOp.h"
// RtpProcessGroup is deprecated, use rtp_llm.distribute.collective_torch instead
// #include "rtp_llm/models_py/bindings/common/RtpProcessGroup.h"
#include "rtp_llm/models_py/bindings/cuda/PerTokenGroupQuantFp8.h"
#include "rtp_llm/models_py/bindings/cuda/MoETopkSoftmax.h"
#include "3rdparty/flashinfer/flashinfer.h"
#include "rtp_llm/models_py/bindings/cuda/TrtFp8QuantOp.h"
#include "rtp_llm/models_py/bindings/cuda/ReuseKVCacheOp.h"
#include "rtp_llm/models_py/bindings/cuda/MlaKMergeOp.h"
#include "rtp_llm/models_py/bindings/cuda/FastTopkOp.h"
#include "rtp_llm/models_py/bindings/cuda/DebugKernelOp.h"
#include "rtp_llm/models_py/bindings/cuda/MlaQuantOp.h"

using namespace rtp_llm;

namespace torch_ext {

void registerBasicCudaOps(py::module& rtp_ops_m) {
    rtp_ops_m.def("debug_kernel",
                  &debugKernel,
                  "Debug kernel to print 2D data blocks from GPU tensor",
                  py::arg("data"),
                  py::arg("start_row"),
                  py::arg("start_col"),
                  py::arg("m"),
                  py::arg("n"),
                  py::arg("row_len"),  // Will use data.sizes()[1] if 0
                  py::arg("info_id"));

    rtp_ops_m.def("write_cache_store",
                  &WriteCacheStoreOp,
                  "WriteCacheStoreOp kernel",
                  py::arg("input_lengths"),
                  py::arg("prefix_lengths"),
                  py::arg("kv_cache_block_id_host"),
                  py::arg("cache_store_member"),
                  py::arg("kv_cache"));

    rtp_ops_m.def("rmsnorm",
                  &rmsnorm,
                  "RMSNorm kernel",
                  py::arg("output"),
                  py::arg("input"),
                  py::arg("weight"),
                  py::arg("eps"),
                  py::arg("cuda_stream") = 0);

    rtp_ops_m.def("fused_add_rmsnorm",
                  &fused_add_rmsnorm,
                  "Fused Add RMSNorm kernel",
                  py::arg("input"),
                  py::arg("residual"),
                  py::arg("weight"),
                  py::arg("eps"),
                  py::arg("cuda_stream") = 0);

    rtp_ops_m.def("silu_and_mul",
                  &silu_and_mul,
                  "SiLU and Multiply kernel",
                  py::arg("output"),
                  py::arg("input"),
                  py::arg("cuda_stream") = 0);

    rtp_ops_m.def("fused_qk_rmsnorm",
                  &FusedQKRMSNorm,
                  "Fused QK RMSNorm kernel",
                  py::arg("IO"),
                  py::arg("q_gamma"),
                  py::arg("k_gamma"),
                  py::arg("layernorm_eps"),
                  py::arg("q_group_num"),
                  py::arg("k_group_num"),
                  py::arg("m"),
                  py::arg("n"),
                  py::arg("norm_size"));

    rtp_ops_m.def("layernorm",
                  &layernorm,
                  "LayerNorm kernel",
                  py::arg("output"),
                  py::arg("input"),
                  py::arg("weight"),
                  py::arg("beta"),
                  py::arg("eps"));

    rtp_ops_m.def("fused_add_layernorm",
                  &fused_add_layernorm,
                  "Fused Add LayerNorm kernel",
                  py::arg("input"),
                  py::arg("residual"),
                  py::arg("bias"),
                  py::arg("weight"),
                  py::arg("beta"),
                  py::arg("eps"));

    rtp_ops_m.def("per_token_group_quant_int8",
                  &per_token_group_quant_int8,
                  "Int8 Gemm Per Token Group",
                  py::arg("input"),
                  py::arg("output_q"),
                  py::arg("output_s"),
                  py::arg("group_size"),
                  py::arg("eps"),
                  py::arg("int8_min"),
                  py::arg("int8_max"),
                  py::arg("scale_ue8m0"));

    rtp_ops_m.def("per_token_group_quant_fp8",
                  &per_token_group_quant_fp8,
                  "Fp8 Gemm Per Token Group",
                  py::arg("input"),
                  py::arg("output_q"),
                  py::arg("output_s"),
                  py::arg("group_size"),
                  py::arg("eps"),
                  py::arg("fp8_min"),
                  py::arg("fp8_max"),
                  py::arg("scale_ue8m0"));

    rtp_ops_m.def("per_token_group_quant_fp8_v2",
                  &per_token_group_quant_fp8_v2,
                  "Fp8 Gemm Per Token Group",
                  py::arg("input"),
                  py::arg("output_q"),
                  py::arg("output_s"),
                  py::arg("group_size"),
                  py::arg("eps"),
                  py::arg("fp8_min"),
                  py::arg("fp8_max"),
                  py::arg("scale_ue8m0"),
                  py::arg("fuse_silu_and_mul"),
                  py::arg("masked_m"));

    rtp_ops_m.def("moe_topk_softmax",
                  &moe_topk_softmax,
                  "MoE Topk Softmax kernel",
                  py::arg("topk_weights"),
                  py::arg("topk_indices"),
                  py::arg("token_expert_indices"),
                  py::arg("gating_output"));

    rtp_ops_m.def(
        "embedding", &embedding, "Embedding lookup kernel", py::arg("output"), py::arg("input"), py::arg("weight"));
    rtp_ops_m.def("embedding_bert",
                  &embeddingBert,
                  "EmbeddingBert lookup kernel",
                  py::arg("output"),
                  py::arg("input"),
                  py::arg("weight"),
                  py::arg("combo_position_ids"),
                  py::arg("position_encoding"),
                  py::arg("combo_tokens_type_ids"),
                  py::arg("token_type_embedding"),
                  py::arg("input_embedding_scalar") = 1.0f);

    rtp_ops_m.def("reuse_kv_cache_indexed_batched",
                  &rtp_llm::ReuseKVCacheIndexedBatched,
                  "Reuse KV cache indexed batched kernel",
                  py::arg("final_compressed_kv"),
                  py::arg("final_k_pe"),
                  py::arg("compressed_kv"),
                  py::arg("k_pe"),
                  py::arg("kv_cache_base"),
                  py::arg("reuse_cache_page_indice"),
                  py::arg("batch_reuse_info_vec"),
                  py::arg("qo_indptr"),
                  py::arg("tokens_per_block"));

    rtp_ops_m.def("mla_k_merge",
                  &rtp_llm::MlaKMerge,
                  "Fused kernel to merge k_nope and k_pe efficiently",
                  py::arg("k_out"),
                  py::arg("k_nope"),
                  py::arg("k_pe"));

    // CUDA Graph Copy Kernel Functions
    rtp_ops_m.def("cuda_graph_copy_small2large",
                  &cuda_graph_copy_small2large,
                  "CUDA Graph copy kernel: Small to Large tensor copy",
                  py::arg("input_tensor"),
                  py::arg("output_tensor"),
                  py::arg("batch_size"),
                  py::arg("max_batch_size"),
                  py::arg("max_seq_len"),
                  py::arg("input_lengths"),
                  py::arg("hidden_size"),
                  py::arg("cu_seq_len"));

    rtp_ops_m.def("cuda_graph_copy_large2small",
                  &cuda_graph_copy_large2small,
                  "CUDA Graph copy kernel: Large to Small tensor copy",
                  py::arg("input_tensor"),
                  py::arg("output_tensor"),
                  py::arg("batch_size"),
                  py::arg("max_batch_size"),
                  py::arg("max_seq_len"),
                  py::arg("input_lengths"),
                  py::arg("hidden_size"),
                  py::arg("cu_seq_len"));

    rtp_ops_m.def("fast_topk_v2",
                  &fast_topk_v2,
                  "Fast TopK v2 kernel",
                  py::arg("score"),
                  py::arg("indices"),
                  py::arg("lengths"),
                  py::arg("row_starts") = py::none());

    rtp_ops_m.def("fast_topk_transform_fused",
                  &fast_topk_transform_fused,
                  "Fast TopK Transform Fused kernel",
                  py::arg("score"),
                  py::arg("lengths"),
                  py::arg("dst_page_table"),
                  py::arg("src_page_table") = py::none(),
                  py::arg("cu_seqlens_q"),
                  py::arg("row_starts") = py::none());

    rtp_ops_m.def("fast_topk_transform_ragged_fused",
                  &fast_topk_transform_ragged_fused,
                  "Fast TopK Transform Ragged Fused kernel",
                  py::arg("score"),
                  py::arg("lengths"),
                  py::arg("topk_indices_ragged"),
                  py::arg("topk_indices_offset"),
                  py::arg("row_starts") = py::none());

    rtp_ops_m.def("indexer_k_quant_and_cache",
                  &indexer_k_quant_and_cache,
                  "Indexer K quantization and cache kernel",
                  py::arg("k"),
                  py::arg("kv_cache"),
                  py::arg("slot_mapping"),
                  py::arg("quant_block_size"),
                  py::arg("scale_fmt"));

    rtp_ops_m.def("cp_gather_indexer_k_quant_cache",
                  &cp_gather_indexer_k_quant_cache,
                  "Gather indexer K quantized cache kernel",
                  py::arg("kv_cache"),
                  py::arg("dst_k"),
                  py::arg("dst_scale"),
                  py::arg("block_table"),
                  py::arg("cu_seq_lens"));

    rtp_ops_m.def("cp_gather_and_upconvert_fp8_kv_cache",
                  &cp_gather_and_upconvert_fp8_kv_cache,
                  "Gather and upconvert FP8 KV cache to BF16 workspace (MLA DeepSeek V3 layout)",
                  py::arg("src_cache"),
                  py::arg("dst_compressed_kv"),
                  py::arg("dst_k_pe"),
                  py::arg("block_table"),
                  py::arg("seq_lens"),
                  py::arg("workspace_starts"),
                  py::arg("batch_size"));

    rtp_ops_m.def("concat_and_cache_mla",
                  &concat_and_cache_mla,
                  "Concat and cache MLA (Multi-Head Latent Attention) kernel",
                  py::arg("kv_c"),
                  py::arg("k_pe"),
                  py::arg("kv_cache"),
                  py::arg("slot_mapping"),
                  py::arg("kv_cache_dtype"),
                  py::arg("scale"));
}

void registerBaseCudaBindings(py::module& rtp_ops_m) {
    registerBasicCudaOps(rtp_ops_m);
    registerFusedMoEOp(rtp_ops_m);
    registerSelectTopkOp(rtp_ops_m);
    registerGroupTopKOp(rtp_ops_m);
    // RtpProcessGroup is deprecated, use rtp_llm.distribute.collective_torch instead
    // registerRtpProcessGroup(rtp_ops_m);
    registerTrtFp8QuantOp(rtp_ops_m);
}

}  // namespace torch_ext
