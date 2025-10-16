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
#include "rtp_llm/models_py/bindings/common/RtpProcessGroup.h"
#include "rtp_llm/models_py/bindings/cuda/PerTokenGroupQuantFp8.h"
#include "rtp_llm/models_py/bindings/cuda/MoETopkSoftmax.h"
#include "3rdparty/flashinfer/flashinfer.h"
#include "rtp_llm/models_py/bindings/cuda/TrtFp8QuantOp.h"

using namespace rtp_llm;

namespace torch_ext {

void registerBasicCudaOps(py::module& rtp_ops_m) {
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
}

void registerBaseCudaBindings(py::module& rtp_ops_m) {
    registerBasicCudaOps(rtp_ops_m);
    registerFusedMoEOp(rtp_ops_m);
    registerSelectTopkOp(rtp_ops_m);
    registerGroupTopKOp(rtp_ops_m);
    registerRtpProcessGroup(rtp_ops_m);
    registerTrtFp8QuantOp(rtp_ops_m);
}

}  // namespace torch_ext