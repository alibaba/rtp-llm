#pragma once

#include <ATen/ATen.h>
#include <cuda_runtime_api.h>
#include "rtp_llm/models_py/bindings/cuda/kernels/cuda_graph_prepare.h"

namespace rtp_llm {

struct AttentionInputMetadataParams {
    const int32_t* input_lengths           = nullptr;
    const int32_t* prefix_lengths          = nullptr;
    int32_t*       cu_seqlens              = nullptr;
    int32_t*       cu_kv_seqlens           = nullptr;
    int32_t*       padding_offset          = nullptr;
    int32_t*       sequence_lengths_plus_1 = nullptr;
    int32_t        batch_size              = 0;
    int32_t        total_tokens            = 0;
    int32_t        device                  = -1;
};

AttentionInputMetadataParams prepareAttentionInputMetadata(const at::Tensor& input_lengths,
                                                           const at::Tensor& prefix_lengths,
                                                           const at::Tensor& cu_seqlens,
                                                           const at::Tensor& cu_kv_seqlens,
                                                           const at::Tensor& padding_offset,
                                                           const at::Tensor& sequence_lengths_plus_1 = {});

void invokeBuildAttentionInputMetadata(AttentionInputMetadataParams  params,
                                       cudaStream_t                  stream,
                                       const CudaGraphPrepareReplay& replay = {});

cudaGraphNode_t addAttentionInputMetadataNode(cudaGraph_t graph, cudaGraphNode_t before);

void invokeBuildAttentionInputMetadata(const at::Tensor& input_lengths,
                                       const at::Tensor& prefix_lengths,
                                       at::Tensor&       cu_seqlens,
                                       at::Tensor&       cu_kv_seqlens,
                                       at::Tensor&       padding_offset,
                                       cudaStream_t      stream,
                                       const at::Tensor& sequence_lengths_plus_1 = {});

}  // namespace rtp_llm
