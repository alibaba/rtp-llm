#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

namespace rtp_llm {

// Consume the contiguous AllGather layout [tp, batch, local_vocab] directly.
// vocab_size excludes the padded tail in the final rank. Probabilities retain
// draft-vocabulary order; only returned token IDs apply the optional d2t map.
// Optional sequence/position inputs are read-only CUDA int32 tensors. Their
// +1 outputs own separate storage. Unsupported inputs leave every output alone.
// Inputs and allocations must belong to the caller's current CUDA stream/device.
bool tryInvokeMtpGreedyRankMajor(const torch::Tensor& logits,
                                 const torch::Tensor& d2t_map,
                                 int64_t              vocab_size,
                                 const torch::Tensor& sequence_lengths,
                                 const torch::Tensor& position_ids,
                                 torch::Tensor&       token_ids,
                                 torch::Tensor&       probabilities,
                                 torch::Tensor&       next_sequence_lengths,
                                 torch::Tensor&       next_position_ids,
                                 cudaStream_t         stream);

}  // namespace rtp_llm
