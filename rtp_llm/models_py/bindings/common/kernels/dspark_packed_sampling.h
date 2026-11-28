#pragma once

#include "rtp_llm/models_py/bindings/core/OpData.h"
#if USING_CUDA
#include <cuda_runtime.h>
#elif USING_ROCM
#include <hip/hip_runtime.h>
#endif

namespace rtp_llm {
#if USING_CUDA
using SamplingStream = cudaStream_t;
#elif USING_ROCM
using SamplingStream = hipStream_t;
#endif
// Select parameters and optional per-logical-row history using an int64 row map.
void invokeSamplingGather(const GreedyParams&  dense,
                          const GreedyParams&  packed,
                          const torch::Tensor& packed_to_dense,
                          const torch::Tensor& history,
                          const torch::Tensor& tokens,
                          SamplingStream       stream);

struct SamplingResult {
    torch::Tensor tokens, probs, indices, sampling_probs, cum_log_probs, success;
};

// Restores live output fields and initializes omitted rows in a single launch.
// Tokens contain only the sampled column; histories are not output consumers.
void invokeSamplingScatter(const SamplingResult& packed,
                           const SamplingResult& dense,
                           const torch::Tensor&  dense_to_packed,
                           SamplingStream        stream);
}  // namespace rtp_llm
