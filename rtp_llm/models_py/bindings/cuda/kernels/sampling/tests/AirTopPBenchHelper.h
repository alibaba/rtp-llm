#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace rtp_llm {

size_t airTopPWorkspaceBytes(uint32_t batch_size, uint32_t vocab_size);

void invokeAirTopPRenormBench(float*       probs,
                              float*       renormed_prob,
                              float*       top_p_arr,
                              uint32_t     batch_size,
                              float        top_p_val,
                              uint32_t     vocab_size,
                              void*        workspace,
                              cudaStream_t stream);

void invokeFlashInferSamplingFromProbBench(const float* probs,
                                           int32_t*     output,
                                           bool*        valid,
                                           uint32_t     batch_size,
                                           uint32_t     vocab_size,
                                           uint64_t*    seed_arr,
                                           uint64_t*    offset_arr,
                                           cudaStream_t stream);

void invokeFlashInferTopKTopPSamplingBench(const float*   probs,
                                           const int32_t* top_k_arr,
                                           const float*   top_p_arr,
                                           int32_t*       output,
                                           bool*          valid,
                                           uint32_t       batch_size,
                                           uint32_t       vocab_size,
                                           uint64_t*      seed_arr,
                                           uint64_t*      offset_arr,
                                           cudaStream_t   stream);

}  // namespace rtp_llm
