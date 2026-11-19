#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>

namespace rtp_llm {

cudaError_t invokeContiguousSiluFp8(const void* input,
                                   const int32_t* expert_ids,
                                   const int32_t* expert_end,
                                   uint8_t* output,
                                   float* scales,
                                   int rows,
                                   int width,
                                   int scale_ld,
                                   cudaStream_t stream,
                                   int padding_begin,
                                   int padding_end);

}  // namespace rtp_llm
