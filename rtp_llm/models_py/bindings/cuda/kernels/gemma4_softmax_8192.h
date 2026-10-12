#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstdint>

namespace rtp_llm {

void invokeGemma4Softmax8192Bf16(const __nv_bfloat16* input,
                                 __nv_bfloat16*       output,
                                 int64_t              rows,
                                 int32_t              query_length,
                                 int32_t              query_start,
                                 int32_t              window_left,
                                 cudaStream_t         stream);

}  // namespace rtp_llm
