#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstdint>

namespace rtp_llm {

void invokeGemma4ExpandKvHeads8Bf16(const __nv_bfloat16* k,
                                    const __nv_bfloat16* v,
                                    __nv_bfloat16*       expanded_k,
                                    __nv_bfloat16*       expanded_v,
                                    int64_t              tokens,
                                    cudaStream_t         stream);

void invokeGemma4ExpandKvHeads2Bf16(const __nv_bfloat16* k,
                                    const __nv_bfloat16* v,
                                    __nv_bfloat16*       expanded_k,
                                    __nv_bfloat16*       expanded_v,
                                    int64_t              tokens,
                                    cudaStream_t         stream);

}  // namespace rtp_llm
