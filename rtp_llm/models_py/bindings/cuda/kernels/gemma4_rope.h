#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstdint>

namespace rtp_llm {

void invokeGemma4RopeCosSinBf16(const int32_t* positions,
                                const float*   inv_freq,
                                __nv_bfloat16* cos,
                                __nv_bfloat16* sin,
                                int64_t        tokens,
                                int32_t        half_dim,
                                cudaStream_t   stream);

void invokeGemma4QkRopeBf16(const __nv_bfloat16* q,
                            const __nv_bfloat16* k,
                            const __nv_bfloat16* cos,
                            const __nv_bfloat16* sin,
                            __nv_bfloat16*       q_out,
                            __nv_bfloat16*       k_out,
                            int64_t              tokens,
                            int32_t              query_heads,
                            int32_t              kv_heads,
                            int32_t              head_dim,
                            cudaStream_t         stream);

}  // namespace rtp_llm
