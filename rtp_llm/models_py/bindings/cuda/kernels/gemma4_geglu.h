#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstdint>

namespace rtp_llm {

void invokeGemma4GegluTanhBf16(
    const __nv_bfloat16* gate_up, __nv_bfloat16* output, int64_t rows, int32_t intermediate_size, cudaStream_t stream);

}  // namespace rtp_llm
