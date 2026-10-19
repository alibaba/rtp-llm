#pragma once
#include <cuda_runtime_api.h>

namespace rtp_llm {
// SM103 / 148 SMs only. All tensors are contiguous except packed scales,
// which are MN-major int32 [M,40] / [1792,40]. No persistent pointer cache.
void invokeV41FrontGemm(void* a, void* sa, void* b, void* sb, void* x, void* head,
                        void* out, void* head_out, unsigned m, cudaStream_t stream);
}
