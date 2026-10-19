#pragma once
#include <cuda_runtime.h>

namespace rtp_llm {
void invokeV41QBGemm(void* a, void* sa, void* b, void* sb, void* out,
                     void* payload, void* sf, void* scaled_weights, void* weights,
                     void* frequencies, void* positions, bool positions_i64,
                     unsigned m, unsigned max_position, cudaStream_t stream);
}
