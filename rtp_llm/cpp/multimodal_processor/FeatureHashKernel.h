#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace rtp_llm {

cudaError_t
invokeFeatureHash(const void* input, int64_t num_rows, int64_t row_bytes, int32_t* output, cudaStream_t stream);

}  // namespace rtp_llm
