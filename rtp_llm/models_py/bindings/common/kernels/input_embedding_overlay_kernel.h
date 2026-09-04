#pragma once

#include <cstdint>

#if USING_CUDA
#include <cuda_runtime.h>
#endif
#if USING_ROCM
#include <hip/hip_runtime.h>
#include "rtp_llm/models_py/bindings/rocm/cuda_shims.h"
#endif

namespace rtp_llm {

// Apply graph-staged input embeddings in place. metadata has one int32 entry
// per token row: zero preserves the token embedding and non-zero replaces it
// with the corresponding override row. Consumed entries are reset to zero by
// the kernel so the common no-override path needs no host-side clear.
template<typename T>
void invokeInputEmbeddingOverlay(
    T* inputs_embeds, const T* overrides, int32_t* metadata, int token_count, int hidden_size, cudaStream_t stream);

}  // namespace rtp_llm
