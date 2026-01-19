#pragma once

#include <stdint.h>

#if USING_CUDA
#include <cuda_runtime.h>
#endif

#if USING_ROCM
#include "rtp_llm/cpp/rocm/cuda_shims.h"
#endif

namespace rtp_llm {

/**
 * @brief Mapping draft tokens to target tokens,
 *        tokens[batch_size, token_offset:] = d2t_map[tokens[batch_size, token_offset:]]

 * @param tokens: The tokens to be mapped shape: [batch_size, token_stride]
 * @param batch_size: The batch size
 * @param token_offset: The offset of the tokens
 * @param token_stride: The stride of the tokens
 * @param d2t_map: The mapping from draft tokens to target tokens
 * @param stream: The stream to be used
 */
template<typename IdType>
void invokeMappingDraft2Target(
    IdType* tokens, int batch_size, int token_offset, int token_stride, int64_t* d2t_map, cudaStream_t stream);
}  // namespace rtp_llm