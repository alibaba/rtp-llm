#include "rtp_llm/cpp/kernels/vocab_prune/mapping.h"
#include <cstdint>

namespace rtp_llm {

template<typename IdType>
__global__ void
mapping_draft2target_kernel(IdType* tokens, int batch_size, int token_offset, int token_stride, int64_t* d2t_map) {
    const int idx       = blockIdx.x * blockDim.x + threadIdx.x;
    const int token_idx = idx % token_stride;

    if (token_idx < token_offset || idx >= batch_size * token_stride) {
        return;
    }

    tokens[idx] = d2t_map[tokens[idx]];
}

template<typename IdType>
void invokeMappingDraft2Target(
    IdType* tokens, int batch_size, int token_offset, int token_stride, int64_t* d2t_map, cudaStream_t stream) {
    const int block_size = 256;
    const int grid_size  = (batch_size * token_stride + block_size - 1) / block_size;
    mapping_draft2target_kernel<<<grid_size, block_size, 0, stream>>>(
        tokens, batch_size, token_offset, token_stride, d2t_map);
}

#define INSTANTIATE_MAPPING_DRAFT2TARGET(IdType)                                                                       \
    template void invokeMappingDraft2Target(                                                                           \
        IdType* tokens, int batch_size, int token_offset, int token_stride, int64_t* d2t_map, cudaStream_t stream);

INSTANTIATE_MAPPING_DRAFT2TARGET(int32_t);
INSTANTIATE_MAPPING_DRAFT2TARGET(int64_t);

}  // namespace rtp_llm