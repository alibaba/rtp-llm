#include "rtp_llm/cpp/kernels/finished_mask_logits.h"
#include "rtp_llm/cpp/kernels/logits_util.h"

namespace rtp_llm {

// Batch version kernel for processing multiple beams
template<typename T>
__global__ void finished_mask_logits(
    T* logits_batch, const uint8_t* __restrict__ finished_mask, int batch_size, int vocab_size, int end_token_id) {
    int batch_idx = blockIdx.y;
    int vocab_idx = threadIdx.x + (blockIdx.x * blockDim.x);

    if (batch_idx < batch_size) {
        if (finished_mask[batch_idx] == 1) {
            if (vocab_idx != end_token_id) {
                logits_batch[vocab_idx] = NegativeInfinity<T>();
            }
        }
    }
}

template<typename T>
void invokeFinishedMaskLogits(T* logits_batch,
                              const uint8_t* __restrict__ finished_mask,
                              int          batch_size,
                              int          vocab_size,
                              int          end_token_id,
                              cudaStream_t stream) {
    dim3 block, grid;

    block.x = 256;
    block.y = 1;
    block.z = 1;
    grid.y  = batch_size;
    grid.z  = 1;
    grid.x  = (vocab_size + block.x - 1) / block.x;
    finished_mask_logits<<<grid, block, 0, stream>>>(logits_batch, finished_mask, batch_size, vocab_size, end_token_id);
#if USING_CUDA
    check_cuda_value(cudaPeekAtLastError());
#endif
    check_cuda_error();
}

template void invokeFinishedMaskLogits<float>(float* logits_batch,
                                              const uint8_t* __restrict__ finished_mask,
                                              int          batch_size,
                                              int          vocab_size,
                                              int          end_token_id,
                                              cudaStream_t stream);
template void invokeFinishedMaskLogits<half>(half* logits_batch,
                                             const uint8_t* __restrict__ finished_mask,
                                             int          batch_size,
                                             int          vocab_size,
                                             int          end_token_id,
                                             cudaStream_t stream);
template void invokeFinishedMaskLogits<__nv_bfloat16>(__nv_bfloat16* logits_batch,
                                                      const uint8_t* __restrict__ finished_mask,
                                                      int          batch_size,
                                                      int          vocab_size,
                                                      int          end_token_id,
                                                      cudaStream_t stream);

}  // namespace rtp_llm