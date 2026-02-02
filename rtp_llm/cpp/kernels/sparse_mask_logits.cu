#include "rtp_llm/cpp/kernels/sparse_mask_logits.h"
#include "rtp_llm/cpp/kernels/logits_util.h"

namespace rtp_llm {

// Batch version kernel for processing multiple beams
template<typename T>
__global__ void sparse_mask_logits(const int batch_size,
                                   const int vocab_size,
                                   const int mask_size,
                                   T*        logits_batch,
                                   const int* __restrict__ batch_idx,
                                   const int* __restrict__ mask_idx,
                                   T* valid_score) {
    int m_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int b_idx      = 0;
    int total_size = batch_size * 2;
    for (int i = 0; i < total_size; i += 2) {
        if (m_idx < batch_idx[i]) {
            b_idx = batch_idx[i + 1];
            break;
        }
    }
    if (m_idx < mask_size) {
        int v_idx = mask_idx[m_idx];
        if (b_idx < batch_size && v_idx < vocab_size) {
            int global_idx           = b_idx * vocab_size + v_idx;
            logits_batch[global_idx] = valid_score[m_idx];
        }
    }
}

template<typename T>
void invokeSparseMaskLogits(T* logits_batch,
                            const int* __restrict__ batch_idx,
                            const int* __restrict__ mask_idx,
                            T*           valid_scores,
                            const int    batch_size,
                            const int    vocab_size,
                            const int    mask_size,
                            cudaStream_t stream) {
    dim3 block, grid;

    block.x = 256;
    block.y = 1;
    block.z = 1;
    grid.y  = 1;
    grid.z  = 1;

    // first store valid scores
    grid.x = (mask_size + block.x - 1) / block.x;
    extract_valid_scores<<<grid, block, 0, stream>>>(
        batch_size, vocab_size, mask_size, logits_batch, batch_idx, mask_idx, valid_scores);

    // fill logits with -INF
    grid.y = batch_size;
    grid.x = (vocab_size + block.x - 1) / block.x;
    fill_logits_with_neg_inf<<<grid, block, 0, stream>>>(batch_size, vocab_size, logits_batch);

    // restore valid scores
    grid.y = 1;
    grid.x = (mask_size + block.x - 1) / block.x;

    sparse_mask_logits<<<grid, block, 0, stream>>>(
        batch_size, vocab_size, mask_size, logits_batch, batch_idx, mask_idx, valid_scores);
#if USING_CUDA
    check_cuda_value(cudaPeekAtLastError());
#endif
    check_cuda_error();
}

template void invokeSparseMaskLogits<float>(float* logits_batch,
                                            const int* __restrict__ batch_idx,
                                            const int* __restrict__ mask_idx,
                                            float*       valid_scores,
                                            const int    batch_size,
                                            const int    vocab_size,
                                            const int    maskt_size,
                                            cudaStream_t stream);
template void invokeSparseMaskLogits<half>(half* logits_batch,
                                           const int* __restrict__ batch_idx,
                                           const int* __restrict__ mask_idx,
                                           half*        valid_scores,
                                           const int    batch_size,
                                           const int    vocab_size,
                                           const int    maskt_size,
                                           cudaStream_t stream);
template void invokeSparseMaskLogits<__nv_bfloat16>(__nv_bfloat16* logits_batch,
                                                    const int* __restrict__ batch_idx,
                                                    const int* __restrict__ mask_idx,
                                                    __nv_bfloat16* valid_scores,
                                                    const int      batch_size,
                                                    const int      vocab_size,
                                                    const int      maskt_size,
                                                    cudaStream_t   stream);

}  // namespace rtp_llm