#include "rtp_llm/cpp/kernels/weight_logits.h"
#include "rtp_llm/cpp/kernels/logits_util.h"

namespace rtp_llm {

// Batch version kernel for processing multiple beams
template<typename T>
__global__ void weight_logits(const int batch_size,
                              const int vocab_size,
                              const int weight_size,
                              T*        logits_batch,
                              const int* __restrict__ batch_idx,
                              const int* __restrict__ vocab_idx,
                              const float* __restrict__ weight_batch) {
    int weight_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int b_idx      = 0;
    int total_size = batch_size * 2;
    for (int i = 0; i < total_size; i += 2) {
        if (weight_idx < batch_idx[i]) {
            b_idx = batch_idx[i + 1];
            break;
        }
    }
    if (weight_idx < weight_size) {
        int v_idx = vocab_idx[weight_idx];
        if (b_idx < batch_size && v_idx < vocab_size) {
            int global_idx           = b_idx * vocab_size + v_idx;
            logits_batch[global_idx] = weight_batch[weight_idx];
        }
    }
}

template<typename T>
void invokeWeightLogits(T* logits_batch,
                        const int* __restrict__ batch_idx,
                        const int* __restrict__ vocab_idx,
                        float* __restrict__ weight_batch,
                        const int    batch_size,
                        const int    vocab_size,
                        const int    weight_size,
                        cudaStream_t stream) {
    dim3 block, grid;

    block.x = 256;
    block.y = 1;
    block.z = 1;
    grid.y  = 1;
    grid.z  = 1;

    grid.x = (weight_size + block.x - 1) / block.x;
    extract_valid_scores_to_weights<<<grid, block, 0, stream>>>(
        batch_size, vocab_size, weight_size, logits_batch, batch_idx, vocab_idx, weight_batch);
    check_cuda_error();

    // fill logits with -INF
    grid.y = batch_size;
    grid.x = (vocab_size + block.x - 1) / block.x;
    fill_logits_with_neg_inf<<<grid, block, 0, stream>>>(batch_size, vocab_size, logits_batch);
    check_cuda_error();

    // add weight to valid scores
    grid.y = 1;
    grid.x = (weight_size + block.x - 1) / block.x;

    weight_logits<<<grid, block, 0, stream>>>(
        batch_size, vocab_size, weight_size, logits_batch, batch_idx, vocab_idx, weight_batch);
#if USING_CUDA
    check_cuda_value(cudaPeekAtLastError());
#endif
    check_cuda_error();
}

template void invokeWeightLogits<float>(float* logits_batch,
                                        const int* __restrict__ batch_idx,
                                        const int* __restrict__ vocab_idx,
                                        float* __restrict__ weight_batch,
                                        const int    batch_size,
                                        const int    vocab_size,
                                        const int    weight_size,
                                        cudaStream_t stream);
template void invokeWeightLogits<half>(half* logits_batch,
                                       const int* __restrict__ batch_idx,
                                       const int* __restrict__ vocab_idx,
                                       float* __restrict__ weight_batch,
                                       const int    batch_size,
                                       const int    vocab_size,
                                       const int    weight_size,
                                       cudaStream_t stream);
template void invokeWeightLogits<__nv_bfloat16>(__nv_bfloat16* logits_batch,
                                                const int* __restrict__ batch_idx,
                                                const int* __restrict__ vocab_idx,
                                                float* __restrict__ weight_batch,
                                                const int    batch_size,
                                                const int    vocab_size,
                                                const int    weight_size,
                                                cudaStream_t stream);

}  // namespace rtp_llm