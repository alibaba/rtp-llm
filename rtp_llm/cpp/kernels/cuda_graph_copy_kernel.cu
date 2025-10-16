
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include "rtp_llm/cpp/kernels/cuda_graph_copy_kernel.h"
#include <stdio.h>
#if USING_CUDA
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#ifdef ENABLE_BF16
#include <cuda_bf16.h>
#endif
#endif
#if USING_ROCM
#include <hip/hip_runtime.h>
#include "rtp_llm/cpp/rocm/cuda_shims.h"
#endif

// Helper function to convert to float (specialized for each type)
__device__ float convert_to_float(__nv_bfloat16 val) {
    return __bfloat162float(val);
}

__device__ float convert_to_float(__half val) {
    return __half2float(val);
}

__device__ float convert_to_float(float val) {
    return val;
}

__device__ float convert_to_float(int val) {
    return float(val);
}

namespace rtp_llm {

// Device function to find batch index using binary search on cu_seq_len
__device__ __forceinline__ int findBatchIndex(const int token_idx, const int* cu_seq_len, const int batch_size) {
    int left = 0, right = batch_size;
    while (left < right) {
        int mid = (left + right + 1) / 2;
        if (cu_seq_len[mid] < token_idx + 1) {
            left = mid;
        } else {
            right = mid - 1;
        }
    }
    return left;
}

template<typename T>
__global__ void cudaGraphCopySmall2LargeKernel(T*         input_tensor,
                                               T*         output_tensor,
                                               const int* input_lengths,
                                               const int* batch_size,
                                               const int  max_seq_len,
                                               const int  hidden_size,
                                               const int* cu_seq_len) {
    const int tid           = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = blockDim.x * gridDim.x;
    // Calculate total_valid_elements using cu_seq_len array
    // cu_seq_len[i] contains cumulative length up to batch i
    // cu_seq_len[batch_size] contains total elements
    const int total_valid_elements = cu_seq_len[*batch_size] * hidden_size;

    // Each thread processes multiple elements with stride = total_threads
    // This design handles cases where grid_size is limited to 65536
    // Each thread will process: tid, tid+total_threads, tid+2*total_threads, ...
    for (int idx = tid; idx < total_valid_elements; idx += total_threads) {
        // Find which batch and sequence this element belongs to using cu_seq_len
        // Convert linear index to token index first
        const int token_idx  = idx / hidden_size;
        const int hidden_idx = idx % hidden_size;

        // Binary search to find which batch this token belongs to
        const int batch_idx = findBatchIndex(token_idx, cu_seq_len, *batch_size);

        // Calculate sequence index within this batch
        const int seq_idx = token_idx - cu_seq_len[batch_idx];

        // Calculate source index in compact tensor (linear index)
        const int source_idx = idx;

        // Calculate destination index in aligned tensor
        const int dest_idx = batch_idx * max_seq_len * hidden_size + seq_idx * hidden_size + hidden_idx;

        // Perform the copy
        output_tensor[dest_idx] = input_tensor[source_idx];
    }
}

template<typename T>
void invokeCudaGraphCopySmall2Large(T*        input_tensor,
                                    T*        output_tensor,
                                    int*      batch_size,
                                    const int max_batch_size,
                                    const int max_seq_len,
                                    int*      input_lengths,
                                    const int hidden_size,
                                    int*      cu_seq_len,
#if USING_CUDA
                                    cudaStream_t stream) {
#elif USING_ROCM
                                    hipStream_t stream) {
#endif
    // Validate input parameters
    if (input_tensor == nullptr || output_tensor == nullptr || input_lengths == nullptr || *batch_size <= 0
        || max_seq_len <= 0 || hidden_size <= 0 || cu_seq_len == nullptr) {
        return;
    }

    // Calculate grid and block dimensions
    // Use cu_seq_len[batch_size] which contains total token count
    const int total_elements = cu_seq_len[*batch_size] * hidden_size;
    dim3      block(256);
    const int grid_size = min((total_elements + block.x - 1) / block.x, 65536);
    dim3      grid(grid_size);

    // Launch kernel
    cudaGraphCopySmall2LargeKernel<T><<<grid, block, 0, stream>>>(
        input_tensor, output_tensor, input_lengths, batch_size, max_seq_len, hidden_size, cu_seq_len);
}

template<typename T>
__global__ void cudaGraphCopyLarge2SmallKernel(T*         input_tensor,
                                               T*         output_tensor,
                                               const int* input_lengths,
                                               const int* batch_size,
                                               const int  max_seq_len,
                                               const int  hidden_size,
                                               const int* cu_seq_len) {
    const int tid           = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = blockDim.x * gridDim.x;

    // Calculate total_valid_elements using cu_seq_len array
    // cu_seq_len[i] contains cumulative length up to batch i
    // cu_seq_len[batch_size] contains total elements
    const int total_valid_elements = cu_seq_len[*batch_size] * hidden_size;

    // Each thread processes multiple elements with stride = total_threads
    // This design handles cases where grid_size is limited to 65536
    // Each thread will process: tid, tid+total_threads, tid+2*total_threads, ...
    for (int idx = tid; idx < total_valid_elements; idx += total_threads) {
        // Find which batch and sequence this element belongs to using cu_seq_len
        // Convert linear index to token index first
        const int token_idx  = idx / hidden_size;
        const int hidden_idx = idx % hidden_size;

        // Binary search to find which batch this token belongs to
        const int batch_idx = findBatchIndex(token_idx, cu_seq_len, *batch_size);

        // Calculate sequence index within this batch
        const int seq_idx = token_idx - cu_seq_len[batch_idx];

        // Calculate source index in aligned tensor
        const int source_idx = batch_idx * max_seq_len * hidden_size + seq_idx * hidden_size + hidden_idx;

        // Calculate destination index in compact tensor (linear index)
        const int dest_idx = idx;

        // Perform the copy
        output_tensor[dest_idx] = input_tensor[source_idx];
    }
}

template<typename T>
void invokeCudaGraphCopyLarge2Small(T*        input_tensor,
                                    T*        output_tensor,
                                    int*      batch_size,
                                    const int max_batch_size,
                                    const int max_seq_len,
                                    int*      input_lengths,
                                    const int hidden_size,
                                    int*      cu_seq_len,
#if USING_CUDA
                                    cudaStream_t stream) {
#elif USING_ROCM
                                    hipStream_t stream) {
#endif
    // Validate input parameters
    if (input_tensor == nullptr || output_tensor == nullptr || input_lengths == nullptr || *batch_size <= 0
        || max_seq_len <= 0 || hidden_size <= 0 || cu_seq_len == nullptr) {
        return;
    }

    // Calculate grid and block dimensions
    // Use cu_seq_len[batch_size] which contains total token count
    const int total_elements = cu_seq_len[*batch_size] * hidden_size;

    dim3      block(256);
    const int grid_size = min((total_elements + block.x - 1) / block.x, 65536);
    dim3      grid(grid_size);

    // Launch kernel
    cudaGraphCopyLarge2SmallKernel<T><<<grid, block, 0, stream>>>(
        input_tensor, output_tensor, input_lengths, batch_size, max_seq_len, hidden_size, cu_seq_len);
}

// Template instantiations
#if USING_CUDA
template void invokeCudaGraphCopySmall2Large<half>(half*        input_tensor,
                                                   half*        output_tensor,
                                                   int*         batch_size,
                                                   const int    max_batch_size,
                                                   const int    max_seq_len,
                                                   int*         input_lengths,
                                                   const int    hidden_size,
                                                   int*         cu_seq_len,
                                                   cudaStream_t stream);

template void invokeCudaGraphCopySmall2Large<float>(float*       input_tensor,
                                                    float*       output_tensor,
                                                    int*         batch_size,
                                                    const int    max_batch_size,
                                                    const int    max_seq_len,
                                                    int*         input_lengths,
                                                    const int    hidden_size,
                                                    int*         cu_seq_len,
                                                    cudaStream_t stream);

template void invokeCudaGraphCopyLarge2Small<half>(half*        input_tensor,
                                                   half*        output_tensor,
                                                   int*         batch_size,
                                                   const int    max_batch_size,
                                                   const int    max_seq_len,
                                                   int*         input_lengths,
                                                   const int    hidden_size,
                                                   int*         cu_seq_len,
                                                   cudaStream_t stream);

template void invokeCudaGraphCopyLarge2Small<float>(float*       input_tensor,
                                                    float*       output_tensor,
                                                    int*         batch_size,
                                                    const int    max_batch_size,
                                                    const int    max_seq_len,
                                                    int*         input_lengths,
                                                    const int    hidden_size,
                                                    int*         cu_seq_len,
                                                    cudaStream_t stream);

#ifdef ENABLE_BF16
template void invokeCudaGraphCopySmall2Large<__nv_bfloat16>(__nv_bfloat16* input_tensor,
                                                            __nv_bfloat16* output_tensor,
                                                            int*           batch_size,
                                                            const int      max_batch_size,
                                                            const int      max_seq_len,
                                                            int*           input_lengths,
                                                            const int      hidden_size,
                                                            int*           cu_seq_len,
                                                            cudaStream_t   stream);

template void invokeCudaGraphCopyLarge2Small<__nv_bfloat16>(__nv_bfloat16* input_tensor,
                                                            __nv_bfloat16* output_tensor,
                                                            int*           batch_size,
                                                            const int      max_batch_size,
                                                            const int      max_seq_len,
                                                            int*           input_lengths,
                                                            const int      hidden_size,
                                                            int*           cu_seq_len,
                                                            cudaStream_t   stream);
#endif

#elif USING_ROCM
template void invokeCudaGraphCopySmall2Large<half>(half*       input_tensor,
                                                   half*       output_tensor,
                                                   int*        batch_size,
                                                   const int   max_batch_size,
                                                   const int   max_seq_len,
                                                   int*        input_lengths,
                                                   const int   hidden_size,
                                                   int*        cu_seq_len,
                                                   hipStream_t stream);

template void invokeCudaGraphCopySmall2Large<float>(float*      input_tensor,
                                                    float*      output_tensor,
                                                    int*        batch_size,
                                                    const int   max_batch_size,
                                                    const int   max_seq_len,
                                                    int*        input_lengths,
                                                    const int   hidden_size,
                                                    int*        cu_seq_len,
                                                    hipStream_t stream);

template void invokeCudaGraphCopyLarge2Small<half>(half*       input_tensor,
                                                   half*       output_tensor,
                                                   int*        batch_size,
                                                   const int   max_batch_size,
                                                   const int   max_seq_len,
                                                   int*        input_lengths,
                                                   const int   hidden_size,
                                                   int*        cu_seq_len,
                                                   hipStream_t stream);

template void invokeCudaGraphCopyLarge2Small<float>(float*      input_tensor,
                                                    float*      output_tensor,
                                                    int*        batch_size,
                                                    const int   max_batch_size,
                                                    const int   max_seq_len,
                                                    int*        input_lengths,
                                                    const int   hidden_size,
                                                    int*        cu_seq_len,
                                                    hipStream_t stream);
#endif

}  // namespace rtp_llm
