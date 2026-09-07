#include <algorithm>
#include "rtp_llm/models_py/bindings/common/kernels/cuda_graph_copy_kernel.h"
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
#include "rtp_llm/models_py/bindings/rocm/cuda_shims.h"
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
__device__ __forceinline__ int findBatchIndex(const int64_t token_idx, const int* cu_seq_len, const int batch_size) {
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
__global__ void cudaGraphCopySmall2LargeKernel(T*            input_tensor,
                                               T*            output_tensor,
                                               const int*    input_lengths,
                                               const int*    batch_size,
                                               const int64_t max_batch_size,
                                               const int64_t max_seq_len,
                                               const int64_t hidden_size,
                                               const int*    cu_seq_len) {

    const int64_t tid           = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total_threads = static_cast<int64_t>(blockDim.x) * gridDim.x;

    // batch_size may live in graph-private device memory, so validate it in
    // the kernel instead of dereferencing it on the host launcher.
    const int current_batch_size = *batch_size;
    if (current_batch_size <= 0 || current_batch_size > max_batch_size) {
        return;
    }

    // Calculate total_valid_elements using cu_seq_len array
    // cu_seq_len[i] contains cumulative length up to batch i
    // cu_seq_len[batch_size] contains total elements
    const int64_t total_valid_elements = static_cast<int64_t>(cu_seq_len[current_batch_size]) * hidden_size;

    // Each thread processes multiple elements with stride = total_threads
    // This design handles cases where grid_size is limited to 65536
    // Each thread will process: tid, tid+total_threads, tid+2*total_threads, ...
    for (int64_t idx = tid; idx < total_valid_elements; idx += total_threads) {
        // Find which batch and sequence this element belongs to using cu_seq_len
        // Convert linear index to token index first
        const int64_t token_idx  = idx / hidden_size;
        const int64_t hidden_idx = idx % hidden_size;

        // Binary search to find which batch this token belongs to
        const int batch_idx = findBatchIndex(token_idx, cu_seq_len, current_batch_size);

        // Calculate sequence index within this batch
        const int64_t seq_idx = token_idx - cu_seq_len[batch_idx];

        // Calculate source index in compact tensor (linear index)
        const int64_t source_idx = idx;

        // Calculate destination index in right-aligned tensor
        const int64_t padding  = max_seq_len - input_lengths[batch_idx];
        const int64_t dest_idx = static_cast<int64_t>(batch_idx) * max_seq_len * hidden_size
                                 + (padding + seq_idx) * hidden_size + hidden_idx;

        // Perform the copy
        output_tensor[dest_idx] = input_tensor[source_idx];
    }
}

template<typename T>
void invokeCudaGraphCopySmall2Large(T*            input_tensor,
                                    T*            output_tensor,
                                    const int*    batch_size,
                                    const int64_t max_batch_size,
                                    const int64_t max_seq_len,
                                    const int*    input_lengths,
                                    const int64_t hidden_size,
                                    const int*    cu_seq_len,
#if USING_CUDA
                                    cudaStream_t stream) {
#elif USING_ROCM
                                    hipStream_t stream) {
#endif
    // Validate input parameters
    if (input_tensor == nullptr || output_tensor == nullptr || batch_size == nullptr || input_lengths == nullptr
        || max_batch_size <= 0 || max_seq_len <= 0 || hidden_size <= 0 || cu_seq_len == nullptr) {
        return;
    }

    // use fixed block and grid size for cuda graph
    dim3 block(256);
    dim3 grid(1024);

    cudaGraphCopySmall2LargeKernel<T><<<grid, block, 0, stream>>>(
        input_tensor, output_tensor, input_lengths, batch_size, max_batch_size, max_seq_len, hidden_size, cu_seq_len);
}

template<typename T>
__global__ void cudaGraphCopyLarge2SmallKernel(T*            input_tensor,
                                               T*            output_tensor,
                                               const int*    input_lengths,
                                               const int*    batch_size,
                                               const int64_t max_batch_size,
                                               const int64_t max_seq_len,
                                               const int64_t hidden_size,
                                               const int*    cu_seq_len) {
    const int64_t tid           = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total_threads = static_cast<int64_t>(blockDim.x) * gridDim.x;

    // batch_size may live in graph-private device memory, so validate it in
    // the kernel instead of dereferencing it on the host launcher.
    const int current_batch_size = *batch_size;
    if (current_batch_size <= 0 || current_batch_size > max_batch_size) {
        return;
    }

    // Calculate total_valid_elements using cu_seq_len array
    // cu_seq_len[i] contains cumulative length up to batch i
    // cu_seq_len[batch_size] contains total elements
    const int64_t total_valid_elements = static_cast<int64_t>(cu_seq_len[current_batch_size]) * hidden_size;

    // Each thread processes multiple elements with stride = total_threads
    // This design handles cases where grid_size is limited to 65536
    // Each thread will process: tid, tid+total_threads, tid+2*total_threads, ...
    for (int64_t idx = tid; idx < total_valid_elements; idx += total_threads) {
        // Find which batch and sequence this element belongs to using cu_seq_len
        // Convert linear index to token index first
        const int64_t token_idx  = idx / hidden_size;
        const int64_t hidden_idx = idx % hidden_size;

        // Binary search to find which batch this token belongs to
        const int batch_idx = findBatchIndex(token_idx, cu_seq_len, current_batch_size);

        // Calculate sequence index within this batch
        const int64_t seq_idx = token_idx - cu_seq_len[batch_idx];

        // Read back the valid suffix from the right-aligned tensor.
        const int64_t padding    = max_seq_len - input_lengths[batch_idx];
        const int64_t source_idx = static_cast<int64_t>(batch_idx) * max_seq_len * hidden_size
                                   + (padding + seq_idx) * hidden_size + hidden_idx;

        // Calculate destination index in compact tensor (linear index)
        const int64_t dest_idx = idx;

        // Perform the copy
        output_tensor[dest_idx] = input_tensor[source_idx];
    }
}

template<typename T>
void invokeCudaGraphCopyLarge2Small(T*            input_tensor,
                                    T*            output_tensor,
                                    const int*    batch_size,
                                    const int64_t max_batch_size,
                                    const int64_t max_seq_len,
                                    const int*    input_lengths,
                                    const int64_t hidden_size,
                                    const int*    cu_seq_len,
#if USING_CUDA
                                    cudaStream_t stream) {
#elif USING_ROCM
                                    hipStream_t stream) {
#endif
    if (input_tensor == nullptr || output_tensor == nullptr || batch_size == nullptr || input_lengths == nullptr
        || max_batch_size <= 0 || max_seq_len <= 0 || hidden_size <= 0 || cu_seq_len == nullptr) {
        return;
    }

    // use fixed block and grid size for cuda graph
    dim3 block(256);
    dim3 grid(1024);

    cudaGraphCopyLarge2SmallKernel<T><<<grid, block, 0, stream>>>(
        input_tensor, output_tensor, input_lengths, batch_size, max_batch_size, max_seq_len, hidden_size, cu_seq_len);
}

// Template instantiations
#if USING_CUDA
template void invokeCudaGraphCopySmall2Large<half>(half*         input_tensor,
                                                   half*         output_tensor,
                                                   const int*    batch_size,
                                                   const int64_t max_batch_size,
                                                   const int64_t max_seq_len,
                                                   const int*    input_lengths,
                                                   const int64_t hidden_size,
                                                   const int*    cu_seq_len,
                                                   cudaStream_t  stream);

template void invokeCudaGraphCopySmall2Large<float>(float*        input_tensor,
                                                    float*        output_tensor,
                                                    const int*    batch_size,
                                                    const int64_t max_batch_size,
                                                    const int64_t max_seq_len,
                                                    const int*    input_lengths,
                                                    const int64_t hidden_size,
                                                    const int*    cu_seq_len,
                                                    cudaStream_t  stream);

template void invokeCudaGraphCopyLarge2Small<half>(half*         input_tensor,
                                                   half*         output_tensor,
                                                   const int*    batch_size,
                                                   const int64_t max_batch_size,
                                                   const int64_t max_seq_len,
                                                   const int*    input_lengths,
                                                   const int64_t hidden_size,
                                                   const int*    cu_seq_len,
                                                   cudaStream_t  stream);

template void invokeCudaGraphCopyLarge2Small<float>(float*        input_tensor,
                                                    float*        output_tensor,
                                                    const int*    batch_size,
                                                    const int64_t max_batch_size,
                                                    const int64_t max_seq_len,
                                                    const int*    input_lengths,
                                                    const int64_t hidden_size,
                                                    const int*    cu_seq_len,
                                                    cudaStream_t  stream);

#ifdef ENABLE_BF16
template void invokeCudaGraphCopySmall2Large<__nv_bfloat16>(__nv_bfloat16* input_tensor,
                                                            __nv_bfloat16* output_tensor,
                                                            const int*     batch_size,
                                                            const int64_t  max_batch_size,
                                                            const int64_t  max_seq_len,
                                                            const int*     input_lengths,
                                                            const int64_t  hidden_size,
                                                            const int*     cu_seq_len,
                                                            cudaStream_t   stream);

template void invokeCudaGraphCopyLarge2Small<__nv_bfloat16>(__nv_bfloat16* input_tensor,
                                                            __nv_bfloat16* output_tensor,
                                                            const int*     batch_size,
                                                            const int64_t  max_batch_size,
                                                            const int64_t  max_seq_len,
                                                            const int*     input_lengths,
                                                            const int64_t  hidden_size,
                                                            const int*     cu_seq_len,
                                                            cudaStream_t   stream);
#endif

#elif USING_ROCM
template void invokeCudaGraphCopySmall2Large<half>(half*         input_tensor,
                                                   half*         output_tensor,
                                                   const int*    batch_size,
                                                   const int64_t max_batch_size,
                                                   const int64_t max_seq_len,
                                                   const int*    input_lengths,
                                                   const int64_t hidden_size,
                                                   const int*    cu_seq_len,
                                                   hipStream_t   stream);

template void invokeCudaGraphCopySmall2Large<float>(float*        input_tensor,
                                                    float*        output_tensor,
                                                    const int*    batch_size,
                                                    const int64_t max_batch_size,
                                                    const int64_t max_seq_len,
                                                    const int*    input_lengths,
                                                    const int64_t hidden_size,
                                                    const int*    cu_seq_len,
                                                    hipStream_t   stream);

template void invokeCudaGraphCopyLarge2Small<half>(half*         input_tensor,
                                                   half*         output_tensor,
                                                   const int*    batch_size,
                                                   const int64_t max_batch_size,
                                                   const int64_t max_seq_len,
                                                   const int*    input_lengths,
                                                   const int64_t hidden_size,
                                                   const int*    cu_seq_len,
                                                   hipStream_t   stream);

template void invokeCudaGraphCopyLarge2Small<float>(float*        input_tensor,
                                                    float*        output_tensor,
                                                    const int*    batch_size,
                                                    const int64_t max_batch_size,
                                                    const int64_t max_seq_len,
                                                    const int*    input_lengths,
                                                    const int64_t hidden_size,
                                                    const int*    cu_seq_len,
                                                    hipStream_t   stream);

#ifdef ENABLE_BF16
template void invokeCudaGraphCopySmall2Large<amd_bfloat16>(amd_bfloat16* input_tensor,
                                                           amd_bfloat16* output_tensor,
                                                           const int*    batch_size,
                                                           const int64_t max_batch_size,
                                                           const int64_t max_seq_len,
                                                           const int*    input_lengths,
                                                           const int64_t hidden_size,
                                                           const int*    cu_seq_len,
                                                           hipStream_t   stream);

template void invokeCudaGraphCopyLarge2Small<amd_bfloat16>(amd_bfloat16* input_tensor,
                                                           amd_bfloat16* output_tensor,
                                                           const int*    batch_size,
                                                           const int64_t max_batch_size,
                                                           const int64_t max_seq_len,
                                                           const int*    input_lengths,
                                                           const int64_t hidden_size,
                                                           const int*    cu_seq_len,
                                                           hipStream_t   stream);
#endif
#endif

}  // namespace rtp_llm
