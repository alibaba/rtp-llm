// Adapted from src/fastertransformer/cutlass/cutlass_kernels/moe_gemm/moe_kernels.cu
// remove all moeGemm part, and minor modified

#include "moe_topKSoftmax_kernels.h"
#include "src/fastertransformer/cuda/cuda_type_utils.cuh"

#if USING_CUDA
#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11050)
#include <cub/cub.cuh>
#else
#include "3rdparty/cub/cub.cuh"
#endif
#include "src/fastertransformer/cuda/cuda_utils.h"
#endif

#if USING_ROCM
#include <hipcub/hipcub.hpp>
#include "src/fastertransformer/rocm/hip_utils.h"
using namespace fastertransformer::rocm;
#endif

namespace fastertransformer {
static constexpr int WARP_SIZE = 32;

/// Aligned array type
template<typename T,
         /// Number of elements in the array
         int N,
         /// Alignment requirement in bytes
         int Alignment = sizeof(T) * N>
class alignas(Alignment) AlignedArray {
    float data[N];
};

// ====================== Softmax things ===============================
// We have our own implementation of softmax here so we can support transposing the topk_scales
// in the softmax kernel when we extend this module to support expert-choice routing.
template<int TPB>
__launch_bounds__(TPB) __global__
    void moeSoftmax(const float* input, const bool* finished, float* topk_scales, const int num_cols) {
    using BlockReduce = cub::BlockReduce<float, TPB>;
    __shared__ typename BlockReduce::TempStorage tmpStorage;

    __shared__ float normalizing_factor;
    __shared__ float float_max;

    const int thread_row_offset = blockIdx.x * num_cols;

    cub::Sum sum;
    float    threadData(-FLT_MAX);

    // Don't touch finished rows.
    if ((finished != nullptr) && finished[blockIdx.x]) {
        return;
    }

    for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
        const int idx = thread_row_offset + ii;
        threadData    = max(input[idx], threadData);
    }

    const float maxElem = BlockReduce(tmpStorage).Reduce(threadData, cub::Max());
    if (threadIdx.x == 0) {
        float_max = maxElem;
    }
    __syncthreads();

    threadData = 0;

    for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
        const int idx = thread_row_offset + ii;
        threadData += exp((static_cast<float>(input[idx]) - float_max));
    }

    const auto Z = BlockReduce(tmpStorage).Reduce(threadData, sum);

    if (threadIdx.x == 0) {
        normalizing_factor = 1.f / Z;
    }
    __syncthreads();

    for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
        const int   idx  = thread_row_offset + ii;
        const float val  = exp((static_cast<float>(input[idx]) - float_max)) * normalizing_factor;
        topk_scales[idx] = val;
    }
}

template<int TPB>
__launch_bounds__(TPB) __global__ void moeTopK(const float* inputs_after_softmax,
                                               const bool*  finished,
                                               float*       topk_scales,
                                               int*         topk_expertID,
                                               int*         topk_rowColID,
                                               const int    num_experts,
                                               const int    num_cols,
                                               const int    start_expert,
                                               const int    end_expert) {

    using cub_kvp     = cub::KeyValuePair<int, float>;
    using BlockReduce = cub::BlockReduce<cub_kvp, TPB>;
    __shared__ typename BlockReduce::TempStorage tmpStorage;

    cub_kvp     thread_kvp;
    cub::ArgMax arg_max;

    const int num_rows  = gridDim.x;
    const int block_row = blockIdx.x;

    const bool row_is_active      = finished ? !finished[block_row] : true;
    const int  thread_read_offset = blockIdx.x * num_experts;
    for (int col_idx = 0; col_idx < num_cols; ++col_idx) {
        thread_kvp.key   = 0;
        thread_kvp.value = -1.f;  // This is OK because inputs are probabilities

        cub_kvp inp_kvp;
        for (int expert = threadIdx.x; expert < num_experts; expert += TPB) {
            const int idx = thread_read_offset + expert;
            inp_kvp.key   = expert;
            inp_kvp.value = inputs_after_softmax[idx];

            for (int prior_k = 0; prior_k < col_idx; ++prior_k) {
                const int prior_winning_expert = topk_expertID[num_cols * block_row + prior_k];

                if (prior_winning_expert == expert) {
                    inp_kvp = thread_kvp;
                }
            }

            thread_kvp = arg_max(inp_kvp, thread_kvp);
        }

        const cub_kvp result_kvp = BlockReduce(tmpStorage).Reduce(thread_kvp, arg_max);
        if (threadIdx.x == 0) {
            // Ignore experts the node isn't responsible for with expert parallelism
            const int  expert             = result_kvp.key;
            const bool node_uses_expert   = expert >= start_expert && expert < end_expert;
            const bool should_process_row = row_is_active && node_uses_expert;

            const int idx      = num_cols * block_row + col_idx;
            topk_scales[idx]   = result_kvp.value;
            topk_expertID[idx] = should_process_row ? (expert - start_expert) : num_experts;
            assert(topk_expertID[idx] >= 0);
            topk_rowColID[idx] = col_idx * num_rows + block_row;
        }
        __syncthreads();
    }
}

// ====================== TopK softmax things ===============================

/*
  A Top-num_cols gating softmax written to exploit when the number of experts in the MoE layers
  are a small power of 2. This allows us to cleanly share the rows among the threads in
  a single warp and eliminate communication between warps (so no need to use shared mem).

  It fuses the softmax, max and argmax into a single kernel.

  Limitations:
  1) This implementation is intended for when the number of experts is a small power of 2.
  2) This implementation assumes num_cols is small, but will work for any num_cols.
*/

template<int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE) __global__ void topkGatingSoftmax(const float* input,
                                                                              const bool*  finished,
                                                                              float*       topk_scales,
                                                                              const int    num_rows,
                                                                              int*         topk_expertID,
                                                                              int*         topk_rowColID,
                                                                              const int    num_cols,
                                                                              const int    start_expert,
                                                                              const int    end_expert) {
    // We begin by enforcing compile time assertions and setting up compile time constants.
    static_assert(VPT == (VPT & -VPT), "VPT must be power of 2");
    static_assert(NUM_EXPERTS == (NUM_EXPERTS & -NUM_EXPERTS), "NUM_EXPERTS must be power of 2");
    static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG), "BYTES_PER_LDG must be power of 2");
    static_assert(BYTES_PER_LDG <= 16, "BYTES_PER_LDG must be leq 16");

    // Number of bytes each thread pulls in per load
    static constexpr int ELTS_PER_LDG    = BYTES_PER_LDG / sizeof(float);
    static constexpr int ELTS_PER_ROW    = NUM_EXPERTS;
    static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
    static constexpr int LDG_PER_THREAD  = VPT / ELTS_PER_LDG;

    // Restrictions based on previous section.
    static_assert(VPT % ELTS_PER_LDG == 0, "The elements per thread must be a multiple of the elements per ldg");
    static_assert(WARP_SIZE % THREADS_PER_ROW == 0, "The threads per row must cleanly divide the threads per warp");
    static_assert(THREADS_PER_ROW == (THREADS_PER_ROW & -THREADS_PER_ROW), "THREADS_PER_ROW must be power of 2");
    static_assert(THREADS_PER_ROW <= WARP_SIZE, "THREADS_PER_ROW can be at most warp size");

    // We have NUM_EXPERTS elements per row. We specialize for small #experts
    static constexpr int ELTS_PER_WARP = WARP_SIZE * VPT;
    static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
    static constexpr int ROWS_PER_CTA  = WARPS_PER_CTA * ROWS_PER_WARP;

    // Restrictions for previous section.
    static_assert(ELTS_PER_WARP % ELTS_PER_ROW == 0, "The elts per row must cleanly divide the total elt per warp");

    // ===================== From this point, we finally start computing run-time variables. ========================

    // Compute CTA and warp rows. We pack multiple rows into a single warp, and a block contains WARPS_PER_CTA warps.
    // This, each block processes a chunk of rows. We start by computing the start row for each block.
    const int cta_base_row = blockIdx.x * ROWS_PER_CTA;

    // Now, using the base row per thread block, we compute the base row per warp.
    const int warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;

    // The threads in a warp are split into sub-groups that will work on a row.
    // We compute row offset for each thread sub-group
    const int thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
    const int thread_row         = warp_base_row + thread_row_in_warp;

    // Threads with topk_expertID out of bounds should early exit here.
    if (thread_row >= num_rows) {
        return;
    }
    const bool row_is_active = finished ? !finished[thread_row] : true;

    // We finally start setting up the read pointers for each thread. First, each thread jumps to the start of the
    // row it will read.
    const float* thread_row_ptr = input + thread_row * ELTS_PER_ROW;

    // Now, we compute the group each thread belong to in order to determine the first column to start loads.
    const int    thread_group_idx         = threadIdx.x % THREADS_PER_ROW;
    const int    first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
    const float* thread_read_ptr          = thread_row_ptr + first_elt_read_by_thread;

    // Determine the pointer type to use to read in the data depending on the BYTES_PER_LDG template param. In theory,
    // this can support all powers of 2 up to 16.
    using AccessType = AlignedArray<float, ELTS_PER_LDG>;

    // Finally, we pull in the data from global mem
    float             row_chunk[VPT];
    AccessType*       row_chunk_vec_ptr   = reinterpret_cast<AccessType*>(&row_chunk);
    const AccessType* vec_thread_read_ptr = reinterpret_cast<const AccessType*>(thread_read_ptr);
#pragma unroll
    for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
        row_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
    }

    // First, we perform a max reduce within the thread. We can do the max in fp16 safely (I think) and just
    // convert to float afterwards for the exp + sum reduction.
    float thread_max = row_chunk[0];
#pragma unroll
    for (int ii = 1; ii < VPT; ++ii) {
        thread_max = max(thread_max, row_chunk[ii]);
    }

// Now, we find the max within the thread group and distribute among the threads. We use a butterfly reduce.
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
        thread_max = max(thread_max, __shfl_xor_sync(0xFFFFFFFF, thread_max, mask, THREADS_PER_ROW));
    }

    // From this point, thread max in all the threads have the max within the row.
    // Now, we subtract the max from each element in the thread and take the exp. We also compute the thread local sum.
    float row_sum = 0;
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii) {
        row_chunk[ii] = expf(row_chunk[ii] - thread_max);
        row_sum += row_chunk[ii];
    }

// Now, we perform the sum reduce within each thread group. Similar to the max reduce, we use a bufferfly pattern.
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
        row_sum += __shfl_xor_sync(0xFFFFFFFF, row_sum, mask, THREADS_PER_ROW);
    }

    // From this point, all threads have the max and the sum for their rows in the thread_max and thread_sum variables
    // respectively. Finally, we can scale the rows for the softmax. Technically, for top-k gating we don't need to
    // compute the entire softmax row. We can likely look at the maxes and only compute for the top-k values in the row.
    // However, this kernel will likely not be a bottle neck and it seems better to closer match torch and find the
    // argmax after computing the softmax.
    const float reciprocal_row_sum = 1.f / row_sum;

#pragma unroll
    for (int ii = 0; ii < VPT; ++ii) {
        row_chunk[ii] = row_chunk[ii] * reciprocal_row_sum;
    }

    // Now, softmax_res contains the softmax of the row chunk. Now, I want to find the topk elements in each row, along
    // with the max index.
    int                  start_col          = first_elt_read_by_thread;
    static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;

    for (int col_idx = 0; col_idx < num_cols; ++col_idx) {
        // First, each thread does the local argmax
        float max_val = row_chunk[0];
        int   expert  = start_col;
#pragma unroll
        for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD; ++ldg, col += COLS_PER_GROUP_LDG) {
#pragma unroll
            for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {
                float val = row_chunk[ldg * ELTS_PER_LDG + ii];

                // No check on the experts here since columns with the smallest index are processed first and only
                // updated if > (not >=)
                if (val > max_val) {
                    max_val = val;
                    expert  = col + ii;
                }
            }
        }

// Now, we perform the argmax reduce. We use the butterfly pattern so threads reach consensus about the max.
// This will be useful for K > 1 so that the threads can agree on "who" had the max value. That thread can
// then blank out their max with -inf and the warp can run more iterations...
#pragma unroll
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
            float other_max    = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, THREADS_PER_ROW);
            int   other_expert = __shfl_xor_sync(0xFFFFFFFF, expert, mask, THREADS_PER_ROW);

            // We want lower topk_expertID to "win" in every thread so we break ties this way
            if (other_max > max_val || (other_max == max_val && other_expert < expert)) {
                max_val = other_max;
                expert  = other_expert;
            }
        }

        // Write the max for this k iteration to global memory.
        if (thread_group_idx == 0) {
            // Add a guard to ignore experts not included by this node
            const bool node_uses_expert   = expert >= start_expert && expert < end_expert;
            const bool should_process_row = row_is_active && node_uses_expert;

            // The lead thread from each sub-group will write out the final results to global memory. (This will be a
            // single) thread per row of the input/topk_scales matrices.
            const int idx      = num_cols * thread_row + col_idx;
            topk_scales[idx]   = max_val;
            topk_expertID[idx] = should_process_row ? (expert - start_expert) : NUM_EXPERTS;
            topk_rowColID[idx] = idx;
        }

        // Finally, we clear the value in the thread with the current max if there is another iteration to run.
        if (col_idx + 1 < num_cols) {
            const int ldg_group_for_expert     = expert / COLS_PER_GROUP_LDG;
            const int thread_to_clear_in_group = (expert / ELTS_PER_LDG) % THREADS_PER_ROW;

            // Only the thread in the group which produced the max will reset the "winning" value to -inf.
            if (thread_group_idx == thread_to_clear_in_group) {
                const int offset_for_expert = expert % ELTS_PER_LDG;
                // Safe to set to any negative value since row_chunk values must be between 0 and 1.
                row_chunk[ldg_group_for_expert * ELTS_PER_LDG + offset_for_expert] = -10000.f;
            }
        }
    }
}

namespace detail {
// Constructs some constants needed to partition the work across threads at compile time.
template<int EXPERTS, int BYTES_PER_LDG>
struct TopkConstants {
    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(float);
    static_assert(EXPERTS / (ELTS_PER_LDG * WARP_SIZE) == 0 || EXPERTS % (ELTS_PER_LDG * WARP_SIZE) == 0, "");
    static constexpr int VECs_PER_THREAD = std::max(1, EXPERTS / (ELTS_PER_LDG * WARP_SIZE));
    static constexpr int VPT             = VECs_PER_THREAD * ELTS_PER_LDG;
    static constexpr int THREADS_PER_ROW = EXPERTS / VPT;
    static constexpr int ROWS_PER_WARP   = WARP_SIZE / THREADS_PER_ROW;
};
}  // namespace detail

template<int EXPERTS, int WARPS_PER_TB>
void topkGatingSoftmaxLauncherHelper(const float* input,
                                     const bool*  finished,
                                     float*       topk_scales,
                                     int*         topk_expertID,
                                     int*         topk_rowColID,
                                     const int    num_rows,
                                     const int    num_cols,
                                     const int    start_expert,
                                     const int    end_expert,
                                     cudaStream_t stream) {
    static constexpr std::size_t MAX_BYTES_PER_LDG = 16;

    static constexpr int BYTES_PER_LDG = std::min(MAX_BYTES_PER_LDG, sizeof(float) * EXPERTS);
    using Constants                    = detail::TopkConstants<EXPERTS, BYTES_PER_LDG>;
    static constexpr int VPT           = Constants::VPT;
    static constexpr int ROWS_PER_WARP = Constants::ROWS_PER_WARP;
    const int            num_warps     = (num_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
    const int            num_blocks    = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB;

    dim3 block_dim(WARP_SIZE, WARPS_PER_TB);
    topkGatingSoftmax<VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG><<<num_blocks, block_dim, 0, stream>>>(
        input, finished, topk_scales, num_rows, topk_expertID, topk_rowColID, num_cols, start_expert, end_expert);
}

void topkGatingSoftmax_KL(const float* input,
                          const bool*  finished,
                          float*       softmax_temp_output,
                          float*       topk_scales,
                          int*         topk_expertID,
                          int*         topk_rowColID,
                          const int    num_rows,
                          const int    num_experts,
                          const int    num_cols,
                          const int    start_expert,
                          const int    end_expert,
                          cudaStream_t stream) {
    static constexpr int WARPS_PER_TB = 4;

    switch (num_experts) {
        case 1: {
            topkGatingSoftmaxLauncherHelper<1, WARPS_PER_TB>(input,
                                                             finished,
                                                             topk_scales,
                                                             topk_expertID,
                                                             topk_rowColID,
                                                             num_rows,
                                                             num_cols,
                                                             start_expert,
                                                             end_expert,
                                                             stream);
            break;
        }
        case 2: {
            topkGatingSoftmaxLauncherHelper<2, WARPS_PER_TB>(input,
                                                             finished,
                                                             topk_scales,
                                                             topk_expertID,
                                                             topk_rowColID,
                                                             num_rows,
                                                             num_cols,
                                                             start_expert,
                                                             end_expert,
                                                             stream);
            break;
        }
        case 4: {
            topkGatingSoftmaxLauncherHelper<4, WARPS_PER_TB>(input,
                                                             finished,
                                                             topk_scales,
                                                             topk_expertID,
                                                             topk_rowColID,
                                                             num_rows,
                                                             num_cols,
                                                             start_expert,
                                                             end_expert,
                                                             stream);
            break;
        }
        case 8: {
            topkGatingSoftmaxLauncherHelper<8, WARPS_PER_TB>(input,
                                                             finished,
                                                             topk_scales,
                                                             topk_expertID,
                                                             topk_rowColID,
                                                             num_rows,
                                                             num_cols,
                                                             start_expert,
                                                             end_expert,
                                                             stream);
            break;
        }
        case 16: {
            topkGatingSoftmaxLauncherHelper<16, WARPS_PER_TB>(input,
                                                              finished,
                                                              topk_scales,
                                                              topk_expertID,
                                                              topk_rowColID,
                                                              num_rows,
                                                              num_cols,
                                                              start_expert,
                                                              end_expert,
                                                              stream);
            break;
        }
        case 32: {
            topkGatingSoftmaxLauncherHelper<32, WARPS_PER_TB>(input,
                                                              finished,
                                                              topk_scales,
                                                              topk_expertID,
                                                              topk_rowColID,
                                                              num_rows,
                                                              num_cols,
                                                              start_expert,
                                                              end_expert,
                                                              stream);
            break;
        }
        case 64: {
            topkGatingSoftmaxLauncherHelper<64, WARPS_PER_TB>(input,
                                                              finished,
                                                              topk_scales,
                                                              topk_expertID,
                                                              topk_rowColID,
                                                              num_rows,
                                                              num_cols,
                                                              start_expert,
                                                              end_expert,
                                                              stream);
            break;
        }
        case 128: {
            topkGatingSoftmaxLauncherHelper<128, WARPS_PER_TB>(input,
                                                               finished,
                                                               topk_scales,
                                                               topk_expertID,
                                                               topk_rowColID,
                                                               num_rows,
                                                               num_cols,
                                                               start_expert,
                                                               end_expert,
                                                               stream);
            break;
        }
        case 256: {
            topkGatingSoftmaxLauncherHelper<256, WARPS_PER_TB>(input,
                                                               finished,
                                                               topk_scales,
                                                               topk_expertID,
                                                               topk_rowColID,
                                                               num_rows,
                                                               num_cols,
                                                               start_expert,
                                                               end_expert,
                                                               stream);
            break;
        }
        default: {
            static constexpr int TPB = 256;
            FT_CHECK(softmax_temp_output != nullptr);
            moeSoftmax<TPB><<<num_rows, TPB, 0, stream>>>(input, finished, softmax_temp_output, num_experts);
            moeTopK<TPB><<<num_rows, TPB, 0, stream>>>(softmax_temp_output,
                                                       finished,
                                                       topk_scales,
                                                       topk_expertID,
                                                       topk_rowColID,
                                                       num_experts,
                                                       num_cols,
                                                       start_expert,
                                                       end_expert);
        }
    }
}

void sort_KL(const int*   keys_in,
             const int*   values_in,
             int*         keys_out,
             int*         values_out,
             const size_t num_key_value_pairs,
             size_t       num_bits,
             cudaStream_t stream) {

    void*  ws_tmp  = NULL;
    size_t ws_size = 0;
    cub::DeviceRadixSort::SortPairs(
        ws_tmp, ws_size, keys_in, keys_out, values_in, values_out, num_key_value_pairs, 0, num_bits, stream);
    check_cuda_error(cudaMalloc(&ws_tmp, ws_size));
    cub::DeviceRadixSort::SortPairs(
        ws_tmp, ws_size, keys_in, keys_out, values_in, values_out, num_key_value_pairs, 0, num_bits, stream);
    check_cuda_error(cudaFree(ws_tmp));
}

// ============================== Infer GEMM sizes =================================
// TODO Could linear search be better for small # experts
__device__ inline int findTotalEltsLeqTarget(const int* sorted_indices, const int arr_length, const int target) {
    int64_t low = 0, high = arr_length - 1, target_location = -1;
    while (low <= high) {
        int64_t mid = (low + high) / 2;

        if (sorted_indices[mid] > target) {
            high = mid - 1;
        } else {
            low             = mid + 1;
            target_location = mid;
        }
    }
    return target_location + 1;
}

// Sets up the gemm assuming the inputs, experts and outputs are stored in row major order.
// Assumes we want to perform output = matmul(inputs, experts) + bias
//
// "total_rows_before_expert" contains the index one past the last occurrence of the corresponding expert.
// e.g. Index 0 is the start offset of expert 1, the final entry is the total number of active rows
__global__ void computeTotalRowsBeforeExpertKernel(const int*    sorted_experts,
                                                   const int     sorted_experts_len,
                                                   const int64_t num_experts,
                                                   int*          total_rows_before_expert) {
    // First, compute the global tid. We only need 1 thread per expert.
    const int expert = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert >= num_experts) {
        return;
    }

    // This should construct the last index where each expert occurs.
    total_rows_before_expert[expert] = findTotalEltsLeqTarget(sorted_experts, sorted_experts_len, expert);
}
void computeTotalRowsBeforeExpert_KL(const int*   sorted_indices,
                                     const int    total_indices,
                                     const int    num_experts,
                                     int*         total_rows_before_expert,
                                     cudaStream_t stream) {
    const int threads = std::min(1024, num_experts);
    const int blocks  = (num_experts + threads - 1) / threads;

    computeTotalRowsBeforeExpertKernel<<<blocks, threads, 0, stream>>>(
        sorted_indices, total_indices, num_experts, total_rows_before_expert);
}

// ========================== Permutation things =======================================

// Duplicated and permutes rows for MoE. In addition, reverse the permutation map to help with finalizing routing.

// "expanded_x_row" simply means that the number of values is num_rows x k. It is "expanded" since we will have to
// duplicate some rows in the input matrix to match the dimensions. Duplicates will always get routed to separate
// experts in the end.

// Note that the expanded_dest_row_to_expanded_source_row map referred to here has indices in the range (0,
// k*rows_in_input - 1). However, it is set up so that index 0, rows_in_input, 2*rows_in_input ... (k-1)*rows_in_input
// all map to row 0 in the original matrix. Thus, to know where to read in the source matrix, we simply take the modulus
// of the expanded index.

template<typename T, bool CHECK_SKIPPED>
__global__ void permutInputRowsKernel(const T*       unpermuted_input,
                                      T*             permuted_output,
                                      const int*     expanded_dest_row_to_expanded_source_row,
                                      int*           expanded_source_row_to_expanded_dest_row,
                                      const int      num_rows,
                                      const int      num_cols,
                                      const int64_t* num_dest_rows,
                                      const int      k_bits) {

    // Reverse permutation map.
    // I do this so that later, we can use the source -> dest map to do the k-way reduction and unpermuting. I need the
    // reverse map for that reduction to allow each threadblock to do 1 k-way reduce without atomics later in MoE. 1
    // thread block will be responsible for all k summations.
    const int expanded_dest_row   = blockIdx.x;
    const int expanded_source_row = expanded_dest_row_to_expanded_source_row[expanded_dest_row];
    if (threadIdx.x == 0) {
        expanded_source_row_to_expanded_dest_row[expanded_source_row] = expanded_dest_row;
    }

    if (!CHECK_SKIPPED || blockIdx.x < *num_dest_rows) {
        // Duplicate and permute rows
        const int source_row = expanded_source_row >> k_bits;

        const T* source_row_ptr = unpermuted_input + source_row * num_cols;
        T*       dest_row_ptr   = permuted_output + expanded_dest_row * num_cols;

        for (int tid = threadIdx.x; tid < num_cols; tid += blockDim.x) {
            dest_row_ptr[tid] = source_row_ptr[tid];
        }
    }
}

template<typename T>
void permutInputRows_KL(const T*       unpermuted_input,
                        T*             permuted_output,
                        const int*     expanded_dest_row_to_expanded_source_row,
                        int*           expanded_source_row_to_expanded_dest_row,
                        const int      num_rows,
                        const int      num_cols,
                        const int64_t* num_valid_tokens_ptr,
                        const int      k,
                        cudaStream_t   stream) {
    const int k_bits  = log2(k);
    const int blocks  = num_rows * k;
    const int threads = std::min(num_cols, 1024);
    auto func = (num_valid_tokens_ptr != nullptr) ? permutInputRowsKernel<T, true> : permutInputRowsKernel<T, false>;
    func<<<blocks, threads, 0, stream>>>(unpermuted_input,
                                         permuted_output,
                                         expanded_dest_row_to_expanded_source_row,
                                         expanded_source_row_to_expanded_dest_row,
                                         num_rows,
                                         num_cols,
                                         num_valid_tokens_ptr,
                                         k_bits);
}

#define INSTANT_permutInputRows_KL(T)                                                                                  \
    template void permutInputRows_KL(const T*       unpermuted_input,                                                  \
                                     T*             permuted_output,                                                   \
                                     const int*     expanded_dest_row_to_expanded_source_row,                          \
                                     int*           expanded_source_row_to_expanded_dest_row,                          \
                                     const int      num_rows,                                                          \
                                     const int      num_cols,                                                          \
                                     const int64_t* num_valid_tokens_ptr,                                              \
                                     const int      k,                                                                 \
                                     cudaStream_t   stream)
INSTANT_permutInputRows_KL(float);
INSTANT_permutInputRows_KL(half);
#ifdef ENABLE_BF16
INSTANT_permutInputRows_KL(__nv_bfloat16);
#endif
#undef INSTANT_permutInputRows_KL

enum class ScaleMode2 : int {
    NO_SCALE     = 0,
    DEFAULT      = 1,
    RENORM_SCALE = 2,
};
// Final kernel to unpermute and scale
// This kernel unpermutes the original data, does the k-way reduction and performs the final skip connection.
template<typename T, int RESIDUAL_NUM, bool HAS_BIAS, ScaleMode2 SCALE_MODE, bool CHECK_SKIPPED>
__global__ void finalizeMoeRoutingKernel(const T*       expanded_permuted_rows,
                                         T*             reduced_unpermuted_output,
                                         const T*       skip_1,
                                         const T*       skip_2,
                                         const T*       bias,
                                         const float*   scales,
                                         const int*     expanded_source_row_to_expanded_dest_row,
                                         const int*     expert_for_source_row,
                                         const int      cols,
                                         const int      k,
                                         const int64_t* num_valid_ptr) {
    const int  original_row    = blockIdx.x;
    const int  num_rows        = gridDim.x;
    const auto offset          = original_row * cols;
    T*         reduced_row_ptr = reduced_unpermuted_output + offset;
    const T*   skip_1_row_ptr{};
    const T*   skip_2_row_ptr{};

    if (RESIDUAL_NUM >= 1) {
        skip_1_row_ptr = skip_1 + offset;
    }

    if (RESIDUAL_NUM == 2) {
        skip_2_row_ptr = skip_2 + offset;
    }
    const int64_t num_valid = *num_valid_ptr;
    for (int tid = threadIdx.x; tid < cols; tid += blockDim.x) {
        T     thread_output{0.f};
        float row_rescale{0.f};
        for (int k_idx = 0; k_idx < k; ++k_idx) {
            const int64_t k_offset              = original_row * k + k_idx;
            const int     expanded_permuted_row = expanded_source_row_to_expanded_dest_row[k_offset];

            const float row_scale = (SCALE_MODE == ScaleMode2::NO_SCALE) ? 1.f : scales[k_offset];
            if constexpr (SCALE_MODE == ScaleMode2::RENORM_SCALE) {
                row_rescale = row_rescale + row_scale;
            }

            // Check after row sum has accumulated
            if (CHECK_SKIPPED && expanded_permuted_row >= num_valid) {
                continue;
            }

            const T* expanded_permuted_rows_row_ptr = expanded_permuted_rows + expanded_permuted_row * cols;

            const int expert_idx = expert_for_source_row[k_offset];

            const T* bias_ptr   = bias + expert_idx * cols;
            const T  bias_value = HAS_BIAS ? bias_ptr[tid] : T(0.f);
            if (blockIdx.x == 0 && threadIdx.x == 0) {
                // printf("permuted_rowID, oriRowID, thread_output = %d, %d, %f, %f, %f, %f, %f, %f\n",
                //        expanded_permuted_row,
                //        expanded_original_row,
                //        static_cast<float>(thread_output),
                //        row_scale,
                //        scales[k_offset],
                //        static_cast<float>(expanded_permuted_rows_row_ptr[tid]),
                //        static_cast<float>(expanded_permuted_rows_row_ptr[tid] + bias_value),
                //        static_cast<float>(thread_output)
                //            + row_scale * static_cast<float>(expanded_permuted_rows_row_ptr[tid] + bias_value));
            }
            thread_output = static_cast<float>(thread_output)
                            + row_scale * static_cast<float>(expanded_permuted_rows_row_ptr[tid] + bias_value);
        }

        if (SCALE_MODE == ScaleMode2::RENORM_SCALE && (!CHECK_SKIPPED || thread_output)) {
            assert(row_rescale != 0.f);
            thread_output = static_cast<float>(thread_output) / row_rescale;
        }

        if (RESIDUAL_NUM == 1) {
            thread_output = thread_output + skip_1_row_ptr[tid];
        } else if (RESIDUAL_NUM == 2) {
            thread_output = thread_output + skip_1_row_ptr[tid] + skip_2_row_ptr[tid];
        }
        reduced_row_ptr[tid] = thread_output;
    }
}

template<typename T, int RESIDUAL_NUM>
void finalizeMoeRoutingKernelLauncherSelectBias(const T*       expanded_permuted_rows,
                                                T*             reduced_unpermuted_output,
                                                const T*       skip_1,
                                                const T*       skip_2,
                                                const T*       bias,
                                                const float*   scales,
                                                const int*     expanded_source_row_to_expanded_dest_row,
                                                const int*     expert_for_source_row,
                                                const int      num_rows,
                                                const int      cols,
                                                const int      k,
                                                const int64_t* num_valid_ptr,
                                                const int      tp_rank,
                                                int            normalization_mode,
                                                cudaStream_t   stream) {
    const int blocks  = num_rows;
    const int threads = std::min(cols, 1024);

    // Only add bias on rank 0 for tensor parallelism
    // const bool is_rank_0 = parallelism_config.tp_rank == 0;
    const bool is_rank_0 = (tp_rank == 0);
    const bool has_bias  = bias != nullptr && is_rank_0;

    const bool check_finished = num_valid_ptr != nullptr;

    int renorm_scales = normalization_mode;
    // ScaleMode2 renorm_scales = ScaleMode2::DEFAULT;
    // if (normalization_mode == MOEExpertScaleNormalizationMode::RENORMALIZE) {
    //     renorm_scales = k == 1 ? ScaleMode2::NO_SCALE : ScaleMode2::RENORM_SCALE;
    // }

    using FuncPtr             = decltype(&finalizeMoeRoutingKernel<T, RESIDUAL_NUM, false, ScaleMode2::DEFAULT, false>);
    FuncPtr func_map[2][3][2] = {
        {
            {&finalizeMoeRoutingKernel<T, RESIDUAL_NUM, false, ScaleMode2::NO_SCALE, false>,
             &finalizeMoeRoutingKernel<T, RESIDUAL_NUM, true, ScaleMode2::NO_SCALE, false>},

            {&finalizeMoeRoutingKernel<T, RESIDUAL_NUM, false, ScaleMode2::DEFAULT, false>,
             &finalizeMoeRoutingKernel<T, RESIDUAL_NUM, true, ScaleMode2::DEFAULT, false>},

            {&finalizeMoeRoutingKernel<T, RESIDUAL_NUM, false, ScaleMode2::RENORM_SCALE, false>,
             &finalizeMoeRoutingKernel<T, RESIDUAL_NUM, true, ScaleMode2::RENORM_SCALE, false>},
        },
        {
            {&finalizeMoeRoutingKernel<T, RESIDUAL_NUM, false, ScaleMode2::NO_SCALE, true>,
             &finalizeMoeRoutingKernel<T, RESIDUAL_NUM, true, ScaleMode2::NO_SCALE, true>},

            {&finalizeMoeRoutingKernel<T, RESIDUAL_NUM, false, ScaleMode2::DEFAULT, true>,
             &finalizeMoeRoutingKernel<T, RESIDUAL_NUM, true, ScaleMode2::DEFAULT, true>},

            {&finalizeMoeRoutingKernel<T, RESIDUAL_NUM, false, ScaleMode2::RENORM_SCALE, true>,
             &finalizeMoeRoutingKernel<T, RESIDUAL_NUM, true, ScaleMode2::RENORM_SCALE, true>},
        }};
    auto* const func = func_map[check_finished][int(renorm_scales)][has_bias];
    func<<<blocks, threads, 0, stream>>>(expanded_permuted_rows,
                                         reduced_unpermuted_output,
                                         skip_1,
                                         skip_2,
                                         bias,
                                         scales,
                                         expanded_source_row_to_expanded_dest_row,
                                         expert_for_source_row,
                                         cols,
                                         k,
                                         num_valid_ptr);
}

template<typename T>
void finalizeMoeRoutingKernelLauncher(const T*       expanded_permuted_rows,
                                      T*             reduced_unpermuted_output,
                                      const T*       skip_1,
                                      const T*       skip_2,
                                      const T*       bias,
                                      const float*   scales,
                                      const int*     expanded_source_row_to_expanded_dest_row,
                                      const int*     expert_for_source_row,
                                      const int      num_rows,
                                      const int      cols,
                                      const int      k,
                                      const int64_t* num_valid_ptr,
                                      const int      tp_rank,
                                      int            normalization_mode,
                                      cudaStream_t   stream) {
    finalizeMoeRoutingKernelLauncherSelectBias<T, 0>(expanded_permuted_rows,
                                                     reduced_unpermuted_output,
                                                     skip_1,
                                                     skip_2,
                                                     bias,
                                                     scales,
                                                     expanded_source_row_to_expanded_dest_row,
                                                     expert_for_source_row,
                                                     num_rows,
                                                     cols,
                                                     k,
                                                     num_valid_ptr,
                                                     tp_rank,
                                                     normalization_mode,
                                                     stream);
}

#define INSTANT_finalizeMoeRoutingKernelLauncher(T)                                                                    \
    template void finalizeMoeRoutingKernelLauncher(const T*       expanded_permuted_rows,                              \
                                                   T*             reduced_unpermuted_output,                           \
                                                   const T*       skip_1,                                              \
                                                   const T*       skip_2,                                              \
                                                   const T*       bias,                                                \
                                                   const float*   scales,                                              \
                                                   const int*     expanded_source_row_to_expanded_dest_row,            \
                                                   const int*     expert_for_source_row,                               \
                                                   const int      num_rows,                                            \
                                                   const int      cols,                                                \
                                                   const int      k,                                                   \
                                                   const int64_t* num_valid_ptr,                                       \
                                                   const int      tp_rank,                                             \
                                                   int            normalization_mode,                                  \
                                                   cudaStream_t   stream)
INSTANT_finalizeMoeRoutingKernelLauncher(float);
INSTANT_finalizeMoeRoutingKernelLauncher(half);
#ifdef ENABLE_BF16
INSTANT_finalizeMoeRoutingKernelLauncher(__nv_bfloat16);
#endif
#undef INSTANT_finalizeMoeRoutingKernelLauncher

}  // namespace fastertransformer