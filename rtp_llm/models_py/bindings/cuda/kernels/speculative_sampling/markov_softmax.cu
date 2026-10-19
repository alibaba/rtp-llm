#include "rtp_llm/models_py/bindings/cuda/kernels/speculative_sampling/markov_softmax.h"

#if USING_CUDA

#include <math.h>
#include <algorithm>
#include <cub/block/block_reduce.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace rtp_llm {
namespace {

constexpr int kMarkovSoftmaxThreads = 1024;

// Target enough (row x split) CTAs to keep the GPU fed even at small batch;
// every split keeps at least ~2k elements of work so the second kernel's
// per-CTA partial combine stays noise.
constexpr int kMarkovSoftmaxTargetCtas   = 256;
constexpr int kMarkovSoftmaxMaxSplits    = 16;
constexpr int kMarkovSoftmaxMinChunkSize = 2048;

// Local functors keep the kernel independent of the cub::Max/cub::Sum header
// location, which moves between cub versions (thread_operators.hpp vs .cuh).
struct MarkovMaxOp {
    __device__ __forceinline__ float operator()(const float& a, const float& b) const {
        return fmaxf(a, b);
    }
};

struct MarkovArgmaxMaxOp {
    __device__ __forceinline__ float operator()(const float& a, const float& b) const {
        return isnan(a) ? a : (isnan(b) ? b : fmaxf(a, b));
    }
};

// NaNs win argmax; an ascending scan retains the first NaN or finite maximum.
__device__ __forceinline__ bool markovArgmaxGreater(float a, float b) {
    return a > b || (isnan(a) && !isnan(b));
}

struct MarkovSumOp {
    __device__ __forceinline__ float operator()(const float& a, const float& b) const {
        return a + b;
    }
};

struct MarkovMinIdxOp {
    __device__ __forceinline__ int operator()(const int& a, const int& b) const {
        return min(a, b);
    }
};

template<typename T>
struct alignas(4 * sizeof(T)) MarkovVec4 {
    T x, y, z, w;
};

template<typename T>
__device__ __forceinline__ float markovBiasToF32(T value) {
    return static_cast<float>(value);
}

template<>
__device__ __forceinline__ float markovBiasToF32<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template<>
__device__ __forceinline__ float markovBiasToF32<__nv_half>(__nv_half value) {
    return __half2float(value);
}

// Mirrors the eager chain markov_bias.to(fp32) -> logits + bias -> div_(T):
// an exact FP32 cast, an FP32 add, and an FP32 divide, in that order, so the
// per-element softmax inputs are bitwise identical to the materialized chain.
template<typename BiasT>
__device__ __forceinline__ float scaledMarkovLogit(float logit, BiasT bias, float temperature) {
    return (logit + markovBiasToF32<BiasT>(bias)) / temperature;
}

// Online max/sum fold (single read pass): keeps m = max(x_so_far) and
// s = sum(exp(x - m)) without materializing the row.
__device__ __forceinline__ void onlineSoftmaxFold(float& m, float& s, float x) {
    // Masked elements contribute no mass, including before the first finite logit.
    if (x == -INFINITY) {
        return;
    }
    if (x > m) {
        s = s * expf(m - x) + 1.0f;
        m = x;
    } else {
        s += expf(x - m);
    }
}

// Per-(row, split) chunk bounds in kernel-native units (float4 groups on the
// vector path, elements on the scalar path).
__device__ __forceinline__ void chunkBounds(int split, int units, int chunk, int& begin, int& end) {
    begin = split * chunk;
    end   = min(units, begin + chunk);
}

// Kernel 1: fold each (row, split) chunk into an online (max, sum) partial.
template<typename BiasT, bool kVectorized>
__global__ void markovSoftmaxPartialsKernel(const float* __restrict__ logits,
                                            int64_t logits_row_stride,
                                            const BiasT* __restrict__ markov_bias,
                                            const float* __restrict__ temperature,
                                            float* __restrict__ partials_max,
                                            float* __restrict__ partials_sum,
                                            const int vocab_size,
                                            const int chunk) {
    const int    split    = blockIdx.x;
    const int    row      = blockIdx.y;
    const int    splits   = gridDim.x;
    const float  row_temp = temperature[row];
    const float* lrow     = logits + static_cast<int64_t>(row) * logits_row_stride;
    const BiasT* brow     = markov_bias + static_cast<int64_t>(row) * vocab_size;

    float m = -INFINITY;
    float s = 0.0f;
    int   begin, end;
    if constexpr (kVectorized) {
        const int                units = vocab_size / 4;
        const float4*            l4    = reinterpret_cast<const float4*>(lrow);
        const MarkovVec4<BiasT>* b4    = reinterpret_cast<const MarkovVec4<BiasT>*>(brow);
        chunkBounds(split, units, chunk, begin, end);
        for (int i = begin + threadIdx.x; i < end; i += kMarkovSoftmaxThreads) {
            const float4            l = l4[i];
            const MarkovVec4<BiasT> b = b4[i];
            onlineSoftmaxFold(m, s, scaledMarkovLogit(l.x, b.x, row_temp));
            onlineSoftmaxFold(m, s, scaledMarkovLogit(l.y, b.y, row_temp));
            onlineSoftmaxFold(m, s, scaledMarkovLogit(l.z, b.z, row_temp));
            onlineSoftmaxFold(m, s, scaledMarkovLogit(l.w, b.w, row_temp));
        }
    } else {
        chunkBounds(split, vocab_size, chunk, begin, end);
        for (int i = begin + threadIdx.x; i < end; i += kMarkovSoftmaxThreads) {
            onlineSoftmaxFold(m, s, scaledMarkovLogit(lrow[i], brow[i], row_temp));
        }
    }

    // Combine the per-thread online folds: reduce the thread maxima, rescale
    // every thread's sum onto that common max, then reduce the sums. Threads
    // that saw no elements carry (m=-INF, s=0) and rescale to exactly 0.
    typedef cub::BlockReduce<float, kMarkovSoftmaxThreads> BlockReduceT;
    __shared__ typename BlockReduceT::TempStorage          temp_storage;
    __shared__ float                                       shared_stat;

    const float block_max = BlockReduceT(temp_storage).Reduce(m, MarkovMaxOp());
    if (threadIdx.x == 0) {
        shared_stat = block_max;
    }
    __syncthreads();
    const float chunk_max = shared_stat;
    const float rescaled  = s == 0.0f ? 0.0f : s * expf(m - chunk_max);
    const float block_sum = BlockReduceT(temp_storage).Reduce(rescaled, MarkovSumOp());
    if (threadIdx.x == 0) {
        partials_max[static_cast<int64_t>(row) * splits + split] = chunk_max;
        partials_sum[static_cast<int64_t>(row) * splits + split] = block_sum;
    }
}

// Kernel 2: combine per-row partials and write q to both outputs.
template<typename BiasT, bool kVectorized>
__global__ void markovSoftmaxWriteKernel(const float* __restrict__ logits,
                                         int64_t logits_row_stride,
                                         const BiasT* __restrict__ markov_bias,
                                         const float* __restrict__ temperature,
                                         const float* __restrict__ partials_max,
                                         const float* __restrict__ partials_sum,
                                         float* __restrict__ q_all_probs,
                                         int64_t q_all_probs_row_stride,
                                         float* __restrict__ q_contig,
                                         const int vocab_size,
                                         const int chunk) {
    const int   split    = blockIdx.x;
    const int   row      = blockIdx.y;
    const int   splits   = gridDim.x;
    const float row_temp = temperature[row];

    float row_max = -INFINITY;
    for (int i = 0; i < splits; ++i) {
        row_max = fmaxf(row_max, partials_max[static_cast<int64_t>(row) * splits + i]);
    }
    float row_sum = 0.0f;
    for (int i = 0; i < splits; ++i) {
        const float partial_sum = partials_sum[static_cast<int64_t>(row) * splits + i];
        if (partial_sum != 0.0f) {
            row_sum += partial_sum * expf(partials_max[static_cast<int64_t>(row) * splits + i] - row_max);
        }
    }

    const float* lrow   = logits + static_cast<int64_t>(row) * logits_row_stride;
    const BiasT* brow   = markov_bias + static_cast<int64_t>(row) * vocab_size;
    float*       q_row  = q_all_probs + static_cast<int64_t>(row) * q_all_probs_row_stride;
    float*       q_crow = q_contig != nullptr ? q_contig + static_cast<int64_t>(row) * vocab_size : nullptr;

    int begin, end;
    if constexpr (kVectorized) {
        const int                units = vocab_size / 4;
        const float4*            l4    = reinterpret_cast<const float4*>(lrow);
        const MarkovVec4<BiasT>* b4    = reinterpret_cast<const MarkovVec4<BiasT>*>(brow);
        float4*                  q4    = reinterpret_cast<float4*>(q_row);
        float4*                  qc4   = q_crow != nullptr ? reinterpret_cast<float4*>(q_crow) : nullptr;
        chunkBounds(split, units, chunk, begin, end);
        for (int i = begin + threadIdx.x; i < end; i += kMarkovSoftmaxThreads) {
            const float4            l = l4[i];
            const MarkovVec4<BiasT> b = b4[i];
            float4                  q;
            q.x   = expf(scaledMarkovLogit(l.x, b.x, row_temp) - row_max) / row_sum;
            q.y   = expf(scaledMarkovLogit(l.y, b.y, row_temp) - row_max) / row_sum;
            q.z   = expf(scaledMarkovLogit(l.z, b.z, row_temp) - row_max) / row_sum;
            q.w   = expf(scaledMarkovLogit(l.w, b.w, row_temp) - row_max) / row_sum;
            q4[i] = q;
            if (qc4 != nullptr) {
                qc4[i] = q;
            }
        }
    } else {
        chunkBounds(split, vocab_size, chunk, begin, end);
        for (int i = begin + threadIdx.x; i < end; i += kMarkovSoftmaxThreads) {
            const float q = expf(scaledMarkovLogit(lrow[i], brow[i], row_temp) - row_max) / row_sum;
            q_row[i]      = q;
            if (q_crow != nullptr) {
                q_crow[i] = q;
            }
        }
    }
}

template<typename BiasT, bool kVectorized>
cudaError_t launchMarkovSoftmaxStep(const float* logits,
                                    int64_t      logits_row_stride,
                                    const BiasT* markov_bias,
                                    const float* temperature,
                                    float*       q_all_probs,
                                    int64_t      q_all_probs_row_stride,
                                    float*       q_contig,
                                    int          batch_size,
                                    int          vocab_size,
                                    float*       partials,
                                    cudaStream_t stream) {
    const int  splits = markovSoftmaxStepSplits(batch_size, vocab_size);
    const int  units  = kVectorized ? vocab_size / 4 : vocab_size;
    const int  chunk  = (units + splits - 1) / splits;
    const dim3 grid(splits, batch_size);
    const dim3 block(kMarkovSoftmaxThreads);
    float*     partials_max = partials;
    float*     partials_sum = partials + static_cast<size_t>(batch_size) * splits;

    markovSoftmaxPartialsKernel<BiasT, kVectorized><<<grid, block, 0, stream>>>(
        logits, logits_row_stride, markov_bias, temperature, partials_max, partials_sum, vocab_size, chunk);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }
    markovSoftmaxWriteKernel<BiasT, kVectorized><<<grid, block, 0, stream>>>(logits,
                                                                             logits_row_stride,
                                                                             markov_bias,
                                                                             temperature,
                                                                             partials_max,
                                                                             partials_sum,
                                                                             q_all_probs,
                                                                             q_all_probs_row_stride,
                                                                             q_contig,
                                                                             vocab_size,
                                                                             chunk);
    return cudaGetLastError();
}

// Argmax kernel 1: reduce each (row, split) chunk of the scaled Markov logits
// to a (max, first-argmax) partial. The per-element value function is exactly
// scaledMarkovLogit, so the winning element is bitwise the same row maximum
// the softmax path exponentiates; the tie-break keeps the lowest index, the
// convention of the greedy target sampler and torch::argmax. The argmax index
// is stored as a float; callers gate vocab < 2^24 so the int -> float -> int
// round trip is exact.
template<typename BiasT, bool kVectorized>
__global__ void markovArgmaxPartialsKernel(const float* __restrict__ logits,
                                           int64_t logits_row_stride,
                                           const BiasT* __restrict__ markov_bias,
                                           const float* __restrict__ temperature,
                                           float* __restrict__ partials_max,
                                           float* __restrict__ partials_idx,
                                           const int vocab_size,
                                           const int chunk) {
    const int    split    = blockIdx.x;
    const int    row      = blockIdx.y;
    const int    splits   = gridDim.x;
    const float  row_temp = temperature[row];
    const float* lrow     = logits + static_cast<int64_t>(row) * logits_row_stride;
    const BiasT* brow     = markov_bias + static_cast<int64_t>(row) * vocab_size;

    float m   = -INFINITY;
    int   idx = INT_MAX;
    int   begin, end;
    if constexpr (kVectorized) {
        const int                units = vocab_size / 4;
        const float4*            l4    = reinterpret_cast<const float4*>(lrow);
        const MarkovVec4<BiasT>* b4    = reinterpret_cast<const MarkovVec4<BiasT>*>(brow);
        chunkBounds(split, units, chunk, begin, end);
        int i = begin + threadIdx.x;
        if (i < end) {
            // Seed with the thread's first group so an all-equal chunk still
            // records its first element, matching the first-maximum contract.
            const float4            l = l4[i];
            const MarkovVec4<BiasT> b = b4[i];
            m                         = scaledMarkovLogit(l.x, b.x, row_temp);
            idx                       = i * 4;
            const float vy            = scaledMarkovLogit(l.y, b.y, row_temp);
            if (markovArgmaxGreater(vy, m)) {
                m   = vy;
                idx = i * 4 + 1;
            }
            const float vz = scaledMarkovLogit(l.z, b.z, row_temp);
            if (markovArgmaxGreater(vz, m)) {
                m   = vz;
                idx = i * 4 + 2;
            }
            const float vw = scaledMarkovLogit(l.w, b.w, row_temp);
            if (markovArgmaxGreater(vw, m)) {
                m   = vw;
                idx = i * 4 + 3;
            }
            i += kMarkovSoftmaxThreads;
        }
        for (; i < end; i += kMarkovSoftmaxThreads) {
            const float4            l = l4[i];
            const MarkovVec4<BiasT> b = b4[i];
            // The thread's ascending walk keeps the first NaN or maximum.
            const float vx = scaledMarkovLogit(l.x, b.x, row_temp);
            if (markovArgmaxGreater(vx, m)) {
                m   = vx;
                idx = i * 4;
            }
            const float vy = scaledMarkovLogit(l.y, b.y, row_temp);
            if (markovArgmaxGreater(vy, m)) {
                m   = vy;
                idx = i * 4 + 1;
            }
            const float vz = scaledMarkovLogit(l.z, b.z, row_temp);
            if (markovArgmaxGreater(vz, m)) {
                m   = vz;
                idx = i * 4 + 2;
            }
            const float vw = scaledMarkovLogit(l.w, b.w, row_temp);
            if (markovArgmaxGreater(vw, m)) {
                m   = vw;
                idx = i * 4 + 3;
            }
        }
    } else {
        chunkBounds(split, vocab_size, chunk, begin, end);
        int i = begin + threadIdx.x;
        if (i < end) {
            m   = scaledMarkovLogit(lrow[i], brow[i], row_temp);
            idx = i;
            i += kMarkovSoftmaxThreads;
        }
        for (; i < end; i += kMarkovSoftmaxThreads) {
            const float v = scaledMarkovLogit(lrow[i], brow[i], row_temp);
            if (markovArgmaxGreater(v, m)) {
                m   = v;
                idx = i;
            }
        }
    }

    // Block reduce: the chunk max, then the lowest index among the threads
    // that saw that max. Empty chunks carry (m = -inf, idx = INT_MAX) and
    // never win unless every chunk is empty (impossible for vocab > 0).
    typedef cub::BlockReduce<float, kMarkovSoftmaxThreads> BlockReduceF;
    typedef cub::BlockReduce<int, kMarkovSoftmaxThreads>   BlockReduceI;
    __shared__ typename BlockReduceF::TempStorage          temp_f;
    __shared__ typename BlockReduceI::TempStorage          temp_i;
    __shared__ float                                       shared_max;

    const float block_max = BlockReduceF(temp_f).Reduce(m, MarkovArgmaxMaxOp());
    if (threadIdx.x == 0) {
        shared_max = block_max;
    }
    __syncthreads();
    const int candidate = (m == shared_max || (isnan(m) && isnan(shared_max))) ? idx : INT_MAX;
    const int block_idx = BlockReduceI(temp_i).Reduce(candidate, MarkovMinIdxOp());
    if (threadIdx.x == 0) {
        // Empty chunks carry INT_MAX; fold it to vocab_size, which stays exact
        // through the float partial slot (callers gate vocab < 2^24) and can
        // never beat a real index in the combine below.
        partials_max[static_cast<int64_t>(row) * splits + split] = block_max;
        partials_idx[static_cast<int64_t>(row) * splits + split] = static_cast<float>(min(block_idx, vocab_size));
    }
}

// Argmax kernel 2: combine the per-row partials and write the row token.
// One small CTA per row; the <= kMarkovSoftmaxMaxSplits partials are a few
// cached scalar loads, so no second sync or redundant work is warranted.
__global__ void markovArgmaxWriteKernel(int32_t* __restrict__ tokens_out,
                                        const float* __restrict__ partials_max,
                                        const float* __restrict__ partials_idx,
                                        const int splits,
                                        const int vocab_size) {
    const int row = blockIdx.x;
    if (threadIdx.x == 0) {
        float row_max = -INFINITY;
        int   row_arg = vocab_size;
        for (int i = 0; i < splits; ++i) {
            const float m   = partials_max[static_cast<int64_t>(row) * splits + i];
            const int   idx = static_cast<int>(partials_idx[static_cast<int64_t>(row) * splits + i]);
            if (markovArgmaxGreater(m, row_max) || row_arg == vocab_size) {
                row_max = m;
                row_arg = idx;
            } else if ((m == row_max || (isnan(m) && isnan(row_max))) && idx < row_arg) {
                row_arg = idx;
            }
        }
        tokens_out[row] = static_cast<int32_t>(row_arg);
    }
}

template<typename BiasT, bool kVectorized>
cudaError_t launchMarkovArgmaxStep(const float* logits,
                                   int64_t      logits_row_stride,
                                   const BiasT* markov_bias,
                                   const float* temperature,
                                   int32_t*     tokens_out,
                                   int          batch_size,
                                   int          vocab_size,
                                   float*       partials,
                                   cudaStream_t stream) {
    const int  splits = markovSoftmaxStepSplits(batch_size, vocab_size);
    const int  units  = kVectorized ? vocab_size / 4 : vocab_size;
    const int  chunk  = (units + splits - 1) / splits;
    const dim3 grid(splits, batch_size);
    const dim3 block(kMarkovSoftmaxThreads);
    float*     partials_max = partials;
    float*     partials_idx = partials + static_cast<size_t>(batch_size) * splits;

    markovArgmaxPartialsKernel<BiasT, kVectorized><<<grid, block, 0, stream>>>(
        logits, logits_row_stride, markov_bias, temperature, partials_max, partials_idx, vocab_size, chunk);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }
    markovArgmaxWriteKernel<<<batch_size, 32, 0, stream>>>(tokens_out, partials_max, partials_idx, splits, vocab_size);
    return cudaGetLastError();
}

}  // namespace

int markovSoftmaxStepSplits(int batch_size, int vocab_size) {
    if (batch_size <= 0 || vocab_size <= 0) {
        return 1;
    }
    int splits = (kMarkovSoftmaxTargetCtas + batch_size - 1) / batch_size;
    splits     = std::min(std::max(splits, 1), kMarkovSoftmaxMaxSplits);
    splits     = std::min(splits, std::max(1, vocab_size / kMarkovSoftmaxMinChunkSize));
    return splits;
}

size_t markovSoftmaxStepPartialsBytes(int batch_size, int vocab_size) {
    if (batch_size <= 0) {
        return 0;
    }
    return 2ull * static_cast<size_t>(batch_size) * markovSoftmaxStepSplits(batch_size, vocab_size) * sizeof(float);
}

cudaError_t invokeMarkovSoftmaxStep(const float*    logits,
                                    int64_t         logits_row_stride,
                                    const void*     markov_bias,
                                    MarkovBiasDType bias_dtype,
                                    const float*    temperature,
                                    float*          q_all_probs,
                                    int64_t         q_all_probs_row_stride,
                                    float*          q_contig,
                                    int             batch_size,
                                    int             vocab_size,
                                    float*          partials,
                                    cudaStream_t    stream) {
    if (batch_size == 0) {
        return cudaSuccess;
    }
    if (logits == nullptr || markov_bias == nullptr || temperature == nullptr || q_all_probs == nullptr
        || partials == nullptr || vocab_size <= 0 || logits_row_stride < vocab_size
        || q_all_probs_row_stride < vocab_size) {
        return cudaErrorInvalidValue;
    }
    // Contiguous views can have storage offsets that break vector alignment.
    const bool vectorized = (vocab_size % 4 == 0) && (logits_row_stride % 4 == 0) && (q_all_probs_row_stride % 4 == 0)
                            && reinterpret_cast<uintptr_t>(logits) % alignof(float4) == 0
                            && reinterpret_cast<uintptr_t>(q_all_probs) % alignof(float4) == 0
                            && reinterpret_cast<uintptr_t>(q_contig) % alignof(float4) == 0;

#define LAUNCH_MARKOV_SOFTMAX(BiasT)                                                                                   \
    do {                                                                                                               \
        if (vectorized && reinterpret_cast<uintptr_t>(bias_typed) % alignof(MarkovVec4<BiasT>) == 0) {                 \
            return launchMarkovSoftmaxStep<BiasT, true>(logits,                                                        \
                                                        logits_row_stride,                                             \
                                                        bias_typed,                                                    \
                                                        temperature,                                                   \
                                                        q_all_probs,                                                   \
                                                        q_all_probs_row_stride,                                        \
                                                        q_contig,                                                      \
                                                        batch_size,                                                    \
                                                        vocab_size,                                                    \
                                                        partials,                                                      \
                                                        stream);                                                       \
        }                                                                                                              \
        return launchMarkovSoftmaxStep<BiasT, false>(logits,                                                           \
                                                     logits_row_stride,                                                \
                                                     bias_typed,                                                       \
                                                     temperature,                                                      \
                                                     q_all_probs,                                                      \
                                                     q_all_probs_row_stride,                                           \
                                                     q_contig,                                                         \
                                                     batch_size,                                                       \
                                                     vocab_size,                                                       \
                                                     partials,                                                         \
                                                     stream);                                                          \
    } while (0)

    switch (bias_dtype) {
        case MarkovBiasDType::kFloat32: {
            const float* bias_typed = static_cast<const float*>(markov_bias);
            LAUNCH_MARKOV_SOFTMAX(float);
        }
        case MarkovBiasDType::kBFloat16: {
            const __nv_bfloat16* bias_typed = static_cast<const __nv_bfloat16*>(markov_bias);
            LAUNCH_MARKOV_SOFTMAX(__nv_bfloat16);
        }
        case MarkovBiasDType::kFloat16: {
            const __nv_half* bias_typed = static_cast<const __nv_half*>(markov_bias);
            LAUNCH_MARKOV_SOFTMAX(__nv_half);
        }
        default:
            return cudaErrorInvalidValue;
    }

#undef LAUNCH_MARKOV_SOFTMAX
}

size_t markovArgmaxStepPartialsBytes(int batch_size, int vocab_size) {
    if (batch_size <= 0) {
        return 0;
    }
    return 2ull * static_cast<size_t>(batch_size) * markovSoftmaxStepSplits(batch_size, vocab_size) * sizeof(float);
}

cudaError_t invokeMarkovArgmaxStep(const float*    logits,
                                   int64_t         logits_row_stride,
                                   const void*     markov_bias,
                                   MarkovBiasDType bias_dtype,
                                   const float*    temperature,
                                   int32_t*        tokens_out,
                                   int             batch_size,
                                   int             vocab_size,
                                   float*          partials,
                                   cudaStream_t    stream) {
    if (batch_size == 0) {
        return cudaSuccess;
    }
    // vocab < 2^24 keeps the argmax index exact through its float partial
    // slot; the caller gate must fall back to the dense softmax chain beyond
    // that.
    if (logits == nullptr || markov_bias == nullptr || temperature == nullptr || tokens_out == nullptr
        || partials == nullptr || vocab_size <= 0 || vocab_size >= (1 << 24) || logits_row_stride < vocab_size
        || batch_size < 0) {
        return cudaErrorInvalidValue;
    }
    const bool vectorized = (vocab_size % 4 == 0) && (logits_row_stride % 4 == 0)
                            && reinterpret_cast<uintptr_t>(logits) % alignof(float4) == 0;

#define LAUNCH_MARKOV_ARGMAX(BiasT)                                                                                    \
    do {                                                                                                               \
        if (vectorized && reinterpret_cast<uintptr_t>(bias_typed) % alignof(MarkovVec4<BiasT>) == 0) {                 \
            return launchMarkovArgmaxStep<BiasT, true>(logits,                                                         \
                                                       logits_row_stride,                                              \
                                                       bias_typed,                                                     \
                                                       temperature,                                                    \
                                                       tokens_out,                                                     \
                                                       batch_size,                                                     \
                                                       vocab_size,                                                     \
                                                       partials,                                                       \
                                                       stream);                                                        \
        }                                                                                                              \
        return launchMarkovArgmaxStep<BiasT, false>(                                                                   \
            logits, logits_row_stride, bias_typed, temperature, tokens_out, batch_size, vocab_size, partials, stream); \
    } while (0)

    switch (bias_dtype) {
        case MarkovBiasDType::kFloat32: {
            const float* bias_typed = static_cast<const float*>(markov_bias);
            LAUNCH_MARKOV_ARGMAX(float);
        }
        case MarkovBiasDType::kBFloat16: {
            const __nv_bfloat16* bias_typed = static_cast<const __nv_bfloat16*>(markov_bias);
            LAUNCH_MARKOV_ARGMAX(__nv_bfloat16);
        }
        case MarkovBiasDType::kFloat16: {
            const __nv_half* bias_typed = static_cast<const __nv_half*>(markov_bias);
            LAUNCH_MARKOV_ARGMAX(__nv_half);
        }
        default:
            return cudaErrorInvalidValue;
    }

#undef LAUNCH_MARKOV_ARGMAX
}

}  // namespace rtp_llm

#endif  // USING_CUDA
