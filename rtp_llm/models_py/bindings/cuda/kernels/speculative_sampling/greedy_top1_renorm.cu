#include "rtp_llm/models_py/bindings/cuda/kernels/speculative_sampling/greedy_top1_renorm.h"

#if USING_CUDA

#include <math.h>
#include <algorithm>
#include <climits>
#include <cub/block/block_reduce.cuh>

namespace rtp_llm {
namespace {

constexpr int kGreedyTop1Threads = 1024;

// Target enough (row x split) CTAs to keep the GPU fed even at small batch;
// every split keeps at least ~4k elements of work so the second kernel's
// per-CTA partial combine stays noise. Same row-split shape as the Markov
// draft-softmax kernels, with a higher CTA target because both kernels here
// are pure single-pass memory streams (read argmax / write one-hot).
constexpr int kGreedyTop1TargetCtas   = 1024;
constexpr int kGreedyTop1MaxSplits    = 16;
constexpr int kGreedyTop1MinChunkSize = 4096;

// Local functors keep the kernel independent of the cub::Max header location,
// which moves between cub versions (thread_operators.hpp vs .cuh).
struct GreedyMaxOp {
    __device__ __forceinline__ float operator()(const float& a, const float& b) const {
        return isnan(a) ? a : (isnan(b) ? b : fmaxf(a, b));
    }
};

// NaNs win argmax; an ascending scan retains the first NaN or finite maximum.
__device__ __forceinline__ bool greedyArgmaxGreater(float a, float b) {
    return a > b || (isnan(a) && !isnan(b));
}

struct GreedyMinOp {
    __device__ __forceinline__ int operator()(const int& a, const int& b) const {
        return min(a, b);
    }
};

// Per-(row, split) chunk bounds in kernel-native units (float4 groups on the
// vector path, elements on the scalar path).
__device__ __forceinline__ void greedyChunkBounds(int split, int units, int chunk, int& begin, int& end) {
    begin = split * chunk;
    end   = min(units, begin + chunk);
}

// Kernel 1: reduce each (row, split) chunk to a (max, first-argmax) partial.
// The argmax index is stored as a float; callers guarantee vocab < 2^24 so
// the int -> float -> int round trip is exact.
template<bool kVectorized>
__global__ void greedyTop1PartialsKernel(const float* __restrict__ logits,
                                         float* __restrict__ partials_max,
                                         float* __restrict__ partials_idx,
                                         const int vocab_size,
                                         const int chunk) {
    const int    split  = blockIdx.x;
    const int    row    = blockIdx.y;
    const int    splits = gridDim.x;
    const float* lrow   = logits + static_cast<int64_t>(row) * vocab_size;

    float m   = -INFINITY;
    int   idx = INT_MAX;
    int   begin, end;

    if constexpr (kVectorized) {
        const int     units = vocab_size / 4;
        const float4* l4    = reinterpret_cast<const float4*>(lrow);
        greedyChunkBounds(split, units, chunk, begin, end);
        int i = begin + threadIdx.x;
        if (i < end) {
            // Seed with the thread's first float4 so an all-equal (or all
            // -inf) chunk still records its first element, matching
            // torch::argmax's first-maximum behavior.
            const float4 l = l4[i];
            m              = l.x;
            idx            = i * 4;
            if (greedyArgmaxGreater(l.y, m)) {
                m   = l.y;
                idx = i * 4 + 1;
            }
            if (greedyArgmaxGreater(l.z, m)) {
                m   = l.z;
                idx = i * 4 + 2;
            }
            if (greedyArgmaxGreater(l.w, m)) {
                m   = l.w;
                idx = i * 4 + 3;
            }
            i += kGreedyTop1Threads;
        }
        for (; i < end; i += kGreedyTop1Threads) {
            const float4 l = l4[i];
            // The thread's ascending walk keeps the first NaN or maximum.
            if (greedyArgmaxGreater(l.x, m)) {
                m   = l.x;
                idx = i * 4;
            }
            if (greedyArgmaxGreater(l.y, m)) {
                m   = l.y;
                idx = i * 4 + 1;
            }
            if (greedyArgmaxGreater(l.z, m)) {
                m   = l.z;
                idx = i * 4 + 2;
            }
            if (greedyArgmaxGreater(l.w, m)) {
                m   = l.w;
                idx = i * 4 + 3;
            }
        }
    } else {
        greedyChunkBounds(split, vocab_size, chunk, begin, end);
        int i = begin + threadIdx.x;
        if (i < end) {
            m   = lrow[i];
            idx = i;
            i += kGreedyTop1Threads;
        }
        for (; i < end; i += kGreedyTop1Threads) {
            if (greedyArgmaxGreater(lrow[i], m)) {
                m   = lrow[i];
                idx = i;
            }
        }
    }

    // Block reduce: the chunk max, then the lowest index among the threads
    // that saw that max. Empty chunks carry (m = -inf, idx = INT_MAX) and
    // never win unless every chunk is empty (impossible for vocab > 0).
    typedef cub::BlockReduce<float, kGreedyTop1Threads> BlockReduceF;
    typedef cub::BlockReduce<int, kGreedyTop1Threads>   BlockReduceI;
    __shared__ typename BlockReduceF::TempStorage       temp_f;
    __shared__ typename BlockReduceI::TempStorage       temp_i;
    __shared__ float                                    shared_max;

    const float block_max = BlockReduceF(temp_f).Reduce(m, GreedyMaxOp());
    if (threadIdx.x == 0) {
        shared_max = block_max;
    }
    __syncthreads();
    const int candidate = (m == shared_max || (isnan(m) && isnan(shared_max))) ? idx : INT_MAX;
    const int block_idx = BlockReduceI(temp_i).Reduce(candidate, GreedyMinOp());
    if (threadIdx.x == 0) {
        // Empty chunks carry INT_MAX; fold it to vocab_size, which stays exact
        // through the float partial slot (callers gate vocab < 2^24) and can
        // never beat a real index in kernel 2's combine.
        partials_max[static_cast<int64_t>(row) * splits + split] = block_max;
        partials_idx[static_cast<int64_t>(row) * splits + split] = static_cast<float>(min(block_idx, vocab_size));
    }
}

// Kernel 2: combine the per-row partials, zero-fill the row's one-hot chunk,
// and scatter the 1.0 at the row argmax; split 0 also writes the token.
// kWriteOnehot=false is the token-only point-mass mode: the chunk zero-fill
// and the scatter are skipped (the caller consumes the distribution as a
// point mass at samples_out), leaving the partials combine + token write.
template<bool kVectorized, bool kWriteOnehot>
__global__ void greedyTop1WriteKernel(float* __restrict__ onehot_out,
                                      int32_t* __restrict__ samples_out,
                                      const float* __restrict__ partials_max,
                                      const float* __restrict__ partials_idx,
                                      const int vocab_size,
                                      const int chunk) {
    const int split  = blockIdx.x;
    const int row    = blockIdx.y;
    const int splits = gridDim.x;

    // Every CTA redundantly combines the <= kGreedyTop1MaxSplits partials of
    // its row (a few cached 4-byte loads) instead of paying a second sync.
    // `vocab_size` is the empty-chunk sentinel written by kernel 1; split 0 is
    // always non-empty for vocab > 0, so the sentinel never survives a full
    // combine of a real row.
    float row_max = -INFINITY;
    int   row_arg = vocab_size;
    for (int i = 0; i < splits; ++i) {
        const float m   = partials_max[static_cast<int64_t>(row) * splits + i];
        const int   idx = static_cast<int>(partials_idx[static_cast<int64_t>(row) * splits + i]);
        if (greedyArgmaxGreater(m, row_max) || row_arg == vocab_size) {
            row_max = m;
            row_arg = idx;
        } else if ((m == row_max || (isnan(m) && isnan(row_max))) && idx < row_arg) {
            row_arg = idx;
        }
    }

    if constexpr (kWriteOnehot) {
        float* orow = onehot_out + static_cast<int64_t>(row) * vocab_size;
        int    begin, end;
        if constexpr (kVectorized) {
            const int units = vocab_size / 4;
            float4*   o4    = reinterpret_cast<float4*>(orow);
            greedyChunkBounds(split, units, chunk, begin, end);
            const float4 zero4 = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int i = begin + threadIdx.x; i < end; i += kGreedyTop1Threads) {
                o4[i] = zero4;
            }
            begin *= 4;
            end *= 4;
        } else {
            greedyChunkBounds(split, vocab_size, chunk, begin, end);
            for (int i = begin + threadIdx.x; i < end; i += kGreedyTop1Threads) {
                orow[i] = 0.0f;
            }
        }

        // The scatter must land after this CTA's zero-fill of the same range.
        __syncthreads();
        if (threadIdx.x == 0) {
            if (row_arg >= begin && row_arg < end) {
                orow[row_arg] = 1.0f;
            }
            if (split == 0) {
                samples_out[row] = static_cast<int32_t>(row_arg);
            }
        }
    } else if (threadIdx.x == 0 && split == 0) {
        samples_out[row] = static_cast<int32_t>(row_arg);
    }
}

template<bool kVectorized, bool kWriteOnehot>
cudaError_t launchGreedyTop1Renorm(const float* logits,
                                   float*       onehot_out,
                                   int32_t*     samples_out,
                                   int          batch_size,
                                   int          vocab_size,
                                   float*       partials,
                                   cudaStream_t stream) {
    const int  splits = greedyTop1RenormSplits(batch_size, vocab_size);
    const int  units  = kVectorized ? vocab_size / 4 : vocab_size;
    const int  chunk  = (units + splits - 1) / splits;
    const dim3 grid(splits, batch_size);
    const dim3 block(kGreedyTop1Threads);
    float*     partials_max = partials;
    float*     partials_idx = partials + static_cast<size_t>(batch_size) * splits;

    greedyTop1PartialsKernel<kVectorized>
        <<<grid, block, 0, stream>>>(logits, partials_max, partials_idx, vocab_size, chunk);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }
    greedyTop1WriteKernel<kVectorized, kWriteOnehot>
        <<<grid, block, 0, stream>>>(onehot_out, samples_out, partials_max, partials_idx, vocab_size, chunk);
    return cudaGetLastError();
}

}  // namespace

int greedyTop1RenormSplits(int batch_size, int vocab_size) {
    if (batch_size <= 0 || vocab_size <= 0) {
        return 1;
    }
    int splits = (kGreedyTop1TargetCtas + batch_size - 1) / batch_size;
    splits     = std::min(std::max(splits, 1), kGreedyTop1MaxSplits);
    splits     = std::min(splits, std::max(1, vocab_size / kGreedyTop1MinChunkSize));
    return splits;
}

size_t greedyTop1RenormPartialsBytes(int batch_size, int vocab_size) {
    if (batch_size <= 0) {
        return 0;
    }
    return 2ull * static_cast<size_t>(batch_size) * greedyTop1RenormSplits(batch_size, vocab_size) * sizeof(float);
}

cudaError_t invokeGreedyTop1Renorm(const float* logits,
                                   float*       onehot_out,
                                   int32_t*     samples_out,
                                   int          batch_size,
                                   int          vocab_size,
                                   float*       partials,
                                   cudaStream_t stream,
                                   bool         write_onehot) {
    if (batch_size == 0) {
        return cudaSuccess;
    }
    // vocab < 2^24 keeps the argmax index exact through its float partial
    // slot; the caller gate must fall back to the legacy chain beyond that.
    // onehot_out is only dereferenced when write_onehot is set.
    if (logits == nullptr || (write_onehot && onehot_out == nullptr) || samples_out == nullptr || partials == nullptr
        || vocab_size <= 0 || vocab_size >= (1 << 24) || batch_size < 0) {
        return cudaErrorInvalidValue;
    }
    // Contiguous views may start at an unaligned storage offset.
    if (vocab_size % 4 == 0 && reinterpret_cast<uintptr_t>(logits) % alignof(float4) == 0
        && (!write_onehot || reinterpret_cast<uintptr_t>(onehot_out) % alignof(float4) == 0)) {
        return write_onehot ? launchGreedyTop1Renorm<true, true>(
                                  logits, onehot_out, samples_out, batch_size, vocab_size, partials, stream) :
                              launchGreedyTop1Renorm<true, false>(
                                  logits, onehot_out, samples_out, batch_size, vocab_size, partials, stream);
    }
    return write_onehot ? launchGreedyTop1Renorm<false, true>(
                              logits, onehot_out, samples_out, batch_size, vocab_size, partials, stream) :
                          launchGreedyTop1Renorm<false, false>(
                              logits, onehot_out, samples_out, batch_size, vocab_size, partials, stream);
}

}  // namespace rtp_llm

#endif  // USING_CUDA
