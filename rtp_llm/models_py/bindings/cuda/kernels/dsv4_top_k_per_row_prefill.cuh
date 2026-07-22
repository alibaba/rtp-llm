/*
 * Per-row TopK (prefill) — vendored from vLLM csrc/sampler.cu.
 *
 * Self-contained: pulls in only cub + cuda runtime.  All decode-only template
 * specializations of the helper kernels (multipleBlocksPerRow/mergeBlocks)
 * are kept for code parity but only the prefill driver is exposed.
 */

#ifndef DSV4_TOP_K_PER_ROW_PREFILL_CUH_
#define DSV4_TOP_K_PER_ROW_PREFILL_CUH_

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cfloat>
#include <cstdint>
#include <type_traits>

namespace vllm {
namespace dsv4_prefill {

#ifndef DSV4_PREFILL_WARP_SIZE
#define DSV4_PREFILL_WARP_SIZE 32
#endif

__device__ __forceinline__ uint32_t convert_to_uint32_dsv4(float x) {
    uint32_t bits = __float_as_uint(x);
    return (bits & 0x80000000) ? bits : ~bits & 0x7fffffff;
}

__device__ __forceinline__ uint64_t make_deterministic_sort_key(float logit, int index) {
    const uint32_t bits = __float_as_uint(logit);
    // Map IEEE-754 bits to an unsigned key whose natural order matches the
    // numeric float order.  The low bits make a smaller token index rank
    // first when scores are bitwise equal.
    const uint32_t ordered = (bits & 0x80000000U) ? ~bits : (bits ^ 0x80000000U);
    return (static_cast<uint64_t>(ordered) << 32) | (0xffffffffU - static_cast<uint32_t>(index));
}

template<int step>
static inline __device__ uint32_t extractBinIdx(float x) {
    if constexpr (step == 0) {
        __half   hx   = __float2half(x);
        uint16_t bits = __half_as_ushort(hx);
        bits          = (bits & 0x8000) ? bits : ~bits & 0x7fff;
        return bits >> 5;
    } else {
        uint32_t bits = __float_as_uint(x);
        bits          = (bits & 0x80000000) ? bits : ~bits & 0x7fffffff;
        if constexpr (step == 1) {
            return bits >> 21;
        } else if constexpr (step == 2) {
            return (bits >> 10) & 0x7ff;
        } else if constexpr (step == 3) {
            return bits & 0x3ff;
        }
    }
    return 0;
}

template<int shift>
static inline __device__ bool isPartialMatch(float x, uint32_t pattern) {
    if constexpr (shift == 0) {
        return true;
    }
    uint32_t bits = __float_as_uint(x);
    bits          = (bits & 0x80000000) ? bits : ~bits & 0x7fffffff;
    return (bits ^ pattern) >> shift == 0;
}

template<typename T, typename idxT, typename Func>
__device__ void vectorized_process(size_t thread_rank, size_t num_threads, const T* in, idxT len, Func f) {
    constexpr int kWarpSize = DSV4_PREFILL_WARP_SIZE;
    using WideT             = float4;
    if constexpr (sizeof(T) >= sizeof(WideT)) {
        for (idxT i = thread_rank; i < len; i += num_threads) {
            f(in[i], i);
        }
    } else {
        static_assert(sizeof(WideT) % sizeof(T) == 0);
        constexpr int items_per_scalar = sizeof(WideT) / sizeof(T);
        union {
            WideT scalar;
            T     array[items_per_scalar];
        } wide;

        int skip_cnt = (reinterpret_cast<size_t>(in) % sizeof(WideT)) ?
                           ((sizeof(WideT) - reinterpret_cast<size_t>(in) % sizeof(WideT)) / sizeof(T)) :
                           0;
        if (skip_cnt > len) {
            skip_cnt = len;
        }
        const WideT* in_cast  = reinterpret_cast<decltype(in_cast)>(in + skip_cnt);
        const idxT   len_cast = (len - skip_cnt) / items_per_scalar;

        for (idxT i = thread_rank; i < len_cast; i += num_threads) {
            wide.scalar       = in_cast[i];
            const idxT real_i = skip_cnt + i * items_per_scalar;
#pragma unroll
            for (int j = 0; j < items_per_scalar; ++j) {
                f(wide.array[j], real_i + j);
            }
        }

        static_assert(kWarpSize >= items_per_scalar);
        if (thread_rank < (size_t)skip_cnt) {
            f(in[thread_rank], thread_rank);
        }
        const idxT remain_i = skip_cnt + len_cast * items_per_scalar + thread_rank;
        if (remain_i < len) {
            f(in[remain_i], remain_i);
        }
    }
}

template<int  step,
         int  kNumThreadsPerBlock,
         int  kNumBins,
         int  kNumFinalItems,
         bool multipleBlocksPerRow,
         bool mergeBlocks,
         typename SmemFinalType,
         typename SmemOutputType>
__device__ bool processHistogramStep(const int*      indices,
                                     const float*    logits,
                                     int             rowEnd,
                                     uint32_t&       logitPattern,
                                     int&            thresholdBinIdx,
                                     SmemOutputType& smemOutput,
                                     int*            smemThresholdBinIdx,
                                     int*            smemFinalDstIdx,
                                     int*            smemFinalBinSize,
                                     int*            smemFoundTopKValues,
                                     SmemFinalType&  smemFinal,
                                     int             stride1,
                                     int             rowStart,
                                     int             topK) {
#pragma unroll
    for (int idx = threadIdx.x; idx < kNumBins; idx += kNumThreadsPerBlock) {
        smemFinal.histo.data[idx] = 0;
    }
    __syncthreads();

    constexpr auto patternShift = step < 2 ? 0 : step == 2 ? 21 : 10;
    if constexpr (step == 2) {
        logitPattern = static_cast<uint32_t>(thresholdBinIdx & 0x7ff) << patternShift;
    } else if constexpr (step == 3) {
        logitPattern |= static_cast<uint32_t>(thresholdBinIdx & 0x7ff) << patternShift;
    }

    auto distributeToBins = [&](float logit, int /* idx */ = 0) {
        if (isPartialMatch<patternShift>(logit, logitPattern)) {
            uint32_t binIdx = extractBinIdx<step>(logit);
            atomicAdd(&smemFinal.histo.data[binIdx], 1);
        }
    };

    if (stride1 == 1) {
        vectorized_process(threadIdx.x, kNumThreadsPerBlock, logits + rowStart, rowEnd - rowStart, distributeToBins);
    } else {
        for (int idx = rowStart + threadIdx.x; idx < rowEnd; idx += kNumThreadsPerBlock) {
            float logit = logits[idx * stride1];
            distributeToBins(logit, idx);
        }
    }
    __syncthreads();

    int lastValue = smemFoundTopKValues[0];

    for (int round = 0; round < kNumBins / kNumThreadsPerBlock; round++) {
        int idx      = threadIdx.x + kNumThreadsPerBlock * round;
        int binCount = smemFinal.histo.data[idx];
        __syncthreads();

        int prefixSum{0}, totalSum{0};
        using Scan = cub::BlockScan<int, kNumThreadsPerBlock>;
        Scan(smemFinal.histo.scan).ExclusiveSum(binCount, prefixSum, totalSum);

        prefixSum += lastValue;
        totalSum += lastValue;
        smemFinal.histo.data[idx] = prefixSum;
        __syncthreads();

        bool foundThreshold = false;
        if (prefixSum < topK) {
            int nextPrefixSum = threadIdx.x == kNumThreadsPerBlock - 1 ? totalSum : smemFinal.histo.data[idx + 1];
            if (nextPrefixSum >= topK) {
                smemThresholdBinIdx[0] = idx;
                smemFinalBinSize[0]    = nextPrefixSum - prefixSum;
                foundThreshold         = true;
            }
        }
        if (__syncthreads_or(foundThreshold)) {
            break;
        }
        lastValue = totalSum;
    }
    __syncthreads();

    thresholdBinIdx = smemThresholdBinIdx[0];

    auto processBins = [&](float logit, int idx) {
        if (isPartialMatch<patternShift>(logit, logitPattern)) {
            uint32_t binIdx              = extractBinIdx<step>(logit);
            bool     shouldWriteDirectly = (step == 0 && smemFinalBinSize[0] <= kNumFinalItems) || (step >= 1);
            if (binIdx < (uint32_t)thresholdBinIdx && shouldWriteDirectly) {
                int dstIdx = atomicAdd(&smemFoundTopKValues[0], 1);
                if constexpr (mergeBlocks) {
                    smemOutput[dstIdx] = indices[idx];
                } else if constexpr (multipleBlocksPerRow) {
                    smemOutput[dstIdx]                                  = idx + rowStart;
                    reinterpret_cast<float*>(smemOutput + topK)[dstIdx] = logit;
                } else {
                    smemOutput[dstIdx] = idx;
                }
            }
            if constexpr (step < 3) {
                if (binIdx == (uint32_t)thresholdBinIdx && smemFinalBinSize[0] <= kNumFinalItems) {
                    int dstIdx                     = atomicAdd(&smemFinalDstIdx[0], 1);
                    smemFinal.items.logits[dstIdx] = logit;
                    if constexpr (mergeBlocks) {
                        smemFinal.items.indices[dstIdx] = indices[idx];
                    } else if constexpr (multipleBlocksPerRow) {
                        smemFinal.items.indices[dstIdx] = idx + rowStart;
                    } else {
                        smemFinal.items.indices[dstIdx] = idx;
                    }
                }
            }
        }
    };

    if (stride1 == 1) {
        vectorized_process(threadIdx.x, kNumThreadsPerBlock, logits + rowStart, rowEnd - rowStart, processBins);
    } else {
        for (int idx = rowStart + threadIdx.x; idx < rowEnd; idx += kNumThreadsPerBlock) {
            float logit = logits[idx * stride1];
            processBins(logit, idx);
        }
    }
    __syncthreads();

    if constexpr (step == 3) {
        // The last radix bin represents one exact float bit pattern.  The old
        // atomicAdd path let whichever warp arrived first claim the remaining
        // TopK slots, so exact-score ties could change the selected *set*
        // across launches or score chunks.  This slow path is reached only
        // when an exact-score threshold bin contains more than
        // kNumFinalItems candidates.  Scan it in key-index order to make the
        // tie break deterministic; higher-score bins were already emitted in
        // parallel above.
        if (threadIdx.x == 0) {
            int dstIdx = smemFoundTopKValues[0];
            for (int absoluteIdx = rowStart; absoluteIdx < rowEnd && dstIdx < topK; ++absoluteIdx) {
                float logit = stride1 == 1 ? logits[absoluteIdx] : logits[absoluteIdx * stride1];
                if (isPartialMatch<10>(logit, logitPattern)
                    && extractBinIdx<3>(logit) == static_cast<uint32_t>(thresholdBinIdx)) {
                    const int candidateIdx = stride1 == 1 ? absoluteIdx - rowStart : absoluteIdx;
                    if constexpr (mergeBlocks) {
                        smemOutput[dstIdx] = indices[candidateIdx];
                    } else if constexpr (multipleBlocksPerRow) {
                        smemOutput[dstIdx]                                  = absoluteIdx;
                        reinterpret_cast<float*>(smemOutput + topK)[dstIdx] = logit;
                    } else {
                        smemOutput[dstIdx] = stride1 == 1 ? absoluteIdx - rowStart : absoluteIdx;
                    }
                    ++dstIdx;
                }
            }
        }
        __syncthreads();
    }

    return smemFinalBinSize[0] > kNumFinalItems;
}

template<int  kNumThreadsPerBlock,
         int  kNumBins,
         bool useRadixSort,
         bool multipleBlocksPerRow = false,
         bool mergeBlocks          = false>
static __device__ void topKPerRowJob(const int*   indices,
                                     const float* logits,
                                     int          rowStart,
                                     int          rowEnd,
                                     int*         outIndices,
                                     float*       outLogits,
                                     int          stride1,
                                     int          topK) {
    static constexpr int kNumFinalItems          = 2048;
    static constexpr int kNumFinalItemsPerThread = kNumFinalItems / kNumThreadsPerBlock;
    using FinalSort = cub::BlockRadixSort<uint64_t, kNumThreadsPerBlock, kNumFinalItemsPerThread, int>;
    using FinalSortTempStorage = std::conditional_t<useRadixSort, typename FinalSort::TempStorage, int>;
    using Scan                 = cub::BlockScan<int, kNumThreadsPerBlock>;

    struct FinalItems {
        int   indices[kNumFinalItems];
        float logits[kNumFinalItems];
    };
    struct Histogram {
        typename Scan::TempStorage scan;
        int                        data[kNumBins];
    };

    __shared__ union {
        FinalItems           items;
        FinalSortTempStorage finalSort;
        Histogram            histo;
    } smemFinal;

    extern __shared__ int32_t smemOutput[];

    __shared__ int smemThresholdBinIdx[1];
    __shared__ int smemFinalDstIdx[1];
    __shared__ int smemFinalBinSize[1];
    __shared__ int smemFoundTopKValues[1];

    int rowLen = rowEnd - rowStart;

    if (rowLen <= topK) {
        for (int rowIt = threadIdx.x; rowIt < rowLen; rowIt += kNumThreadsPerBlock) {
            if constexpr (multipleBlocksPerRow) {
                outIndices[rowIt] = rowIt + rowStart;
                outLogits[rowIt]  = logits[rowIt + rowStart];
            } else {
                outIndices[rowIt] = rowIt;
            }
        }
        for (int rowIt = rowLen + threadIdx.x; rowIt < topK; rowIt += kNumThreadsPerBlock) {
            outIndices[rowIt] = -1;
            if constexpr (multipleBlocksPerRow) {
                outLogits[rowIt] = -FLT_MAX;
            }
        }
        return;
    }

    if (threadIdx.x == 0) {
        smemFinalDstIdx[0]     = 0;
        smemFoundTopKValues[0] = 0;
    }
    __syncthreads();
    int      thresholdBinIdx = -1;
    uint32_t logitPattern    = 0;

    bool continueToNextStep =
        processHistogramStep<0, kNumThreadsPerBlock, kNumBins, kNumFinalItems, multipleBlocksPerRow, mergeBlocks>(
            indices,
            logits,
            rowEnd,
            logitPattern,
            thresholdBinIdx,
            smemOutput,
            smemThresholdBinIdx,
            smemFinalDstIdx,
            smemFinalBinSize,
            smemFoundTopKValues,
            smemFinal,
            stride1,
            rowStart,
            topK);

    if (continueToNextStep) {
        continueToNextStep =
            processHistogramStep<1, kNumThreadsPerBlock, kNumBins, kNumFinalItems, multipleBlocksPerRow, mergeBlocks>(
                indices,
                logits,
                rowEnd,
                logitPattern,
                thresholdBinIdx,
                smemOutput,
                smemThresholdBinIdx,
                smemFinalDstIdx,
                smemFinalBinSize,
                smemFoundTopKValues,
                smemFinal,
                stride1,
                rowStart,
                topK);
    }
    if (continueToNextStep) {
        continueToNextStep =
            processHistogramStep<2, kNumThreadsPerBlock, kNumBins, kNumFinalItems, multipleBlocksPerRow, mergeBlocks>(
                indices,
                logits,
                rowEnd,
                logitPattern,
                thresholdBinIdx,
                smemOutput,
                smemThresholdBinIdx,
                smemFinalDstIdx,
                smemFinalBinSize,
                smemFoundTopKValues,
                smemFinal,
                stride1,
                rowStart,
                topK);
    }
    if (continueToNextStep) {
        processHistogramStep<3, kNumThreadsPerBlock, kNumBins, kNumFinalItems, multipleBlocksPerRow, mergeBlocks>(
            indices,
            logits,
            rowEnd,
            logitPattern,
            thresholdBinIdx,
            smemOutput,
            smemThresholdBinIdx,
            smemFinalDstIdx,
            smemFinalBinSize,
            smemFoundTopKValues,
            smemFinal,
            stride1,
            rowStart,
            topK);
    }

    if (!continueToNextStep) {
        if constexpr (useRadixSort) {
            uint64_t finalKeys[kNumFinalItemsPerThread];
            int      finalIndices[kNumFinalItemsPerThread];
#pragma unroll
            for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii) {
                finalKeys[ii] = 0;
            }
#pragma unroll
            for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii) {
                int srcIdx = ii * kNumThreadsPerBlock + threadIdx.x;
                if (srcIdx < smemFinalDstIdx[0]) {
                    finalIndices[ii] = smemFinal.items.indices[srcIdx];
                    finalKeys[ii] = make_deterministic_sort_key(
                        smemFinal.items.logits[srcIdx], smemFinal.items.indices[srcIdx]);
                }
            }
            __syncthreads();
            FinalSort(smemFinal.finalSort).SortDescendingBlockedToStriped(finalKeys, finalIndices);
            int baseIdx = smemFoundTopKValues[0];
#pragma unroll
            for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii) {
                int srcIdx = ii * kNumThreadsPerBlock + threadIdx.x;
                int dstIdx = baseIdx + srcIdx;
                if (dstIdx < topK) {
                    smemOutput[dstIdx] = finalIndices[ii];
                    if constexpr (multipleBlocksPerRow) {
                        reinterpret_cast<float*>(smemOutput + topK)[dstIdx] = logits[finalIndices[ii] * stride1];
                    }
                }
            }
        } else {
            auto baseIdx = smemFoundTopKValues[0];
            for (int i = threadIdx.x; i < smemFinalDstIdx[0]; i += kNumThreadsPerBlock) {
                int  outIndex = 0;
                auto logit    = smemFinal.items.logits[i];
                for (int j = 0; j < smemFinalDstIdx[0]; j++) {
                    auto otherLogit = smemFinal.items.logits[j];
                    if (logit < otherLogit
                        || (logit == otherLogit && smemFinal.items.indices[i] > smemFinal.items.indices[j])) {
                        outIndex++;
                    }
                }
                if (outIndex + baseIdx < topK) {
                    smemOutput[outIndex + baseIdx] = smemFinal.items.indices[i];
                    if constexpr (multipleBlocksPerRow) {
                        reinterpret_cast<float*>(smemOutput + topK)[outIndex + baseIdx] = smemFinal.items.logits[i];
                    }
                }
            }
        }
        __syncthreads();
    }

    for (int i = threadIdx.x; i < topK; i += kNumThreadsPerBlock) {
        if constexpr (multipleBlocksPerRow) {
            outIndices[i] = smemOutput[i];
            outLogits[i]  = reinterpret_cast<float*>(smemOutput + topK)[i];
        } else {
            if (stride1 == 1) {
                outIndices[i] = smemOutput[i];
            } else {
                outIndices[i] = smemOutput[i] - rowStart;
            }
        }
    }
}

template<int kNumThreadsPerBlock, bool useRadixSort>
static __global__ __launch_bounds__(kNumThreadsPerBlock) void topKPerRowPrefill(const float* logits,
                                                                                const int*   rowStarts,
                                                                                const int*   rowEnds,
                                                                                const int*   rowIndices,
                                                                                int*         outIndices,
                                                                                int          stride0,
                                                                                int          stride1,
                                                                                const int    topK,
                                                                                const int    offsetIndex) {
    static constexpr int kNumBins = 2048;

    int scheduleIdx = blockIdx.x + offsetIndex;
    int rowIdx      = rowIndices == nullptr ? scheduleIdx : rowIndices[scheduleIdx];

    int rowStart = rowStarts[rowIdx];
    int rowEnd   = rowEnds[rowIdx];

    outIndices += static_cast<int64_t>(rowIdx) * topK;
    logits += static_cast<int64_t>(rowIdx) * stride0;

    topKPerRowJob<kNumThreadsPerBlock, kNumBins, useRadixSort>(
        nullptr, logits, rowStart, rowEnd, outIndices, nullptr, stride1, topK);
}

}  // namespace dsv4_prefill
}  // namespace vllm

#endif  // DSV4_TOP_K_PER_ROW_PREFILL_CUH_
