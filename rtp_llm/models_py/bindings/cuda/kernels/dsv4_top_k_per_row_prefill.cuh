/*
 * Per-row TopK (prefill) — vendored & optimized from vLLM csrc/sampler.cu.
 *
 * Self-contained: pulls in only cub + cuda runtime.  All decode-only template
 * specializations of the helper kernels (multipleBlocksPerRow/mergeBlocks)
 * are kept for code parity but only the prefill driver is exposed.
 *
 * Optimizations vs upstream vLLM:
 *  - Step-0 float4 vectorized loads with single-VOTE warp-uniform shortcut
 *    (128->1 atomic for uniform zero/-inf warps) and quad-equal collapse (4->1),
 *    with zero slow __match_any_sync overhead on random inputs.
 *  - 2-round blocked BlockScan (4 bins/thread, 1024 bins/round), so positive
 *    and +0.0 thresholds terminate in a single BlockScan.
 *  - Step < 3 register-tracked min/max key + speculative fill for constant-tie
 *    buckets (e.g. dense ReLU 0.0f / -inf), exiting in 1 pass instead of 4.
 *  - Union-fit 1024-item SMEM rank-count (9.2 KB total static SMEM, zero
 *    cub::BlockRadixSort register/SMEM bloat) for both force_radix_sort paths.
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

// Canonical descending uint32 order key normalizing -0.0f to +0.0f for exact
// threshold tie-equality checks without collapsing genuine tiny nonzero diffs.
__device__ __forceinline__ uint32_t convert_to_uint32_canon(float x) {
    uint32_t bits = __float_as_uint(x);
    if (bits == 0x80000000u) {
        return 0x7fffffffu;
    }
    return (bits & 0x80000000u) ? bits : (~bits & 0x7fffffffu);
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

template<int kNumThreadsPerBlock>
__device__ void step0VectorizedHisto(const float* in, int len, int* histo) {
    using WideT                    = float4;
    constexpr int items_per_scalar = 4;

    int skip_cnt =
        (reinterpret_cast<size_t>(in) % sizeof(WideT)) ?
            static_cast<int>((sizeof(WideT) - reinterpret_cast<size_t>(in) % sizeof(WideT)) / sizeof(float)) :
            0;
    if (skip_cnt > len) {
        skip_cnt = len;
    }
    const WideT* in_cast  = reinterpret_cast<const WideT*>(in + skip_cnt);
    const int    len_cast = (len - skip_cnt) / items_per_scalar;
    const int    full_end = (len_cast / kNumThreadsPerBlock) * kNumThreadsPerBlock;

    int i = static_cast<int>(threadIdx.x);
    for (; i < full_end; i += kNumThreadsPerBlock) {
        const WideT    v     = in_cast[i];
        const uint32_t b0    = extractBinIdx<0>(v.x);
        const uint32_t b1    = extractBinIdx<0>(v.y);
        const uint32_t b2    = extractBinIdx<0>(v.z);
        const uint32_t b3    = extractBinIdx<0>(v.w);
        const bool     allEq = (b0 == b1) && (b2 == b3) && (b0 == b2);
        const uint32_t ref   = __shfl_sync(0xFFFFFFFFu, b0, 0);
        if (__all_sync(0xFFFFFFFFu, allEq && (b0 == ref))) {
            if ((threadIdx.x & 31) == 0) {
                atomicAdd(&histo[ref], 128);
            }
        } else if (allEq) {
            atomicAdd(&histo[b0], 4);
        } else {
            atomicAdd(&histo[b0], 1);
            atomicAdd(&histo[b1], 1);
            atomicAdd(&histo[b2], 1);
            atomicAdd(&histo[b3], 1);
        }
    }

    for (; i < len_cast; i += kNumThreadsPerBlock) {
        const WideT    v  = in_cast[i];
        const uint32_t b0 = extractBinIdx<0>(v.x);
        const uint32_t b1 = extractBinIdx<0>(v.y);
        const uint32_t b2 = extractBinIdx<0>(v.z);
        const uint32_t b3 = extractBinIdx<0>(v.w);
        if ((b0 == b1) && (b2 == b3) && (b0 == b2)) {
            atomicAdd(&histo[b0], 4);
        } else {
            atomicAdd(&histo[b0], 1);
            atomicAdd(&histo[b1], 1);
            atomicAdd(&histo[b2], 1);
            atomicAdd(&histo[b3], 1);
        }
    }

    if (static_cast<int>(threadIdx.x) < skip_cnt) {
        atomicAdd(&histo[extractBinIdx<0>(in[threadIdx.x])], 1);
    }
    const int remain_i = skip_cnt + len_cast * items_per_scalar + static_cast<int>(threadIdx.x);
    if (remain_i < len) {
        atomicAdd(&histo[extractBinIdx<0>(in[remain_i])], 1);
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
                                     bool            hasStep0Filter,
                                     int             step0ThreshBin,
                                     int&            thresholdBinIdx,
                                     SmemOutputType& smemOutput,
                                     int*            smemThresholdBinIdx,
                                     int*            smemFinalDstIdx,
                                     int*            smemFinalBinSize,
                                     int*            smemFoundTopKValues,
                                     int*            smemThreshOutIdx,
                                     uint32_t*       smemThreshMinKey,
                                     uint32_t*       smemThreshMaxKey,
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

    if constexpr (step == 0) {
        if (stride1 == 1) {
            step0VectorizedHisto<kNumThreadsPerBlock>(logits + rowStart, rowEnd - rowStart, smemFinal.histo.data);
        } else {
            for (int idx = rowStart + static_cast<int>(threadIdx.x); idx < rowEnd; idx += kNumThreadsPerBlock) {
                const float logit = logits[static_cast<int64_t>(idx) * stride1];
                atomicAdd(&smemFinal.histo.data[extractBinIdx<0>(logit)], 1);
            }
        }
    } else {
        auto distributeToBins = [&](float logit, int /* idx */ = 0) {
            const bool stepMatch = (!hasStep0Filter || static_cast<int>(extractBinIdx<0>(logit)) == step0ThreshBin)
                                   && isPartialMatch<patternShift>(logit, logitPattern);
            if (stepMatch) {
                atomicAdd(&smemFinal.histo.data[extractBinIdx<step>(logit)], 1);
            }
        };

        if (stride1 == 1) {
            vectorized_process(
                threadIdx.x, kNumThreadsPerBlock, logits + rowStart, rowEnd - rowStart, distributeToBins);
        } else {
            for (int idx = rowStart + static_cast<int>(threadIdx.x); idx < rowEnd; idx += kNumThreadsPerBlock) {
                const float logit = logits[static_cast<int64_t>(idx) * stride1];
                distributeToBins(logit, idx);
            }
        }
    }
    __syncthreads();

    // 2-round blocked BlockScan: kScanHalf = 1024 bins per round (4 bins/thread).
    // Positive values and +0.0f always live in [0..1023], exiting in 1 BlockScan.
    constexpr int kBinsPerScan = kNumThreadsPerBlock * 4;
    static_assert(kNumBins % kBinsPerScan == 0);
    constexpr int kNumRounds = kNumBins / kBinsPerScan;
    int           lastValue  = smemFoundTopKValues[0];

    for (int round = 0; round < kNumRounds; ++round) {
        const int binBase = round * kBinsPerScan + static_cast<int>(threadIdx.x) * 4;
        int       threadBins[4];
        int       threadSum = 0;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            threadBins[b] = smemFinal.histo.data[binBase + b];
            threadSum += threadBins[b];
        }

        int prefixSum = 0;
        int totalSum  = 0;
        using Scan    = cub::BlockScan<int, kNumThreadsPerBlock>;
        Scan(smemFinal.histo.scan).ExclusiveSum(threadSum, prefixSum, totalSum);
        prefixSum += lastValue;
        totalSum += lastValue;

        bool foundThreshold = false;
        if (prefixSum < topK && prefixSum + threadSum >= topK) {
            int running = prefixSum;
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const int cnt  = threadBins[b];
                const int next = running + cnt;
                if (next >= topK) {
                    smemThresholdBinIdx[0] = binBase + b;
                    smemFinalBinSize[0]    = cnt;
                    smemThreshOutIdx[0]    = running;
                    smemFinalDstIdx[0]     = 0;
                    smemThreshMinKey[0]    = 0xFFFFFFFFu;
                    smemThreshMaxKey[0]    = 0u;
                    foundThreshold         = true;
                    break;
                }
                running = next;
            }
        }
        if (__syncthreads_or(foundThreshold)) {
            break;
        }
        lastValue = totalSum;
    }

    thresholdBinIdx       = smemThresholdBinIdx[0];
    const int  binSize    = smemFinalBinSize[0];
    const bool smallFinal = (binSize <= kNumFinalItems);

    uint32_t localMinKey   = 0xFFFFFFFFu;
    uint32_t localMaxKey   = 0u;
    bool     hasThreshItem = false;

    auto processBins = [&](float logit, int idx) {
        const bool stepMatch = (step == 0)
                               || ((!hasStep0Filter || static_cast<int>(extractBinIdx<0>(logit)) == step0ThreshBin)
                                   && isPartialMatch<patternShift>(logit, logitPattern));
        if (!stepMatch) {
            return;
        }
        const uint32_t binIdx = extractBinIdx<step>(logit);
        if (binIdx < static_cast<uint32_t>(thresholdBinIdx)) {
            const int dstIdx = atomicAdd(smemFoundTopKValues, 1);
            if constexpr (mergeBlocks) {
                smemOutput[dstIdx] = indices[idx];
            } else if constexpr (multipleBlocksPerRow) {
                smemOutput[dstIdx]                                  = idx + rowStart;
                reinterpret_cast<float*>(smemOutput + topK)[dstIdx] = logit;
            } else {
                smemOutput[dstIdx] = idx;
            }
        } else if (binIdx == static_cast<uint32_t>(thresholdBinIdx)) {
            if constexpr (step < 3) {
                const uint32_t kCanon = convert_to_uint32_canon(logit);
                localMinKey           = kCanon < localMinKey ? kCanon : localMinKey;
                localMaxKey           = kCanon > localMaxKey ? kCanon : localMaxKey;
                hasThreshItem         = true;
                if (smallFinal) {
                    const int dstIdx               = atomicAdd(smemFinalDstIdx, 1);
                    smemFinal.items.logits[dstIdx] = logit;
                    if constexpr (mergeBlocks) {
                        smemFinal.items.indices[dstIdx] = indices[idx];
                    } else if constexpr (multipleBlocksPerRow) {
                        smemFinal.items.indices[dstIdx] = idx + rowStart;
                    } else {
                        smemFinal.items.indices[dstIdx] = idx;
                    }
                } else if (*reinterpret_cast<const volatile int*>(smemThreshOutIdx) < topK) {
                    // Speculative write for equal-tie fast exit (minKey == maxKey).
                    const int dstIdx = atomicAdd(smemThreshOutIdx, 1);
                    if (dstIdx < topK) {
                        if constexpr (mergeBlocks) {
                            smemOutput[dstIdx] = indices[idx];
                        } else if constexpr (multipleBlocksPerRow) {
                            smemOutput[dstIdx]                                  = idx + rowStart;
                            reinterpret_cast<float*>(smemOutput + topK)[dstIdx] = logit;
                        } else {
                            smemOutput[dstIdx] = idx;
                        }
                    }
                }
            } else {
                if (*reinterpret_cast<const volatile int*>(smemThreshOutIdx) < topK) {
                    const int dstIdx = atomicAdd(smemThreshOutIdx, 1);
                    if (dstIdx < topK) {
                        if constexpr (mergeBlocks) {
                            smemOutput[dstIdx] = indices[idx];
                        } else if constexpr (multipleBlocksPerRow) {
                            smemOutput[dstIdx]                                  = idx + rowStart;
                            reinterpret_cast<float*>(smemOutput + topK)[dstIdx] = logit;
                        } else {
                            smemOutput[dstIdx] = idx;
                        }
                    }
                }
            }
        }
    };

    if (stride1 == 1) {
        vectorized_process(threadIdx.x, kNumThreadsPerBlock, logits + rowStart, rowEnd - rowStart, processBins);
    } else {
        for (int idx = rowStart + static_cast<int>(threadIdx.x); idx < rowEnd; idx += kNumThreadsPerBlock) {
            const float logit = logits[static_cast<int64_t>(idx) * stride1];
            processBins(logit, idx);
        }
    }

    if constexpr (step < 3) {
        if (hasThreshItem) {
            atomicMin(smemThreshMinKey, localMinKey);
            atomicMax(smemThreshMaxKey, localMaxKey);
        }
    }
    __syncthreads();

    if constexpr (step == 3) {
        return false;
    } else {
        if (smallFinal) {
            return false;
        }
        return (smemThreshMinKey[0] != smemThreshMaxKey[0]);
    }
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
    // kNumFinalItems = 1024 matches Histogram::data[2048] size (8192 bytes)
    // for zero extra SMEM and high CTA occupancy on both useRadixSort paths.
    static constexpr int kNumFinalItems = 1024;
    using Scan                          = cub::BlockScan<int, kNumThreadsPerBlock>;
    (void)sizeof(std::bool_constant<useRadixSort>);

    struct FinalItems {
        int   indices[kNumFinalItems];
        float logits[kNumFinalItems];
    };
    struct Histogram {
        typename Scan::TempStorage scan;
        int                        data[kNumBins];
    };

    __shared__ union {
        FinalItems items;
        Histogram  histo;
    } smemFinal;

    extern __shared__ int32_t smemOutput[];

    __shared__ int      smemThresholdBinIdx[1];
    __shared__ int      smemFinalDstIdx[1];
    __shared__ int      smemFinalBinSize[1];
    __shared__ int      smemFoundTopKValues[1];
    __shared__ int      smemThreshOutIdx[1];
    __shared__ uint32_t smemThreshMinKey[1];
    __shared__ uint32_t smemThreshMaxKey[1];

    const int rowLen     = rowEnd - rowStart;
    const int safeRowLen = rowLen > 0 ? rowLen : 0;

    if (rowLen <= topK) {
        for (int rowIt = threadIdx.x; rowIt < safeRowLen; rowIt += kNumThreadsPerBlock) {
            if constexpr (multipleBlocksPerRow) {
                outIndices[rowIt] = rowIt + rowStart;
                outLogits[rowIt]  = logits[rowIt + rowStart];
            } else {
                outIndices[rowIt] = rowIt;
            }
        }
        for (int rowIt = safeRowLen + threadIdx.x; rowIt < topK; rowIt += kNumThreadsPerBlock) {
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
    bool     hasStep0Filter  = false;
    int      step0ThreshBin  = -1;

    bool continueToNextStep =
        processHistogramStep<0, kNumThreadsPerBlock, kNumBins, kNumFinalItems, multipleBlocksPerRow, mergeBlocks>(
            indices,
            logits,
            rowEnd,
            logitPattern,
            false,
            -1,
            thresholdBinIdx,
            smemOutput,
            smemThresholdBinIdx,
            smemFinalDstIdx,
            smemFinalBinSize,
            smemFoundTopKValues,
            smemThreshOutIdx,
            smemThreshMinKey,
            smemThreshMaxKey,
            smemFinal,
            stride1,
            rowStart,
            topK);

    bool needFinalSelect = (smemFinalBinSize[0] <= kNumFinalItems);

    if (continueToNextStep) {
        step0ThreshBin = thresholdBinIdx;
        hasStep0Filter = true;
        continueToNextStep =
            processHistogramStep<1, kNumThreadsPerBlock, kNumBins, kNumFinalItems, multipleBlocksPerRow, mergeBlocks>(
                indices,
                logits,
                rowEnd,
                logitPattern,
                hasStep0Filter,
                step0ThreshBin,
                thresholdBinIdx,
                smemOutput,
                smemThresholdBinIdx,
                smemFinalDstIdx,
                smemFinalBinSize,
                smemFoundTopKValues,
                smemThreshOutIdx,
                smemThreshMinKey,
                smemThreshMaxKey,
                smemFinal,
                stride1,
                rowStart,
                topK);
        needFinalSelect = (smemFinalBinSize[0] <= kNumFinalItems);
    }
    if (continueToNextStep) {
        continueToNextStep =
            processHistogramStep<2, kNumThreadsPerBlock, kNumBins, kNumFinalItems, multipleBlocksPerRow, mergeBlocks>(
                indices,
                logits,
                rowEnd,
                logitPattern,
                hasStep0Filter,
                step0ThreshBin,
                thresholdBinIdx,
                smemOutput,
                smemThresholdBinIdx,
                smemFinalDstIdx,
                smemFinalBinSize,
                smemFoundTopKValues,
                smemThreshOutIdx,
                smemThreshMinKey,
                smemThreshMaxKey,
                smemFinal,
                stride1,
                rowStart,
                topK);
        needFinalSelect = (smemFinalBinSize[0] <= kNumFinalItems);
    }
    if (continueToNextStep) {
        processHistogramStep<3, kNumThreadsPerBlock, kNumBins, kNumFinalItems, multipleBlocksPerRow, mergeBlocks>(
            indices,
            logits,
            rowEnd,
            logitPattern,
            hasStep0Filter,
            step0ThreshBin,
            thresholdBinIdx,
            smemOutput,
            smemThresholdBinIdx,
            smemFinalDstIdx,
            smemFinalBinSize,
            smemFoundTopKValues,
            smemThreshOutIdx,
            smemThreshMinKey,
            smemThreshMaxKey,
            smemFinal,
            stride1,
            rowStart,
            topK);
        needFinalSelect = false;
    }

    if (needFinalSelect) {
        const int baseIdx = smemFoundTopKValues[0];
        const int nFinal  = smemFinalDstIdx[0];
        if (smemThreshMinKey[0] == smemThreshMaxKey[0]) {
            const int need = topK - baseIdx;
            const int take = nFinal < need ? nFinal : need;
            for (int i = static_cast<int>(threadIdx.x); i < take; i += kNumThreadsPerBlock) {
                smemOutput[baseIdx + i] = smemFinal.items.indices[i];
                if constexpr (multipleBlocksPerRow) {
                    reinterpret_cast<float*>(smemOutput + topK)[baseIdx + i] = smemFinal.items.logits[i];
                }
            }
        } else {
            for (int i = static_cast<int>(threadIdx.x); i < nFinal; i += kNumThreadsPerBlock) {
                int         outIndex = 0;
                const float logit    = smemFinal.items.logits[i];
                for (int j = 0; j < nFinal; ++j) {
                    const float otherLogit = smemFinal.items.logits[j];
                    if (logit < otherLogit || (logit == otherLogit && i < j)) {
                        outIndex++;
                    }
                }
                if (outIndex + baseIdx < topK) {
                    smemOutput[outIndex + baseIdx] = smemFinal.items.indices[i];
                    if constexpr (multipleBlocksPerRow) {
                        reinterpret_cast<float*>(smemOutput + topK)[outIndex + baseIdx] = logit;
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
