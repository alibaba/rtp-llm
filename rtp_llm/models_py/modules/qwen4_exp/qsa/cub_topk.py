"""QSA TopK with native CUDA ordering and fused token expansion.

The model-local kernel uses NVIDIA CUB's installed stable block sort APIs. It does
not vendor vLLM/CUB source. Stable uint32 keys retain the pinned Torch TopK
selection and raw radix order, including signed zero and both NaN signs.
Compilation occurs during eager warmup; an unprimed CUDA graph falls back.
"""

import hashlib
import logging
import threading

import torch

_TORCH_GIT = "70d99e998b4955e0049d13a98d77ae1b14db1f45"
_MODULE = None
_BUILD_FAILED = False
_BUILD_LOCK = threading.Lock()
_LOGGER = logging.getLogger(__name__)

_CUDA_SOURCE = r"""

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_merge_sort.cuh>
#include <cub/block/block_scan.cuh>
#include <type_traits>

namespace rtp_llm {

// Streaming radix selection: fixed shared storage, runtime visible length.
// The coarse histogram and boundary-only refinement follow SGLang fast_topk.
// A full boundary buffer uses exact streaming refinement, never truncation or
// native TopK. Stable compaction/sort preserve the existing selection order.
__device__ __forceinline__ uint32_t qsaSelectionKey(uint32_t bits) {
    const bool nan = (bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu);
    return nan ? 0xffffffffu : (bits & 0x80000000u) ? ~bits : bits ^ 0x80000000u;
}
__device__ __forceinline__ uint32_t qsaCoarseKey(uint32_t bits) {
    if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu)) return 255;
    const uint16_t h = __half_as_ushort(__float2half_rn(__uint_as_float(bits)));
    return ((h & 0x8000u) ? uint16_t(~h) : uint16_t(h | 0x8000u)) >> 8;
}
struct QsaMinimum {
    __device__ uint32_t operator()(uint32_t a, uint32_t b) const { return a < b ? a : b; }
};
struct QsaGreater {
    __device__ bool operator()(uint32_t a, uint32_t b) const { return a > b; }
};
template<int Threads>
__global__ __launch_bounds__(Threads) void qsaStreamingTopK(
    const float* scores, const int32_t* lengths, int32_t* output,
    int columns, bool finite_only) {
    static_assert(Threads == 256 || Threads == 512);
    constexpr int K = 512, SortItems = K / Threads;
    using Scan = cub::BlockScan<uint32_t, Threads>;
    using PairScan = cub::BlockScan<uint64_t, Threads>;
    using Sort = cub::BlockMergeSort<uint32_t, Threads, SortItems, uint32_t>;
    __shared__ union {
        typename Scan::TempStorage scan;
        typename PairScan::TempStorage pairs;
        typename Sort::TempStorage sort;
    } ops;
    __shared__ uint32_t histogram[256];
    constexpr int CandidateCapacity = 2048;
    __shared__ uint32_t candidates[2][CandidateCapacity];
    __shared__ uint32_t counter, input_count;
    __shared__ int overflow;
    __shared__ uint32_t selected[512];
    __shared__ uint32_t threshold, coarse_boundary, remaining;
    __shared__ int done;
    const int lane = threadIdx.x;
    const int64_t length = lengths[blockIdx.x];
    const int64_t complete = (length - (length < 0 ? 3 : 0)) / 4;
    const int visible = int(max(int64_t(0), min(int64_t(columns), complete)));
    const int count = max(K, visible);
    const float* row = scores + int64_t(blockIdx.x) * columns;
    if (lane == 0) {
        threshold = 0;
        remaining = K;
        done = count == K;
    }
    __syncthreads();
    if (count == K) {
        uint32_t mine = 0xffffffffu, minimum;
        for (int col = lane; col < count; col += Threads) {
            const uint32_t bits = col < visible ? __float_as_uint(row[col]) : 0xff800000u;
            mine = min(mine, qsaSelectionKey(bits));
        }
        Scan(ops.scan).InclusiveScan(mine, minimum, QsaMinimum{});
        if (lane == Threads - 1) threshold = minimum;
        __syncthreads();
    }
    if (count > K) {
        if (lane < 256) histogram[lane] = 0;
        __syncthreads();
        for (int col = lane; col < count; col += Threads) {
            const uint32_t bits = col < visible ? __float_as_uint(row[col]) : 0xff800000u;
            atomicAdd(histogram + qsaCoarseKey(bits), 1u);
        }
        __syncthreads();
        uint32_t inclusive, total;
        const uint32_t frequency = lane < 256 ? histogram[lane] : 0;
        Scan(ops.scan).InclusiveSum(frequency, inclusive, total);
        const uint32_t above = total - inclusive;
        if (above < K && above + frequency >= K) {
            coarse_boundary = lane;
            remaining = K - above;
            input_count = frequency;
            overflow = frequency > CandidateCapacity;
        }
        __syncthreads();
        if (lane == 0) counter = 0;
        if (lane < 256) histogram[lane] = 0;
        __syncthreads();
        // Candidate collection and the first exact-byte histogram share one scan.
        if (!overflow) {
            for (int col = lane; col < count; col += Threads) {
                const uint32_t bits = col < visible ? __float_as_uint(row[col]) : 0xff800000u;
                if (qsaCoarseKey(bits) == coarse_boundary) {
                    const uint32_t rank = atomicAdd(&counter, 1u);
                    candidates[0][rank] = col;
                    atomicAdd(histogram + (qsaSelectionKey(bits) >> 24), 1u);
                }
            }
        }
        __syncthreads();
        for (int pass = 0; pass < 4; ++pass) {
            const uint32_t prefix = threshold;
            const uint32_t target = remaining;
            const int shift = 24 - pass * 8;
            const uint32_t mask = pass == 0 ? 0u : 0xffffffffu << (32 - pass * 8);
            if (overflow) {
                if (lane < 256) histogram[lane] = 0;
                __syncthreads();
                for (int first = 0; first < count; first += Threads) {
                    const int col = first + lane;
                    const uint32_t bits = col < visible ? __float_as_uint(row[col]) : 0xff800000u;
                    const uint32_t key = qsaSelectionKey(bits);
                    const bool active = col < count && qsaCoarseKey(bits) == coarse_boundary && (key & mask) == prefix;
                    const uint32_t bin = active ? (key >> shift) & 255u : 256u;
                    const unsigned peers = __match_any_sync(0xffffffffu, bin);
                    if (active && (lane & 31) == __ffs(peers) - 1)
                        atomicAdd(histogram + bin, __popc(peers));
                }
                __syncthreads();
            }
            const uint32_t frequency = lane < 256 ? histogram[lane] : 0;
            Scan(ops.scan).InclusiveSum(frequency, inclusive, total);
            const uint32_t above = total - inclusive;
            if (above < target && above + frequency >= target) {
                threshold = prefix | (uint32_t(lane) << shift);
                remaining = target - above;
                done = frequency == target - above || pass == 3;
            }
            __syncthreads();
            if (done) {
                // Restore the true boundary before stable compaction; a partial
                // radix prefix alone can also include keys in lower coarse bins.
                uint32_t mine = 0xffffffffu, minimum;
                const uint32_t chosen = threshold;
                const uint32_t current_mask = 0xffffffffu << shift;
                const uint32_t n = overflow ? count : input_count;
                for (uint32_t i = lane; i < n; i += Threads) {
                    const uint32_t col = overflow ? i : candidates[pass & 1][i];
                    const uint32_t bits = col < visible ? __float_as_uint(row[col]) : 0xff800000u;
                    const uint32_t key = qsaSelectionKey(bits);
                    if (qsaCoarseKey(bits) == coarse_boundary && (key & current_mask) == chosen)
                        mine = min(mine, key);
                }
                Scan(ops.scan).InclusiveScan(mine, minimum, QsaMinimum{});
                if (lane == Threads - 1) threshold = minimum;
                __syncthreads();
                break;
            }
            if (!overflow) {
                const uint32_t n = input_count;
                if (lane == 0) counter = 0;
                if (lane < 256) histogram[lane] = 0;
                __syncthreads();
                const uint32_t chosen = threshold;
                for (uint32_t i = lane; i < n; i += Threads) {
                    const uint32_t col = candidates[pass & 1][i];
                    const uint32_t bits = col < visible ? __float_as_uint(row[col]) : 0xff800000u;
                    const uint32_t key = qsaSelectionKey(bits);
                    if ((key & (0xffffffffu << shift)) == chosen) {
                        const uint32_t rank = atomicAdd(&counter, 1u);
                        candidates[(pass + 1) & 1][rank] = col;
                        atomicAdd(histogram + ((key >> (shift - 8)) & 255u), 1u);
                    }
                }
                __syncthreads();
                if (lane == 0) input_count = counter;
                __syncthreads();
            }
        }
    }
    // The exact native tie membership is independent of histogram atomics:
    // compact above-threshold columns first, then the lowest-column boundary ties.
    const int items = (count + Threads - 1) / Threads;
    const int first = lane * items;
    uint32_t above = 0, equal = 0;
    bool has_nan = false;
    for (int i = 0; i < items && first + i < count; ++i) {
        const int col = first + i;
        const uint32_t bits = col < visible ? __float_as_uint(row[col]) : 0xff800000u;
        const uint32_t key = qsaSelectionKey(bits);
        above += key > threshold;
        equal += key == threshold;
        has_nan |= key == 0xffffffffu;
    }
    uint64_t prefix, total;
    PairScan(ops.pairs).ExclusiveSum((uint64_t(above) << 32) | equal, prefix, total);
    __syncthreads();
    uint32_t above_rank = prefix >> 32;
    uint32_t equal_rank = uint32_t(prefix) + uint32_t(total >> 32);
    for (int i = 0; i < items && first + i < count; ++i) {
        const int col = first + i;
        const uint32_t bits = col < visible ? __float_as_uint(row[col]) : 0xff800000u;
        const uint32_t key = qsaSelectionKey(bits);
        if (key > threshold) selected[above_rank++] = col;
        else if (key == threshold && equal_rank < K) selected[equal_rank++] = col;
    }
    const bool any_nan = __syncthreads_or(has_nan);
    uint32_t key[SortItems], index[SortItems];
#pragma unroll
    for (int i = 0; i < SortItems; ++i) {
        index[i] = selected[lane * SortItems + i];
        uint32_t bits = index[i] < visible ? __float_as_uint(row[index[i]]) : 0xff800000u;
        if (!any_nan && (bits & 0x7fffffffu) == 0) bits = 0;
        key[i] = (bits & 0x80000000u) ? ~bits : bits ^ 0x80000000u;
    }
    Sort(ops.sort).StableSort(key, index, QsaGreater{});
    __syncthreads();
#pragma unroll
    for (int i = 0; i < SortItems; ++i) {
        const uint32_t col = index[i];
        const uint32_t bits = col < visible ? __float_as_uint(row[col]) : 0xff800000u;
        const bool valid = col < visible && ((bits & 0x7f800000u) != 0x7f800000u ||
                                            (!finite_only && bits == 0x7f800000u));
        selected[lane * SortItems + i] = valid ? col : 0xffffffffu;
    }
    __syncthreads();
    int32_t* out = output + int64_t(blockIdx.x) * 2051;
    for (int token = lane; token < 2051; token += Threads) {
        if (token < 2048) {
            const int col = int(selected[token / 4]);
            out[token] = col >= 0 ? col * 4 + token % 4 : -1;
        } else {
            const int tail = token - 2048;
            out[token] = tail < length - complete * 4 ? int(complete * 4 + tail) : -1;
        }
    }
}

template<int N, int Threads, bool UseMerge = false, bool DraftPacked = false>
__global__ void qsaCubStableTopK(const float*   scores,
                                 const int32_t* lengths,
                                 int32_t*       output,
                                 int            columns,
                                 bool           finite_only,
                                 const int32_t* draft_lengths = nullptr,
                                 const int32_t* draft_cu      = nullptr,
                                 int            draft_batch   = 0) {
    constexpr int Items = N / Threads;
    using FullSort      = std::conditional_t<UseMerge,
                                             cub::BlockMergeSort<uint32_t, Threads, Items, uint32_t>,
                                             cub::BlockRadixSort<uint32_t, Threads, Items, uint32_t>>;
    using SelectedSort  = cub::BlockRadixSort<uint32_t, Threads, 512 / Threads, uint32_t>;
    __shared__ union SharedSort {
        typename FullSort::TempStorage     full;
        typename SelectedSort::TempStorage selected;
    } storage;
    __shared__ uint32_t chosen[512];
    __shared__ uint32_t cutoff;
    const int           row        = blockIdx.x;
    int                 output_row = row;
    if constexpr (DraftPacked) {
        const int request = row / 4, lane = row % 4, count = draft_lengths[request];
        if (lane < count) {
            output_row = draft_cu[request] + lane;
        } else {
            output_row = draft_cu[draft_batch] + request * 4 - draft_cu[request] + lane - count;
            for (int j = threadIdx.x; j < 2051; j += Threads)
                output[output_row * 2051 + j] = -1;
            return;
        }
    }

    const int64_t length   = lengths[row];
    const int64_t complete = (length - (length < 0 ? 3 : 0)) / 4;
    uint32_t      key[Items], index[Items];
    bool          special = false;
#pragma unroll
    for (int i = 0; i < Items; ++i) {
        uint32_t col = threadIdx.x * Items + i;
        uint32_t bits =
            (col < columns && col < complete) ?
                (col < columns && col < complete ? __float_as_uint(scores[row * columns + col]) : 0xff800000u) :
                0xff800000u;
        uint32_t ordered = (bits & 0x80000000u) ? ~bits : bits ^ 0x80000000u;
        bool     isnan   = (bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu);
        special |= isnan || bits == 0x80000000u;
        key[i]   = isnan ? 0xffffffffu : ordered;
        index[i] = col;
    }
    bool any_special = __syncthreads_or(special);
    if constexpr (UseMerge) {
        FullSort(storage.full).StableSort(key, index, QsaGreater{});
    } else {
        FullSort(storage.full).SortDescending(key, index);
    }
    if (any_special) {
        // Torch first gathers values strictly above the selected radix, then
        // its equal boundary, in column order. Its float radix sort treats
        // signed zeros as equal. Recreate that order only for NaNs/-0.

#pragma unroll
        for (int i = 0; i < Items; ++i) {
            int rank = threadIdx.x * Items + i;
            if (rank < 512)
                chosen[rank] = index[i];
            if (rank == 511)
                cutoff = key[i];
        }
        __syncthreads();
        uint32_t raw_key[512 / Threads], raw_index[512 / Threads];
#pragma unroll
        for (int i = 0; i < 512 / Threads; ++i) {
            uint32_t rank = threadIdx.x * (512 / Threads) + i;
            uint32_t col  = chosen[rank];
            uint32_t bits =
                (col < columns && col < complete ? __float_as_uint(scores[row * columns + col]) : 0xff800000u);
            uint32_t ordered   = (bits & 0x80000000u) ? ~bits : bits ^ 0x80000000u;
            bool     isnan     = (bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu);
            uint32_t selection = isnan ? 0xffffffffu : ordered;
            raw_key[i]         = 2 * N - ((selection == cutoff ? N : 0) + col);
            raw_index[i]       = col;
        }
        SelectedSort(storage.selected).SortDescending(raw_key, raw_index);
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 512 / Threads; ++i) {
            uint32_t col = raw_index[i];
            uint32_t bits =
                (col < columns && col < complete ? __float_as_uint(scores[row * columns + col]) : 0xff800000u);
            // CUB float keys canonicalize -0/+0; payloads retain original IDs.
            if ((bits & 0x7fffffffu) == 0)
                bits = 0;
            raw_key[i] = (bits & 0x80000000u) ? ~bits : bits ^ 0x80000000u;
        }
        SelectedSort(storage.selected).SortDescending(raw_key, raw_index);
#pragma unroll
        for (int i = 0; i < 512 / Threads; ++i) {
            uint32_t rank = threadIdx.x * (512 / Threads) + i;
            uint32_t col  = raw_index[i];
            uint32_t bits =
                (col < columns && col < complete ? __float_as_uint(scores[row * columns + col]) : 0xff800000u);
            bool valid = ((bits & 0x7f800000u) != 0x7f800000u || (!finite_only && bits == 0x7f800000u)) && col < columns
                         && col < complete;
#pragma unroll
            for (int j = 0; j < 4; ++j)
                output[output_row * 2051 + rank * 4 + j] = valid ? static_cast<int32_t>(col * 4 + j) : -1;
        }
    } else {
#pragma unroll
        for (int i = 0; i < Items; ++i) {
            uint32_t rank = threadIdx.x * Items + i;
            if (rank < 512) {
                uint32_t col = index[i];
                uint32_t bits =
                    (col < columns && col < complete ? __float_as_uint(scores[row * columns + col]) : 0xff800000u);
                bool valid = ((bits & 0x7f800000u) != 0x7f800000u || (!finite_only && bits == 0x7f800000u))
                             && col < columns && col < complete;
                if constexpr (N == 1024 && UseMerge) {
                    chosen[rank] = valid ? col : 0xffffffffu;
                } else {
#pragma unroll
                    for (int j = 0; j < 4; ++j)
                        output[output_row * 2051 + rank * 4 + j] = valid ? static_cast<int32_t>(col * 4 + j) : -1;
                }
            }
        }
        if constexpr (N == 1024 && UseMerge) {
            // Sorted ranks are blocked across threads. Stage the selected
            // columns once, then let adjacent lanes write adjacent tokens.
            __syncthreads();
            for (int token = threadIdx.x; token < 2048; token += Threads) {
                const int col = static_cast<int>(chosen[token / 4]);
                output[output_row * 2051 + token] = col >= 0 ? col * 4 + token % 4 : -1;
            }
        }
    }
    if (threadIdx.x == 0) {
#pragma unroll
        for (int j = 0; j < 3; ++j)
            output[output_row * 2051 + 2048 + j] = j < length - complete * 4 ? complete * 4 + j : -1;
    }
}
}  // namespace rtp_llm

void qsa_cub_topk(
    torch::Tensor scores, torch::Tensor lengths, torch::Tensor output, int64_t threads, bool finite_only, bool merge) {
    TORCH_CHECK(scores.is_cuda() && lengths.is_cuda() && output.is_cuda());
    TORCH_CHECK(scores.scalar_type() == at::kFloat && lengths.scalar_type() == at::kInt
                && output.scalar_type() == at::kInt);
    TORCH_CHECK(scores.dim() == 2 && lengths.dim() == 1 && output.dim() == 2);
    TORCH_CHECK(scores.is_contiguous() && lengths.is_contiguous() && output.is_contiguous());
    TORCH_CHECK(scores.device() == lengths.device() && scores.device() == output.device());
    TORCH_CHECK(scores.size(0) == lengths.numel() && output.size(0) == scores.size(0) && output.size(1) == 2051);
    TORCH_CHECK(scores.size(1) == 1023 || scores.size(1) == 1024 || scores.size(1) == 2047 || scores.size(1) == 2048
                || scores.size(1) == 2144);
    TORCH_CHECK(threads == 128 || threads == 256 || threads == 512);
    c10::cuda::CUDAGuard guard(scores.device());
    auto                 stream = at::cuda::getCurrentCUDAStream().stream();
    const int            rows = scores.size(0), columns = scores.size(1);
    if (!rows)
        return;
    // A 4K decode graph reserves 1023 compressed columns. Sort the bounded
    // row directly and expand in the same launch, preserving native tie order.
    if (merge && columns <= 1024 && threads == 512) {
        rtp_llm::qsaCubStableTopK<1024, 512, true><<<rows, 512, 0, stream>>>(
            scores.data_ptr<float>(), lengths.data_ptr<int32_t>(),
            output.data_ptr<int32_t>(), columns, finite_only);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }
    // Measured 8K domains only. StableSort preserves equal-key column order;
    // the special-value second pass below remains the pinned Torch radix order.
    if (merge && (columns == 2047 || columns == 2048 || columns == 2144)) {
        if (columns == 2144) {
            rtp_llm::qsaCubStableTopK<2304, 256, true><<<rows, 256, 0, stream>>>(scores.data_ptr<float>(),
                                                                                 lengths.data_ptr<int32_t>(),
                                                                                 output.data_ptr<int32_t>(),
                                                                                 columns,
                                                                                 finite_only);
        } else {
            rtp_llm::qsaCubStableTopK<2048, 512, true><<<rows, 512, 0, stream>>>(scores.data_ptr<float>(),
                                                                                 lengths.data_ptr<int32_t>(),
                                                                                 output.data_ptr<int32_t>(),
                                                                                 columns,
                                                                                 finite_only);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }
#define LAUNCH(N, T)                                                                                                   \
    rtp_llm::qsaCubStableTopK<N, T><<<rows, T, 0, stream>>>(                                                           \
        scores.data_ptr<float>(), lengths.data_ptr<int32_t>(), output.data_ptr<int32_t>(), columns, finite_only)
    if (columns <= 1024) {
        if (threads == 128) {
            LAUNCH(1024, 128);
        } else if (threads == 256) {
            LAUNCH(1024, 256);
        } else {
            LAUNCH(1024, 512);
        }
    } else if (columns == 2144) {
        if (threads == 128) {
            LAUNCH(2176, 128);
        } else if (threads == 256) {
            LAUNCH(2304, 256);
        } else {
            LAUNCH(2560, 512);
        }
    } else {
        if (threads == 128) {
            LAUNCH(2048, 128);
        } else if (threads == 256) {
            LAUNCH(2048, 256);
        } else {
            LAUNCH(2048, 512);
        }
    }
#undef LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void qsa_cub_topk_draft(
    torch::Tensor scores, torch::Tensor lengths, torch::Tensor output, torch::Tensor row_lengths, torch::Tensor cu) {
    TORCH_CHECK(scores.is_cuda() && lengths.is_cuda() && output.is_cuda() && row_lengths.is_cuda() && cu.is_cuda());
    TORCH_CHECK(scores.scalar_type() == at::kFloat && lengths.scalar_type() == at::kInt
                && output.scalar_type() == at::kInt && row_lengths.scalar_type() == at::kInt
                && cu.scalar_type() == at::kInt);
    TORCH_CHECK(scores.is_contiguous() && lengths.is_contiguous() && output.is_contiguous()
                && row_lengths.is_contiguous() && cu.is_contiguous());
    TORCH_CHECK(scores.device() == lengths.device() && scores.device() == output.device()
                && scores.device() == row_lengths.device() && scores.device() == cu.device());
    TORCH_CHECK(scores.dim() == 2 && lengths.dim() == 1 && row_lengths.dim() == 1 && cu.dim() == 1
                && output.dim() == 2);
    int rows = scores.size(0), cols = scores.size(1), batch = row_lengths.numel();
    TORCH_CHECK(rows == batch * 4 && batch >= 1 && batch <= 8 && lengths.numel() == rows && output.size(0) == rows
                && output.size(1) == 2051 && cu.numel() == batch + 1 && cols == 2144);
    c10::cuda::CUDAGuard guard(scores.device());
    auto                 stream = at::cuda::getCurrentCUDAStream().stream();
    rtp_llm::qsaCubStableTopK<2304, 256, true, true><<<rows, 256, 0, stream>>>(scores.data_ptr<float>(),
                                                                               lengths.data_ptr<int32_t>(),
                                                                               output.data_ptr<int32_t>(),
                                                                               cols,
                                                                               true,
                                                                               row_lengths.data_ptr<int32_t>(),
                                                                               cu.data_ptr<int32_t>(),
                                                                               batch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void qsa_radix_topk(torch::Tensor scores, torch::Tensor lengths, torch::Tensor output, bool finite_only) {
    TORCH_CHECK(scores.is_cuda() && lengths.is_cuda() && output.is_cuda());
    TORCH_CHECK(scores.scalar_type() == at::kFloat && lengths.scalar_type() == at::kInt
                && output.scalar_type() == at::kInt);
    TORCH_CHECK(scores.dim() == 2 && lengths.dim() == 1 && output.dim() == 2);
    TORCH_CHECK(scores.is_contiguous() && lengths.is_contiguous() && output.is_contiguous());
    TORCH_CHECK(scores.device() == lengths.device() && scores.device() == output.device());
    TORCH_CHECK(scores.size(0) >= 1 && scores.size(0) <= 65536);
    // Expanded token indices use int32; there is no context-capacity cutoff.
    TORCH_CHECK(scores.size(1) >= 512 && scores.size(1) <= INT32_MAX / 4);
    const int rows = scores.size(0), columns = scores.size(1);
    TORCH_CHECK(lengths.numel() == rows && output.size(0) == rows && output.size(1) == 2051);
    c10::cuda::CUDAGuard guard(scores.device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    if (rows <= 64) {
        rtp_llm::qsaStreamingTopK<512><<<rows, 512, 0, stream>>>(
            scores.data_ptr<float>(), lengths.data_ptr<int32_t>(),
            output.data_ptr<int32_t>(), columns, finite_only);
    } else {
        rtp_llm::qsaStreamingTopK<256><<<rows, 256, 0, stream>>>(
            scores.data_ptr<float>(), lengths.data_ptr<int32_t>(),
            output.data_ptr<int32_t>(), columns, finite_only);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

"""


def is_supported(scores, lengths):
    return (
        scores.is_cuda
        and torch.version.hip is None
        and torch.version.git_version == _TORCH_GIT
        and scores.dtype == torch.float32
        and scores.ndim == 2
        and 1 <= scores.shape[0] <= 64
        # The 8K MTP draft graph reserves three additional 128-token pages:
        # (64 + 3) * 32 compressed columns. Preserve the full-width Torch order.
        and scores.shape[1] in (1023, 1024, 2047, 2048, 2144)
        and scores.is_contiguous()
        and lengths.dtype == torch.int32
        and lengths.shape == (scores.shape[0],)
        and lengths.device == scores.device
        and lengths.is_contiguous()
        and not (scores.requires_grad or lengths.requires_grad)
        and (
            (
                torch.cuda.get_device_capability(scores.device) == (12, 0)
                and scores.shape[0] <= 32
                and scores.shape[1] in (2047, 2048, 2144)
            )
            or (
                torch.cuda.get_device_capability(scores.device) == (9, 0)
                and scores.shape[1] in (1023, 1024)
            )
        )
    )


def _load_module(device):
    global _MODULE, _BUILD_FAILED
    if _MODULE is not None:
        return _MODULE
    if _BUILD_FAILED:
        return None
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            return None
        with _BUILD_LOCK:
            if _MODULE is not None:
                return _MODULE
            if _BUILD_FAILED:
                return None
            try:
                from torch.utils.cpp_extension import load_inline

                major, minor = torch.cuda.get_device_capability(device)
                arch = f"{major}{minor}"
                flags = [
                    "-O3",
                    "--expt-relaxed-constexpr",
                    f"-gencode=arch=compute_{arch},code=sm_{arch}",
                    # The runtime image resolves a CUDA 13.0 cudart under an
                    # nvcc 13.2 toolchain; the CCCL compatibility guard rejects
                    # that pair even though the kernels only use stable CUB /
                    # runtime APIs. NVIDIA documents this define as the escape
                    # hatch for mismatched compiler/toolkit pairs.
                    "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",
                ]
                identity = hashlib.sha256(
                    (_CUDA_SOURCE + _TORCH_GIT + repr(flags)).encode()
                ).hexdigest()[:16]
                _MODULE = load_inline(
                    name="qwen4_qsa_topk_" + identity,
                    cpp_sources="void qsa_cub_topk(torch::Tensor scores, torch::Tensor lengths, torch::Tensor output, int64_t threads, bool finite_only, bool merge); void qsa_cub_topk_draft(torch::Tensor scores, torch::Tensor lengths, torch::Tensor output, torch::Tensor row_lengths, torch::Tensor cu); void qsa_radix_topk(torch::Tensor scores, torch::Tensor lengths, torch::Tensor output, bool finite_only);",
                    cuda_sources=_CUDA_SOURCE,
                    functions=[
                        "qsa_cub_topk",
                        "qsa_cub_topk_draft",
                        "qsa_radix_topk",
                    ],
                    extra_cflags=["-O2"],
                    extra_cuda_cflags=flags,
                    with_cuda=True,
                    verbose=False,
                )
            except Exception:
                _BUILD_FAILED = True
                _LOGGER.warning(
                    "Qwen4 QSA CUB compilation unavailable; using native selection",
                    exc_info=True,
                )
            return _MODULE


def try_cub_qsa_tokens(scores, lengths, *, finite_only=True):
    if not is_supported(scores, lengths):
        return None
    module = _load_module(scores.device)
    if module is None:
        return None
    output = torch.empty(
        (scores.shape[0], 2051), device=scores.device, dtype=torch.int32
    )
    module.qsa_cub_topk(
        scores,
        lengths,
        output,
        512 if scores.shape[1] <= 1024 else 256,
        finite_only,
        True,
    )
    return output


def try_cub_draft_qsa_tokens(scores, visible, row_lengths, cu):
    """Select directly into packed draft order; metadata is owner-validated.

    Fixed four-lane input rows and packed output rows have a bijection even
    for empty requests. Inactive rows fill the output padding with -1.
    """
    if not (
        is_supported(scores, visible)
        and scores.shape[1] == 2144
        and row_lengths.ndim == cu.ndim == 1
        and 1 <= row_lengths.numel() <= 8
        and scores.shape[0] == 4 * row_lengths.numel()
        and cu.numel() == row_lengths.numel() + 1
        and all(
            t.dtype == torch.int32
            and t.device == scores.device
            and t.is_contiguous()
            and not t.requires_grad
            for t in (row_lengths, cu)
        )
    ):
        return None
    module = _load_module(scores.device)
    if module is None:
        return None
    output = torch.empty(
        (scores.shape[0], 2051), device=scores.device, dtype=torch.int32
    )
    module.qsa_cub_topk_draft(scores, visible, output, row_lengths, cu)
    return output


def try_radix_qsa_tokens(scores, lengths, *, finite_only=True):
    """SM90 exact TopK with runtime visible lengths and fixed shared storage."""
    if not (
        scores.is_cuda
        and torch.version.hip is None
        and torch.version.git_version == _TORCH_GIT
        and scores.dtype == torch.float32
        and scores.ndim == 2
        and 1 <= scores.shape[0] <= 65536
        and 512 <= scores.shape[1] <= (2**31 - 1) // 4
        and scores.is_contiguous()
        and lengths.dtype == torch.int32
        and lengths.shape == (scores.shape[0],)
        and lengths.device == scores.device
        and lengths.is_contiguous()
        and not (scores.requires_grad or lengths.requires_grad)
        and torch.cuda.get_device_capability(scores.device) == (9, 0)
    ):
        return None
    module = _load_module(scores.device)
    if module is None:
        return None
    output = torch.empty(
        (scores.shape[0], 2051), device=scores.device, dtype=torch.int32
    )
    module.qsa_radix_topk(scores, lengths, output, finite_only)
    return output
