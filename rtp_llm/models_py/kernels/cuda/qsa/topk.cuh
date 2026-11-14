

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_merge_sort.cuh>
#include <type_traits>

namespace rtp_llm {
struct QsaGreater {
    __device__ bool operator()(uint32_t a, uint32_t b) const {
        return a > b;
    }
};
template<int N, int Threads, bool UseMerge>
union QsaSortStorage {
    using FullSort     = std::conditional_t<UseMerge,
                                            cub::BlockMergeSort<uint32_t, Threads, N / Threads, uint32_t>,
                                            cub::BlockRadixSort<uint32_t, Threads, N / Threads, uint32_t>>;
    using SelectedSort = cub::BlockRadixSort<uint32_t, Threads, 512 / Threads, uint32_t>;
    typename FullSort::TempStorage     full;
    typename SelectedSort::TempStorage selected;
};
template<int N, int Threads, bool UseMerge = false, bool DraftPacked = false>
__global__ __launch_bounds__(Threads) void qsaCubStableTopK(const float*   scores,
                                                            const int32_t* lengths,
                                                            int32_t*       output,
                                                            int            columns,
                                                            bool           finite_only,
                                                            const int32_t* draft_lengths = nullptr,
                                                            const int32_t* draft_cu      = nullptr,
                                                            int            draft_batch   = 0
#ifdef RTP_QSA_HISTOGRAM
                                                            ,
                                                            const bool* fallback = nullptr
#endif
) {
    constexpr int Items = N / Threads;
    using FullSort      = std::conditional_t<UseMerge,
                                             cub::BlockMergeSort<uint32_t, Threads, Items, uint32_t>,
                                             cub::BlockRadixSort<uint32_t, Threads, Items, uint32_t>>;
    using SelectedSort  = cub::BlockRadixSort<uint32_t, Threads, 512 / Threads, uint32_t>;
    using SharedSort    = QsaSortStorage<N, Threads, UseMerge>;
    __shared__ __align__(16) unsigned char local_storage[N > 4096 ? 16 : sizeof(SharedSort)];
    extern __shared__ __align__(16) unsigned char dynamic_storage[];
    auto&               storage = *reinterpret_cast<SharedSort*>(N > 4096 ? dynamic_storage : local_storage);
    __shared__ uint32_t chosen[512];
    __shared__ uint32_t cutoff;
    const int           row = blockIdx.x;
#ifdef RTP_QSA_HISTOGRAM
    if (fallback && !fallback[row])
        return;
#endif
    int output_row = row;
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
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    output[output_row * 2051 + rank * 4 + j] = valid ? static_cast<int32_t>(col * 4 + j) : -1;
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

void qsa_cub_init() {
    C10_CUDA_CHECK(cudaFuncSetAttribute(rtp_llm::qsaCubStableTopK<8192, 512>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        sizeof(rtp_llm::QsaSortStorage<8192, 512, false>)));
    C10_CUDA_CHECK(cudaFuncSetAttribute(rtp_llm::qsaCubStableTopK<16384, 512>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        sizeof(rtp_llm::QsaSortStorage<16384, 512, false>)));
}

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
                || (scores.size(1) >= 2144 && scores.size(1) <= 16384));
    TORCH_CHECK(threads == 128 || threads == 256 || threads == 512);
    c10::cuda::CUDAGuard guard(scores.device());
    auto                 stream = at::cuda::getCurrentCUDAStream().stream();
    const int            rows = scores.size(0), columns = scores.size(1);
    if (!rows)
        return;
    if (columns > 2144) {
        if (columns <= 4096) {
            rtp_llm::qsaCubStableTopK<4096, 512><<<rows, 512, 0, stream>>>(scores.data_ptr<float>(),
                                                                           lengths.data_ptr<int32_t>(),
                                                                           output.data_ptr<int32_t>(),
                                                                           columns,
                                                                           finite_only);
        } else if (columns <= 8192) {
            rtp_llm::qsaCubStableTopK<8192, 512>
                <<<rows, 512, sizeof(rtp_llm::QsaSortStorage<8192, 512, false>), stream>>>(scores.data_ptr<float>(),
                                                                                           lengths.data_ptr<int32_t>(),
                                                                                           output.data_ptr<int32_t>(),
                                                                                           columns,
                                                                                           finite_only);
        } else {
            rtp_llm::qsaCubStableTopK<16384, 512>
                <<<rows, 512, sizeof(rtp_llm::QsaSortStorage<16384, 512, false>), stream>>>(scores.data_ptr<float>(),
                                                                                            lengths.data_ptr<int32_t>(),
                                                                                            output.data_ptr<int32_t>(),
                                                                                            columns,
                                                                                            finite_only);
        }
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
