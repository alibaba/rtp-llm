
#include <cuda_fp16.h>

namespace qwen4_qsa_histogram {
constexpr int Threads = 512, Capacity = 4096;
using Sort = cub::BlockRadixSort<uint64_t, Threads, Capacity / Threads>;
union Storage {
    uint64_t          candidates[Capacity];
    Sort::TempStorage sort;
};

template<int Count>
__device__ __forceinline__ void sortCandidates(Storage& storage, int count, int32_t* output) {
    using LocalSort = cub::BlockRadixSort<uint64_t, Threads, Count / Threads>;
    uint64_t keys[Count / Threads];
#pragma unroll
    for (int i = 0; i < Count / Threads; ++i) {
        const int position = threadIdx.x * (Count / Threads) + i;
        keys[i]            = position < count ? storage.candidates[position] : 0;
    }
    __syncthreads();
    auto& scratch = *reinterpret_cast<typename LocalSort::TempStorage*>(&storage);
    LocalSort(scratch).SortDescending(keys, 0, 46);
#pragma unroll
    for (int i = 0; i < Count / Threads; ++i) {
        const int rank = threadIdx.x * (Count / Threads) + i;
        if (rank < 512) {
            const int col = 16383 - static_cast<int>(keys[i] & 0x3fffu);
#pragma unroll
            for (int j = 0; j < 4; ++j)
                output[rank * 4 + j] = col * 4 + j;
        }
    }
}

__device__ __forceinline__ uint32_t coarse(float value) {
    const uint16_t bits = __half_as_ushort(__float2half_rn(value));
    const uint16_t key  = (bits & 0x8000u) ? uint16_t(~bits) : uint16_t(bits | 0x8000u);
    return key >> 8;
}

__global__ __launch_bounds__(Threads) void histogramTopK(
    const float* scores, const int32_t* lengths, int32_t* output, bool* fallback, int columns) {
    const int     row = blockIdx.x, tid = threadIdx.x;
    const int64_t length   = lengths[row];
    const int64_t complete = (length - (length < 0 ? 3 : 0)) / 4;
    const int     visible  = static_cast<int>(min(complete, static_cast<int64_t>(columns)));
    if (visible <= 512) {
        if (tid == 0)
            fallback[row] = true;
        return;
    }
    __shared__ int    histogram[256];
    __shared__ int    threshold, candidate_count, cursor;
    extern __shared__ __align__(16) unsigned char shared_bytes[];
    auto&                                         storage = *reinterpret_cast<Storage*>(shared_bytes);
    if (tid < 256)
        histogram[tid] = 0;
    if (tid == 0) {
        fallback[row] = true;
        cursor        = 0;
    }
    __syncthreads();
    bool special = false;
    for (int col = tid; col < visible; col += Threads) {
        const float    value = scores[static_cast<int64_t>(row) * columns + col];
        const uint32_t bits  = __float_as_uint(value);
        special |= (bits & 0x7f800000u) == 0x7f800000u || bits == 0x80000000u;
        atomicAdd(histogram + coarse(value), 1);
    }
    if (__syncthreads_or(special))
        return;
    if (tid == 0) {
        int count = 0;
        for (int bin = 255; bin >= 0; --bin) {
            count += histogram[bin];
            if (count >= 512) {
                threshold       = bin;
                candidate_count = count;
                break;
            }
        }
    }
    __syncthreads();
    if (candidate_count > Capacity)
        return;
    for (int col = tid; col < visible; col += Threads) {
        const float value = scores[static_cast<int64_t>(row) * columns + col];
        if (coarse(value) >= static_cast<uint32_t>(threshold)) {
            const uint32_t bits     = __float_as_uint(value);
            const uint32_t ordered  = (bits & 0x80000000u) ? ~bits : bits ^ 0x80000000u;
            const uint64_t key      = (static_cast<uint64_t>(ordered) << 14) | (16383 - col);
            const int      position = atomicAdd(&cursor, 1);
            if (position < Capacity)
                storage.candidates[position] = key;
        }
    }
    __syncthreads();
    // Keep the exact score/index keys, but avoid sorting thousands of zero
    // padding entries when the coarse tail contains only about 512 scores.
    if (candidate_count <= 1024)
        sortCandidates<1024>(storage, candidate_count, output + row * 2051);
    else if (candidate_count <= 2048)
        sortCandidates<2048>(storage, candidate_count, output + row * 2051);
    else
        sortCandidates<4096>(storage, candidate_count, output + row * 2051);
    if (tid == 0) {
        for (int j = 0; j < 3; ++j)
            output[row * 2051 + 2048 + j] = j < length - complete * 4 ? complete * 4 + j : -1;
        fallback[row] = false;
    }
}
}  // namespace qwen4_qsa_histogram

void qsa_radix_init() {
    qsa_cub_init();
    C10_CUDA_CHECK(cudaFuncSetAttribute(qwen4_qsa_histogram::histogramTopK,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        sizeof(qwen4_qsa_histogram::Storage)));
}

void qsa_radix_topk(
    torch::Tensor scores, torch::Tensor lengths, torch::Tensor output, torch::Tensor fallback, bool finite_only) {
    TORCH_CHECK(scores.is_cuda() && lengths.is_cuda() && output.is_cuda() && fallback.is_cuda());
    TORCH_CHECK(scores.scalar_type() == at::kFloat && lengths.scalar_type() == at::kInt
                && output.scalar_type() == at::kInt && fallback.scalar_type() == at::kBool);
    TORCH_CHECK(scores.is_contiguous() && lengths.is_contiguous() && output.is_contiguous()
                && fallback.is_contiguous());
    TORCH_CHECK(scores.device() == lengths.device() && scores.device() == output.device()
                && scores.device() == fallback.device());
    TORCH_CHECK(scores.dim() == 2 && lengths.dim() == 1 && output.dim() == 2 && fallback.dim() == 1);
    const int rows = scores.size(0), columns = scores.size(1);
    TORCH_CHECK(columns >= 2048 && columns <= 16384 && lengths.numel() == rows && output.size(0) == rows
                && output.size(1) == 2051 && fallback.numel() == rows);
    if (!rows)
        return;
    c10::cuda::CUDAGuard guard(scores.device());
    auto                 stream = at::cuda::getCurrentCUDAStream().stream();
    qwen4_qsa_histogram::histogramTopK<<<rows, 512, sizeof(qwen4_qsa_histogram::Storage), stream>>>(
        scores.data_ptr<float>(),
        lengths.data_ptr<int32_t>(),
        output.data_ptr<int32_t>(),
        fallback.data_ptr<bool>(),
        columns);
#define QWEN4_FALLBACK(N, SHARED)                                                                                      \
    rtp_llm::qsaCubStableTopK<N, 512><<<rows, 512, SHARED, stream>>>(scores.data_ptr<float>(),                         \
                                                                     lengths.data_ptr<int32_t>(),                      \
                                                                     output.data_ptr<int32_t>(),                       \
                                                                     columns,                                          \
                                                                     finite_only,                                      \
                                                                     nullptr,                                          \
                                                                     nullptr,                                          \
                                                                     0,                                                \
                                                                     fallback.data_ptr<bool>())
    if (columns <= 2048) {
        QWEN4_FALLBACK(2048, 0);
    } else if (columns <= 4096) {
        QWEN4_FALLBACK(4096, 0);
    } else if (columns <= 8192) {
        QWEN4_FALLBACK(8192, sizeof(rtp_llm::QsaSortStorage<8192, 512, false>));
    } else {
        QWEN4_FALLBACK(16384, sizeof(rtp_llm::QsaSortStorage<16384, 512, false>));
    }
#undef QWEN4_FALLBACK
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
