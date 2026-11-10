#include "rtp_llm/models_py/bindings/cuda/kernels/deepselect_fp32.h"
#include <cstdint>
#include <limits>

#if defined(RTP_DEEPSELECT_FP32_ENABLED) && !defined(USE_ROCM)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cuda/functional>

// CUTLASS 4.0 does not recognize SM103a's supported TMA instructions.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1030 && defined(__CUDA_ARCH_FEAT_SM103_ALL)
#ifndef CUTE_ARCH_TMA_SM90_ENABLED
#define CUTE_ARCH_TMA_SM90_ENABLED
#endif
#endif
#include "rtp_llm/models_py/bindings/cuda/kernels/deepselect/cuda_kernels/v3_fp32/topk_select.cuh"
#endif

namespace torch_ext {
static std::tuple<torch::Tensor, torch::Tensor>
deepselect_fp32_topk_impl(const torch::Tensor& logits, int64_t k, bool* used_deepselect, bool int64_indices);


#if defined(RTP_DEEPSELECT_FP32_ENABLED) && !defined(USE_ROCM)
namespace {
// Sorting only the candidates makes the nucleus boundary inexpensive. The
// full-row count accounts for boundary ties that exactly-K selection omits.
__global__ void deferredTopKTopPThreshold(const float* logits,
                                        const float* selected,
                                        float* thresholds,
                                        int vocab,
                                        float top_p) {
    constexpr int threads = 256;
    constexpr int items = 4;
    using Sort = cub::BlockRadixSort<float, threads, items>;
    using Reduce = cub::BlockReduce<float, threads>;
    using Scan = cub::BlockScan<float, threads>;
    __shared__ union {
        Sort::TempStorage sort;
        Reduce::TempStorage reduce;
        Scan::TempStorage scan;
    } temporary;
    __shared__ float maximum, boundary, denominator;
    const int row = blockIdx.x;
    float values[items];
#pragma unroll
    for (int i = 0; i < items; ++i)
        values[i] = selected[row * 1024 + threadIdx.x * items + i];
    Sort(temporary.sort).SortDescending(values);
    __syncthreads();
    if (threadIdx.x == 0)
        maximum = values[0];
    if (threadIdx.x == threads - 1)
        boundary = values[items - 1];
    __syncthreads();
    if (!isfinite(maximum)) {
        if (threadIdx.x == 0)
            thresholds[row] = nanf("");
        return;
    }
    float missing_ties = 0;
    for (int token = threadIdx.x; token < vocab; token += threads) {
        const float value = logits[static_cast<int64_t>(row) * vocab + token];
        missing_ties += isnan(value) ? nanf("") : static_cast<float>(value == boundary);
    }
    float probabilities[items];
    float local_sum = 0;
#pragma unroll
    for (int i = 0; i < items; ++i) {
        missing_ties -= values[i] == boundary;
        probabilities[i] = expf(values[i] - maximum);
        local_sum += probabilities[i];
    }
    const float boundary_mass = expf(boundary - maximum);
    local_sum += missing_ties * boundary_mass;
    const float total = Reduce(temporary.reduce).Sum(local_sum);
    if (threadIdx.x == 0)
        denominator = total;
    __syncthreads();
    if (isnan(denominator)) {
        if (threadIdx.x == 0)
            thresholds[row] = denominator;
        return;
    }
    if (top_p >= 1.0f) {
        if (threadIdx.x == 0)
            thresholds[row] = boundary;
        return;
    }
    float prefix[items];
    Scan(temporary.scan).InclusiveSum(probabilities, prefix);
    __syncthreads();
    float first = 1024.0f;
#pragma unroll
    for (int i = 0; i < items; ++i) {
        if (prefix[i] >= top_p * denominator)
            first = fminf(first, static_cast<float>(threadIdx.x * items + i));
    }
    const float first_global = Reduce(temporary.reduce).Reduce(first, cuda::minimum<float>());
    if (threadIdx.x == 0)
        denominator = first_global;
    __syncthreads();
    const int cutoff_index = static_cast<int>(denominator);
    if (cutoff_index == 1024) {
        if (threadIdx.x == 0)
            thresholds[row] = boundary;
    } else if (threadIdx.x == cutoff_index / items) {
        thresholds[row] = values[cutoff_index % items];
    }
}

struct CompactMinimum {
    float value;
    int slot;
    bool nonfinite;
};

struct CompactMinimumOp {
    __device__ CompactMinimum operator()(const CompactMinimum& a, const CompactMinimum& b) const {
        auto result = b.value < a.value || (b.value == a.value && b.slot > a.slot) ? b : a;
        result.nonfinite = a.nonfinite || b.nonfinite;
        return result;
    }
};

template<int items>
__device__ void compactTopKTopPBody(float* dense, const float* selected, const int32_t* selected_ids,
                                  float* compact, int32_t* ids, int32_t* fallback,
                                  int vocab, float top_p, int removed_slot, float removed_value) {
    constexpr int threads = 256;
    using Sort = cub::BlockRadixSort<float, threads, items, int32_t>;
    using Reduce = cub::BlockReduce<float, threads>;
    using Scan = cub::BlockScan<float, threads>;
    __shared__ union {
        typename Sort::TempStorage sort;
        typename Reduce::TempStorage reduce;
        typename Scan::TempStorage scan;
    } temporary;
    __shared__ float maximum, boundary, extra, total, cutoff;
    __shared__ int overflow;
    const int row = blockIdx.x;
    float values[items];
    int32_t keys[items];
#pragma unroll
    for (int i = 0; i < items; ++i) {
        const int slot = threadIdx.x * items + i;
        const int source = items == 4 && slot == removed_slot ? 1024 : slot;
        values[i] = source < 1025 ? selected[row * 1025 + source] : -INFINITY;
        keys[i] = source < 1025 ? selected_ids[row * 1025 + source] : -1;
    }
    Sort(temporary.sort).SortDescending(values, keys);
    __syncthreads();
    if (threadIdx.x == 0) maximum = values[0];
    if constexpr (items == 4) {
        if (threadIdx.x == 255) boundary = values[3];
        if (threadIdx.x == 0) extra = removed_value;
    } else {
        if (threadIdx.x == 127) boundary = values[7];
        if (threadIdx.x == 128) extra = values[0];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        overflow = boundary == extra || !isfinite(maximum);
        fallback[row] = overflow;
    }
    __syncthreads();
    float missing_ties = 0;
    if (overflow) {
        for (int token = threadIdx.x; token < vocab; token += threads) {
            const float value = dense[static_cast<int64_t>(row) * vocab + token];
            missing_ties += isnan(value) ? nanf("") : static_cast<float>(value == boundary);
        }
    }
    float probabilities[items], prefix[items];
    float subtotal = 0;
#pragma unroll
    for (int i = 0; i < items; ++i) {
        const int slot = threadIdx.x * items + i;
        probabilities[i] = slot < 1024 ? expf(values[i] - maximum) : 0.0f;
        subtotal += probabilities[i];
        if (overflow && slot < 1024) missing_ties -= values[i] == boundary;
    }
    subtotal += missing_ties * expf(boundary - maximum);
    const float sum = Reduce(temporary.reduce).Sum(subtotal);
    if (threadIdx.x == 0) total = sum;
    __syncthreads();
    Scan(temporary.scan).InclusiveSum(probabilities, prefix);
    __syncthreads();
    float first = 1024;
#pragma unroll
    for (int i = 0; i < items; ++i)
        if (threadIdx.x * items + i < 1024 && prefix[i] >= top_p * total)
            first = fminf(first, static_cast<float>(threadIdx.x * items + i));
    const float first_global = Reduce(temporary.reduce).Reduce(first, cuda::minimum<float>());
    if (threadIdx.x == 0) {
        total = first_global;
        cutoff = boundary;
    }
    __syncthreads();
    const int index = static_cast<int>(total);
    if (top_p < 1.0f && index < 1024 && threadIdx.x == index / items)
        cutoff = values[index % items];
    __syncthreads();
#pragma unroll
    for (int i = 0; i < items; ++i) {
        const int slot = threadIdx.x * items + i;
        if (slot < 1024) {
            compact[row * 1024 + slot] = values[i] >= cutoff ? values[i] : -INFINITY;
            ids[row * 1024 + slot] = keys[i];
        }
    }
    if (overflow) {
        const float threshold = isfinite(maximum) ? cutoff : nanf("");
        for (int token = threadIdx.x; token < vocab; token += threads) {
            const auto offset = static_cast<int64_t>(row) * vocab + token;
            const float value = dense[offset];
            dense[offset] = isnan(threshold) ? threshold : (value >= threshold ? value : -INFINITY);
        }
    }
}

__global__ void compactTopKTopP(float* dense, const float* selected, const int32_t* selected_ids,
                               float* compact, int32_t* ids, int32_t* fallback, int vocab, float top_p) {
    using Reduce = cub::BlockReduce<CompactMinimum, 256>;
    __shared__ Reduce::TempStorage temporary;
    __shared__ CompactMinimum minimum;
    const int row = blockIdx.x;
    CompactMinimum local{INFINITY, -1, false};
    const CompactMinimumOp op;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int slot = threadIdx.x * 4 + i;
        const float value = selected[row * 1025 + slot];
        local = op(local, CompactMinimum{value, slot, !isfinite(value)});
    }
    if (threadIdx.x == 0) {
        const float value = selected[row * 1025 + 1024];
        local = op(local, CompactMinimum{value, 1024, !isfinite(value)});
    }
    const auto result = Reduce(temporary).Reduce(local, op);
    if (threadIdx.x == 0) minimum = result;
    __syncthreads();
    if (minimum.nonfinite) {
        // Preserve the existing radix ordering and fallback behavior for NaN,
        // infinities and sparse rows rather than imposing a numeric min order.
        compactTopKTopPBody<8>(dense, selected, selected_ids, compact, ids, fallback,
                              vocab, top_p, -1, 0.0f);
    } else {
        compactTopKTopPBody<4>(dense, selected, selected_ids, compact, ids, fallback,
                              vocab, top_p, minimum.slot, minimum.value);
    }
}

__global__ void deferredMaskLogits(float* logits, const float* thresholds, int vocab) {
    const int row = blockIdx.y;
    const int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token < vocab) {
        const auto offset = static_cast<int64_t>(row) * vocab + token;
        const float threshold = thresholds[row];
        const float value = logits[offset];
        logits[offset] = isnan(threshold) ? threshold : (value >= threshold ? value : -INFINITY);
    }
}
}  // namespace
#endif

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
deferred_compact_topk_topp(const torch::Tensor& logits, int64_t k, float top_p) {
#if defined(RTP_DEEPSELECT_FP32_ENABLED) && !defined(USE_ROCM)
    if (k != 1024 || !logits.is_cuda() || logits.scalar_type() != torch::kFloat32 || logits.dim() != 2
        || !logits.is_contiguous() || logits.size(1) <= k || logits.size(1) >= MAX_VOCAB_SIZE
        || logits.size(0) > 65535 || logits.stride(0) % 256 != 0
        || reinterpret_cast<uintptr_t>(logits.data_ptr()) % 16 != 0 || !(top_p > 0 && top_p <= 1))
        return {};
    const auto* properties = at::cuda::getDeviceProperties(logits.get_device());
    if (properties->major != 10 || (properties->minor != 0 && properties->minor != 3)) return {};
    const c10::cuda::CUDAGuard guard(logits.device());
    auto selected = deepselect_fp32_topk_impl(logits, 1025, nullptr, false);
    auto compact = torch::empty({logits.size(0), 1024}, logits.options());
    auto ids = torch::empty({logits.size(0), 1024}, logits.options().dtype(torch::kInt32));
    auto fallback = torch::empty({logits.size(0)}, logits.options().dtype(torch::kInt32));
    const auto stream = at::cuda::getCurrentCUDAStream(logits.get_device());
    if (logits.size(0)) {
        compactTopKTopP<<<logits.size(0), 256, 0, stream>>>(
            logits.data_ptr<float>(), std::get<0>(selected).data_ptr<float>(),
            std::get<1>(selected).data_ptr<int32_t>(), compact.data_ptr<float>(), ids.data_ptr<int32_t>(),
            fallback.data_ptr<int32_t>(), logits.size(1), top_p);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {compact, ids, fallback};
#else
    return {};
#endif
}

bool deferred_topk_topp_filter(const torch::Tensor& logits, int64_t k, float top_p) {
#if defined(RTP_DEEPSELECT_FP32_ENABLED) && !defined(USE_ROCM)
    if (k != 1024 || !logits.is_cuda() || logits.scalar_type() != torch::kFloat32 || logits.dim() != 2
        || !logits.is_contiguous() || logits.size(0) > 65535 || logits.size(1) <= k || logits.size(1) >= MAX_VOCAB_SIZE
        || logits.stride(0) % 256 != 0 || reinterpret_cast<uintptr_t>(logits.data_ptr()) % 16 != 0
        || !(top_p > 0.0f && top_p <= 1.0f))
        return false;
    const auto* properties = at::cuda::getDeviceProperties(logits.get_device());
    if (properties->major != 10 || (properties->minor != 0 && properties->minor != 3))
        return false;
    const c10::cuda::CUDAGuard guard(logits.device());
    auto selected = deepselect_fp32_topk(logits, k);
    auto thresholds = torch::empty({logits.size(0)}, logits.options());
    const auto stream = at::cuda::getCurrentCUDAStream(logits.get_device());
    if (logits.size(0)) {
        deferredTopKTopPThreshold<<<logits.size(0), 256, 0, stream>>>(
            logits.data_ptr<float>(), std::get<0>(selected).data_ptr<float>(), thresholds.data_ptr<float>(),
            logits.size(1), top_p);
        deferredMaskLogits<<<dim3((logits.size(1) + 255) / 256, logits.size(0)), 256, 0, stream>>>(
            logits.data_ptr<float>(), thresholds.data_ptr<float>(), logits.size(1));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return true;
#else
    return false;
#endif
}
static std::tuple<torch::Tensor, torch::Tensor>
deepselect_fp32_topk_impl(const torch::Tensor& logits, int64_t k, bool* used_deepselect, bool int64_indices) {
    if (used_deepselect)
        *used_deepselect = false;
#if defined(RTP_DEEPSELECT_FP32_ENABLED) && !defined(USE_ROCM)
    if ((k == 1024 || k == 1025 || k == 4096) && logits.is_cuda() && logits.scalar_type() == torch::kFloat32 && logits.dim() == 2
        && logits.size(0) <= std::numeric_limits<int32_t>::max() && logits.size(1) >= k
        && logits.size(1) < MAX_VOCAB_SIZE && logits.is_contiguous() && logits.stride(0) % 256 == 0
        && reinterpret_cast<uintptr_t>(logits.data_ptr()) % 16 == 0) {
        const auto* properties = at::cuda::getDeviceProperties(logits.get_device());
        if (properties->major == 10 && (properties->minor == 0 || properties->minor == 3)) {
            const c10::cuda::CUDAGuard guard(logits.device());
            const auto stream = at::cuda::getCurrentCUDAStream(logits.get_device());
            auto values = torch::empty({logits.size(0), k}, logits.options());
            auto indices = torch::empty({logits.size(0), k}, logits.options().dtype(torch::kInt32));
            if (logits.size(0) > 0) {
                TopkSelectArgs args{};
                args.batch_size                = static_cast<uint32_t>(logits.size(0));
                args.vocab_size                = static_cast<uint32_t>(logits.size(1));
                args.topk                      = static_cast<uint32_t>(k);
                args.input                     = logits.data_ptr();
                args.output_index              = indices.data_ptr();
                args.output_value              = values.data_ptr();
                args.return_value              = true;
                args.end_ptr                   = nullptr;
                args.stride_input_batch        = logits.stride(0);
                args.stride_output_index_batch = indices.stride(0);
                args.stride_output_value_batch = values.stride(0);
                args.idx_oob_fill_value        = -1;
                args.value_oob_fill_value      = -std::numeric_limits<float>::infinity();
                args.shared_memory_size_per_sm = properties->sharedMemPerBlockOptin;
                args.stream                    = stream;
                if (k == 1024) {
                    using Config = TopkSelectConfig<float, int32_t, false, false, true, 1024, 512, 1, 8192, 4096, 3>;
                    topk_select_fp32::run_topk_select_kernel<Config, false>(args);
                } else if (k == 1025) {
                    // Keep the extra boundary candidate while using the wide scan
                    // of K1024. The smaller survivor capacity fits in one SM.
                    using Config = TopkSelectConfig<float, int32_t, false, false, true, 1536, 512, 1, 8192, 4096, 3>;
                    using Kernel = topk_select_fp32::TopkSelectKernelFP32<Config, false>;
                    // Static shared variables also consume the opt-in limit.
                    // Leave 8 KiB headroom and validate the loaded function's
                    // actual static allocation before setting its dynamic size.
                    constexpr size_t dynamic_bytes = sizeof(typename Kernel::SharedMemoryPlanFP32);
                    static_assert(dynamic_bytes <= (227 - 8) * 1024);
                    cudaFuncAttributes attributes{};
                    C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, topk_select_fp32::topk_kernel<Kernel>));
                    TORCH_CHECK(dynamic_bytes + attributes.sharedSizeBytes <= properties->sharedMemPerBlockOptin,
                                "K1025 selector exceeds the combined static and dynamic shared-memory limit");
                    topk_select_fp32::run_topk_select_kernel<Config, false>(args);
                } else {
                    using Config = TopkSelectConfig<float, int32_t, false, false, true, 4096, 256, 1, 4096, 4096, 3>;
                    topk_select_fp32::run_topk_select_kernel<Config, false>(args);
                }
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
            if (used_deepselect)
                *used_deepselect = true;
            return {values, int64_indices ? indices.to(torch::kInt64) : indices};
        }
    }
#endif
    auto result = torch::topk(logits, k, -1, true, false);
    if (!int64_indices)
        std::get<1>(result) = std::get<1>(result).to(torch::kInt32);
    return result;
}

std::tuple<torch::Tensor, torch::Tensor>
deepselect_fp32_topk(const torch::Tensor& logits, int64_t k, bool* used_deepselect) {
    return deepselect_fp32_topk_impl(logits, k, used_deepselect, true);
}
}  // namespace torch_ext
