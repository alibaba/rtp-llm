#include "rtp_llm/models_py/bindings/cuda/kernels/deepselect_fp32.h"

#include <cstdint>
#include <cub/cub.cuh>
#include <math_constants.h>
#include <limits>

#if defined(RTP_DEEPSELECT_FP32_ENABLED) && !defined(USE_ROCM)
#include <ATen/MemoryOverlap.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// Older CUTLASS headers recognize TMA on SM100a only.
// The exact SM103a cubin supports these same SM90 TMA load instructions.
// Enable only the feature used here, without globally claiming SM100 MMA.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1030 && defined(__CUDA_ARCH_FEAT_SM103_ALL)
#ifndef CUTE_ARCH_TMA_SM90_ENABLED
#define CUTE_ARCH_TMA_SM90_ENABLED
#endif
#endif

#include "rtp_llm/models_py/bindings/cuda/kernels/deepselect/cuda_kernels/v3_fp32/topk_select.cuh"
#endif

namespace torch_ext {

#if defined(RTP_DEEPSELECT_FP32_ENABLED) && !defined(USE_ROCM)
namespace {
template<typename BiasT, bool InvalidOnly = false>
__global__ void markovScaledLogitsKernel(const float* __restrict__ logits,
                                         int64_t logits_row_stride,
                                         const BiasT* __restrict__ bias,
                                         const float* __restrict__ temperature,
                                         float* __restrict__ output,
                                         int        vocab_size,
                                         const int* candidate_counts) {
    const int row = blockIdx.y;
    if constexpr (InvalidOnly) {
        const int count = candidate_counts[row];
        if (count >= 4096 && count <= 8192)
            return;
    }
    const int     col           = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const int64_t offset        = static_cast<int64_t>(row) * vocab_size + col;
    const int64_t logits_offset = static_cast<int64_t>(row) * logits_row_stride + col;
    const float   scale         = temperature[row];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (col + i < vocab_size) {
            output[offset + i] = (logits[logits_offset + i] + static_cast<float>(bias[offset + i])) / scale;
        }
    }
}

constexpr int kPrefilterVocab   = 129280;
constexpr int kPrefilterK       = 4096;
constexpr int kPrefilterCap     = 8192;
constexpr int kPrefilterThreads = 256;
constexpr int kPrefilterChunk   = 2048;

template<bool WithMarkov>
__device__ __forceinline__ float prefilterValue(
    const float* logits, int64_t stride, const __nv_bfloat16* bias, const float* temperature, int row, int col) {
    float value = logits[int64_t(row) * stride + col];
    if constexpr (WithMarkov) {
        value = (value + __bfloat162float(bias[int64_t(row) * kPrefilterVocab + col])) / temperature[row];
    }
    return value;
}

template<bool WithMarkov>
__global__ void prefilterThresholdKernel(const float*         logits,
                                         int64_t              stride,
                                         const __nv_bfloat16* bias,
                                         const float*         temperature,
                                         float*               threshold,
                                         int*                 counts,
                                         int*                 full_ends) {
    constexpr int items = WithMarkov ? 8 : 16;
    constexpr int rank  = WithMarkov ? 100 : 200;
    using Sort          = cub::BlockRadixSort<float, kPrefilterThreads, items>;
    __shared__ typename Sort::TempStorage storage;
    float                                 keys[items];
    const int                             row = blockIdx.x;
#pragma unroll
    for (int i = 0; i < items; ++i) {
        const int col = (threadIdx.x * items + i) * (kPrefilterVocab / (kPrefilterThreads * items));
        keys[i]       = prefilterValue<WithMarkov>(logits, stride, bias, temperature, row, col);
    }
    Sort(storage).SortDescending(keys);
    if (threadIdx.x == (rank - 1) / items)
        threshold[row] = keys[(rank - 1) % items];
    if (threadIdx.x == 0) {
        counts[row]    = 0;
        full_ends[row] = kPrefilterVocab;
    }
}

template<bool WithMarkov>
__global__ void prefilterCompactKernel(const float*         logits,
                                       int64_t              stride,
                                       const __nv_bfloat16* bias,
                                       const float*         temperature,
                                       const float*         threshold,
                                       float*               candidate_values,
                                       int*                 candidate_indices,
                                       int*                 counts) {
    using Scan = cub::BlockScan<int, kPrefilterThreads>;
    __shared__ typename Scan::TempStorage scan_storage;
    __shared__ int                        block_base;
    const int                             row    = blockIdx.y;
    const int                             start  = blockIdx.x * kPrefilterChunk + threadIdx.x * 8;
    const float                           cutoff = threshold[row];
    const bool bad_scale = WithMarkov && (!isfinite(temperature[row]) || temperature[row] <= 0.0f);
    float      values[8];
    if (start + 7 < kPrefilterVocab) {
        const float4* source = reinterpret_cast<const float4*>(logits + int64_t(row) * stride + start);
        const float4  a = source[0], b = source[1];
        const float   raw[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            float value = raw[i];
            if constexpr (WithMarkov) {
                value = (value + __bfloat162float(bias[int64_t(row) * kPrefilterVocab + start + i])) / temperature[row];
            }
            values[i] = value;
        }
    } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int col = start + i;
            values[i]     = col < kPrefilterVocab ?
                                prefilterValue<WithMarkov>(logits, stride, bias, temperature, row, col) :
                                -CUDART_INF_F;
        }
    }
    bool keep[8];
    int  kept = 0;
    bool bad  = bad_scale || isnan(cutoff) || cutoff == CUDART_INF_F;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const bool in_range = start + i < kPrefilterVocab;
        bad |= in_range && (isnan(values[i]) || values[i] == CUDART_INF_F);
        keep[i] = in_range && values[i] >= cutoff;
        kept += keep[i];
    }
    int prefix, total;
    Scan(scan_storage).ExclusiveSum(kept, prefix, total);
    if (threadIdx.x == 0) {
        block_base = atomicAdd(counts + row, total) & 0x7fffffff;
    }
    __syncthreads();
    if (bad)
        atomicOr(reinterpret_cast<unsigned int*>(counts + row), 0x80000000u);
    int destination = block_base + prefix;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (keep[i]) {
            if (destination < kPrefilterCap) {
                const int offset          = row * kPrefilterCap + destination;
                candidate_values[offset]  = values[i];
                candidate_indices[offset] = start + i;
            }
            ++destination;
        }
    }
}
}  // namespace
#endif

torch::Tensor
markov_scaled_logits(const torch::Tensor& logits, const torch::Tensor& markov_bias, const torch::Tensor& temperature) {
    TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == torch::kFloat32 && logits.dim() == 2
                    && logits.stride(1) == 1 && markov_bias.is_cuda() && markov_bias.is_contiguous()
                    && markov_bias.sizes() == logits.sizes()
                    && (markov_bias.scalar_type() == torch::kFloat32 || markov_bias.scalar_type() == torch::kBFloat16
                        || markov_bias.scalar_type() == torch::kFloat16)
                    && temperature.is_cuda() && temperature.scalar_type() == torch::kFloat32
                    && temperature.is_contiguous() && temperature.dim() == 1 && temperature.size(0) == logits.size(0)
                    && logits.device() == markov_bias.device() && logits.device() == temperature.device(),
                "DSpARK scaled logits require same-device FP32 logits [B,V], contiguous bias [B,V]"
                " in FP32/BF16/FP16, and contiguous FP32 temperature [B]");
#if defined(RTP_DEEPSELECT_FP32_ENABLED) && !defined(USE_ROCM)
    auto output = torch::empty(logits.sizes(), logits.options());
    if (logits.numel() == 0) {
        return output;
    }
    const c10::cuda::CUDAGuard guard(logits.device());
    constexpr int              threads = 256;
    const dim3                 grid((logits.size(1) + threads * 4 - 1) / (threads * 4), logits.size(0));
    auto                       stream = at::cuda::getCurrentCUDAStream(logits.get_device());
    if (markov_bias.scalar_type() == torch::kFloat32) {
        markovScaledLogitsKernel<float><<<grid, threads, 0, stream>>>(logits.data_ptr<float>(),
                                                                      logits.stride(0),
                                                                      markov_bias.data_ptr<float>(),
                                                                      temperature.data_ptr<float>(),
                                                                      output.data_ptr<float>(),
                                                                      logits.size(1),
                                                                      nullptr);
    } else if (markov_bias.scalar_type() == torch::kBFloat16) {
        markovScaledLogitsKernel<__nv_bfloat16>
            <<<grid, threads, 0, stream>>>(logits.data_ptr<float>(),
                                           logits.stride(0),
                                           reinterpret_cast<const __nv_bfloat16*>(markov_bias.data_ptr()),
                                           temperature.data_ptr<float>(),
                                           output.data_ptr<float>(),
                                           logits.size(1),
                                           nullptr);
    } else {
        markovScaledLogitsKernel<__half>
            <<<grid, threads, 0, stream>>>(logits.data_ptr<float>(),
                                           logits.stride(0),
                                           reinterpret_cast<const __half*>(markov_bias.data_ptr()),
                                           temperature.data_ptr<float>(),
                                           output.data_ptr<float>(),
                                           logits.size(1),
                                           nullptr);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
#else
    return (logits + markov_bias.to(torch::kFloat32)).div(temperature.unsqueeze(1));
#endif
}

bool deepselect_fp32_available() {
#if defined(RTP_DEEPSELECT_FP32_ENABLED) && !defined(USE_ROCM)
    return true;
#else
    return false;
#endif
}

static void deepselect_fp32_impl(const torch::Tensor& logits,
                                 const torch::Tensor& ends,
                                 torch::Tensor&       output,
                                 bool                 filter_finite,
                                 torch::Tensor*       output_values,
                                 const torch::Tensor* candidate_counts,
                                 bool                 skip_valid_candidates,
                                 const torch::Tensor* index_map) {
#if defined(RTP_DEEPSELECT_FP32_ENABLED) && !defined(USE_ROCM)
    TORCH_CHECK(logits.is_cuda() && ends.is_cuda() && output.is_cuda(), "DeepSelect requires CUDA tensors");
    TORCH_CHECK(logits.device() == ends.device() && logits.device() == output.device(),
                "DeepSelect tensors must share a device");
    TORCH_CHECK(logits.scalar_type() == torch::kFloat32, "DeepSelect logits must be FP32");
    TORCH_CHECK(ends.scalar_type() == torch::kInt32 && output.scalar_type() == torch::kInt32,
                "DeepSelect ends/output must be int32");
    TORCH_CHECK(logits.dim() == 2 && logits.size(0) <= std::numeric_limits<int32_t>::max(),
                "DeepSelect logits must be [M,N] with M <= INT32_MAX");
    const int64_t rows  = logits.size(0);
    const int64_t width = logits.size(1);
    TORCH_CHECK(width >= 512 && width < MAX_VOCAB_SIZE, "DeepSelect width must be in [512, 2^23)");
    TORCH_CHECK(logits.stride(1) == 1 && logits.stride(0) >= width && logits.stride(0) % 256 == 0
                    && reinterpret_cast<uintptr_t>(logits.data_ptr()) % 16 == 0,
                "DeepSelect logits require unit inner stride, 1024B row stride and 16B base alignment");
    if (rows > 0 && width % 32 != 0) {
        // TMA describes complete 128B rows. Stride padding alone does not prove
        // the last row's allocation backs those extra elements, so check the
        // actual storage extent. Division avoids overflow in (rows - 1) * stride.
        const uint64_t storage_elements = logits.storage().nbytes() / sizeof(float);
        const uint64_t offset           = static_cast<uint64_t>(logits.storage_offset());
        const uint64_t padded_width     = (static_cast<uint64_t>(width) + 31) / 32 * 32;
        TORCH_CHECK(offset <= storage_elements && padded_width <= storage_elements - offset
                        && static_cast<uint64_t>(rows - 1)
                               <= (storage_elements - offset - padded_width) / static_cast<uint64_t>(logits.stride(0)),
                    "DeepSelect requires backed storage through the last 128B-padded input row");
    }
    TORCH_CHECK(ends.dim() == 1 && ends.size(0) == rows && ends.is_contiguous(),
                "DeepSelect ends must be contiguous [M]");
    const int64_t topk = output.dim() == 2 ? output.size(1) : 0;
    TORCH_CHECK(output.dim() == 2 && output.size(0) == rows && (topk == 512 || topk == 4096) && output.stride(1) == 1
                    && output.stride(0) >= topk && output.stride(0) % 8 == 0
                    && reinterpret_cast<uintptr_t>(output.data_ptr()) % 32 == 0,
                "DeepSelect output must be [M,512|4096] with unit inner stride and 32B row/base alignment");
    // Strided views can make the overlap analyzer return TooHard. Reject
    // shared storage conservatively, matching the Python metadata gate.
    TORCH_CHECK(!output.is_alias_of(logits) && !output.is_alias_of(ends),
                "DeepSelect output must not share storage with logits or ends");
    at::assert_no_overlap(output, logits);
    at::assert_no_overlap(output, ends);
    if (output_values) {
        TORCH_CHECK(topk == 4096 && output_values->is_cuda() && output_values->device() == logits.device()
                        && output_values->scalar_type() == torch::kFloat32 && output_values->sizes() == output.sizes()
                        && output_values->is_contiguous() && !output_values->is_alias_of(logits)
                        && !output_values->is_alias_of(ends) && !output_values->is_alias_of(output),
                    "DeepSelect value output requires separate contiguous CUDA FP32 [M,4096] storage");
    }
    if (candidate_counts) {
        TORCH_CHECK(candidate_counts->is_cuda() && candidate_counts->device() == logits.device()
                        && candidate_counts->scalar_type() == torch::kInt32 && candidate_counts->sizes() == ends.sizes()
                        && candidate_counts->is_contiguous() && topk == 4096,
                    "candidate count mask requires CUDA int32 [M] and K4096");
    }
    if (index_map) {
        TORCH_CHECK(index_map->is_cuda() && index_map->device() == logits.device()
                        && index_map->scalar_type() == torch::kInt32 && index_map->dim() == 2
                        && index_map->size(0) == rows && index_map->size(1) == 8192 && index_map->is_contiguous()
                        && topk == 4096,
                    "candidate vocabulary map requires CUDA int32 [M,8192] and K4096");
    }
    const c10::cuda::CUDAGuard device_guard(logits.device());
    const cudaDeviceProp*      properties = at::cuda::getDeviceProperties(logits.get_device());
    TORCH_CHECK(properties->major == 10 && (properties->minor == 0 || properties->minor == 3),
                "DeepSelect FP32 requires SM100 or SM103");
    if (rows == 0) {
        return;
    }

    TopkSelectArgs args{};
    args.batch_size                = static_cast<uint32_t>(rows);
    args.vocab_size                = static_cast<uint32_t>(width);
    args.topk                      = static_cast<uint32_t>(topk);
    args.input                     = logits.data_ptr();
    args.output_index              = output.data_ptr();
    args.output_value              = output_values ? output_values->data_ptr() : nullptr;
    args.return_value              = output_values != nullptr;
    args.end_ptr                   = ends.data_ptr<int32_t>();
    args.stride_input_batch        = logits.stride(0);
    args.stride_output_index_batch = output.stride(0);
    args.stride_output_value_batch = output_values ? output_values->stride(0) : 0;
    args.idx_oob_fill_value        = -1;
    args.value_oob_fill_value      = -std::numeric_limits<float>::infinity();
    args.shared_memory_size_per_sm = properties->sharedMemPerBlockOptin;
    args.stream                    = at::cuda::getCurrentCUDAStream(logits.get_device());
    args.candidate_counts          = candidate_counts ? candidate_counts->data_ptr<int32_t>() : nullptr;
    args.skip_valid_candidates     = skip_valid_candidates;
    args.index_map                 = index_map ? index_map->data_ptr<int32_t>() : nullptr;
    args.stride_index_map_batch    = index_map ? index_map->stride(0) : 0;

    // Keep upstream's single-wave configuration. For multiple waves, reduce
    // the round and TMA buffers so that two selecting CTAs fit on each SM.
    auto dispatch = [&](auto config) {
        using Config = decltype(config);
        if (filter_finite) {
            topk_select_fp32::run_topk_select_kernel<Config, true>(args);
        } else {
            topk_select_fp32::run_topk_select_kernel<Config, false>(args);
        }
    };
    using MultiWaveConfig = TopkSelectConfig<float, int32_t, false, false, false, 512, 256, 2, 4096, 4096, 2>;
    using MultiWaveKernel = topk_select_fp32::TopkSelectKernelFP32<MultiWaveConfig, true>;
    if (topk == 4096) {
        if (output_values) {
            dispatch(TopkSelectConfig<float, int32_t, false, false, true, 4096, 256, 1, 4096, 4096, 3>{});
        } else {
            dispatch(TopkSelectConfig<float, int32_t, false, false, false, 4096, 256, 1, 4096, 4096, 3>{});
        }
    } else if (rows > properties->multiProcessorCount
               && sizeof(MultiWaveKernel::SharedMemoryPlanFP32) * MultiWaveConfig::target_occupancy
                      <= properties->sharedMemPerBlockOptin) {
        dispatch(MultiWaveConfig{});
    } else {
        dispatch(TopkSelectConfig<float, int32_t, false, false, false, 512, 512, 1, 8192, 4096, 3>{});
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
#else
    TORCH_CHECK(false, "DeepSelect FP32 requires a CUDA >= 12.9 Blackwell build");
#endif
}

void deepselect_fp32(const torch::Tensor& logits,
                     const torch::Tensor& ends,
                     torch::Tensor&       output,
                     bool                 filter_finite,
                     torch::Tensor*       output_values) {
    deepselect_fp32_impl(logits, ends, output, filter_finite, output_values, nullptr, false, nullptr);
}

void deepselect_fp32(const torch::Tensor& logits,
                     const torch::Tensor& ends,
                     torch::Tensor&       output,
                     bool                 filter_finite) {
    deepselect_fp32(logits, ends, output, filter_finite, nullptr);
}

std::tuple<torch::Tensor, torch::Tensor> deepselect_fp32_topk(const torch::Tensor& logits, int64_t k) {
#if defined(RTP_DEEPSELECT_FP32_ENABLED) && !defined(USE_ROCM)
    if ((k == 4096 || (k > 0 && k <= 512)) && logits.is_cuda() && logits.scalar_type() == torch::kFloat32
        && logits.dim() == 2 && logits.size(1) >= (k == 4096 ? 4096 : 512) && logits.is_contiguous()
        && logits.stride(0) % 256 == 0 && at::cuda::getDeviceProperties(logits.get_device())->major == 10
        && (at::cuda::getDeviceProperties(logits.get_device())->minor == 0
            || at::cuda::getDeviceProperties(logits.get_device())->minor == 3)) {
        const int64_t width   = k == 4096 ? 4096 : 512;
        auto          indices = torch::empty({logits.size(0), width}, logits.options().dtype(torch::kInt32));
        auto          ends    = torch::full({logits.size(0)}, logits.size(1), logits.options().dtype(torch::kInt32));
        auto          values  = k == 4096 ? torch::empty({logits.size(0), width}, logits.options()) : torch::Tensor();
        deepselect_fp32(logits, ends, indices, /*filter_finite=*/false, values.defined() ? &values : nullptr);
        auto indices64 = indices.to(torch::kInt64);
        if (!values.defined()) {
            values = logits.gather(1, indices64);
        }
        if (k == width) {
            return {values, indices64};
        }
        auto selected = torch::topk(values, k, -1, /*largest=*/true, /*sorted=*/false);
        return {std::get<0>(selected), indices64.gather(1, std::get<1>(selected))};
    }
#endif
    auto selected = torch::topk(logits, k, -1, /*largest=*/true, /*sorted=*/false);
    return {std::get<0>(selected), std::get<1>(selected)};
}

#if defined(RTP_DEEPSELECT_FP32_ENABLED) && !defined(USE_ROCM)
static std::tuple<torch::Tensor, torch::Tensor>
prefilter_topk4096(const torch::Tensor& logits, const torch::Tensor* markov_bias, const torch::Tensor* temperature) {
    const bool with_markov = markov_bias != nullptr;
    TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == torch::kFloat32 && logits.dim() == 2
                    && logits.size(1) == kPrefilterVocab && logits.stride(1) == 1 && logits.stride(0) % 256 == 0,
                "K4096 prefilter requires aligned CUDA FP32 [B,129280] logits");
    if (with_markov) {
        TORCH_CHECK(markov_bias->is_cuda() && markov_bias->device() == logits.device()
                        && markov_bias->scalar_type() == torch::kBFloat16 && markov_bias->sizes() == logits.sizes()
                        && markov_bias->is_contiguous() && temperature->is_cuda()
                        && temperature->device() == logits.device() && temperature->scalar_type() == torch::kFloat32
                        && temperature->dim() == 1 && temperature->size(0) == logits.size(0)
                        && temperature->is_contiguous(),
                    "Markov K4096 prefilter requires BF16 contiguous bias and FP32 temperatures");
    }
    const c10::cuda::CUDAGuard guard(logits.device());
    const int                  rows            = logits.size(0);
    auto                       int_options     = logits.options().dtype(torch::kInt32);
    auto                       thresholds      = torch::empty({rows}, logits.options());
    auto                       candidates      = torch::empty({rows, kPrefilterCap}, logits.options());
    auto                       candidate_ids   = torch::empty({rows, kPrefilterCap}, int_options);
    auto                       counts          = torch::empty({rows}, int_options);
    auto                       full_ends       = torch::empty({rows}, int_options);
    auto                       selected_values = torch::empty({rows, kPrefilterK}, logits.options());
    auto                       selected_ids    = torch::empty({rows, kPrefilterK}, int_options);
    auto                       stream          = at::cuda::getCurrentCUDAStream(logits.get_device());
    const auto*  bias_ptr = with_markov ? reinterpret_cast<const __nv_bfloat16*>(markov_bias->data_ptr()) : nullptr;
    const float* temperature_ptr = with_markov ? temperature->data_ptr<float>() : nullptr;
    if (with_markov) {
        prefilterThresholdKernel<true><<<rows, kPrefilterThreads, 0, stream>>>(logits.data_ptr<float>(),
                                                                               logits.stride(0),
                                                                               bias_ptr,
                                                                               temperature_ptr,
                                                                               thresholds.data_ptr<float>(),
                                                                               counts.data_ptr<int>(),
                                                                               full_ends.data_ptr<int>());
        prefilterCompactKernel<true>
            <<<dim3((kPrefilterVocab + kPrefilterChunk - 1) / kPrefilterChunk, rows), kPrefilterThreads, 0, stream>>>(
                logits.data_ptr<float>(),
                logits.stride(0),
                bias_ptr,
                temperature_ptr,
                thresholds.data_ptr<float>(),
                candidates.data_ptr<float>(),
                candidate_ids.data_ptr<int>(),
                counts.data_ptr<int>());
    } else {
        prefilterThresholdKernel<false><<<rows, kPrefilterThreads, 0, stream>>>(logits.data_ptr<float>(),
                                                                                logits.stride(0),
                                                                                nullptr,
                                                                                nullptr,
                                                                                thresholds.data_ptr<float>(),
                                                                                counts.data_ptr<int>(),
                                                                                full_ends.data_ptr<int>());
        prefilterCompactKernel<false>
            <<<dim3((kPrefilterVocab + kPrefilterChunk - 1) / kPrefilterChunk, rows), kPrefilterThreads, 0, stream>>>(
                logits.data_ptr<float>(),
                logits.stride(0),
                nullptr,
                nullptr,
                thresholds.data_ptr<float>(),
                candidates.data_ptr<float>(),
                candidate_ids.data_ptr<int>(),
                counts.data_ptr<int>());
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // The mask is evaluated on the GPU. The compact selector writes only valid
    // rows; invalid rows are overwritten by the exact full-vocabulary selector.
    deepselect_fp32_impl(candidates, counts, selected_ids, false, &selected_values, &counts, false, &candidate_ids);
    torch::Tensor fallback_logits = logits;
    if (with_markov) {
        fallback_logits       = torch::empty(logits.sizes(), logits.options());
        constexpr int threads = 256;
        const dim3    grid((kPrefilterVocab + threads * 4 - 1) / (threads * 4), rows);
        markovScaledLogitsKernel<__nv_bfloat16, true><<<grid, threads, 0, stream>>>(logits.data_ptr<float>(),
                                                                                    logits.stride(0),
                                                                                    bias_ptr,
                                                                                    temperature_ptr,
                                                                                    fallback_logits.data_ptr<float>(),
                                                                                    kPrefilterVocab,
                                                                                    counts.data_ptr<int>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    deepselect_fp32_impl(fallback_logits, full_ends, selected_ids, false, &selected_values, &counts, true, nullptr);
    return {selected_values, selected_ids.to(torch::kInt64)};
}
#endif

std::tuple<torch::Tensor, torch::Tensor> deepselect_fp32_topk_prefilter(const torch::Tensor& logits) {
#if defined(RTP_DEEPSELECT_FP32_ENABLED) && !defined(USE_ROCM)
    return prefilter_topk4096(logits, nullptr, nullptr);
#else
    return deepselect_fp32_topk(logits, 4096);
#endif
}

std::tuple<torch::Tensor, torch::Tensor> deepselect_fp32_markov_topk_prefilter(const torch::Tensor& logits,
                                                                               const torch::Tensor& markov_bias,
                                                                               const torch::Tensor& temperature) {
#if defined(RTP_DEEPSELECT_FP32_ENABLED) && !defined(USE_ROCM)
    return prefilter_topk4096(logits, &markov_bias, &temperature);
#else
    return deepselect_fp32_topk(markov_scaled_logits(logits, markov_bias, temperature), 4096);
#endif
}

void deepselect_fp32_joint_topk_topp(const torch::Tensor& probs,
                                     torch::Tensor&       output,
                                     const torch::Tensor& top_k,
                                     const torch::Tensor& top_p,
                                     int64_t              max_top_k) {
    TORCH_CHECK(probs.is_cuda() && probs.scalar_type() == torch::kFloat32 && probs.dim() == 2 && probs.is_contiguous()
                    && output.sizes() == probs.sizes() && output.device() == probs.device()
                    && output.scalar_type() == probs.scalar_type() && output.is_contiguous()
                    && !output.is_alias_of(probs),
                "joint top-k/top-p expects separate contiguous CUDA FP32 [B,V] input/output");
    TORCH_CHECK(top_k.is_cuda() && top_p.is_cuda() && top_k.device() == probs.device()
                    && top_p.device() == probs.device() && top_k.scalar_type() == torch::kInt32
                    && top_p.scalar_type() == torch::kFloat32 && top_k.numel() == probs.size(0)
                    && top_p.numel() == probs.size(0) && max_top_k > 0 && max_top_k <= 512
                    && max_top_k <= probs.size(1),
                "joint top-k/top-p expects CUDA row parameters and 1 <= max_top_k <= 512");

    auto selected = std::get<0>(deepselect_fp32_topk(probs, max_top_k));
    auto ranked   = std::get<0>(torch::sort(selected, -1, /*descending=*/true));
    // If the full-distribution top-p boundary lies past K, the Kth value
    // defines the intersection. Comparing all V values with the cutoff keeps
    // every token tied at either boundary.
    auto p_rank = ranked.cumsum(-1).lt(top_p.unsqueeze(1)).sum(-1, true).clamp_max(max_top_k - 1);
    // FP32 prefix sums can round to one before the last positive top-k value.
    // A top-p value of one must leave the explicit top-k support unchanged.
    p_rank.masked_fill_(top_p.unsqueeze(1).eq(1.0f), max_top_k - 1);
    auto k_rank = top_k.to(torch::kLong).sub(1).unsqueeze(1);
    auto cutoff = ranked.gather(1, torch::minimum(p_rank, k_rank));
    output.copy_(probs);
    output.masked_fill_(probs.lt(cutoff), 0.0);
    output.div_(output.sum(-1, true).clamp_min_(1e-10));
}

}  // namespace torch_ext
