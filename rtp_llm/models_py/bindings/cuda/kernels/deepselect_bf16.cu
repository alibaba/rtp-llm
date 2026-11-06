#include "rtp_llm/models_py/bindings/cuda/kernels/deepselect_bf16.h"

#include <cstdint>
#include <limits>
#include <type_traits>

#if defined(RTP_DEEPSELECT_BF16_ENABLED) && !defined(USE_ROCM)
#include <ATen/MemoryOverlap.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

// RTP's CUTLASS 3.8 predates SM103 and recognizes TMA on SM100a only.
// The exact SM103a cubin supports these same SM90 TMA load instructions.
// Enable only the feature used here, without globally claiming SM100 MMA.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1030 && defined(__CUDA_ARCH_FEAT_SM103_ALL)
#ifndef CUTE_ARCH_TMA_SM90_ENABLED
#define CUTE_ARCH_TMA_SM90_ENABLED
#endif
#endif

#include "rtp_llm/models_py/bindings/cuda/kernels/deepselect/cuda_kernels/v3/topk_select.cuh"
#endif

namespace torch_ext {

bool deepselect_bf16_available() {
#if defined(RTP_DEEPSELECT_BF16_ENABLED) && !defined(USE_ROCM)
    return true;
#else
    return false;
#endif
}

namespace {

template<bool DENSE, bool CANDIDATES = false, bool SORTED_TOKENS = false>
void deepselect_bf16_impl(const torch::Tensor& logits,
                          const torch::Tensor& ends,
                          torch::Tensor&       output,
                          bool                 filter_finite) {
#if defined(RTP_DEEPSELECT_BF16_ENABLED) && !defined(USE_ROCM)
    TORCH_CHECK(logits.is_cuda() && ends.is_cuda() && output.is_cuda(), "DeepSelect requires CUDA tensors");
    TORCH_CHECK(logits.device() == ends.device() && logits.device() == output.device(),
                "DeepSelect tensors must share a device");
    TORCH_CHECK(logits.scalar_type() == torch::kBFloat16, "DeepSelect logits must be BF16");
    TORCH_CHECK(ends.scalar_type() == torch::kInt32 && output.scalar_type() == torch::kInt32,
                "DeepSelect ends/output must be int32");
    TORCH_CHECK(logits.dim() == 2 && logits.size(0) <= std::numeric_limits<int32_t>::max(),
                "DeepSelect logits must be [M,N] with M <= INT32_MAX");
    const int64_t rows  = logits.size(0);
    const int64_t width = logits.size(1);
    TORCH_CHECK(width >= 512 && width < MAX_VOCAB_SIZE && width % 512 == 0,
                "DeepSelect width must be a multiple of 512 in [512, 2^23)");
    TORCH_CHECK(logits.stride(1) == 1 && logits.stride(0) >= width && logits.stride(0) % 512 == 0
                    && reinterpret_cast<uintptr_t>(logits.data_ptr()) % 16 == 0,
                "DeepSelect logits require unit inner stride, 1024B row stride and 16B base alignment");
    TORCH_CHECK(ends.dim() == 1 && ends.size(0) == rows && ends.is_contiguous(),
                "DeepSelect ends must be contiguous [M]");
    constexpr int64_t topk = CANDIDATES ? 2048 : 512;
    TORCH_CHECK(output.dim() == 2 && output.size(0) == rows && output.size(1) == topk && output.stride(1) == 1
                    && output.stride(0) >= topk && output.stride(0) % 8 == 0
                    && reinterpret_cast<uintptr_t>(output.data_ptr()) % 32 == 0,
                "DeepSelect output must be [M,",
                topk,
                "] with unit inner stride and 32B row/base alignment");
    at::assert_no_overlap(output, logits);
    at::assert_no_overlap(output, ends);
    const c10::cuda::CUDAGuard device_guard(logits.device());
    const cudaDeviceProp*      properties = at::cuda::getDeviceProperties(logits.get_device());
    TORCH_CHECK(properties->major == 10 && (properties->minor == 0 || properties->minor == 3),
                "DeepSelect BF16 requires SM100 or SM103");
    if (rows == 0) {
        return;
    }

    TopkSelectArgs args{};
    args.batch_size                = static_cast<uint32_t>(rows);
    args.vocab_size                = static_cast<uint32_t>(width);
    args.topk                      = topk;
    args.sorted_index              = CANDIDATES || SORTED_TOKENS;
    args.input                     = logits.data_ptr();
    args.output_index              = output.data_ptr();
    args.end_ptr                   = ends.data_ptr<int32_t>();
    args.stride_input_batch        = logits.stride(0);
    args.stride_output_index_batch = output.stride(0);
    args.idx_oob_fill_value        = -1;
    args.value_oob_fill_value      = -std::numeric_limits<float>::infinity();
    args.shared_memory_size_per_sm = properties->sharedMemPerBlockOptin;
    args.stream                    = at::cuda::getCurrentCUDAStream(logits.get_device());

    if constexpr (CANDIDATES) {
        // Upstream normal BF16 has capacity buckets 512/1024/4096, not 2048.
        // Select runtime K2048 using the 4096 bucket. Keep occupancy at one:
        // its survivor buffers are larger than the token K512 instantiations.
        using Config = TopkSelectConfig<nv_bfloat16,
                                        int32_t,
                                        false,
                                        true,
                                        false,
                                        4096,
                                        256,
                                        1,
                                        4096,
                                        4096,
                                        4,
                                        512,
                                        1,
                                        true,
                                        false,
                                        true>;
        topk_select_bf16_normal::run_topk_select_kernel<Config>(args);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }

    // Match the upstream BF16 normal dispatcher, but instantiate only K512,
    // int32 indices and optional index sorting, without value output.
    // Sorting uses upstream's window epilogue within the selecting CTA.
    const auto launch = [&](auto finite) {
        constexpr bool FILTER_FINITE = decltype(finite)::value;
        if (rows <= properties->multiProcessorCount) {
            using Config = TopkSelectConfig<nv_bfloat16,
                                            int32_t,
                                            false,
                                            SORTED_TOKENS,
                                            false,
                                            512,
                                            512,
                                            1,
                                            8192,
                                            4096,
                                            5,
                                            512,
                                            1,
                                            DENSE,
                                            FILTER_FINITE>;
            topk_select_bf16_normal::run_topk_select_kernel<Config>(args);
        } else {
            using Config = TopkSelectConfig<nv_bfloat16,
                                            int32_t,
                                            false,
                                            SORTED_TOKENS,
                                            false,
                                            512,
                                            256,
                                            2,
                                            4096,
                                            4096,
                                            4,
                                            512,
                                            1,
                                            DENSE,
                                            FILTER_FINITE>;
            topk_select_bf16_normal::run_topk_select_kernel<Config>(args);
        }
    };
    if constexpr (DENSE && !SORTED_TOKENS) {
        if (filter_finite) {
            launch(std::true_type{});
        } else {
            launch(std::false_type{});
        }
    } else {
        launch(std::false_type{});
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
#else
    TORCH_CHECK(false, "DeepSelect BF16 requires a CUDA >= 12.9 Blackwell build");
#endif
}

#if defined(RTP_DEEPSELECT_BF16_ENABLED) && !defined(USE_ROCM)
// scores[b, pitch] = bf16_rn(logits[b, V] + float(bias[b, V])), tail padded -inf.
__global__ void correctedBf16Kernel(const float* __restrict__ logits,
                                    int64_t logits_row_stride,
                                    const __nv_bfloat16* __restrict__ bias,
                                    __nv_bfloat16* __restrict__ scores,
                                    int vocab_size,
                                    int pitch) {
    const int      row = blockIdx.y;
    const int      col = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    __nv_bfloat16* out = scores + static_cast<int64_t>(row) * pitch;
    if (col >= vocab_size) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (col + i < pitch) {
                out[col + i] = __float2bfloat16_rn(-std::numeric_limits<float>::infinity());
            }
        }
        return;
    }
    const int64_t logits_offset = static_cast<int64_t>(row) * logits_row_stride + col;
    const int64_t bias_offset   = static_cast<int64_t>(row) * vocab_size + col;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (col + i < vocab_size) {
            out[col + i] = __float2bfloat16_rn(logits[logits_offset + i] + static_cast<float>(bias[bias_offset + i]));
        } else if (col + i < pitch) {
            out[col + i] = __float2bfloat16_rn(-std::numeric_limits<float>::infinity());
        }
    }
}

// values[b, k] = float(scores[b, ids[b, k]]) / temperature[b]. Selection already
// happened before temperature scaling, which preserves order for positive temperature.
__global__ void scaleSelectedKernel(const __nv_bfloat16* __restrict__ scores,
                                    int pitch,
                                    const int32_t* __restrict__ ids,
                                    const float* __restrict__ temperature,
                                    float* __restrict__ values,
                                    int k) {
    const int row = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= k) {
        return;
    }
    const int32_t id = ids[static_cast<int64_t>(row) * k + col];
    // Keep division correctly rounded under this translation unit's fast-math flags.
    const float value = __fdiv_rn(static_cast<float>(scores[static_cast<int64_t>(row) * pitch + id]), temperature[row]);
    values[static_cast<int64_t>(row) * k + col] = value;
}

#endif

}  // namespace

void deepselect_bf16(const torch::Tensor& logits, const torch::Tensor& ends, torch::Tensor& output) {
    deepselect_bf16_impl<false>(logits, ends, output, false);
}

void deepselect_bf16_dense(const torch::Tensor& logits,
                           const torch::Tensor& ends,
                           torch::Tensor&       output,
                           bool                 filter_finite) {
    deepselect_bf16_impl<true>(logits, ends, output, filter_finite);
}

void deepselect_bf16_candidates(const torch::Tensor& scores, const torch::Tensor& ends, torch::Tensor& output) {
    deepselect_bf16_impl<true, true>(scores, ends, output, false);
}

void deepselect_bf16_dense_sorted(const torch::Tensor& logits, const torch::Tensor& ends, torch::Tensor& output) {
    // Decode publication already filters nonfinite scores, after selection.
    deepselect_bf16_impl<true, false, true>(logits, ends, output, false);
}

bool draft_markov_topk_bf16_supported(const torch::Tensor& logits, int64_t k) {
#if defined(RTP_DEEPSELECT_BF16_ENABLED) && !defined(USE_ROCM)
    if (!logits.defined() || !logits.is_cuda() || logits.scalar_type() != torch::kFloat32 || logits.dim() != 2
        || logits.stride(1) != 1 || logits.size(0) > 65535) {
        return false;
    }
    if (k != 512 && k != 1024 && k != 4096) {
        return false;
    }
    const auto vocab = logits.size(1);
    if (vocab < k || (vocab + 511) / 512 * 512 >= MAX_VOCAB_SIZE) {
        return false;
    }
    const auto* properties = at::cuda::getDeviceProperties(logits.get_device());
    return properties->major == 10 && (properties->minor == 0 || properties->minor == 3);
#else
    (void)logits;
    (void)k;
    return false;
#endif
}

std::tuple<torch::Tensor, torch::Tensor> draft_markov_topk_bf16(const torch::Tensor& logits,
                                                                const torch::Tensor& markov_bias,
                                                                const torch::Tensor& temperature,
                                                                int64_t              k) {
#if defined(RTP_DEEPSELECT_BF16_ENABLED) && !defined(USE_ROCM)
    TORCH_CHECK(draft_markov_topk_bf16_supported(logits, k),
                "draft BF16 topk requires SM100/SM103, FP32 [B,V] logits with unit inner stride, B <= 65535, "
                "K in {512,1024,4096}, K <= V and padded V < 2^23");
    const int64_t rows  = logits.size(0);
    const int64_t vocab = logits.size(1);
    TORCH_CHECK(markov_bias.is_cuda() && markov_bias.scalar_type() == torch::kBFloat16 && markov_bias.is_contiguous()
                    && markov_bias.sizes() == logits.sizes(),
                "draft bf16 markov topk requires contiguous BF16 bias [B,V] matching the logits");
    TORCH_CHECK(temperature.is_cuda() && temperature.scalar_type() == torch::kFloat32 && temperature.is_contiguous()
                    && temperature.dim() == 1 && temperature.size(0) == rows,
                "draft bf16 markov topk requires contiguous FP32 temperature [B]");
    TORCH_CHECK(logits.device() == markov_bias.device() && logits.device() == temperature.device(),
                "draft bf16 markov topk tensors must share a device");
    const c10::cuda::CUDAGuard device_guard(logits.device());
    const cudaDeviceProp*      properties = at::cuda::getDeviceProperties(logits.get_device());

    // BF16 selector contract: N % 512 == 0 and row stride % 512 == 0.
    const int64_t pitch        = (vocab + 511) / 512 * 512;
    auto          scores       = torch::empty({rows, pitch}, logits.options().dtype(torch::kBFloat16));
    auto          values       = torch::empty({rows, k}, logits.options());
    auto          selected_ids = torch::empty({rows, k}, logits.options().dtype(torch::kInt32));
    if (rows == 0) {
        return {values, selected_ids};
    }
    auto stream = at::cuda::getCurrentCUDAStream(logits.get_device());

    {
        constexpr int threads = 256;
        const dim3 grid(static_cast<uint32_t>((pitch + threads * 4 - 1) / (threads * 4)), static_cast<uint32_t>(rows));
        correctedBf16Kernel<<<grid, threads, 0, stream>>>(
            logits.data_ptr<float>(),
            logits.stride(0),
            reinterpret_cast<const __nv_bfloat16*>(markov_bias.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(scores.data_ptr()),
            static_cast<int>(vocab),
            static_cast<int>(pitch));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    TopkSelectArgs args{};
    args.batch_size                = static_cast<uint32_t>(rows);
    args.vocab_size                = static_cast<uint32_t>(vocab);
    args.topk                      = static_cast<uint32_t>(k);
    args.sorted_index              = false;
    args.abort_when_nan_found      = true;
    args.input                     = scores.data_ptr();
    args.output_index              = selected_ids.data_ptr();
    args.stride_input_batch        = pitch;
    args.stride_output_index_batch = k;
    args.idx_oob_fill_value        = -1;
    args.value_oob_fill_value      = -std::numeric_limits<float>::infinity();
    args.shared_memory_size_per_sm = properties->sharedMemPerBlockOptin;
    args.stream                    = stream;

    // Use the native capacity buckets with padding excluded from -INF tie quotas.
    if (k == 512) {
        using Config = TopkSelectConfig<nv_bfloat16,
                                        int32_t,
                                        false,
                                        false,
                                        false,
                                        512,
                                        512,
                                        1,
                                        8192,
                                        4096,
                                        5,
                                        512,
                                        1,
                                        false,
                                        false,
                                        false,
                                        true>;
        topk_select_bf16_normal::run_topk_select_kernel<Config>(args);
    } else if (k == 1024) {
        using Config = TopkSelectConfig<nv_bfloat16,
                                        int32_t,
                                        false,
                                        false,
                                        false,
                                        1024,
                                        512,
                                        1,
                                        8192,
                                        4096,
                                        3,
                                        512,
                                        1,
                                        false,
                                        false,
                                        false,
                                        true>;
        topk_select_bf16_normal::run_topk_select_kernel<Config>(args);
    } else {
        using Config = TopkSelectConfig<nv_bfloat16,
                                        int32_t,
                                        false,
                                        false,
                                        false,
                                        4096,
                                        256,
                                        1,
                                        4096,
                                        4096,
                                        4,
                                        512,
                                        1,
                                        false,
                                        false,
                                        false,
                                        true>;
        topk_select_bf16_normal::run_topk_select_kernel<Config>(args);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    {
        constexpr int threads = 256;
        const dim3    grid(static_cast<uint32_t>((k + threads - 1) / threads), static_cast<uint32_t>(rows));
        scaleSelectedKernel<<<grid, threads, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(scores.data_ptr()),
                                                          static_cast<int>(pitch),
                                                          selected_ids.data_ptr<int32_t>(),
                                                          temperature.data_ptr<float>(),
                                                          values.data_ptr<float>(),
                                                          static_cast<int>(k));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {values, selected_ids};
#else
    TORCH_CHECK(false, "draft bf16 markov topk requires a CUDA >= 12.9 Blackwell build");
#endif
}

}  // namespace torch_ext
