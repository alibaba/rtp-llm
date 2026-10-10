#include "rtp_llm/models_py/bindings/cuda/kernels/deepselect_fp32.h"

#include <cstdint>
#include <limits>

#if defined(RTP_DEEPSELECT_FP32_ENABLED) && !defined(USE_ROCM)
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

#include "rtp_llm/models_py/bindings/cuda/kernels/deepselect/cuda_kernels/v3_fp32/topk_select.cuh"
#endif

namespace torch_ext {

bool deepselect_fp32_available() {
#if defined(RTP_DEEPSELECT_FP32_ENABLED) && !defined(USE_ROCM)
    return true;
#else
    return false;
#endif
}

void deepselect_fp32(const torch::Tensor& logits,
                     const torch::Tensor& ends,
                     torch::Tensor&       output,
                     bool                 filter_finite) {
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
    TORCH_CHECK(output.dim() == 2 && output.size(0) == rows && output.size(1) == 512 && output.stride(1) == 1
                    && output.stride(0) >= 512 && output.stride(0) % 8 == 0
                    && reinterpret_cast<uintptr_t>(output.data_ptr()) % 32 == 0,
                "DeepSelect output must be [M,512] with unit inner stride and 32B row/base alignment");
    // Strided views can make the overlap analyzer return TooHard. Reject
    // shared storage conservatively, matching the Python metadata gate.
    TORCH_CHECK(!output.is_alias_of(logits) && !output.is_alias_of(ends),
                "DeepSelect output must not share storage with logits or ends");
    at::assert_no_overlap(output, logits);
    at::assert_no_overlap(output, ends);
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
    args.topk                      = 512;
    args.input                     = logits.data_ptr();
    args.output_index              = output.data_ptr();
    args.end_ptr                   = ends.data_ptr<int32_t>();
    args.stride_input_batch        = logits.stride(0);
    args.stride_output_index_batch = output.stride(0);
    args.idx_oob_fill_value        = -1;
    args.value_oob_fill_value      = -std::numeric_limits<float>::infinity();
    args.shared_memory_size_per_sm = properties->sharedMemPerBlockOptin;
    args.stream                    = at::cuda::getCurrentCUDAStream(logits.get_device());

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
    if (rows > properties->multiProcessorCount
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

}  // namespace torch_ext
