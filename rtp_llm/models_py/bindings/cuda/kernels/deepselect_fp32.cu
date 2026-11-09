#include "rtp_llm/models_py/bindings/cuda/kernels/deepselect_fp32.h"
#include <cstdint>
#include <limits>

#if defined(RTP_DEEPSELECT_FP32_ENABLED) && !defined(USE_ROCM)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

// CUTLASS 4.0 does not recognize SM103a's supported TMA instructions.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1030 && defined(__CUDA_ARCH_FEAT_SM103_ALL)
#ifndef CUTE_ARCH_TMA_SM90_ENABLED
#define CUTE_ARCH_TMA_SM90_ENABLED
#endif
#endif
#include "rtp_llm/models_py/bindings/cuda/kernels/deepselect/cuda_kernels/v3_fp32/topk_select.cuh"
#endif

namespace torch_ext {
std::tuple<torch::Tensor, torch::Tensor>
deepselect_fp32_topk(const torch::Tensor& logits, int64_t k, bool* used_deepselect) {
    if (used_deepselect)
        *used_deepselect = false;
#if defined(RTP_DEEPSELECT_FP32_ENABLED) && !defined(USE_ROCM)
    if (k == 4096 && logits.is_cuda() && logits.scalar_type() == torch::kFloat32 && logits.dim() == 2
        && logits.size(0) <= std::numeric_limits<int32_t>::max() && logits.size(1) >= k
        && logits.size(1) < MAX_VOCAB_SIZE && logits.is_contiguous() && logits.stride(0) % 256 == 0
        && reinterpret_cast<uintptr_t>(logits.data_ptr()) % 16 == 0) {
        const auto* properties = at::cuda::getDeviceProperties(logits.get_device());
        if (properties->major == 10 && (properties->minor == 0 || properties->minor == 3)) {
            const c10::cuda::CUDAGuard guard(logits.device());
            auto indices = torch::empty({logits.size(0), k}, logits.options().dtype(torch::kInt32));
            auto values  = torch::empty({logits.size(0), k}, logits.options());
            auto ends    = torch::full({logits.size(0)}, logits.size(1), logits.options().dtype(torch::kInt32));
            if (logits.size(0) > 0) {
                TopkSelectArgs args{};
                args.batch_size                = static_cast<uint32_t>(logits.size(0));
                args.vocab_size                = static_cast<uint32_t>(logits.size(1));
                args.topk                      = 4096;
                args.input                     = logits.data_ptr();
                args.output_index              = indices.data_ptr();
                args.output_value              = values.data_ptr();
                args.return_value              = true;
                args.end_ptr                   = ends.data_ptr<int32_t>();
                args.stride_input_batch        = logits.stride(0);
                args.stride_output_index_batch = indices.stride(0);
                args.stride_output_value_batch = values.stride(0);
                args.idx_oob_fill_value        = -1;
                args.value_oob_fill_value      = -std::numeric_limits<float>::infinity();
                args.shared_memory_size_per_sm = properties->sharedMemPerBlockOptin;
                args.stream                    = at::cuda::getCurrentCUDAStream(logits.get_device());
                using Config = TopkSelectConfig<float, int32_t, false, false, true, 4096, 256, 1, 4096, 4096, 3>;
                topk_select_fp32::run_topk_select_kernel<Config, false>(args);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
            if (used_deepselect)
                *used_deepselect = true;
            return {values, indices.to(torch::kInt64)};
        }
    }
#endif
    return torch::topk(logits, k, -1, true, false);
}
}  // namespace torch_ext
