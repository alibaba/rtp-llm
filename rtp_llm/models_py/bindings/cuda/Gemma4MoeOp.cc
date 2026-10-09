#include "rtp_llm/models_py/bindings/cuda/Gemma4MoeOp.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/gemma4_moe.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>
#include <limits>

namespace torch_ext {

std::tuple<at::Tensor, at::Tensor> gemma4_topk_8_bf16(const at::Tensor& input) {
    TORCH_CHECK(input.is_cuda(), "gemma4_topk_8_bf16: input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == at::kBFloat16, "gemma4_topk_8_bf16: input must be bfloat16");
    TORCH_CHECK(input.is_contiguous(), "gemma4_topk_8_bf16: input must be contiguous");
    TORCH_CHECK(input.dim() == 2 && input.size(1) == 128, "gemma4_topk_8_bf16: input must be [tokens,128]");

    const c10::cuda::CUDAGuard device_guard(input.device());
    auto                       values  = at::empty({input.size(0), 8}, input.options());
    auto                       indices = at::empty({input.size(0), 8}, input.options().dtype(at::kLong));
    rtp_llm::invokeGemma4TopK8Bf16(reinterpret_cast<const __nv_bfloat16*>(input.const_data_ptr<at::BFloat16>()),
                                   reinterpret_cast<__nv_bfloat16*>(values.mutable_data_ptr<at::BFloat16>()),
                                   indices.mutable_data_ptr<int64_t>(),
                                   input.size(0),
                                   at::cuda::getCurrentCUDAStream(input.get_device()).stream());
    return {values, indices};
}

at::Tensor gemma4_weighted_reorder_bf16(const at::Tensor& expert_output,
                                        const at::Tensor& sorted_weight,
                                        const at::Tensor& inverse_permutation) {
    TORCH_CHECK(expert_output.is_cuda() && sorted_weight.is_cuda() && inverse_permutation.is_cuda(),
                "gemma4_weighted_reorder_bf16: inputs must be CUDA tensors");
    TORCH_CHECK(expert_output.scalar_type() == at::kBFloat16 && sorted_weight.scalar_type() == at::kBFloat16,
                "gemma4_weighted_reorder_bf16: values must be bfloat16");
    TORCH_CHECK(inverse_permutation.scalar_type() == at::kLong,
                "gemma4_weighted_reorder_bf16: inverse_permutation must be int64");
    TORCH_CHECK(expert_output.is_contiguous() && sorted_weight.is_contiguous() && inverse_permutation.is_contiguous(),
                "gemma4_weighted_reorder_bf16: inputs must be contiguous");
    TORCH_CHECK(expert_output.dim() == 2 && sorted_weight.dim() == 1 && inverse_permutation.dim() == 1,
                "gemma4_weighted_reorder_bf16: expected [N,H], [N], [N]");
    TORCH_CHECK(expert_output.size(0) == sorted_weight.numel() && expert_output.size(0) == inverse_permutation.numel(),
                "gemma4_weighted_reorder_bf16: row counts must match");
    TORCH_CHECK(expert_output.size(1) % 8 == 0, "gemma4_weighted_reorder_bf16: hidden size must be divisible by 8");
    TORCH_CHECK(expert_output.size(1) <= std::numeric_limits<int32_t>::max(),
                "gemma4_weighted_reorder_bf16: hidden size exceeds int32 limit");
    TORCH_CHECK(expert_output.get_device() == sorted_weight.get_device()
                    && expert_output.get_device() == inverse_permutation.get_device(),
                "gemma4_weighted_reorder_bf16: inputs must share a device");

    const c10::cuda::CUDAGuard device_guard(expert_output.device());
    auto                       output = at::empty_like(expert_output);
    rtp_llm::invokeGemma4WeightedReorderBf16(
        reinterpret_cast<const __nv_bfloat16*>(expert_output.const_data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(sorted_weight.const_data_ptr<at::BFloat16>()),
        inverse_permutation.const_data_ptr<int64_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.mutable_data_ptr<at::BFloat16>()),
        expert_output.size(0),
        static_cast<int32_t>(expert_output.size(1)),
        at::cuda::getCurrentCUDAStream(expert_output.get_device()).stream());
    return output;
}

at::Tensor
gemma4_gather_sorted_expert_input_bf16(const at::Tensor& input, const at::Tensor& permutation, int64_t top_k) {
    TORCH_CHECK(input.is_cuda() && permutation.is_cuda(),
                "gemma4_gather_sorted_expert_input_bf16: inputs must be CUDA tensors");
    TORCH_CHECK(input.scalar_type() == at::kBFloat16, "gemma4_gather_sorted_expert_input_bf16: input must be bfloat16");
    TORCH_CHECK(permutation.scalar_type() == at::kLong,
                "gemma4_gather_sorted_expert_input_bf16: permutation must be int64");
    TORCH_CHECK(input.is_contiguous() && permutation.is_contiguous(),
                "gemma4_gather_sorted_expert_input_bf16: inputs must be contiguous");
    TORCH_CHECK(input.dim() == 2 && input.size(1) % 8 == 0,
                "gemma4_gather_sorted_expert_input_bf16: input must be [tokens,hidden] with aligned hidden size");
    TORCH_CHECK(top_k > 0 && permutation.numel() == input.size(0) * top_k,
                "gemma4_gather_sorted_expert_input_bf16: permutation length mismatch");
    TORCH_CHECK(top_k <= std::numeric_limits<int32_t>::max() && input.size(1) <= std::numeric_limits<int32_t>::max(),
                "gemma4_gather_sorted_expert_input_bf16: dimensions exceed int32 limits");
    TORCH_CHECK(input.get_device() == permutation.get_device(),
                "gemma4_gather_sorted_expert_input_bf16: inputs must share a device");

    const c10::cuda::CUDAGuard device_guard(input.device());
    auto                       output = at::empty({permutation.numel(), input.size(1)}, input.options());
    rtp_llm::invokeGemma4GatherSortedExpertInputBf16(
        reinterpret_cast<const __nv_bfloat16*>(input.const_data_ptr<at::BFloat16>()),
        permutation.const_data_ptr<int64_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.mutable_data_ptr<at::BFloat16>()),
        permutation.numel(),
        static_cast<int32_t>(top_k),
        static_cast<int32_t>(input.size(1)),
        at::cuda::getCurrentCUDAStream(input.get_device()).stream());
    return output;
}

at::Tensor gemma4_top8_sum_bf16(const at::Tensor& input) {
    TORCH_CHECK(input.is_cuda(), "gemma4_top8_sum_bf16: input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == at::kBFloat16, "gemma4_top8_sum_bf16: input must be bfloat16");
    TORCH_CHECK(input.is_contiguous(), "gemma4_top8_sum_bf16: input must be contiguous");
    TORCH_CHECK(input.dim() == 3 && input.size(1) == 8 && input.size(2) % 8 == 0,
                "gemma4_top8_sum_bf16: input must be [tokens,8,aligned_hidden]");
    TORCH_CHECK(input.size(2) <= std::numeric_limits<int32_t>::max(),
                "gemma4_top8_sum_bf16: hidden size exceeds int32 limit");

    const c10::cuda::CUDAGuard device_guard(input.device());
    auto                       output = at::empty({input.size(0), input.size(2)}, input.options());
    rtp_llm::invokeGemma4Top8SumBf16(reinterpret_cast<const __nv_bfloat16*>(input.const_data_ptr<at::BFloat16>()),
                                     reinterpret_cast<__nv_bfloat16*>(output.mutable_data_ptr<at::BFloat16>()),
                                     input.size(0),
                                     static_cast<int32_t>(input.size(2)),
                                     at::cuda::getCurrentCUDAStream(input.get_device()).stream());
    return output;
}

at::Tensor gemma4_finalize_router_weights_bf16(const at::Tensor& top_weights,
                                               const at::Tensor& top_indices,
                                               const at::Tensor& expert_scales) {
    TORCH_CHECK(top_weights.is_cuda() && top_indices.is_cuda() && expert_scales.is_cuda(),
                "gemma4_finalize_router_weights_bf16: inputs must be CUDA tensors");
    TORCH_CHECK(top_weights.scalar_type() == at::kBFloat16 && expert_scales.scalar_type() == at::kBFloat16,
                "gemma4_finalize_router_weights_bf16: weights and scales must be bfloat16");
    TORCH_CHECK(top_indices.scalar_type() == at::kLong, "gemma4_finalize_router_weights_bf16: indices must be int64");
    TORCH_CHECK(top_weights.is_contiguous() && top_indices.is_contiguous() && expert_scales.is_contiguous(),
                "gemma4_finalize_router_weights_bf16: inputs must be contiguous");
    TORCH_CHECK(top_weights.dim() == 2 && top_weights.size(1) == 8,
                "gemma4_finalize_router_weights_bf16: top_weights must be [tokens,8]");
    TORCH_CHECK(top_indices.sizes() == top_weights.sizes(),
                "gemma4_finalize_router_weights_bf16: top_indices shape must match top_weights");
    TORCH_CHECK(expert_scales.dim() == 1, "gemma4_finalize_router_weights_bf16: expert_scales must be a vector");
    TORCH_CHECK(top_weights.get_device() == top_indices.get_device()
                    && top_weights.get_device() == expert_scales.get_device(),
                "gemma4_finalize_router_weights_bf16: inputs must share a device");

    const c10::cuda::CUDAGuard device_guard(top_weights.device());
    auto                       output = at::empty_like(top_weights);
    rtp_llm::invokeGemma4FinalizeRouterWeightsBf16(
        reinterpret_cast<const __nv_bfloat16*>(top_weights.const_data_ptr<at::BFloat16>()),
        top_indices.const_data_ptr<int64_t>(),
        reinterpret_cast<const __nv_bfloat16*>(expert_scales.const_data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(output.mutable_data_ptr<at::BFloat16>()),
        top_weights.size(0),
        at::cuda::getCurrentCUDAStream(top_weights.get_device()).stream());
    return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
gemma4_prepare_grouped_moe(const at::Tensor& expert_ids, const at::Tensor& weights, int64_t num_experts) {
    TORCH_CHECK(expert_ids.is_cuda() && weights.is_cuda(), "gemma4_prepare_grouped_moe: inputs must be CUDA tensors");
    TORCH_CHECK(expert_ids.scalar_type() == at::kLong && weights.scalar_type() == at::kBFloat16,
                "gemma4_prepare_grouped_moe: expected int64 IDs and bfloat16 weights");
    TORCH_CHECK(expert_ids.is_contiguous() && weights.is_contiguous(),
                "gemma4_prepare_grouped_moe: inputs must be contiguous");
    TORCH_CHECK(expert_ids.dim() == 1 && weights.dim() == 1 && expert_ids.numel() == weights.numel(),
                "gemma4_prepare_grouped_moe: inputs must be matching vectors");
    TORCH_CHECK(num_experts > 0 && num_experts <= std::numeric_limits<int32_t>::max(),
                "gemma4_prepare_grouped_moe: invalid expert count");
    TORCH_CHECK(expert_ids.numel() <= std::numeric_limits<int32_t>::max(),
                "gemma4_prepare_grouped_moe: row count exceeds int32 limit");
    TORCH_CHECK(expert_ids.get_device() == weights.get_device(),
                "gemma4_prepare_grouped_moe: inputs must share a device");

    const c10::cuda::CUDAGuard device_guard(expert_ids.device());
    auto                       permutation         = at::empty_like(expert_ids);
    auto                       inverse_permutation = at::empty_like(expert_ids);
    auto                       sorted_weights      = at::empty_like(weights);
    auto                       int_options         = expert_ids.options().dtype(at::kInt);
    auto                       offsets             = at::empty({num_experts}, int_options);
    auto                       scratch             = at::empty({2, num_experts}, int_options);
    rtp_llm::invokeGemma4PrepareGroupedMoe(
        expert_ids.const_data_ptr<int64_t>(),
        reinterpret_cast<const __nv_bfloat16*>(weights.const_data_ptr<at::BFloat16>()),
        permutation.mutable_data_ptr<int64_t>(),
        inverse_permutation.mutable_data_ptr<int64_t>(),
        reinterpret_cast<__nv_bfloat16*>(sorted_weights.mutable_data_ptr<at::BFloat16>()),
        offsets.mutable_data_ptr<int32_t>(),
        scratch.mutable_data_ptr<int32_t>(),
        expert_ids.numel(),
        static_cast<int32_t>(num_experts),
        at::cuda::getCurrentCUDAStream(expert_ids.get_device()).stream());
    return {permutation, inverse_permutation, sorted_weights, offsets};
}

}  // namespace torch_ext
