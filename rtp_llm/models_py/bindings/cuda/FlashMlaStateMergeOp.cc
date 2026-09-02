#include "rtp_llm/models_py/bindings/cuda/FlashMlaStateMergeOp.h"

#include "rtp_llm/models_py/bindings/cuda/kernels/flashmla_state_merge.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>

#include <cstdint>

namespace rtp_llm {
namespace {

constexpr int64_t kHeadSize = 128;

bool hasSupportedLseLayout(const torch::Tensor& tensor) {
    if (tensor.is_contiguous()) {
        return true;
    }
    return tensor.stride(0) == 1 && tensor.stride(1) >= tensor.size(0);
}

bool hasVectorAlignment(const torch::Tensor& tensor) {
    const uintptr_t address   = reinterpret_cast<uintptr_t>(tensor.data_ptr());
    const uintptr_t alignment = tensor.scalar_type() == at::kFloat ? 16 : 8;
    return address % alignment == 0;
}

void validateStateTensors(const torch::Tensor& output,
                          const torch::Tensor& output_lse,
                          const torch::Tensor& partial_output,
                          const torch::Tensor& partial_lse) {
    TORCH_CHECK(output.dim() == 3 && partial_output.dim() == 3,
                "FlashMLA output states must have shape [tokens, heads, 128]");
    TORCH_CHECK(output.size(1) > 0 && output.size(1) == partial_output.size(1) && output.size(2) == kHeadSize
                    && partial_output.size(2) == kHeadSize,
                "FlashMLA state merge requires matching 128-wide heads");
    TORCH_CHECK(output_lse.sizes() == output.sizes().slice(0, 2)
                    && partial_lse.sizes() == partial_output.sizes().slice(0, 2),
                "FlashMLA LSE tensors must have shape [tokens, heads]");
    TORCH_CHECK(output.scalar_type() == at::kBFloat16 || output.scalar_type() == at::kFloat,
                "FlashMLA accumulator must be BF16 or FP32");
    TORCH_CHECK(partial_output.scalar_type() == at::kBFloat16, "FlashMLA partial output must be BF16");
    TORCH_CHECK(output_lse.scalar_type() == at::kFloat && partial_lse.scalar_type() == at::kFloat,
                "FlashMLA LSE tensors must be FP32");
    TORCH_CHECK(output.is_contiguous() && partial_output.is_contiguous(), "FlashMLA output states must be contiguous");
    TORCH_CHECK(hasSupportedLseLayout(output_lse) && hasSupportedLseLayout(partial_lse),
                "FlashMLA LSE tensors must have supported non-overlapping layouts");
    TORCH_CHECK(hasVectorAlignment(output) && hasVectorAlignment(partial_output),
                "FlashMLA output states do not satisfy vector alignment");
}

void validateMetadata(const torch::Tensor& metadata, const char* name) {
    TORCH_CHECK(metadata.scalar_type() == at::kInt && metadata.dim() == 1 && metadata.is_contiguous(),
                name,
                " must be a contiguous int32 vector");
}

}  // namespace

void FlashMlaMergeAttentionStatesSegmentedInPlace(torch::Tensor output,
                                                  torch::Tensor output_lse,
                                                  torch::Tensor partial_output,
                                                  torch::Tensor partial_lse,
                                                  torch::Tensor partial_q_indptr,
                                                  torch::Tensor destination_starts) {
    validateStateTensors(output, output_lse, partial_output, partial_lse);
    validateMetadata(partial_q_indptr, "partial_q_indptr");
    validateMetadata(destination_starts, "destination_starts");
    const int64_t segment_count = destination_starts.numel();
    TORCH_CHECK(partial_q_indptr.numel() == segment_count + 1,
                "segmented FlashMLA metadata requires S+1 indptr entries and S destinations");
    TORCH_CHECK(partial_output.size(0) == 0 || output.size(0) > 0,
                "segmented FlashMLA merge requires output tokens for nonempty partial state");
    if (partial_output.size(0) == 0) {
        return;
    }

    const c10::cuda::CUDAGuard device_guard(output.device());
    const auto                 stream = at::cuda::getCurrentCUDAStream(output.get_device()).stream();
    invokeFlashMlaStateMergeSegmented(
        output, output_lse, partial_output, partial_lse, partial_q_indptr, destination_starts, stream);
}

}  // namespace rtp_llm
