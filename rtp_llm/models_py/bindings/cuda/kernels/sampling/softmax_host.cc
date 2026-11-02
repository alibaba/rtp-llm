#include "rtp_llm/models_py/bindings/cuda/kernels/sampling/softmax_host.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/sampling/split_softmax.h"
#include "rtp_llm/models_py/bindings/common/kernels/sampling_host_utils.h"

namespace rtp_llm {
namespace {
bool canUseSplitSoftmax(const torch::Tensor& logits) {
    return sampler::optimizationEnabled("RTP_LLM_SAMPLER_SPLIT_SOFTMAX") && logits.is_contiguous()
           && logits.scalar_type() == torch::kFloat32 && logits.dim() == 2 && logits.size(0) > 0 && logits.size(0) <= 32
           && logits.size(1) == 248320 && sampler::isSm120();
}
}  // namespace

torch::Tensor samplerSoftmaxInplace(const torch::Tensor& logits) {
    if (canUseSplitSoftmax(logits)) {
        splitSoftmaxInplace(logits);
        return logits;
    }
    auto probs = torch::softmax(logits, -1);
    logits.copy_(probs, true);
    return probs;
}

}  // namespace rtp_llm
