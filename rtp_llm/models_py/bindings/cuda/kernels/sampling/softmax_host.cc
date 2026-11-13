#include "rtp_llm/models_py/bindings/cuda/kernels/sampling/softmax_host.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/sampling/split_softmax.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/sampling/host_utils.h"
#include "rtp_llm/models_py/bindings/cuda/ops/StandaloneOps.h"

namespace rtp_llm {
namespace {
bool canUseSplitSoftmax(const torch::Tensor& logits) {
    return logits.is_contiguous() && logits.scalar_type() == torch::kFloat32 && logits.dim() == 2 && logits.size(0) > 0
           && logits.size(0) <= 32 && logits.size(1) == 248320 && sampler::isSm120();
}
}  // namespace

torch::Tensor samplerSoftmaxInplace(const torch::Tensor& logits, bool batch_invariant) {
#if USING_CUDA && !USE_PPU
    if (batch_invariant) {
        auto stats         = cudaLogProbStats(logits);
        auto probabilities = ((logits - stats.narrow(1, 0, 1)) - stats.narrow(1, 1, 1)).exp();
        logits.copy_(probabilities, true);
        return logits;
    }
#else
    (void)batch_invariant;
#endif
    if (canUseSplitSoftmax(logits)) {
        splitSoftmaxInplace(logits);
        return logits;
    }
    auto probs = torch::softmax(logits, -1);
    logits.copy_(probs, true);
    return probs;
}

}  // namespace rtp_llm
