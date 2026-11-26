#include "rtp_llm/models_py/bindings/cuda/kernels/sampling/softmax_host.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/sampling/split_softmax.h"

namespace rtp_llm {
namespace {
// The split kernel trio is architecture-generic (cub BlockReduce + warp
// shuffles, vocabulary passed as a runtime arg) and replicates PyTorch's
// exact FP32 accumulation order, so the historical SM120/248320-only gate
// only excluded every other deployment. Split softmax pays off whenever a
// 1024-thread single-CTA row softmax (torch's dispatch) is latency-bound on
// a large vocab; keep torch::softmax for small vocabs and CPU tensors. The
// % 4 requirement keeps every row's base pointer 16B-aligned for the vec4
// loads in samplerOrderedWarpSums.
bool canUseSplitSoftmax(const torch::Tensor& logits) {
    return logits.is_cuda() && logits.is_contiguous() && logits.scalar_type() == torch::kFloat32 && logits.dim() == 2
           && logits.size(0) > 0 && logits.size(0) <= 32 && logits.size(1) >= 65536 && logits.size(1) % 4 == 0;
}
}  // namespace

bool canSamplerSplitSoftmax(const torch::Tensor& logits) {
    return canUseSplitSoftmax(logits);
}

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
