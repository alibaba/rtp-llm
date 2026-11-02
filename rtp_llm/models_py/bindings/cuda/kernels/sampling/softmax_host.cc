#include "rtp_llm/models_py/bindings/cuda/kernels/sampling/softmax_host.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/sampling/split_softmax.h"

#include <ATen/cuda/CUDAContext.h>
#include <cstdlib>
#include <cstring>

namespace rtp_llm {
namespace {
bool enabled(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && std::strcmp(value, "1") == 0;
}

bool isSm120() {
    const auto* properties = at::cuda::getCurrentDeviceProperties();
    return properties->major == 12 && properties->minor == 0;
}
}  // namespace

torch::Tensor samplerSoftmaxInplace(const torch::Tensor& logits) {
    const auto batch = logits.size(0);
    if (enabled("RTP_LLM_SAMPLER_SPLIT_SOFTMAX") && logits.is_contiguous()
        && logits.scalar_type() == torch::kFloat32 && logits.dim() == 2 && batch > 0 && batch <= 32
        && logits.size(1) == 248320 && isSm120()) {
        splitSoftmaxInplace(logits);
        return logits;
    }
    auto probs = torch::softmax(logits, -1);
    logits.copy_(probs, true);
    return probs;
}

}  // namespace rtp_llm
