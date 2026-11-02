#include "rtp_llm/models_py/bindings/common/kernels/sampling_temperature.h"
#include "rtp_llm/models_py/bindings/common/kernels/sampling_penalty_kernels.h"

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

void applySamplerTemperature(const torch::Tensor& logits, const torch::Tensor& temperatures, cudaStream_t stream) {
    const auto batch = logits.size(0);
    const auto vocab = logits.size(1);
    if (enabled("RTP_LLM_SAMPLER_TILED_TEMPERATURE") && batch > 0 && batch <= 32 && vocab == 248320
        && logits.is_contiguous() && isSm120()) {
        invokeTiledTemperaturePenalty(logits.data_ptr<float>(), temperatures.data_ptr<float>(), batch, vocab, stream);
    } else {
        invokeBatchApplyTemperaturePenalty(logits.data_ptr<float>(), (float*)nullptr, temperatures.data_ptr<float>(),
                                           batch, vocab, vocab, stream);
    }
}

}  // namespace rtp_llm
