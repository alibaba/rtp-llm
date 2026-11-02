#include "rtp_llm/models_py/bindings/common/kernels/sampling_temperature.h"
#include "rtp_llm/models_py/bindings/common/kernels/sampling_penalty_kernels.h"
#include "rtp_llm/models_py/bindings/common/kernels/sampling_host_utils.h"

namespace rtp_llm {
namespace {
bool canUseTiledTemperature(const torch::Tensor& logits) {
    return sampler::optimizationEnabled("RTP_LLM_SAMPLER_TILED_TEMPERATURE") && logits.size(0) > 0
           && logits.size(0) <= 32 && logits.size(1) == 248320 && logits.is_contiguous() && sampler::isSm120();
}
}  // namespace

void applySamplerTemperature(const torch::Tensor& logits, const torch::Tensor& temperatures, cudaStream_t stream) {
    const auto batch = logits.size(0);
    const auto vocab = logits.size(1);
    if (canUseTiledTemperature(logits)) {
        invokeTiledTemperaturePenalty(logits.data_ptr<float>(), temperatures.data_ptr<float>(), batch, vocab, stream);
    } else {
        invokeBatchApplyTemperaturePenalty(logits.data_ptr<float>(), (float*)nullptr, temperatures.data_ptr<float>(),
                                           batch, vocab, vocab, stream);
    }
}

}  // namespace rtp_llm
