#include "rtp_llm/models_py/bindings/cuda/kernels/sampling/sampler_dispatch.h"

#include <ATen/cuda/CUDAContext.h>
#include <cstdlib>
#include <cstring>

#include "rtp_llm/models_py/bindings/common/kernels/sampling_penalty_kernels.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/sampling/split_softmax.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/speculative_sampling/sampling.h"

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

cudaError_t dispatchRejectionSampling(const RejectionSamplingInputs& p, cudaStream_t stream) {
    float* draft_probs = p.point_mass ? nullptr : p.draft_probs.data_ptr<float>();
    if (enabled("RTP_LLM_SPLIT_REJECTION") && p.batch > 0 && p.batch <= 8 && p.steps == 3
        && p.vocab == 248320 && isSm120()) {
        const int tiles = (p.vocab + 4095) / 4096;
        auto sums = torch::empty({p.batch, tiles}, p.target_probs.options());
        auto positions = torch::empty({p.batch}, p.draft_ids.options());
        return invokeSplitRejectionSampling(draft_probs, p.draft_ids.data_ptr<int32_t>(),
                                            p.uniforms.data_ptr<float>(), p.target_probs.data_ptr<float>(),
                                            p.target_ids.data_ptr<int32_t>(), p.target_stride,
                                            p.output_ids.data_ptr<int32_t>(), p.accepted.data_ptr<int32_t>(),
                                            p.do_sample.data_ptr<bool>(), p.batch, p.steps, p.vocab, p.point_mass,
                                            sums.data_ptr<float>(), positions.data_ptr<int32_t>(), stream);
    }
    return invokeRejectionSampling(draft_probs, p.draft_ids.data_ptr<int32_t>(), p.uniforms.data_ptr<float>(),
                                   p.target_probs.data_ptr<float>(), p.target_ids.data_ptr<int32_t>(), p.target_stride,
                                   p.output_ids.data_ptr<int32_t>(), p.accepted.data_ptr<int32_t>(),
                                   p.do_sample.data_ptr<bool>(), p.batch, p.steps, p.vocab, stream, p.point_mass);
}
}  // namespace rtp_llm
