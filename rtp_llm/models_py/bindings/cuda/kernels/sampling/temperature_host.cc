#include "rtp_llm/models_py/bindings/cuda/kernels/sampling/temperature_host.h"
#include "rtp_llm/models_py/bindings/common/kernels/sampling_penalty_kernels.h"

namespace rtp_llm {
namespace {
bool canUseTiledTemperature(const torch::Tensor& logits, const torch::Tensor& temperatures) {
    return logits.is_cuda() && logits.is_contiguous() && logits.scalar_type() == torch::kFloat32 && logits.dim() == 2
           && logits.size(0) > 0 && logits.size(1) > 0 && temperatures.is_cuda() && temperatures.is_contiguous()
           && temperatures.scalar_type() == torch::kFloat32 && temperatures.numel() >= logits.size(0);
}
}  // namespace

void applySamplerTemperature(const torch::Tensor& logits, float scalar_temperature, cudaStream_t stream) {
    const auto batch = logits.size(0);
    const auto vocab = logits.size(1);
    if (logits.is_cuda() && logits.is_contiguous() && logits.scalar_type() == torch::kFloat32 && logits.dim() == 2
        && batch > 0 && vocab > 0) {
        invokeTiledTemperaturePenalty(logits.data_ptr<float>(), scalar_temperature, batch, vocab, stream);
    } else {
        auto temperatures = torch::full(
            {batch}, scalar_temperature, torch::TensorOptions().dtype(torch::kFloat32).device(logits.device()));
        invokeBatchApplyTemperaturePenalty(
            logits.data_ptr<float>(), (float*)nullptr, temperatures.data_ptr<float>(), batch, vocab, vocab, stream);
    }
}

void applySamplerTemperature(const torch::Tensor& logits, const torch::Tensor& temperatures, cudaStream_t stream) {
    const auto batch = logits.size(0);
    const auto vocab = logits.size(1);
    if (canUseTiledTemperature(logits, temperatures)) {
        invokeTiledTemperaturePenalty(logits.data_ptr<float>(), temperatures.data_ptr<float>(), batch, vocab, stream);
    } else {
        invokeBatchApplyTemperaturePenalty(
            logits.data_ptr<float>(), (float*)nullptr, temperatures.data_ptr<float>(), batch, vocab, vocab, stream);
    }
}

}  // namespace rtp_llm
