#include "maga_transformer/cpp/models/Sampler.h"

using namespace std;

namespace rtp_llm {

Sampler::Sampler(const SamplerInitParams& params)
    : device_(params.device)
    {};

SamplerOutput Sampler::forward(const SamplerInputs& inputs) {
    return SamplerOutput();
}

} // namespace rtp_llm
