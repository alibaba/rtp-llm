#pragma once

#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include <array>

namespace rtp_llm {
// Sampler would split logits into appropriate groups (mostly, based on beam size)
// and calls device sampling apis (greedy, beam search, etc) for each group
class Sampler {
public:
    Sampler(const SamplerInitParams& params);
    ~Sampler() {};

    virtual SamplerOutput forward(const SamplerInputs& inputs);

private:
    void                   preprocessLogits(const SamplerInputs& inputs);
    void                   ensureGreedySamplingBuffers(size_t batch_size);
    void                   allocateGreedySamplingBuffers(size_t max_batch_size);
    GreedySamplingBuffers& nextGreedySamplingBuffers(size_t batch_size);

    // Rotate across forward calls so async host-to-device copies do not read buffers that
    // the next decode round has already overwritten.
    static constexpr size_t kGreedySamplingBufferSlots = 3;

    bool                                                          fixed_max_batch_size_         = false;
    size_t                                                        max_batch_size_               = 0;
    size_t                                                        greedy_sampling_buffer_index_ = 0;
    std::array<GreedySamplingBuffers, kGreedySamplingBufferSlots> greedy_sampling_buffers_;
};

}  // namespace rtp_llm
