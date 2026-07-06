#pragma once

#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include <array>
#include <memory>

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
    void                   waitGreedySamplingBufferEvents();
    GreedySamplingBuffers& nextGreedySamplingBuffers(size_t batch_size);
    void                   markGreedySamplingBufferReady();

    struct GreedySamplingBufferSlot {
        GreedySamplingBuffers         buffers;
        std::shared_ptr<torch::Event> ready_event;
    };

    // Rotate across forward calls and wait on a per-slot event before reuse so async host-to-device
    // copies have consumed the pinned host buffers before the CPU overwrites them.
    static constexpr size_t kGreedySamplingBufferSlots = 3;

    bool                                                             fixed_max_batch_size_         = false;
    size_t                                                           max_batch_size_               = 0;
    size_t                                                           greedy_sampling_buffer_index_ = 0;
    GreedySamplingBufferSlot*                                        current_greedy_sampling_slot_ = nullptr;
    std::array<GreedySamplingBufferSlot, kGreedySamplingBufferSlots> greedy_sampling_buffer_slots_;
};

}  // namespace rtp_llm
