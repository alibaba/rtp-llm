#pragma once

#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/models_py/bindings/core/TensorHolder.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include <array>
#if USING_CUDA
#include <ATen/cuda/CUDAEvent.h>
#endif

namespace rtp_llm {
// Sampler would split logits into appropriate groups (mostly, based on beam size)
// and calls device sampling apis (greedy, beam search, etc) for each group
class Sampler {
public:
    Sampler(const SamplerInitParams& params);
    ~Sampler() {};

    virtual SamplerOutput forward(const SamplerInputs& inputs);

    bool optimizedSamplingEnabled() const {
        return use_optimized_sampling_;
    }

private:
    void preprocessLogits(const SamplerInputs& inputs);

#if USING_CUDA
    struct RngSlot {
        torch::Tensor host;
        torch::Tensor device;
        at::cuda::CUDAEvent ready;
        bool disabled = false;
    };
    // Like buffer_holder_, these slots are per-instance and forward is non-reentrant.
    std::array<RngSlot, 3> rng_slots_;
#endif
    std::shared_ptr<DeferredSamplingRngPool> deferred_rng_pool_;
    TensorHolder            buffer_holder_;
    cuda_graph::GraphStream copy_stream_;
    int                     force_topk_ = 0;
    bool                    use_optimized_sampling_ = false;
};

}  // namespace rtp_llm
