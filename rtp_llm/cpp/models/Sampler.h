#pragma once

#include <optional>
#include <vector>

#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include <memory>

namespace rtp_llm {
// Groups requests by beam count and owns parameter preparation, row selection and
// output restoration. forward() is single-threaded and non-reentrant per instance.
class Sampler {
public:
    Sampler(const SamplerInitParams& params);
    virtual ~Sampler() = default;

    virtual SamplerOutput forward(const SamplerInputs& inputs);

private:
    std::vector<std::optional<ErrorInfo>> preprocessLogits(const SamplerInputs& inputs);
    bool                                  fixed_max_batch_size_ = false;
    size_t                                max_batch_size_       = 0;
};

}  // namespace rtp_llm
