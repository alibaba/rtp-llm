#pragma once

#include "rtp_llm/cpp/normal_engine/NormalBatchStreamProcessor.h"

namespace rtp_llm {

class ContextParallelBatchStreamProcessor: public NormalBatchStreamProcessor {
public:
    ContextParallelBatchStreamProcessor(const rtp_llm::GptInitParameter& params,
                                        const CacheConfig&               cache_config,
                                        bool                             warm_up):
        NormalBatchStreamProcessor(params, cache_config, warm_up) {}

    inline const int getContextParallelSize() {
        return device_->getDeviceProperties().cp_size;
    }

    absl::Status dispatch(const StreamGroups& stream_groups, const MergedOutput& merge_outputs) const override;

    absl::StatusOr<GptModelInputs> gatherModelInput(const StreamGroups& stream_groups) const override;

    absl::StatusOr<SamplerInputs> gatherSamplerInput(const StreamGroups&    stream_groups,
                                                     const GptModelInputs&  model_inputs,
                                                     const GptModelOutputs& model_output) const override;
};

}  // namespace rtp_llm
