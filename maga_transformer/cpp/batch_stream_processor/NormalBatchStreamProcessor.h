#pragma once

#include "maga_transformer/cpp/batch_stream_processor/BatchStreamProcessor.h"
#include "src/fastertransformer/devices/DeviceBase.h"
#include "src/fastertransformer/th_op/GptInitParameter.h"

namespace rtp_llm {

class NormalBatchStreamProcessor: public BatchStreamProcessor {
public:
    NormalBatchStreamProcessor(const GptInitParameter& params):
        num_layers_(params.num_layers_),
        use_int8_kv_cache_(params.int8_kv_cache_),
        device_(ft::DeviceFactory::getDevice(ft::DeviceType::Cuda)) {}
    absl::Status                   dispatch(const StreamGroups&                  stream_groups,
                                            const std::unique_ptr<MergedOutput>& merge_outputs) const override;
    absl::StatusOr<GptModelInputs> gatherModelInput(const StreamGroups& stream_groups) const override;
    absl::StatusOr<SamplerInputs>  gatherSamplerInput(const StreamGroups&    stream_groups,
                                                      const GptModelOutputs& model_output) const override;

private:
    size_t          num_layers_;
    bool            use_int8_kv_cache_;
    ft::DeviceBase* device_;
};

}  // namespace rtp_llm
