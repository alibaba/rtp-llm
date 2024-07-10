#pragma once

#include "src/fastertransformer/devices/DeviceBase.h"
#include "src/fastertransformer/th_op/GptInitParameter.h"
#include "maga_transformer/cpp/dataclass/MergedQuery.h"
#include "maga_transformer/cpp/dataclass/StreamGroups.h"
#include "absl/status/statusor.h"
#include "absl/status/status.h"

namespace rtp_llm {

struct MaskParams {
public:
    const ft::Buffer& input_lengths;
    const ft::Buffer& prefix_lengths;
    ft::DataType      dtype;
    bool              is_causal;
    ft::DeviceBase*   device;
};

class NormalBatchStreamProcessor {
public:
    NormalBatchStreamProcessor(const ft::GptInitParameter& params):
        num_layers_(params.num_layers_),
        use_int8_kv_cache_(params.int8_kv_cache_),
        has_positional_encoding_(params.has_positional_encoding_),
        is_multimodal_(params.is_multimodal_),
        cal_mm_tokens_in_rotary_emb_(params.cal_mm_tokens_in_rotary_emb_),
        device_(ft::DeviceFactory::getDefaultDevice()) {}
    absl::Status                   dispatch(const StreamGroups&                  stream_groups,
                                            const SamplerInputs&                 sampler_inputs,
                                            const std::unique_ptr<MergedOutput>& merge_outputs) const;
    absl::StatusOr<GptModelInputs> gatherModelInput(const StreamGroups& stream_groups) const;
    absl::StatusOr<SamplerInputs>  gatherSamplerInput(const StreamGroups&    stream_groups,
                                                      const GptModelInputs&  model_inputs,
                                                      const GptModelOutputs& model_output) const;

    static ft::BufferPtr createAttentionMask(const MaskParams& params);

private:
    size_t          num_layers_;
    bool            use_int8_kv_cache_;
    bool            has_positional_encoding_;
    bool            is_multimodal_;
    bool            cal_mm_tokens_in_rotary_emb_;
    ft::DeviceBase* device_;
};

}  // namespace rtp_llm
