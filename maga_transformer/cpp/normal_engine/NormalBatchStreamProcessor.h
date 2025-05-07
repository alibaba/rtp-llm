#pragma once

#include "maga_transformer/cpp/devices/DeviceBase.h"
#include "maga_transformer/cpp/th_op/GptInitParameter.h"
#include "maga_transformer/cpp/dataclass/MergedQuery.h"
#include "maga_transformer/cpp/stream/StreamGroups.h"
#include "absl/status/statusor.h"
#include "absl/status/status.h"

namespace rtp_llm {

class NormalBatchStreamProcessor {
public:
    NormalBatchStreamProcessor(const rtp_llm::GptInitParameter& params,
                               const CacheConfig& cache_config, bool warm_up):
        num_layers_(params.num_layers_),
        vocab_size_(params.vocab_size_),
        input_vocab_size_(params.input_vocab_size_),
        use_int8_kv_cache_(params.kv_cache_data_type_ == rtp_llm::DataType::TYPE_INT8),
        has_positional_encoding_(params.has_positional_encoding_),
        is_multimodal_(params.is_multimodal_),
        mm_position_ids_style_((PositionIdsStyle)params.mm_position_ids_style_),
        position_id_len_factor_(params.position_id_len_factor_),
        pd_separation_(params.pd_separation_),
        k_block_size_(cache_config.k_block_stride),
        v_block_size_(cache_config.v_block_stride),
        scale_block_size_(cache_config.kv_scale_block_stride),
        seq_size_per_block_(cache_config.seq_size_per_block),
        warm_up_(warm_up),
        device_(rtp_llm::DeviceFactory::getDefaultDevice()) {}
    virtual absl::Status                   dispatch(const StreamGroups& stream_groups,
                                                    const MergedOutput& merge_outputs) const;
    virtual absl::StatusOr<GptModelInputs> gatherModelInput(const StreamGroups& stream_groups) const;
    virtual absl::StatusOr<SamplerInputs>  gatherSamplerInput(const StreamGroups&    stream_groups,
                                                              const GptModelInputs&  model_inputs,
                                                              const GptModelOutputs& model_output) const;


protected:
    SamplerInputs allocateSamplerInputs(const StreamGroups& stream_groups, size_t total_batch_size, const rtp_llm::BufferPtr& sequence_length) const;
    void    setCommonSamplerInputs(SamplerInputs& sampler_inputs, std::list<GenerateStreamPtr>& all_streams, bool score_batch = false) const;
    void    setThinkModeLogitsProcessorInputs(SamplerInputs& sampler_inputs, std::list<GenerateStreamPtr>& all_streams, bool score_batch = false) const;

protected:
    size_t           num_layers_;
    size_t           vocab_size_;
    size_t           input_vocab_size_;
    bool             use_int8_kv_cache_;
    bool             has_positional_encoding_;
    bool             is_multimodal_;
    PositionIdsStyle mm_position_ids_style_;
    size_t           position_id_len_factor_;
    bool             pd_separation_;
    // size_t           block_size_;
    size_t           k_block_size_;
    size_t           v_block_size_;
    size_t           scale_block_size_;
    size_t           seq_size_per_block_;
    bool             warm_up_;
    rtp_llm::DeviceBase*  device_;
};

}  // namespace rtp_llm
