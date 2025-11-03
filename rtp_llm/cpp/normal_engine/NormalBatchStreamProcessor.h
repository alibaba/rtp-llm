#pragma once

#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/engine_base/stream/StreamGroups.h"
#include "absl/status/statusor.h"
#include "absl/status/status.h"

namespace rtp_llm {

class NormalBatchStreamProcessor {
public:
    NormalBatchStreamProcessor(const ModelConfig& model_config,
                               const MMModelConfig& mm_model_config,
                               const PDSepConfig& pd_sep_config,
                               const ProfilingDebugLoggingConfig& profiling_debug_logging_config,
                               const CacheConfig& cache_config,
                               bool warm_up):
        num_layers_(model_config.num_layers_),
        vocab_size_(model_config.vocab_size_),
        input_vocab_size_(model_config.input_vocab_size_),
        use_int8_kv_cache_(model_config.kv_cache_data_type_ == rtp_llm::DataType::TYPE_INT8),
        has_positional_encoding_(model_config.has_positional_encoding_),
        is_multimodal_(mm_model_config.is_multimodal_),
        mm_position_ids_style_((PositionIdsStyle)mm_model_config.mm_position_ids_style_),
        position_id_len_factor_(mm_model_config.position_id_len_factor_),
        role_type_(pd_sep_config.role_type),
        decode_entrance_(pd_sep_config.decode_entrance),
        k_block_size_(cache_config.k_block_stride),
        v_block_size_(cache_config.v_block_stride),
        scale_block_size_(cache_config.kv_scale_block_stride),
        seq_size_per_block_(cache_config.seq_size_per_block),
        warm_up_(warm_up),
        enable_detail_log_(profiling_debug_logging_config.enable_detail_log),
        device_(rtp_llm::DeviceFactory::getDefaultDevice()) {}
    virtual absl::Status dispatch(const StreamGroups& stream_groups, const MergedOutput& merge_outputs) const;
    virtual absl::StatusOr<GptModelInputs> gatherModelInput(const StreamGroups& stream_groups) const;
    virtual absl::StatusOr<SamplerInputs>  gatherSamplerInput(const StreamGroups&    stream_groups,
                                                              const GptModelInputs&  model_inputs,
                                                              const GptModelOutputs& model_output) const;

protected:
    SamplerInputs allocateSamplerInputs(const StreamGroups&       stream_groups,
                                        size_t                    total_batch_size_in,
                                        size_t                    total_batch_size_out,
                                        const rtp_llm::BufferPtr& sequence_length) const;
    void          setCommonSamplerInputs(SamplerInputs&                sampler_inputs,
                                         std::list<GenerateStreamPtr>& all_streams,
                                         bool                          score_batch = false) const;
    void          setLogitsProcessorInputs(SamplerInputs&                sampler_inputs,
                                           std::list<GenerateStreamPtr>& all_streams,
                                           bool                          score_batch = false) const;

protected:
    size_t           num_layers_;
    size_t           vocab_size_;
    size_t           input_vocab_size_;
    bool             use_int8_kv_cache_;
    bool             has_positional_encoding_;
    bool             is_multimodal_;
    PositionIdsStyle mm_position_ids_style_;
    size_t           position_id_len_factor_;
    RoleType         role_type_;
    bool             decode_entrance_;
    // size_t           block_size_;
    size_t k_block_size_;
    size_t v_block_size_;
    size_t scale_block_size_;
    size_t seq_size_per_block_;
    bool   warm_up_;
    bool   enable_detail_log_;

    rtp_llm::DeviceBase* device_;
};

}  // namespace rtp_llm
