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
    NormalBatchStreamProcessor(const ModelConfig&                 model_config,
                               const PDSepConfig&                 pd_sep_config,
                               const ProfilingDebugLoggingConfig& profiling_debug_logging_config,
                               const CacheConfig&                 cache_config,
                               bool                               warm_up):
        num_layers_(model_config.num_layers),
        vocab_size_(model_config.vocab_size),
        input_vocab_size_(model_config.input_vocab_size),
        use_int8_kv_cache_(model_config.attn_config.kv_cache_dtype == rtp_llm::KvCacheDataType::INT8),
        has_positional_encoding_(model_config.has_positional_encoding),
        is_multimodal_(model_config.mm_model_config.is_multimodal),
        mm_position_ids_style_((PositionIdsStyle)model_config.mm_model_config.mm_position_ids_style),
        position_id_len_factor_(model_config.attn_config.rope_config.index_factor),
        role_type_(pd_sep_config.role_type),
        decode_entrance_(pd_sep_config.decode_entrance),
        block_stride_bytes_(cache_config.kv_block_stride_bytes),
        scale_stride_bytes_(cache_config.kv_scale_stride_bytes),
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
                                        const rtp_llm::BufferPtr& sequence_length,
                                        size_t                    propose_step = 0) const;
    void          setCommonSamplerInputs(SamplerInputs&                sampler_inputs,
                                         std::list<GenerateStreamPtr>& all_streams,
                                         bool                          score_batch  = false,
                                         size_t                        propose_step = 0) const;
    void          setLogitsProcessorInputs(SamplerInputs&                sampler_inputs,
                                           std::list<GenerateStreamPtr>& all_streams,
                                           bool                          score_batch = false) const;

    void dispatchSingleStream(GenerateStreamPtr   stream,
                              const MergedOutput& merge_outputs,
                              int                 batch_idx_in,
                              int                 batch_idx_out,
                              int                 token_offset,
                              bool                return_all_probs,
                              const BufferPtr&    new_tokens_all) const;

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
    size_t           block_stride_bytes_;
    size_t           scale_stride_bytes_;
    size_t           seq_size_per_block_;
    bool             warm_up_;
    bool             enable_detail_log_;

    rtp_llm::DeviceBase* device_;
};

}  // namespace rtp_llm
