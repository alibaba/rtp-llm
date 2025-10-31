#pragma once

#include "rtp_llm/cpp/normal_engine/NormalBatchStreamProcessor.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.h"

namespace rtp_llm {

class MtpBatchStreamProcessor: public NormalBatchStreamProcessor {
public:
    MtpBatchStreamProcessor(const rtp_llm::GptInitParameter& params, const CacheConfig& cache_config, bool warm_up):
        NormalBatchStreamProcessor(params, cache_config, warm_up), propose_step_(params.sp_config.gen_num_per_cycle) {}

    absl::Status dispatchPrefill(const StreamGroups& stream_groups,
                                 const MergedOutput& prefill_output,
                                 const MergedOutput& propose_output) const;

    absl::Status dispatchDecode(const StreamGroups&                          stream_groups,
                                const speculative::SpeculativeSamplerOutput& spec_decode_output,
                                const MergedOutput&                          draft_prefill_output) const;

    absl::StatusOr<GptModelInputs> gatherDecodeModelInput(const StreamGroups& stream_groups) const;

    absl::StatusOr<SamplerInputs> gatherSpecSamplerInput(const StreamGroups&    stream_groups,
                                                         const GptModelInputs&  model_inputs,
                                                         const GptModelOutputs& model_output) const;

protected:
    void dispatchProposePrefillSingleStream(GenerateStreamPtr         stream,
                                            const MergedOutput&       propose_output,
                                            int                       batch_idx_in,
                                            int                       batch_idx_out,
                                            int                       token_offset,
                                            bool                      return_all_probs,
                                            const rtp_llm::BufferPtr& new_tokens_all) const;

    void setProposeTokensForAllStreams(const StreamGroups&      stream_groups,
                                       const MergedOutput&      draft_prefill_output,
                                       const rtp_llm::BufferPtr new_tokens_all) const;

    int propose_step_;
};
}  // namespace rtp_llm