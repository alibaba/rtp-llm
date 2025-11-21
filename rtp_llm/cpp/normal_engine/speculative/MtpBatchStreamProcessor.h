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

    void prepareDecodeDraftModelInput(const StreamGroups& stream_groups, GptModelInputs& model_input);

    void prepareOneStepSpecDecodeModelInput(const StreamGroups& stream_groups, GptModelInputs& model_input);

    void updateDecodeDraftModelInput(GptModelInputs&        model_input,
                                     const GptModelOutputs& model_output,
                                     const torch::Tensor&   draft_token_ids);

    void updatePrefillPostDraftModelInput(GptModelInputs&        model_input,
                                          const GptModelOutputs& model_output,
                                          const SamplerOutput&   sampler_output);

    void updateDecodePostDraftModelInput(GptModelInputs&                              model_input,
                                         const GptModelOutputs&                       model_output,
                                         const speculative::SpeculativeSamplerOutput& speculative_sampler_output,
                                         const size_t                                 batch_size,
                                         torch::Tensor&                               hidden_states_d_t,
                                         size_t&                                      total_accept_len);

    void updateOneStepDraftSamplerOutput(const StreamGroups& stream_groups,
                                         SamplerOutput&      draft_sampler_output,
                                         torch::Tensor&      draft_token_probs_d_t);

    void updateMultiStepDraftSamplerOutput(const StreamGroups&         stream_groups,
                                           SamplerOutput&              draft_sampler_output,
                                           torch::Tensor&              draft_token_ids_d_t,
                                           torch::Tensor&              draft_token_probs_d_t,
                                           std::vector<torch::Tensor>& draft_token_probs_list);

protected:
    void dispatchProposePrefillSingleStream(GenerateStreamPtr         stream,
                                            const MergedOutput&       propose_output,
                                            int                       batch_idx_in,
                                            int                       batch_idx_out,
                                            int                       token_offset,
                                            int                       accept_len,
                                            bool                      return_all_probs,
                                            const rtp_llm::BufferPtr& new_tokens_all) const;

    void setProposeTokensForAllStreams(const StreamGroups&      stream_groups,
                                       const MergedOutput&      draft_prefill_output,
                                       const rtp_llm::BufferPtr new_tokens_all) const;

    int propose_step_;
};
}  // namespace rtp_llm