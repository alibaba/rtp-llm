#pragma once

#include <list>
#include <optional>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/normal_engine/NormalBatchStreamProcessor.h"
#include "rtp_llm/cpp/normal_engine/pipeline/PPTypes.h"
#include "rtp_llm/cpp/normal_engine/speculative/MtpCompute.h"
#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.h"

namespace rtp_llm {

struct GrammarPrefetchContext;
struct PPSamplingState;

class PPBatchStreamProcessor: public NormalBatchStreamProcessor {
public:
    PPBatchStreamProcessor(const ModelConfig&                 model_config,
                           const PDSepConfig&                 pd_sep_config,
                           const ProfilingDebugLoggingConfig& profiling_debug_logging_config,
                           const CacheConfig&                 cache_config,
                           const SpeculativeExecutionConfig&  sp_config,
                           bool                               warm_up);

    absl::StatusOr<PPExecutionPlan> gatherExecutionPlan(const StreamGroups&                     stream_groups,
                                                        const std::vector<RequestId>&           finished_request_ids,
                                                        bool                                    shutdown,
                                                        std::unordered_map<RequestId, int32_t>& token_history_lengths,
                                                        TensorHolder&                           host_holder) const;

    GrammarPrefetchContext
    prepareLogitsProcessorsAsync(const PPSamplingPlan&                                sampling_plan,
                                 const std::unordered_map<RequestId, PPSamplingState>& sampling_states) const;

    void prepareSampling(const PPExecutionPlan&                              plan,
                         const std::unordered_map<RequestId, torch::Tensor>& dspark_q,
                         const torch::Tensor&                                d2t_map,
                         std::unordered_map<RequestId, PPSamplingState>&     sampling_states,
                         PPExecutionResult&                                  result,
                         SamplerOutput&                                      draft_sampler_output,
                         speculative::SpeculativeSamplingParams&             verify_params,
                         GrammarPrefetchContext*                             grammar_prefetch = nullptr);

    SamplerInputs gatherSamplerInputs(const PPSamplingPlan&                                sampling_plan,
                                      const PPOutputConfig&                                output_config,
                                      const torch::Tensor&                                 logits,
                                      const std::unordered_map<RequestId, PPSamplingState>& sampling_states,
                                      bool                                                 score_batch   = false,
                                      size_t                                               propose_step  = 0,
                                      const torch::Tensor&                                 verify_tokens = {}) const;

    void fillExecutionResult(const PPExecutionPlan& plan,
                             const GptModelOutputs& model_output,
                             const SamplerOutput&   sampler_output,
                             PPExecutionResult&     result) const;

    void fillExecutionResult(const PPExecutionPlan&                       plan,
                             const GptModelOutputs&                       model_output,
                             const SamplerOutput&                         target_sampler_output,
                             const speculative::SpeculativeSamplerOutput& sp_output,
                             PPExecutionResult&                           result) const;

    void prepareDraftPrefillAfterTargetPrefill(GptModelInputs&      draft_prefill_input,
                                               const torch::Tensor& target_hidden_states,
                                               const torch::Tensor& sampled_token_ids,
                                               const torch::Tensor& next_position_ids,
                                               TensorHolder&        host_holder) const;

    void prepareDraftPrefillAfterVerify(GptModelInputs&       draft_prefill_input,
                                        const torch::Tensor&  target_hidden_states,
                                        const torch::Tensor&  accepted_token_ids,
                                        const torch::Tensor&  accepted_lengths,
                                        mtp::DraftInputLayout layout,
                                        torch::Tensor&        hidden_states_d_t,
                                        TensorHolder&         host_holder) const;

    void
    prepareDSparkCommitInput(GptModelInputs& commit_input, const torch::Tensor& target_features, bool cp_enabled) const;

    mtp::DSparkRoundState buildDSparkRoundState(const GptModelInputs& target_input,
                                                const torch::Tensor&  draft_next_position_ids,
                                                bool                  is_decode,
                                                const torch::Tensor&  new_token_ids,
                                                const torch::Tensor&  new_token_lengths,
                                                TensorHolder&         host_holder) const;

    void prepareDSparkProposeModelInput(const mtp::DSparkRoundState& round_state,
                                        GptModelInputs&              draft_input,
                                        TensorHolder&                host_holder);

    torch::Tensor prepareDSparkDraftTemperature(const PPSamplingPlan& sampling_plan,
                                                torch::Device         device,
                                                TensorHolder&         host_holder) const;

    absl::Status dispatchExecutionResult(const StreamGroups& stream_groups, const PPExecutionResult& result) const;

private:
    void fillTargetVerifyModelInput(const std::list<GenerateStreamPtr>& streams, GptModelInputs& model_input) const;

    std::optional<ErrorInfo> initLogitsProcessors(std::vector<BaseLogitsProcessorPtr>&     processors,
                                                  const PPSamplingPlan&                    sampling_plan,
                                                  const std::vector<std::vector<int32_t>>& token_history,
                                                  int64_t                                  stream_idx,
                                                  int64_t                                  sequence_offset,
                                                  GrammarPrefetchContext*                  grammar_prefetch) const;

    SamplerInputs allocateSamplerInputs(const PPSamplingPlan& sampling_plan,
                                       const PPOutputConfig& output_config,
                                       size_t                total_batch_size,
                                       size_t                propose_step) const;

    void fillSamplerInputs(SamplerInputs&                                       sampler_inputs,
                           const PPSamplingPlan&                                sampling_plan,
                           const std::unordered_map<RequestId, PPSamplingState>& sampling_states,
                           bool                                                 score_batch,
                           size_t                                               propose_step,
                           const torch::Tensor&                                 verify_tokens) const;

    void fillOptionalOutputs(const PPExecutionPlan& plan,
                             const GptModelOutputs& model_output,
                             const SamplerOutput&   sampler_output,
                             PPExecutionResult&     result) const;

    void validateExecutionResult(const StreamGroups& stream_groups, const PPExecutionResult& result) const;

    absl::Status dispatchPrefillOnlyExecutionResult(const StreamGroups& stream_groups,
                                                    const PPExecutionResult& result) const;

    absl::Status dispatchSpeculativeExecutionResult(const StreamGroups& stream_groups,
                                                   const PPExecutionResult& result) const;

    absl::Status dispatchNormalExecutionResult(const StreamGroups& stream_groups,
                                               const PPExecutionResult& result) const;

    void dispatchNormalSingleStream(const GenerateStreamPtr& stream,
                                    const PPExecutionResult& result,
                                    int64_t                  stream_idx,
                                    int64_t                  batch_idx,
                                    int64_t                  stream_batch_size) const;

private:
    const bool                     sp_enabled_;
    const bool                     is_dspark_;
    const int64_t                  propose_step_;
    const int32_t                  dspark_mask_token_id_;
    const bool                     dspark_sample_from_anchor_;
    mtp::DSparkProposeInputBuffers dspark_propose_input_buffers_;
    torch::Tensor                  dspark_verify_q_buffer_;
    const std::vector<int64_t>     output_vocab_ids_;
    const int64_t                  processor_eos_token_id_;
};

}  // namespace rtp_llm
