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

/** Retain sampling products for draft input preparation and final result assembly. */
struct PPSamplingOutput {
    SamplerOutput                         sampler_output;
    speculative::SpeculativeSamplerOutput verify_output;
    std::vector<ErrorInfo>                request_errors;
    int64_t                               sample_input_us = 0;
};

struct PPDraftOutput {
    /** The complete next proposal, including DSpARK Q; result assembly decides which fields to send. */
    SamplerOutput proposals;
    /** Per-request final draft hidden for the MTP/EAGLE P-side handoff; otherwise undefined. */
    torch::Tensor hidden_states;
};

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
    prepareLogitsProcessorsAsync(const PPSamplingPlan&                                 sampling_plan,
                                 const std::unordered_map<RequestId, PPSamplingState>& sampling_states) const;

    void prepareSampling(const PPExecutionPlan&                              plan,
                         std::unordered_map<RequestId, torch::Tensor>&       dspark_q,
                         const torch::Tensor&                                d2t_map,
                         std::unordered_map<RequestId, PPSamplingState>&     sampling_states,
                         std::vector<ErrorInfo>&                              request_errors,
                         SamplerOutput&                                      draft_sampler_output,
                         speculative::SpeculativeSamplingParams&             verify_params,
                         GrammarPrefetchContext*                             grammar_prefetch = nullptr);

    SamplerInputs gatherSamplerInputs(const PPSamplingPlan&                                 sampling_plan,
                                      const PPOutputConfig&                                 output_config,
                                      const torch::Tensor&                                  logits,
                                      const std::unordered_map<RequestId, PPSamplingState>& sampling_states,
                                      bool                                                  score_batch   = false,
                                      size_t                                                propose_step  = 0,
                                      const torch::Tensor&                                  verify_tokens = {}) const;

    /** Called after draft submission; completes target/draft outputs and their CPU copies before sending. */
    void fillExecutionResult(const PPExecutionPlan&  plan,
                             const GptModelOutputs&  model_output,
                             const PPSamplingOutput& sampling_output,
                             const PPDraftOutput&    draft_output,
                             PPExecutionResult&     result) const;

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

    mtp::DSparkRoundState buildDSparkRoundState(const PPExecutionPlan&  plan,
                                                const PPSamplingOutput& sampling_output,
                                                TensorHolder&           host_holder) const;

    void prepareDSparkProposeModelInput(const mtp::DSparkRoundState& round_state,
                                        GptModelInputs&              draft_input,
                                        TensorHolder&                host_holder);

    torch::Tensor prepareDSparkDraftTemperature(const PPSamplingPlan& sampling_plan,
                                                torch::Device         device,
                                                TensorHolder&         host_holder) const;

    /** Validate received results before metrics or stream updates consume them. */
    void validateExecutionResult(const StreamGroups& stream_groups, const PPExecutionResult& result) const;

    void dispatchExecutionResult(const StreamGroups& stream_groups, const PPExecutionResult& result) const;

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

    void fillSamplerInputs(SamplerInputs&                                        sampler_inputs,
                           const PPSamplingPlan&                                 sampling_plan,
                           const std::unordered_map<RequestId, PPSamplingState>& sampling_states,
                           bool                                                  score_batch,
                           size_t                                                propose_step,
                           const torch::Tensor&                                  verify_tokens) const;

    static void collectRequestErrors(const PPSamplingPlan&                        sampling_plan,
                                     const std::vector<std::optional<ErrorInfo>>& processor_errors,
                                     const torch::Tensor&                         success_cpu,
                                     std::vector<ErrorInfo>&                      request_errors);

    void fillOptionalOutputs(const PPExecutionPlan& plan,
                             const GptModelOutputs& model_output,
                             const SamplerOutput&   sampler_output,
                             PPExecutionResult&     result) const;

    void dispatchPrefillOnlyExecutionResult(const StreamGroups& stream_groups, const PPExecutionResult& result) const;

    void dispatchSpeculativeExecutionResult(const StreamGroups& stream_groups, const PPExecutionResult& result) const;

    void dispatchNormalExecutionResult(const StreamGroups& stream_groups, const PPExecutionResult& result) const;

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
