#pragma once

#include "rtp_llm/cpp/normal_engine/NormalBatchStreamProcessor.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsVerifyRunner.h"
#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.h"

namespace rtp_llm {

class MtpBatchStreamProcessor: public NormalBatchStreamProcessor {
public:
    MtpBatchStreamProcessor(const ModelConfig&                 model_config,
                            const PDSepConfig&                 pd_sep_config,
                            const ProfilingDebugLoggingConfig& profiling_debug_logging_config,
                            const CacheConfig&                 cache_config,
                            const SpeculativeExecutionConfig&  sp_config,
                            bool                               warm_up):
        NormalBatchStreamProcessor(model_config, pd_sep_config, profiling_debug_logging_config, cache_config, warm_up),
        propose_step_(sp_config.gen_num_per_cycle),
        vocab_size_(model_config.vocab_size),
        is_dspark_(sp_config.type == SP_TYPE_DSPARK),
        dspark_mask_token_id_(static_cast<int32_t>(sp_config.sp_dspark_mask_token_id)),
        dspark_sample_from_anchor_(sp_config.sp_dspark_sample_from_anchor) {}

    absl::Status dispatchPrefill(const StreamGroups& stream_groups,
                                 const MergedOutput& prefill_output,
                                 const MergedOutput& propose_output) const;

    // CP slices the draft forward output per rank, so the per-request MTP last
    // hidden rows must be handed in explicitly instead of read off
    // propose_output.model_output.all_hidden_states.
    absl::Status dispatchPrefill(const StreamGroups&  stream_groups,
                                 const MergedOutput&  prefill_output,
                                 const MergedOutput&  propose_output,
                                 const torch::Tensor& draft_last_hidden_states) const;

    absl::Status dispatchDecode(const StreamGroups&                          stream_groups,
                                const speculative::SpeculativeSamplerOutput& spec_decode_output,
                                const MergedOutput&                          draft_prefill_output) const;

    absl::StatusOr<GptModelInputs> gatherDecodeModelInput(const StreamGroups& stream_groups,
                                                          TensorHolder&       host_holder) const;

    // Produce the next round's immutable page table by applying the exact
    // linear-cache permutation used by GenerateStream::specUpdate.  FULL/SWA
    // groups are copied unchanged.  Keeping this operation tensor-native lets
    // proposal/verify overlap host bookkeeping without a per-round D2H join.
    static torch::Tensor advanceLinearCacheBlockTable(const torch::Tensor&               current_table,
                                                      const torch::Tensor&               previous_seq_lengths,
                                                      const torch::Tensor&               accept_lengths,
                                                      const std::vector<CacheGroupType>& group_types,
                                                      int64_t                            seq_size_per_block);

    absl::StatusOr<SamplerInputs> gatherSpecSamplerInput(const StreamGroups&                         stream_groups,
                                                         const GptModelOutputs&                      model_output,
                                                         const SpecLogitsVerifyRunner::LaunchResult& spec_logits_result,
                                                         const torch::Tensor& draft_token_ids) const;

    void prepareDecodeDraftModelInput(const StreamGroups& stream_groups,
                                      GptModelInputs&     model_input,
                                      TensorHolder&       host_holder);

    void prepareOneStepSpecDecodeModelInput(const StreamGroups& stream_groups,
                                            GptModelInputs&     model_input,
                                            TensorHolder&       host_holder);

    // Device-state target-verify gather. Returns true only when every stream
    // has CUDA accept_len/tokens/next_seq_len/propose_tokens; otherwise leaves
    // model_input untouched so the caller can use the legacy mixed path.
    bool gatherMtpDecodeModelInputFromDeviceState(const StreamGroups& stream_groups,
                                                  GptModelInputs&     model_input,
                                                  TensorHolder&       host_holder) const;

    void expandTargetVerifyPositionIds(const StreamGroups& stream_groups, GptModelInputs& model_input) const;

    void updateDecodeDraftModelInput(GptModelInputs&        model_input,
                                     const GptModelOutputs& model_output,
                                     const torch::Tensor&   draft_token_ids,
                                     TensorHolder&          host_holder);

    void updatePrefillPostDraftModelInput(const StreamGroups&    stream_groups,
                                          GptModelInputs&        model_input,
                                          const GptModelOutputs& model_output,
                                          const SamplerOutput&   sampler_output,
                                          TensorHolder&          host_holder);

    // DSpARK runs two standard-slot draft calls per round: a commit call
    // (incremental-prefill shape, normalized target feature rows handed off
    // through last_hidden_states) and a
    // fixed-width propose call ([anchor, noise x (gamma - 1)] against the
    // committed feature KV).
    void validatePrefillDSparkCommitInput(const GptModelInputs& model_input) const;

    void buildDSparkProposeInput(GptModelInputs&      model_input,
                                 const torch::Tensor& anchors,
                                 const torch::Tensor& committed_ends,
                                 TensorHolder&        host_holder);

    // Immutable state shared by the proposal and target-verification phases of
    // one DSpARK round. Position bases contain one MRoPE/default tuple per
    // request; each phase expands them to its own token width, so batching
    // cannot shift one request onto another request's positions.
    struct DSparkRoundState {
        torch::Tensor anchors;
        torch::Tensor committed_ends;
        torch::Tensor position_bases;
    };
    DSparkRoundState buildDSparkRoundState(const StreamGroups&   stream_groups,
                                           const GptModelInputs& model_input,
                                           TensorHolder&         host_holder) const;

    void prepareDSparkProposeModelInput(const DSparkRoundState& round_state,
                                        GptModelInputs&         model_input,
                                        TensorHolder&           host_holder);

    void prepareDSparkTargetVerifyModelInput(const DSparkRoundState& round_state,
                                             GptModelInputs&         model_input,
                                             const torch::Tensor&    proposals,
                                             TensorHolder&           host_holder);

    // Async target preparation runs before proposal sampling completes. Build
    // its verify-width positions locally from the TP-synchronized proposal
    // metadata; no second pre-proposal broadcast is needed.
    void expandDSparkTargetVerifyPositionIdsFromProposal(GptModelInputs& model_input) const;

    void updateDecodePostDSparkCommitInput(GptModelInputs&      model_input,
                                           const torch::Tensor& target_features,
                                           size_t               batch_size);

    void updateDecodePostDraftModelInput(GptModelInputs&                              model_input,
                                         const GptModelOutputs&                       model_output,
                                         const speculative::SpeculativeSamplerOutput& speculative_sampler_output,
                                         const size_t                                 batch_size,
                                         torch::Tensor&                               hidden_states_d_t,
                                         TensorHolder&                                host_holder);

    void updateOneStepDraftSamplerOutput(const StreamGroups& stream_groups,
                                         SamplerOutput&      draft_sampler_output,
                                         torch::Tensor&      draft_token_probs_d_t,
                                         TensorHolder&       host_holder);

    void updateMultiStepDraftSamplerOutput(const StreamGroups&         stream_groups,
                                           SamplerOutput&              draft_sampler_output,
                                           torch::Tensor&              draft_token_ids_d_t,
                                           torch::Tensor&              spec_token_ids_d_t,
                                           torch::Tensor&              draft_token_probs_d_t,
                                           std::vector<torch::Tensor>& draft_token_probs_list);

protected:
    void overlayMtpCacheSnapshots(const StreamGroups& stream_groups,
                                  GptModelInputs&     model_input,
                                  TensorHolder&       host_holder) const;

    torch::Tensor collectDSparkPositionBases(const StreamGroups& stream_groups, TensorHolder& host_holder) const;
    torch::Tensor expandDSparkPositionIds(const torch::Tensor& position_bases, int64_t width) const;

    void updateProposeTokens(const StreamGroups&                stream_groups,
                             const MergedOutput&                draft_prefill_output,
                             std::vector<StreamSpecUpdateInfo>& spec_update_infos) const;

    void preparePrefillSpecUpdateInfo(const StreamGroups&                stream_groups,
                                      const MergedOutput&                prefill_output,
                                      const MergedOutput&                propose_output,
                                      const torch::Tensor&               draft_last_hidden_states,
                                      const torch::Tensor&               new_tokens_all,
                                      std::vector<StreamSpecUpdateInfo>& spec_update_infos) const;

    void prepareDecodeSpecUpdateInfo(const StreamGroups&                          stream_groups,
                                     const speculative::SpeculativeSamplerOutput& spec_decode_output,
                                     const MergedOutput&                          draft_prefill_output,
                                     std::vector<StreamSpecUpdateInfo>&           spec_update_infos) const;

    torch::Tensor compactAcceptedPositionIds(const torch::Tensor&    combo_position_ids,
                                             const std::vector<int>& accept_lens,
                                             size_t                  total_accept_len) const;

    void gatherHiddenStates(const StreamGroups& stream_groups, GptModelInputs& model_input) const;

protected:
    torch::Tensor dsparkComboTokens(int64_t batch_size, const torch::Tensor& anchors);
    torch::Tensor dsparkDraftInputLengths(int64_t batch_size);
    torch::Tensor dsparkDraftLmIndexes(int64_t batch_size);
    int64_t dsparkQueryWidth() const {
        return propose_step_ + static_cast<int64_t>(!dspark_sample_from_anchor_);
    }

    int     propose_step_;
    size_t  vocab_size_                   = 0;
    bool    is_dspark_                    = false;
    int32_t dspark_mask_token_id_         = -1;
    bool    dspark_sample_from_anchor_     = true;

    // Decode-round constants are grow-only device buffers.  Keeping them on
    // device is required by RTP_LLM_STREAM_ASYNC: no accept-length D2H is
    // introduced on the scheduling thread.
    torch::Tensor dspark_combo_cache_;
    torch::Tensor dspark_input_lengths_cache_;
    torch::Tensor dspark_lm_indexes_cache_;
};
}  // namespace rtp_llm
