#pragma once

#include "rtp_llm/cpp/normal_engine/NormalBatchStreamProcessor.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsVerifyRunner.h"
#include "rtp_llm/cpp/normal_engine/speculative/MtpCompute.h"
#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.h"

namespace rtp_llm {

class MtpBatchStreamProcessor: public NormalBatchStreamProcessor {
public:
    static torch::Tensor
    advanceEngramTokenWindows(const torch::Tensor& anchor, const torch::Tensor& tokens, const torch::Tensor& lengths);
    MtpBatchStreamProcessor(const ModelConfig&                 model_config,
                            const PDSepConfig&                 pd_sep_config,
                            const ProfilingDebugLoggingConfig& profiling_debug_logging_config,
                            const CacheConfig&                 cache_config,
                            const SpeculativeExecutionConfig&  sp_config,
                            bool                               warm_up,
                            bool                               request_lanes = false):
        NormalBatchStreamProcessor(model_config, pd_sep_config, profiling_debug_logging_config, cache_config, warm_up),
        request_lanes_(request_lanes),
        propose_step_(sp_config.gen_num_per_cycle),
        vocab_size_(model_config.vocab_size),
        is_dspark_(sp_config.type == SP_TYPE_DSPARK),
        dspark_mask_token_id_(static_cast<int32_t>(sp_config.sp_dspark_mask_token_id)),
        dspark_sample_from_anchor_(sp_config.sp_dspark_sample_from_anchor) {}

    size_t positionIdLenFactor() const {
        return model_input_gatherer_config_.position_id_len_factor;
    }

    mtp::DraftInputLayout draftInputLayout() const;

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
    absl::Status                   gatherLinearReplayInputs(const StreamGroups&                             stream_groups,
                                                            const CacheConfig&                              cache_config,
                                                            GptModelInputs&                                 model_input,
                                                            TensorHolder&                                   host_holder,
                                                            std::vector<GenerateStream::LinearReplayRound>& rounds) const;

    // Produce the next round's immutable page table by applying the exact
    // linear-cache permutation used by GenerateStream::specUpdate.  FULL/SWA
    // groups are copied unchanged. Each LINEAR group uses its own token span.
    // Keeping this operation tensor-native lets proposal/verify overlap host
    // bookkeeping without a per-round D2H join. Non-LINEAR spans may be zero.
    static torch::Tensor advanceLinearCacheBlockTable(const torch::Tensor&               current_table,
                                                      const torch::Tensor&               previous_seq_lengths,
                                                      const torch::Tensor&               accept_lengths,
                                                      const std::vector<CacheGroupType>& group_types,
                                                      const std::vector<int64_t>&        group_spans);

    // Prepare history and proposals on the caller's stream. Omitting draft
    // tokens keeps the host-only stage used by non-CUDA background preparation.
    SamplerInputs prepareSpecSamplerInput(const StreamGroups&                  stream_groups,
                                          const torch::Tensor&                 draft_token_ids    = {},
                                          const std::shared_ptr<torch::Event>& draft_tokens_ready = {}) const;
    void          prepareSpecSamplerInputDevice(const StreamGroups&                  stream_groups,
                                                SamplerInputs&                       sampler_inputs,
                                                const torch::Tensor&                 draft_token_ids,
                                                const std::shared_ptr<torch::Event>& draft_tokens_ready = {}) const;
    // Wait for target logits on the caller's stream before finalizing. This
    // stage also joins the processor mask's ready event on that stream.
    SamplerInputs finalizeSpecSamplerInput(SamplerInputs                               sampler_inputs,
                                           const GptModelOutputs&                      model_output,
                                           const SpecLogitsVerifyRunner::LaunchResult& spec_logits_result) const;

    void prepareDecodeDraftModelInput(const StreamGroups& stream_groups,
                                      GptModelInputs&     model_input,
                                      TensorHolder&       host_holder);

    void prepareOneStepSpecDecodeModelInput(const StreamGroups& stream_groups,
                                            GptModelInputs&     model_input,
                                            TensorHolder&       host_holder);

    // Device-state target-verify gather. Fresh rows may use idle host state;
    // steady rows retain their published device tokens, lengths and positions.
    bool gatherMtpDecodeModelInputFromDeviceState(const StreamGroups& stream_groups,
                                                  GptModelInputs&     model_input,
                                                  TensorHolder&       host_holder) const;

    void expandTargetVerifyPositionIds(const StreamGroups&  stream_groups,
                                       GptModelInputs&      model_input,
                                       const torch::Tensor& authoritative_prefix_lengths = torch::Tensor(),
                                       size_t               query_len                    = 0) const;
    void expandTargetVerifyPositionIdsFromDecode(GptModelInputs& model_input) const;

    void updateDecodeDraftModelInput(GptModelInputs&        model_input,
                                     const GptModelOutputs& model_output,
                                     const torch::Tensor&   draft_token_ids,
                                     TensorHolder&          host_holder);

    void updatePrefillPostDraftModelInput(const StreamGroups&    stream_groups,
                                          GptModelInputs&        model_input,
                                          const GptModelOutputs& model_output,
                                          const SamplerOutput&   sampler_output,
                                          TensorHolder&          host_holder);

    /** COMMIT populates draft KV from normalized target features while retaining the target's prefill geometry. */
    void validatePrefillDSparkCommitInput(const GptModelInputs& model_input) const;

    mtp::DSparkRoundState buildDSparkRoundState(const StreamGroups&   stream_groups,
                                                const GptModelInputs& model_input,
                                                TensorHolder&         host_holder) const;

    void prepareDSparkProposeModelInput(const mtp::DSparkRoundState& round_state,
                                        GptModelInputs&              model_input,
                                        TensorHolder&                host_holder);

    void prepareDSparkTargetVerifyModelInput(const mtp::DSparkRoundState& round_state,
                                             GptModelInputs&              model_input,
                                             const torch::Tensor&         proposals,
                                             TensorHolder&                host_holder);

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
    bool request_lanes_ = false;
    void overlayMtpCacheSnapshots(const StreamGroups& stream_groups,
                                  GptModelInputs&     model_input,
                                  TensorHolder&       host_holder) const;

    torch::Tensor collectDSparkPositionBases(const StreamGroups& stream_groups, TensorHolder& host_holder) const;

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

    void gatherHiddenStates(const StreamGroups& stream_groups, GptModelInputs& model_input) const;

protected:
    int64_t dsparkQueryWidth() const {
        return propose_step_ + static_cast<int64_t>(!dspark_sample_from_anchor_);
    }

    int     propose_step_;
    size_t  vocab_size_                = 0;
    bool    is_dspark_                 = false;
    int32_t dspark_mask_token_id_      = -1;
    bool    dspark_sample_from_anchor_ = true;

    // DSpARK proposal constants are grow-only device buffers. Keeping them on
    // device is required by RTP_LLM_STREAM_ASYNC: no accept-length D2H is
    // introduced on the scheduling thread.
    mtp::DSparkProposeInputBuffers dspark_propose_input_buffers_;
};
}  // namespace rtp_llm
