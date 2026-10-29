#pragma once

#include <memory>
#include <optional>

#include "rtp_llm/cpp/normal_engine/NormalBatchStreamProcessor.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsVerifyRunner.h"
#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.h"
#include <vector>

namespace rtp_llm {

// Target-model logprob state. Decode keeps the complete dense target layout as
// a zero-copy identity view until acceptance is known. It then selects only the
// emitted rows belonging to requesting streams and computes reductions for
// those rows. This avoids materializing a near-complete [R,V] copy for dense
// mixed batches. Prefill/tests may still use sparse capture and may populate
// row_max/row_shifted_logsumexp/top_logits before finalizeMtpTargetLogprobs().
struct MtpTargetLogprobs {
    torch::Tensor raw_logits;             // raw dtype [captured_target_positions, real_vocab_size]
    torch::Tensor row_max;                // FP32 [selected_target_positions]
    torch::Tensor row_shifted_logsumexp;  // FP32 [selected_target_positions]
    torch::Tensor top_logits;             // raw dtype [selected_target_positions, max_k]
    // During sparse capture this maps compact raw rows to dense rows. During
    // final selection it maps emitted output rows to compact raw rows.
    torch::Tensor source_row_indices;
    // Pinned H2D source for sparse CUDA indices. Kept through dispatch so the
    // non-blocking copy cannot outlive its host allocation.
    torch::Tensor source_row_indices_cpu_owner;
    torch::Tensor token_logprobs;         // FP32 [selected_target_positions], defined after finalize
    torch::Tensor top_logprob_token_ids;  // INT32 [selected_target_positions, max_k]
    torch::Tensor top_logprobs;           // FP32 [selected_target_positions, max_k]
    // Empty means raw row i is dense target row i (the identity path).
    // Otherwise compact raw row i corresponds to this dense target row.
    std::vector<int64_t> captured_dense_row_indices;
    int64_t              dense_row_count              = 0;
    bool                 retains_full_lm_head_storage = false;
    int64_t              requested_top_logprobs       = 0;

    bool defined() const {
        return raw_logits.defined() || token_logprobs.defined();
    }

    bool finalized() const {
        return token_logprobs.defined() && !raw_logits.defined() && !row_max.defined()
               && !row_shifted_logsumexp.defined() && !top_logits.defined() && !source_row_indices.defined();
    }

    int64_t maxTopLogprobs() const {
        if (top_logprobs.defined()) {
            return top_logprobs.size(1);
        }
        return top_logits.defined() ? top_logits.size(1) : requested_top_logprobs;
    }

    // Reports whether this payload still owns the complete LM-head output.
    // Stream-async decode must finalize such an identity payload after current-
    // step acceptance, overlapping draft-prefill work, before handing the
    // compact result to regular bookkeeping or entering the next process.
    bool retainsFullLmHeadStorage() const {
        return raw_logits.defined() && retains_full_lm_head_storage;
    }
};

// Prefill knows the output phase before it reduces target logits. Only
// requests that have already entered content may determine the shared top-k
// width; a reasoning-only request must not allocate a wider compact payload.
int64_t maxMtpActiveContentTopLogprobs(const StreamGroups& stream_groups);

// Decode uses an identity capture for both all-request and mixed batches.
// Stream-async decode finalizes that payload in the current step so draft work
// and the next process do not retain the complete target LM-head storage.
bool shouldFinalizeMtpTargetLogprobsEarly(bool stream_async_enabled, const MtpTargetLogprobs& target_logprobs);

// Decode deliberately exposes no row-selection argument: even a 127/128 mixed
// batch must remain an O(1) identity view until acceptance selects final rows.
MtpTargetLogprobs
captureMtpDecodeTargetLogprobs(const torch::Tensor& logits, int64_t max_top_logprobs, int64_t real_vocab_size);

// Generic identity capture is O(1). Supplying sparse rows performs one compact
// row copy for prefill/tests, but no logsumexp or top-k is launched.
MtpTargetLogprobs captureMtpTargetLogprobs(const torch::Tensor&        logits,
                                           int64_t                     max_top_logprobs,
                                           int64_t                     real_vocab_size,
                                           const std::vector<int64_t>& captured_dense_row_indices = {});

MtpTargetLogprobs computeMtpTargetLogprobs(const torch::Tensor&        logits,
                                           int64_t                     max_top_logprobs,
                                           int64_t                     real_vocab_size,
                                           const std::vector<int64_t>& source_row_indices = {});

void finalizeMtpTargetLogprobs(MtpTargetLogprobs&   target_logprobs,
                               const torch::Tensor& emitted_token_ids,
                               const torch::Tensor& raw_logits_override = torch::Tensor());

// Select final emitted rows, then compute logsumexp/top-k/selected-token
// logprobs only for those rows. emitted_token_ids and selected_dense_row_indices
// retain the original dense target-row layout; captured metadata maps them back
// to compact raw rows when the batch was sparse-captured.
void finalizeSelectedMtpTargetLogprobs(MtpTargetLogprobs&          target_logprobs,
                                       const torch::Tensor&        emitted_token_ids,
                                       const std::vector<int64_t>& selected_dense_row_indices);

torch::Tensor reshapeMtpTargetAllProbs(const torch::Tensor& all_probs,
                                       int64_t              batch_size,
                                       int64_t              positions_per_batch,
                                       int64_t              logits_width);

class MtpBatchStreamProcessor: public NormalBatchStreamProcessor {
public:
    static torch::Tensor makeEngramVerifyWindows(const torch::Tensor& anchor, const torch::Tensor& tokens);
    static torch::Tensor
    advanceEngramTokenWindows(const torch::Tensor& anchor, const torch::Tensor& tokens, const torch::Tensor& lengths);
    MtpBatchStreamProcessor(const ModelConfig&                 model_config,
                            const PDSepConfig&                 pd_sep_config,
                            const ProfilingDebugLoggingConfig& profiling_debug_logging_config,
                            const CacheConfig&                 cache_config,
                            const SpeculativeExecutionConfig&  sp_config,
                            bool                               warm_up,
                            kmonitor::MetricsReporterPtr       metrics_reporter = nullptr):
        NormalBatchStreamProcessor(model_config,
                                   pd_sep_config,
                                   profiling_debug_logging_config,
                                   cache_config,
                                   warm_up,
                                   0,
                                   std::move(metrics_reporter)),
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

    absl::Status dispatchPrefill(const StreamGroups&      stream_groups,
                                 const MergedOutput&      prefill_output,
                                 const MergedOutput&      propose_output,
                                 const torch::Tensor&     draft_last_hidden_states,
                                 const MtpTargetLogprobs& target_logprobs) const;

    absl::Status dispatchDecode(const StreamGroups&                          stream_groups,
                                const speculative::SpeculativeSamplerOutput& spec_decode_output,
                                const MergedOutput&                          draft_prefill_output) const;

    absl::Status dispatchDecode(const StreamGroups&                          stream_groups,
                                const speculative::SpeculativeSamplerOutput& spec_decode_output,
                                const MergedOutput&                          draft_prefill_output,
                                MtpTargetLogprobs                            target_logprobs) const;

    // Wait for accepted lengths/tokens and reduce only accepted target rows.
    // Idempotent for an already-finalized payload so an early async finalize
    // can hand the compact result to the regular dispatch worker.
    void finalizeDecodeTargetLogprobs(const StreamGroups&                          stream_groups,
                                      const speculative::SpeculativeSamplerOutput& spec_decode_output,
                                      MtpTargetLogprobs&                           target_logprobs) const;

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

    // `draft_tokens_ready` is recorded on the main stream after the proposal
    // that produced `draft_token_ids`. When it is supplied the token-matrix
    // expansion runs on the copy stream, ordered by that event, and overlaps the
    // target-verify forward. The non-DSpARK MTP path assigns its draft tokens
    // after the forward, so it passes nullptr and expands on the main stream.
    absl::StatusOr<SamplerInputs>
    gatherSpecSamplerInput(const StreamGroups&                         stream_groups,
                           const GptModelOutputs&                      model_output,
                           const SpecLogitsVerifyRunner::LaunchResult& spec_logits_result,
                           const torch::Tensor&                        draft_token_ids,
                           const std::shared_ptr<torch::Event>&        draft_tokens_ready = nullptr) const;

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
        torch::Tensor engram_windows;
        // Filled together with anchors/lengths by the CUDA round-head kernel.
        // Each round owns its buffer, including when the batch shrinks/reorders.
        torch::Tensor proposal_tokens;
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
                                             TensorHolder&           host_holder,
                                             torch::Tensor*          owned_proposals = nullptr);

    struct DSparkPackedTargetLayout {
        // CUDA int64 [batch + total_draft_budget], in stable request order.
        // Undefined when the full-width dense path is retained.
        torch::Tensor packed_to_dense;
        // Preserve the sampler/spec-logits and next-round position contracts.
        torch::Tensor dense_tokens;  // [batch, propose_step + 1]
        torch::Tensor dense_position_ids;
    };

    // The round-head planner guarantees 0 <= lengths <= propose_step and
    // sum(lengths) == total_draft_budget. The host budget fixes the output shape
    // without reading a CUDA scalar or computing a dynamic-size nonzero.
    DSparkPackedTargetLayout prepareDSparkPackedTargetVerifyModelInput(const DSparkRoundState& round_state,
                                                                       GptModelInputs&         model_input,
                                                                       const torch::Tensor&    proposals,
                                                                       const torch::Tensor&    valid_draft_lengths,
                                                                       int64_t                 total_draft_budget,
                                                                       TensorHolder&           host_holder,
                                                                       torch::Tensor* owned_proposals = nullptr);

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
                                      const MtpTargetLogprobs&           target_logprobs,
                                      const torch::Tensor&               new_tokens_all,
                                      std::vector<StreamSpecUpdateInfo>& spec_update_infos) const;

    void prepareDecodeSpecUpdateInfo(const StreamGroups&                          stream_groups,
                                     const speculative::SpeculativeSamplerOutput& spec_decode_output,
                                     const MergedOutput&                          draft_prefill_output,
                                     MtpTargetLogprobs&                           target_logprobs,
                                     std::vector<StreamSpecUpdateInfo>&           spec_update_infos) const;

    torch::Tensor compactAcceptedPositionIds(const torch::Tensor&    combo_position_ids,
                                             const std::vector<int>& accept_lens,
                                             size_t                  total_accept_len) const;

    void gatherHiddenStates(const StreamGroups& stream_groups, GptModelInputs& model_input) const;

protected:
    torch::Tensor dsparkComboTokens(int64_t batch_size, const torch::Tensor& anchors);
    torch::Tensor dsparkDraftInputLengths(int64_t batch_size);
    torch::Tensor dsparkDraftLmIndexes(int64_t batch_size);
    int64_t       dsparkQueryWidth() const {
        return propose_step_ + static_cast<int64_t>(!dspark_sample_from_anchor_);
    }

    int     propose_step_;
    size_t  vocab_size_                = 0;
    bool    is_dspark_                 = false;
    int32_t dspark_mask_token_id_      = -1;
    bool    dspark_sample_from_anchor_ = true;

    // gatherSpecSamplerInput runs serially on the executor thread. Keep its
    // history uploads on one stream so allocator reuse stays stream-ordered.
    mutable std::optional<torch::Stream> sampler_input_copy_stream_;

    // Decode-round constants are grow-only device buffers.  Keeping them on
    // device is required by RTP_LLM_STREAM_ASYNC: no accept-length D2H is
    // introduced on the scheduling thread.
    torch::Tensor dspark_combo_cache_;
    torch::Tensor dspark_input_lengths_cache_;
    torch::Tensor dspark_lm_indexes_cache_;
};
}  // namespace rtp_llm
