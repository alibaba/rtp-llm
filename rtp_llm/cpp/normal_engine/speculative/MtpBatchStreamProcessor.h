#pragma once

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
    MtpBatchStreamProcessor(const ModelConfig&                 model_config,
                            const PDSepConfig&                 pd_sep_config,
                            const ProfilingDebugLoggingConfig& profiling_debug_logging_config,
                            const CacheConfig&                 cache_config,
                            const SpeculativeExecutionConfig&  sp_config,
                            bool                               warm_up):
        NormalBatchStreamProcessor(model_config, pd_sep_config, profiling_debug_logging_config, cache_config, warm_up),
        propose_step_(sp_config.gen_num_per_cycle) {}

    absl::Status dispatchPrefill(const StreamGroups& stream_groups,
                                 const MergedOutput& prefill_output,
                                 const MergedOutput& propose_output) const;

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

    absl::StatusOr<SamplerInputs>
    gatherSpecSamplerInput(const StreamGroups&                         stream_groups,
                           const GptModelInputs&                       model_inputs,
                           const GptModelOutputs&                      model_output,
                           const SpecLogitsVerifyRunner::LaunchResult& spec_logits_result = {}) const;

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

    void updateDecodeDraftModelInput(GptModelInputs&        model_input,
                                     const GptModelOutputs& model_output,
                                     const torch::Tensor&   draft_token_ids,
                                     TensorHolder&          host_holder);

    void updatePrefillPostDraftModelInput(GptModelInputs&        model_input,
                                          const GptModelOutputs& model_output,
                                          const SamplerOutput&   sampler_output,
                                          TensorHolder&          host_holder);

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

    void gatherHiddenStates(const StreamGroups& stream_groups, GptModelInputs& model_input) const;

protected:
    int propose_step_;
};
}  // namespace rtp_llm
