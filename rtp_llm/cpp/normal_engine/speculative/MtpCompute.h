#pragma once

#include <cstdint>
#include <functional>

#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsVerifyRunner.h"
#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.h"

namespace rtp_llm {
namespace mtp {

/** Executors select DSpARK PD prefill batches and call this on every TP rank.
 * Wait for both local writers even if one fails, then reduce status across TP.
 * An empty result means all TP ranks succeeded; executors report request errors.
 * An optional reducer replaces the collective after both waits; otherwise use the built-in TP reduction.
 * Executors own sync_stream and status, used only by the built-in reduction when TP > 1. */
std::string finishDSparkCachePublication(ModelBase*                       target,
                                         ModelBase*                       draft,
                                         const ParallelismConfig&         parallelism_config,
                                         const torch::Stream&             sync_stream,
                                         torch::Tensor&                   status,
                                         const std::function<bool(bool)>& reduce_status = {});

/** A tracked writer retains its finished cycle until consumed. Drain both local writers
 * on an early exit after forward; only the normal completion path performs a collective. */
struct DSparkCacheStoreDrainGuard {
    bool       armed;
    ModelBase* target;
    ModelBase* draft;

    ~DSparkCacheStoreDrainGuard();
    void disarm() {
        armed = false;
    }
};

enum class DraftInputLayout {
    COMPACT,     /** Pack valid rows; input_lengths holds their counts. */
    FIXED_WIDTH, /** Keep K+1 rows per request; select the last valid output. */
};

/** Layout follows active device state, independently of where input metadata resides. */
DraftInputLayout selectDraftInputLayout(bool device_state_enabled);

struct AcceptedTokens {
    torch::Tensor token_ids;
    torch::Tensor lengths;
};

/** Preserve the sampler's device tensors without introducing a host wait or transfer. */
AcceptedTokens getDeviceAcceptedTokens(const speculative::SpeculativeSamplerOutput& output);

/** Wait when required by the consumer's transfer contract; materialize missing mirrors only for host consumers. */
AcceptedTokens getHostAcceptedTokens(const speculative::SpeculativeSamplerOutput& output,
                                     bool                                         wait_for_transfer = true);

/** Immutable state shared by proposal construction, sampling and target verification in one DSpARK round.
 * anchors [batch]: each request's newest committed token.
 * committed_ends [batch]: prefix length before the anchor (committed sequence length - 1).
 * position_bases [batch, factor], when present: the anchor's default/MRoPE position tuple.
 * Proposal and verification expand these same per-request bases to their own token widths,
 * preserving request boundaries and positions across both phases.
 * engram_windows [batch, 4], when present: target history at the anchor, newest token first. */
struct DSparkRoundState {
    torch::Tensor anchors;
    torch::Tensor committed_ends;
    torch::Tensor position_bases;
    torch::Tensor engram_windows;
};

struct DSparkProposeInputBuffers {
    torch::Tensor combo_tokens;
    torch::Tensor input_lengths;
    torch::Tensor lm_output_indexes;
};

/** Normalize optional pre-output-projection MTP features into all_hidden_states for the
 * all_hidden_states -> last_hidden_states hand-off. Prefer this forward's mtp_target_hidden_states;
 * otherwise query the model buffer. An explicit output must match hidden_rows unless it is -1
 * (CP-local rows). For the getter fallback, 0 infers rows from a nonempty all_hidden_states tensor.
 * Target verify passes its combo row count because graph replay does not advance Python's row counter. */
void maybeOverrideLastHiddenWithMtpBuffer(GptModelOutputs& model_output, ModelBase& source, int64_t hidden_rows = 0);

void prepareDraftPrefillAfterTargetPrefill(GptModelInputs&      draft_input,
                                           const torch::Tensor& target_hidden_states,
                                           const torch::Tensor& sampled_token_ids,
                                           const torch::Tensor& next_position_ids,
                                           size_t               position_id_len_factor,
                                           TensorHolder&        host_holder);

/** Prepare draft KV prefill for tokens accepted by target verification.
 * accepted_token_ids: [batch_size, propose_step + 1]; lengths include any retained correction/bonus token.
 * The caller ensures input readiness and retains hidden_states_d_t for the draft round. */
void prepareDraftPrefillAfterVerify(GptModelInputs&      draft_input,
                                    const torch::Tensor& target_hidden_states,
                                    const torch::Tensor& accepted_token_ids,
                                    const torch::Tensor& accepted_lengths,
                                    size_t               batch_size,
                                    int64_t              propose_step,
                                    DraftInputLayout     layout,
                                    size_t               position_id_len_factor,
                                    torch::Tensor&       hidden_states_d_t,
                                    TensorHolder&        host_holder);

/** Synchronize post-verification draft inputs across TP ranks.
 * FIXED_WIDTH requires matching CUDA token/hidden/index buffers; other input fields must already agree. */
void syncDraftPrefillAfterVerify(GptModelInputs&          draft_input,
                                 DraftInputLayout         layout,
                                 const ParallelismConfig& parallelism_config);

/** Convert the selected draft-prefill rows into one decode row per request, preserving valid lengths. */
void prepareDraftDecodeAfterPrefill(GptModelInputs&      draft_input,
                                    const torch::Tensor& draft_hidden_states,
                                    const torch::Tensor& draft_token_ids,
                                    size_t               position_id_len_factor);

void advanceDraftInput(GptModelInputs&      draft_input,
                       const torch::Tensor& draft_hidden_states,
                       const torch::Tensor& draft_token_ids,
                       size_t               position_id_len_factor,
                       TensorHolder&        host_holder);

void prepareDSparkProposeInput(GptModelInputs&            draft_input,
                               const DSparkRoundState&    round_state,
                               size_t                     propose_step,
                               int32_t                    mask_token_id,
                               bool                       sample_from_anchor,
                               DSparkProposeInputBuffers& buffers,
                               TensorHolder&              host_holder);

torch::Tensor expandDSparkPositionIds(const torch::Tensor& position_bases, int64_t width);

torch::Tensor makeEngramVerifyWindows(const torch::Tensor& anchor, const torch::Tensor& tokens);

void runRejectionSampling(speculative::SpeculativeSampler&              sampler,
                          const speculative::SpeculativeSamplingParams& params,
                          SamplerOutput&                                draft_sampler_output,
                          SamplerOutput&                                target_sampler_output,
                          const SpecLogitsVerifyRunner::LaunchResult&   verify_result,
                          speculative::SpeculativeSamplerOutput&        output);

}  // namespace mtp
}  // namespace rtp_llm
