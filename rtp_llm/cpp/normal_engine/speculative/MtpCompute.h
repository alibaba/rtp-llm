#pragma once

#include <cstdint>
#include <functional>

#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsVerifyRunner.h"
#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.h"

namespace rtp_llm {

struct CacheConfig;

namespace mtp {

/** Scalar strides describe only a single cache group; clear them for other layouts. */
void applyCacheStrideToModelInput(GptModelInputs& model_input, const CacheConfig& cache_config);

torch::Tensor makeEngramVerifyWindows(const torch::Tensor& anchor, const torch::Tensor& tokens);

/** Normalize optional pre-output-projection MTP features into all_hidden_states for the
 * all_hidden_states -> last_hidden_states hand-off. Prefer this forward's mtp_target_hidden_states;
 * otherwise query the model buffer. An explicit output must match hidden_rows unless it is -1
 * (CP-local rows). For the getter fallback, 0 infers rows from a nonempty all_hidden_states tensor.
 * Target verify passes its combo row count because graph replay does not advance Python's row counter. */
void maybeOverrideLastHiddenWithMtpBuffer(GptModelOutputs& model_output, ModelBase& source, int64_t hidden_rows = 0);

void runRejectionSampling(speculative::SpeculativeSampler&              sampler,
                          const speculative::SpeculativeSamplingParams& params,
                          SamplerOutput&                                draft_sampler_output,
                          SamplerOutput&                                target_sampler_output,
                          const SpecLogitsVerifyRunner::LaunchResult&   verify_result,
                          speculative::SpeculativeSamplerOutput&        output);

struct AcceptedTokens {
    torch::Tensor token_ids;
    torch::Tensor lengths;
};

/** Wait when required by the consumer's transfer contract; materialize missing mirrors only for host consumers. */
AcceptedTokens getHostAcceptedTokens(const speculative::SpeculativeSamplerOutput& output,
                                     bool                                         wait_for_transfer = true);

enum class DraftInputLayout {
    COMPACT,     /** Pack valid rows; input_lengths holds their counts. */
    FIXED_WIDTH, /** Keep K+1 rows per request; select the last valid output. */
};

/** Fixed-width inputs avoid host synchronization on accept_len in device-state mode. */
DraftInputLayout selectDraftInputLayout(bool device_state_enabled);

void prepareDraftPrefillAfterTargetPrefill(GptModelInputs&      draft_input,
                                           const torch::Tensor& target_hidden_states,
                                           const torch::Tensor& sampled_token_ids,
                                           const torch::Tensor& next_position_ids,
                                           size_t               position_id_len_factor,
                                           TensorHolder&        host_holder);

/** Prepare draft KV prefill for tokens accepted by target verification.
 * accept_tokens: [batch_size, propose_step + 1]; accept_len includes any retained correction/bonus token.
 * COMPACT waits for pending host mirrors and copies only missing mirrors before packing accepted rows.
 * FIXED_WIDTH keeps the device layout without waiting for host transfers.
 * The caller retains hidden_states_d_t for the draft round. */
void prepareDraftPrefillAfterVerify(GptModelInputs&                              model_input,
                                    const torch::Tensor&                         target_hidden_states,
                                    const speculative::SpeculativeSamplerOutput& speculative_sampler_output,
                                    size_t                                       batch_size,
                                    int64_t                                      propose_step,
                                    DraftInputLayout                             layout,
                                    size_t                                       position_id_len_factor,
                                    torch::Tensor&                               hidden_states_d_t,
                                    TensorHolder&                                host_holder);

/** Synchronize prepared draft inputs from TP rank 0.
 * COMPACT syncs shapes and data because compaction changes input sizes.
 * FIXED_WIDTH broadcasts only tokens, hidden states and output indexes into matching CUDA buffers;
 * the remaining input fields must already agree across ranks. */
void syncDraftPrefillAfterVerify(GptModelInputs&          draft_input,
                                 DraftInputLayout         layout,
                                 const ParallelismConfig& parallelism_config);

/** Advance lengths and positions out of place to preserve tensors shared with the previous input.
 * Keep their devices unchanged and CPU positions pinned; the caller handles transfers. */
void advanceDraftInput(GptModelInputs&      draft_input,
                       const torch::Tensor& draft_hidden_states,
                       const torch::Tensor& draft_token_ids);

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

void prepareDSparkProposeInput(GptModelInputs&            draft_input,
                               const DSparkRoundState&    round_state,
                               size_t                     propose_step,
                               int32_t                    mask_token_id,
                               bool                       sample_from_anchor,
                               DSparkProposeInputBuffers& buffers,
                               TensorHolder&              host_holder);

torch::Tensor expandDSparkPositionIds(const torch::Tensor& position_bases, int64_t width);

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

}  // namespace mtp
}  // namespace rtp_llm
