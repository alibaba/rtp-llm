#pragma once

#include <cstdint>

#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsVerifyRunner.h"
#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.h"

namespace rtp_llm {
namespace mtp {

enum class DraftInputLayout {
    COMPACT,      /** Pack valid rows; input_lengths holds their counts. */
    FIXED_WIDTH,  /** Keep K+1 rows per request; select the last valid output. */
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
                                     bool                                        wait_for_transfer = true);

struct DSparkProposeInputBuffers {
    torch::Tensor combo_tokens;
    torch::Tensor input_lengths;
    torch::Tensor lm_output_indexes;
};

/** Use model-provided MTP features as all_hidden_states when available.
 * hidden_rows: >0 explicit count, 0 infer from nonempty output, -1 CP-local row count. */
void maybeOverrideLastHiddenWithMtpBuffer(GptModelOutputs& model_output,
                                         ModelBase&       source,
                                         int64_t          hidden_rows = 0);

void prepareDraftPrefillAfterTargetPrefill(GptModelInputs&      draft_input,
                                          const torch::Tensor& target_hidden_states,
                                          const torch::Tensor& sampled_token_ids,
                                          const torch::Tensor& next_position_ids,
                                          size_t               position_id_len_factor,
                                          TensorHolder&        host_holder);

/** Use valid prefixes of [B, K+1] tokens and matching verify hidden rows.
 * Lengths include retained correction/bonus; callers handle caps, errors and transfer readiness. */
void prepareDraftPrefillAfterVerify(GptModelInputs&      draft_input,
                                    const torch::Tensor& target_hidden_states,
                                    const torch::Tensor& accepted_token_ids,
                                    const torch::Tensor& accepted_lengths,
                                    DraftInputLayout     layout,
                                    size_t               position_id_len_factor,
                                    TensorHolder&        host_holder);

/** COMPACT syncs shapes and data. FIXED_WIDTH requires matching CUDA buffers on all ranks
 * (lm_output_indexes: [B]) and consistent remaining inputs. */
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
                               const torch::Tensor&       anchors,
                               const torch::Tensor&       committed_ends,
                               size_t                     propose_step,
                               int32_t                    mask_token_id,
                               DSparkProposeInputBuffers& buffers,
                               TensorHolder&              host_holder);

void prepareDSparkCommitInput(GptModelInputs& draft_input, const torch::Tensor& target_features);

void runRejectionSampling(speculative::SpeculativeSampler&               sampler,
                          const speculative::SpeculativeSamplingParams& params,
                          SamplerOutput&                                draft_sampler_output,
                          SamplerOutput&                                target_sampler_output,
                          const SpecLogitsVerifyRunner::LaunchResult&    verify_result,
                          speculative::SpeculativeSamplerOutput&        output);

}  // namespace mtp
}  // namespace rtp_llm
