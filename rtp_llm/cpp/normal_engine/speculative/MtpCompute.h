#pragma once

#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsVerifyRunner.h"
#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.h"

namespace rtp_llm {
namespace mtp {

enum class DraftInputLayout {
    COMPACT,
    FIXED_WIDTH,
};

void prepareDraftInputForPrefill(GptModelInputs&      draft_input,
                                 const torch::Tensor& target_hidden_states,
                                 const torch::Tensor& sampled_token_ids,
                                 const torch::Tensor& next_position_ids,
                                 size_t               position_id_len_factor,
                                 TensorHolder&        host_holder);

void prepareDraftInputForDecode(GptModelInputs&      draft_input,
                               const torch::Tensor& target_hidden_states,
                               const torch::Tensor& accepted_token_ids,
                               const torch::Tensor& accepted_lengths,
                               DraftInputLayout     layout,
                               size_t               position_id_len_factor,
                               TensorHolder&        host_holder);

void advanceDraftInput(GptModelInputs&      draft_input,
                       const torch::Tensor& draft_hidden_states,
                       const torch::Tensor& draft_token_ids,
                       size_t               position_id_len_factor,
                       TensorHolder&        host_holder);

void runRejectionSampling(speculative::SpeculativeSampler&               sampler,
                          const speculative::SpeculativeSamplingParams& params,
                          SamplerOutput&                                draft_sampler_output,
                          SamplerOutput&                                target_sampler_output,
                          const SpecLogitsVerifyRunner::LaunchResult&    verify_result,
                          speculative::SpeculativeSamplerOutput&        output);

}  // namespace mtp
}  // namespace rtp_llm
