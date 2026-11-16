#pragma once

#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/models_py/bindings/core/TensorHolder.h"

namespace rtp_llm {
// Model inputs and row mappings for one DSpARK verification round. Sampling
// consumes the resulting maps after the model has executed this layout.
struct DSparkRoundState {
    torch::Tensor anchors;
    torch::Tensor committed_ends;
    torch::Tensor position_bases;
    torch::Tensor engram_windows;
    // Filled together with anchors/lengths by the CUDA round-head kernel.
    // Each round owns its buffer, including when the batch shrinks/reorders.
    torch::Tensor proposal_tokens;
};
void prepareDSparkTargetVerifyModelInput(int                     propose_step,
                                         const DSparkRoundState& round_state,
                                         GptModelInputs&         model_input,
                                         const torch::Tensor&    proposals,
                                         TensorHolder&           host_holder,
                                         torch::Tensor*          owned_proposals = nullptr);

struct DSparkPackedTargetLayout {
    // CUDA int64 [batch + total_draft_budget], in stable request order.
    // Undefined when the full-width dense path is retained.
    torch::Tensor packed_to_dense;
    // CUDA int32 [batch * (propose_step + 1)]; omitted rows contain -1.
    torch::Tensor dense_to_packed;
    // Preserve the sampler/spec-logits and next-round position contracts.
    torch::Tensor dense_tokens;  // [batch, propose_step + 1]
    torch::Tensor dense_position_ids;
};

// The round-head planner guarantees 0 <= lengths <= propose_step and
// sum(lengths) == total_draft_budget. The host budget fixes the output shape
// without reading a CUDA scalar or computing a dynamic-size nonzero.
DSparkPackedTargetLayout prepareDSparkPackedTargetVerifyModelInput(int                     propose_step,
                                                                   const DSparkRoundState& round_state,
                                                                   GptModelInputs&         model_input,
                                                                   const torch::Tensor&    proposals,
                                                                   const torch::Tensor&    valid_draft_lengths,
                                                                   int64_t                 total_draft_budget,
                                                                   TensorHolder&           host_holder,
                                                                   torch::Tensor*          owned_proposals = nullptr,
                                                                   bool                    force_packed_layout = false);

// Plan confidence prefixes and materialize packed verification inputs
// together. The host-known budget fixes every output shape; the identity
// permutation must have been frozen before sampling this round's drafts.
// Unsupported CUDA geometries retain the existing planner/packing path.
DSparkPackedTargetLayout prepareDSparkBoundedTargetVerifyModelInput(int                         propose_step,
                                                                    const DSparkRoundState&     round_state,
                                                                    GptModelInputs&             model_input,
                                                                    const torch::Tensor&        proposals,
                                                                    const torch::Tensor&        confidence_rates,
                                                                    const std::vector<int64_t>& stable_order,
                                                                    int64_t                     total_draft_budget,
                                                                    TensorHolder&               host_holder,
                                                                    torch::Tensor&              valid_draft_lengths,
                                                                    torch::Tensor* owned_proposals     = nullptr,
                                                                    bool           force_packed_layout = false);

torch::Tensor expandDSparkPositionIds(const torch::Tensor& position_bases, int64_t width);

namespace mtp_input {
torch::TensorOptions cudaInt32Options();
torch::Tensor        emptyInt32OnCuda(std::initializer_list<int64_t> shape);
torch::Tensor        fullInt32OnCuda(std::initializer_list<int64_t> shape, int64_t value);
torch::Tensor        toCudaInt32(const torch::Tensor& tensor, TensorHolder& host_holder);
void                 setVerifyPairInputs(GptModelInputs& model_input,
                                         torch::Tensor   combo_tokens,
                                         size_t          batch_size,
                                         size_t          score_len,
                                         TensorHolder&   host_holder);
}  // namespace mtp_input
}  // namespace rtp_llm
