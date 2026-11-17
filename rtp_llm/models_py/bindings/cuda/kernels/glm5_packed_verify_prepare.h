#pragma once

#include <cuda_runtime.h>
#include <torch/extension.h>

namespace rtp_llm {

struct Glm5PackedVerifyInputs {
    torch::Tensor rows;
    torch::Tensor tokens;
    torch::Tensor lengths;
    torch::Tensor lm_output_indexes;
    torch::Tensor position_ids;
    torch::Tensor token_type_ids;
};

// Fuses input materialization only: the caller owns the unchanged caps/rows planner.
// Each supported input is a contiguous CUDA int32/int64 vector on the same device.
// Optional vectors may be undefined; defined vectors must cover all dense tokens.
// Returns false without changing output for unsupported metadata. Invalid device
// row/cap values raise a device assertion, as an invalid index_select would.
// Output owns independent allocations. Source allocations are registered on stream
// so replacing the caller's dense tensors immediately after return is safe.
bool tryGlm5PackedVerifyPrepare(const torch::Tensor& dense_tokens,
                                const torch::Tensor& caps,
                                const torch::Tensor& rows,
                                const torch::Tensor& position_ids,
                                const torch::Tensor& token_type_ids,
                                Glm5PackedVerifyInputs& output,
                                cudaStream_t stream);

// Builds the ordered packed row map and its consumers directly from final caps.
// packed_count is the planner's host-known admission total, including anchors.
bool tryGlm5PackedVerifyPrepareFromCaps(const torch::Tensor& dense_tokens,
                                        const torch::Tensor& caps,
                                        int64_t packed_count,
                                        const torch::Tensor& position_ids,
                                        const torch::Tensor& token_type_ids,
                                        Glm5PackedVerifyInputs& output,
                                        cudaStream_t stream);

}  // namespace rtp_llm
