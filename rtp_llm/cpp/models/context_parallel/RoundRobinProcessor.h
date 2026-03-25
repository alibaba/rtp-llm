#pragma once
#include "rtp_llm/cpp/models/context_parallel/ContextParallelProcessorBase.h"

namespace rtp_llm {

/// @brief Token-level round-robin processing implementation for context parallel.
///
/// Distributes tokens across ranks using a simple round-robin pattern:
///   token i → rank (i % cp_size)
///
/// This aligns computation distribution with the CPSlotMapper KV storage
/// pattern, eliminating the need for complex index transformations in
/// sparse MLA attention.
///
/// Distribution pattern (cp_size=2, 8 tokens):
/// - Rank 0: tokens [0, 2, 4, 6]
/// - Rank 1: tokens [1, 3, 5, 7]
///
/// @note Requires (num_tokens + cp_padding_size) to be divisible by cp_size
class RoundRobinProcessor : public IContextParallelProcessor {
public:
    ~RoundRobinProcessor() override = default;

    size_t handleOutputs(DeviceBase*                               device,
                         BufferPtr&                                hidden_states,
                         const GptModelInputs&                     inputs,
                         const torch_ext::PyContextParallelParams& cp_params) override;

protected:
    int cpAlignSize(int cp_size) const override { return cp_size; }
    bool plan(const std::vector<int>& total_input_tokens,
              std::vector<int>&       input_tokens,
              std::vector<int>&       shuffle_indices,
              int                     cp_rank,
              int                     cp_size,
              int                     cp_chunk_size,
              int                     cp_padding_size) override;

    torch::Tensor generateQKVRestoreIndices(const torch::Tensor& prefill_cp_chunk_lengths, int cp_size) override;

    torch::Tensor generateQKVPaddingMask(const torch::Tensor& prefill_cp_chunk_lengths,
                                         const torch::Tensor& prefill_cp_padding_lengths,
                                         int                  cp_size) override;
};

}  // namespace rtp_llm
