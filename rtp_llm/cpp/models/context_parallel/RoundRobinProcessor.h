#pragma once
#include "rtp_llm/cpp/models/context_parallel/ContextParallelProcessorBase.h"

namespace rtp_llm {

/// @brief Round-robin processing implementation for context parallel.
///
/// Computation distribution: page-level round-robin stride.
///   block b → rank (b % cp_size)
///   Each rank processes entire blocks of page_size tokens.
///
/// KV cache storage: page-level block assignment.
///   block b → rank (b % cp_size)
///   Each rank stores only the blocks it owns; decode reads full blocks
///   directly from the owning prefill peer.
///
/// Distribution pattern (cp_size=2, page_size=4, 16 tokens / 4 blocks):
/// - Rank 0: blocks [0, 2] → tokens [0-3, 8-11]
/// - Rank 1: blocks [1, 3] → tokens [4-7, 12-15]
///
/// @note Requires padded_seq_len to be divisible by (page_size * cp_size)
class RoundRobinProcessor: public IContextParallelProcessor {
public:
    explicit RoundRobinProcessor(int page_size = 1): page_size_(page_size) {}
    ~RoundRobinProcessor() override = default;

    size_t handleOutputs(torch::Tensor&                            hidden_states,
                         const GptModelInputs&                     inputs,
                         const torch_ext::PyContextParallelParams& cp_params) override;

protected:
    int cpAlignSize(int cp_size) const override {
        return page_size_ * cp_size;
    }
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

private:
    int page_size_;
};

}  // namespace rtp_llm
