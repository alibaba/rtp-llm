#pragma once
#include "rtp_llm/cpp/models/context_parallel/ContextParallelProcessorBase.h"

namespace rtp_llm {

/// @brief Zig-zag processing implementation for context parallel
///
/// Processes tokens using a zig-zag shuffle pattern where each rank receives
/// tokens from both the start and end of the sequence, ensuring balanced workload
/// for variable-length sequences.
///
/// Distribution pattern:
/// - Rank 0: [0, pair_size) and [seq_len - pair_size, seq_len)
/// - Rank 1: [pair_size, 2*pair_size) and [seq_len - 2*pair_size, seq_len - pair_size)
/// - ...
///
/// @note Requires (num_tokens + cp_padding_size) to be divisible by (2 * cp_size)
class ZigZagProcessor: public IContextParallelProcessor {
public:
    explicit ZigZagProcessor(const ParallelismConfig& parallelism_config):
        IContextParallelProcessor(parallelism_config) {}
    ~ZigZagProcessor() override = default;

    size_t handleOutputs(torch::Tensor&                            hidden_states,
                         const GptModelInputs&                     inputs,
                         const torch_ext::PyContextParallelParams& cp_params) override;

    void handleOutputsLastHidden(torch::Tensor&                            hidden_states,
                                 const GptModelInputs&                     inputs,
                                 const torch_ext::PyContextParallelParams& cp_params) override;

protected:
    /// @brief This rank's contribution to the gathered last-token hidden (no comm).
    ///
    /// Returns a [num_lm, hidden] buffer whose row j is hidden_states[local_off]
    /// when lm_output_indexes[j] resolves (via the zigzag restore indices) to a
    /// position owned by this rank, and all-zeros otherwise. Summing this buffer
    /// across all CP ranks (all-reduce) yields the gathered last-token hidden in
    /// lm_output_indexes order. Pure tensor math on the small cp_params index
    /// tensors — never allocates a full-sequence buffer — so it runs on CPU and is
    /// unit-testable in-process without NCCL.
    torch::Tensor computeLocalLastHidden(const torch::Tensor&                      hidden_states,
                                         const GptModelInputs&                     inputs,
                                         const torch_ext::PyContextParallelParams& cp_params);
    bool          plan(const std::vector<int>& total_input_tokens,
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
