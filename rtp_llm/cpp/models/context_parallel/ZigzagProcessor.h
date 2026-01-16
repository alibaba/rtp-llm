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
    ~ZigZagProcessor() override = default;

    size_t handleOutputs(DeviceBase*                               device,
                         BufferPtr&                                hidden_states,
                         const GptModelInputs&                     inputs,
                         const torch_ext::PyContextParallelParams& cp_params) override;

protected:
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
