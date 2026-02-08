#pragma once
#include <memory>
#include <vector>
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"

namespace rtp_llm {

enum class PlannerType {
    ZIG_ZAG,
    // Future extensions: ROUND_ROBIN, BLOCK_WISE, etc.
};

class IContextParallelPlanner {
public:
    virtual ~IContextParallelPlanner() = default;

    virtual PlannerType getPlannerType() const = 0;

    /// @brief Process and distribute input tokens across context parallel ranks
    ///
    /// Plans the distribution of input tokens by splitting and padding as needed.
    /// Each rank receives a processed chunk according to the planning strategy.
    virtual bool plan(const std::vector<int>& total_input_tokens,
                      std::vector<int>&       input_tokens,
                      std::vector<int>&       shuffle_indices,
                      int                     cp_rank,
                      int                     cp_size,
                      int                     cp_chunk_size,
                      int                     cp_padding_size) = 0;

    /// @brief Generate indices to restore original QKV order after parallel processing
    ///
    /// @param prefill_cp_chunk_lengths Tensor of chunk lengths for each prefill stream
    /// @param cp_size Total number of ranks in context parallel group
    /// @return Tensor containing restore indices mapping
    virtual torch::Tensor generateQKVRestoreIndices(const torch::Tensor& prefill_cp_chunk_lengths, int cp_size) = 0;

    /// @brief Generate padding mask for QKV tensors
    ///
    /// @param prefill_cp_chunk_lengths Tensor of chunk lengths for each prefill stream
    /// @param prefill_cp_padding_lengths Tensor of padding lengths for each prefill stream
    /// @param cp_size Total number of ranks in context parallel group
    /// @return Tensor containing padding mask (1 for valid tokens, 0 for padding)
    virtual torch::Tensor generateQKVPaddingMask(const torch::Tensor& prefill_cp_chunk_lengths,
                                                 const torch::Tensor& prefill_cp_padding_lengths,
                                                 int                  cp_size) = 0;
};

class ContextParallelPlannerFactory {
public:
    static std::unique_ptr<IContextParallelPlanner> create(PlannerType type);
};

}  // namespace rtp_llm
