#pragma once
#include <vector>
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"

namespace rtp_llm {

/// @brief Distribute input tokens across context parallel ranks with load balancing
///
/// Splits a sequence of input tokens into balanced chunks for parallel processing. Each rank receives
/// a subset of tokens determined by the zig-zag shuffle pattern, ensuring even workload distribution.
///
/// @param total_input_tokens Complete input token sequence before splitting
/// @param input_tokens [out] Token chunk assigned to current rank (pre-allocated)
/// @param shuffle_indices [out] Mapping from output position to original position (pre-allocated)
/// @param cp_rank Current rank ID in context parallel group (0-indexed)
/// @param cp_size Total number of ranks in context parallel group
/// @param cp_chunk_size Number of tokens assigned to current rank
/// @param cp_padding_size Padding tokens to add for alignment
/// @return true if split succeeded, false if parameters are invalid
///
/// @note The shuffle_indices output enables restoring the original token order after processing
/// @warning Requires num_tokens + cp_padding_size to be divisible by (2 * cp_size)
bool contextParallelLoadBalanceSplit(const std::vector<int>& total_input_tokens,
                                     std::vector<int>&       input_tokens,
                                     std::vector<int>&       shuffle_indices,
                                     int                     cp_rank,
                                     int                     cp_size,
                                     int                     cp_chunk_size,
                                     int                     cp_padding_size);

torch::Tensor generateQKVRestoreIndices(const torch::Tensor& prefill_cp_chunk_lengths, int cp_size);

torch::Tensor generateQKVPaddingMask(const torch::Tensor& prefill_cp_chunk_lengths,
                                     const torch::Tensor& prefill_cp_padding_lengths,
                                     int                  cp_size);

}  // namespace rtp_llm
