#pragma once
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include <vector>

namespace rtp_llm {

/// @brief Generate zig-zag shuffle indices for context parallel load balancing
///
/// This function creates a shuffle pattern that distributes tokens evenly across context parallel ranks
/// while maintaining locality. The pattern alternates between taking chunks from the start and end of
/// the sequence to balance computational load.
///
/// @param num_padded_input_tokens Total number of input tokens (including padding)
/// @param cp_size Context parallel size (number of ranks)
/// @return Vector of shuffle indices
///
/// @details
/// - Short sequences (num_tokens ≤ cp_size): Returns identity mapping [0, 1, 2, ..., n-1]
/// - Long sequences (num_tokens > cp_size): Returns zig-zag pattern where
///   * chunk_size = num_tokens / (2 * cp_size)
///   * Pattern alternates: first chunk, last chunk, second chunk, second-to-last chunk, ...
///
/// @example
///   cp_size=4, num_tokens=16, chunk_size=2
///   Result: [0, 1, 14, 15, 2, 3, 12, 13, 4, 5, 10, 11, 6, 7, 8, 9]
///           └───┘  └────┘  └───┘ └────┘  └───┘ └────┘  └───┘ └───┘
///           1st   last      2nd 2nd-last 3rd 3rd-last  4th  4th-last
///
/// @example
///   cp_size=2, num_tokens=16, chunk_size=4
///   Result: [0, 1, 2, 3, 12, 13, 14, 15, 4, 5, 6, 7, 8, 9, 10, 11]
///           └─────────┘  └──────────---┘ └────────┘ └──────────┘
///           1st chunk   last chunk        2nd chunk   2nd-last chunk
std::vector<int> generateZigZagShuffleIndices(int num_padded_input_tokens, int cp_size);

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

}  // namespace rtp_llm
