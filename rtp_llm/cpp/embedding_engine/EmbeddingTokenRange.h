#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace rtp_llm {

// Called on host request IDs before enqueueing. This must not run in a GPU
// forward/capture path: no device synchronization or data-dependent graph work.
inline void validateEmbeddingTokenRange(const int32_t* ids, size_t count, int64_t input_vocab_size) {
    for (size_t i = 0; i < count; ++i) {
        if (ids[i] < 0 || ids[i] >= input_vocab_size) {
            throw std::invalid_argument("Embedding input token ID " + std::to_string(ids[i])
                                        + " is outside checkpoint input vocabulary [0, "
                                        + std::to_string(input_vocab_size) + ")");
        }
    }
}

}  // namespace rtp_llm
