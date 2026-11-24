#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace rtp_llm {

// Called on host request IDs before enqueueing. This must not run in a GPU
// forward/capture path: no device synchronization or data-dependent graph work.
inline void
validateEmbeddingTokenRange(const int32_t* ids, size_t count, int64_t rows, const char* kind = "input token") {
    for (size_t i = 0; i < count; ++i) {
        if (ids[i] < 0 || ids[i] >= rows) {
            throw std::invalid_argument(std::string("Embedding ") + kind + " ID " + std::to_string(ids[i])
                                        + " is outside checkpoint range [0, " + std::to_string(rows) + ")");
        }
    }
}

}  // namespace rtp_llm
