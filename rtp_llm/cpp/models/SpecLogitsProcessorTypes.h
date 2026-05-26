#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace rtp_llm {

struct SpecLogitsProcessorId {
    static constexpr uint64_t kInvalidStreamId = std::numeric_limits<uint64_t>::max();

    uint64_t stream_id     = kInvalidStreamId;
    size_t   processor_idx = std::numeric_limits<size_t>::max();

    bool valid() const {
        return stream_id != kInvalidStreamId && processor_idx != std::numeric_limits<size_t>::max();
    }

    bool operator==(const SpecLogitsProcessorId& other) const {
        return stream_id == other.stream_id && processor_idx == other.processor_idx;
    }
};

}  // namespace rtp_llm
