#pragma once

#include <cstddef>
#include <cstdint>

namespace rtp_llm::crc_copy_benchmark {

struct StagedMemcpyProfile {
    size_t   calls{0};
    size_t   bytes{0};
    uint64_t nanoseconds{0};
};

// Only the separate diagnostic executable links --wrap=memcpy.
void                beginStagedMemcpyProfile(size_t payload_bytes);
StagedMemcpyProfile endStagedMemcpyProfile();

}  // namespace rtp_llm::crc_copy_benchmark
