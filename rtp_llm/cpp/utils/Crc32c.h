#pragma once
#include <array>
#include <cstddef>
#include <cstdint>

namespace rtp_llm {
// Portable table-based reflected Castagnoli CRC with initial/final complement.
// No ISA requirements; the table is initialized once, without per-call allocation.
inline uint32_t cpuCrc32c(const uint8_t* data, size_t bytes) {
    static const auto table = [] {
        std::array<uint32_t, 256> result{};
        for (uint32_t i = 0; i < result.size(); ++i) {
            uint32_t crc = i;
            for (unsigned bit = 0; bit < 8; ++bit) {
                crc = (crc >> 1) ^ ((crc & 1) ? 0x82f63b78U : 0);
            }
            result[i] = crc;
        }
        return result;
    }();
    uint32_t crc = ~uint32_t(0);
    for (size_t i = 0; i < bytes; ++i) {
        crc = (crc >> 8) ^ table[(crc ^ data[i]) & 0xff];
    }
    return ~crc;
}
}  // namespace rtp_llm
