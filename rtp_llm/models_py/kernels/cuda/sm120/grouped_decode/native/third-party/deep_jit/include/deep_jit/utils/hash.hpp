#pragma once

#include <cstdint>
#include <fmt/format.h>
#include <string>
#include <string_view>

namespace deep_jit::hash {

class FNV1a {
public:
    uint64_t state_0 = 0xc6a4a7935bd1e995ull;
    uint64_t state_1 = 0xff51afd7ed558ccdull;

    FNV1a& update(const std::string_view data) {
        const auto update_byte = [&](const uint8_t byte) {
            state_0 = (state_0 ^ byte) * 0x100000001b3ull;
            state_1 = (state_1 ^ byte) * 0x9e3779b97f4a7c15ull;
        };

        // Prefix every update with its length so chained binary inputs have unambiguous boundaries
        const int64_t size = data.size();
        for (int shift = 0; shift < 64; shift += 8)
            update_byte(static_cast<uint8_t>(size >> shift));
        for (const char value: data)
            update_byte(static_cast<uint8_t>(value));
        return *this;
    }

    [[nodiscard]] std::string get_hex_digest() const {
        const auto split_mix = [](uint64_t value) {
            value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
            value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
            return value ^ (value >> 31);
        };
        return fmt::format("{:016x}{:016x}", split_mix(state_0), split_mix(state_1));
    }
};

inline std::string get_hex_digest(const std::string_view data) {
    return FNV1a().update(data).get_hex_digest();
}

}  // namespace deep_jit::hash
