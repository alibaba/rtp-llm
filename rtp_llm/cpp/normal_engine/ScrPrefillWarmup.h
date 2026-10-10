#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace rtp_llm {

// Match the startup gRPC warmup plan in start_server.py, including the
// default powers of two and the final max-length request with reserve space.
inline std::vector<int64_t>
scrPrefillWarmupTokenLens(const std::string& configured, int64_t max_seq_len, int64_t reserve_step) {
    const int64_t max_input = max_seq_len - std::max<int64_t>(1, reserve_step);
    if (max_seq_len < 2 || max_input < 1) {
        throw std::invalid_argument("SCR prefill warmup requires input and speculative reserve space");
    }
    std::vector<int64_t> result;
    const auto           append = [&](int64_t target) {
        const int64_t length = std::min(target, max_input);
        if (std::find(result.begin(), result.end(), length) == result.end()) {
            result.push_back(length);
        }
    };
    if (configured.find_first_not_of(" \t\r\n") == std::string::npos) {
        for (int64_t length = 2; length <= max_seq_len;) {
            append(length);
            if (length > max_seq_len / 2) {
                break;
            }
            length *= 2;
        }
        append(max_seq_len);
        return result;
    }
    std::istringstream input(configured);
    std::string        value;
    if (configured.back() == ',') {
        throw std::invalid_argument("empty SCR prefill warmup token length");
    }
    while (std::getline(input, value, ',')) {
        size_t        used   = 0;
        const int64_t target = std::stoll(value, &used);
        while (used < value.size() && std::isspace(static_cast<unsigned char>(value[used]))) {
            ++used;
        }
        if (used != value.size() || target < 2 || target > max_seq_len) {
            throw std::invalid_argument("invalid SCR prefill warmup token length: " + value);
        }
        append(target);
    }
    return result;
}

}  // namespace rtp_llm
