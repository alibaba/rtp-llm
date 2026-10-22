#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace rtp_llm {

inline std::vector<std::pair<int32_t, int32_t>> getMultimodalTokenSpans(
    const std::vector<int32_t>& tokens, const std::vector<std::vector<int64_t>>& separators, bool include_separators) {
    std::vector<std::pair<int32_t, int32_t>> spans;
    for (const auto& sep : separators) {
        if (sep.size() != 1 && sep.size() != 2) {
            throw std::invalid_argument(
                "more than 2 sep tokens or no sep tokens for multimodal model is not supported");
        }
        int32_t start = -1;
        for (int32_t i = 0; i < static_cast<int32_t>(tokens.size()); ++i) {
            if (tokens[i] == sep[0]) {
                if (sep.size() == 1) {
                    spans.emplace_back(i, i + 1);
                } else {
                    if (start != -1) {
                        throw std::invalid_argument("unmatched multimodal tag pairs");
                    }
                    start = include_separators ? i : i + 1;
                }
            } else if (sep.size() == 2 && tokens[i] == sep[1]) {
                if (start == -1) {
                    throw std::invalid_argument("unmatched multimodal tag pairs");
                }
                spans.emplace_back(start, include_separators ? i + 1 : i);
                start = -1;
            }
        }
        if (start != -1) {
            throw std::invalid_argument("unclosed multimodal tag pairs");
        }
    }
    std::sort(spans.begin(), spans.end());
    for (size_t i = 1; i < spans.size(); ++i) {
        if (spans[i].first < spans[i - 1].second) {
            throw std::invalid_argument("overlapping multimodal tags");
        }
    }
    return spans;
}

}  // namespace rtp_llm
