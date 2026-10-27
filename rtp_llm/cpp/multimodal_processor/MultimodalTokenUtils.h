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
        if (sep.size() != 1 && sep.size() != 2 && sep.size() != 3) {
            throw std::invalid_argument(
                "more than 3 sep tokens or no sep tokens for multimodal model is not supported");
        }
        if (sep.size() == 3) {
            // Nested-aware descriptor {start, end, payload}. Official d3 prompts wrap every
            // feature-bearing leaf pair (a non-empty run of payload tokens: the image
            // placeholder's single pad; each 2-frame video group's pad run) inside an outer
            // template pair whose direct content is ordinary text (the group timestamps) plus
            // nested pairs. Pairs must stay balanced exactly like the 2-separator form, but
            // only pairs whose direct interior is entirely payload tokens are leaf spans;
            // every other pair is a wrapper that stays ordinary text. Payload tokens are
            // only legal as the complete direct interior of one pair, so every payload token
            // is covered by exactly one leaf.
            if (sep[0] == sep[1] || sep[2] == sep[0] || sep[2] == sep[1]) {
                throw std::invalid_argument("duplicated multimodal separator tokens");
            }
            struct OpenPair {
                int32_t start;
                int32_t payload;
                int32_t other;
                int32_t nested;
            };
            std::vector<OpenPair> open;
            for (int32_t i = 0; i < static_cast<int32_t>(tokens.size()); ++i) {
                if (tokens[i] == sep[0]) {
                    open.push_back({i, 0, 0, 0});
                } else if (tokens[i] == sep[1]) {
                    if (open.empty()) {
                        throw std::invalid_argument("unmatched multimodal tag pairs");
                    }
                    OpenPair top = open.back();
                    open.pop_back();
                    if (!open.empty()) {
                        open.back().nested++;
                    }
                    if (top.payload > 0 && top.other == 0 && top.nested == 0) {
                        spans.emplace_back(include_separators ? top.start : top.start + 1,
                                           include_separators ? i + 1 : i);
                    } else if (top.payload > 0) {
                        throw std::invalid_argument(
                            "payload tokens mixed with text or nested tags inside one multimodal tag pair");
                    }
                    // A pair without payload tokens (empty or wrapper) is not a span; it stays text.
                } else if (!open.empty()) {
                    if (tokens[i] == sep[2]) {
                        open.back().payload++;
                    } else {
                        open.back().other++;
                    }
                } else if (tokens[i] == sep[2]) {
                    throw std::invalid_argument("multimodal payload token outside tag pairs");
                }
            }
            if (!open.empty()) {
                throw std::invalid_argument("unclosed multimodal tag pairs");
            }
            continue;
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
