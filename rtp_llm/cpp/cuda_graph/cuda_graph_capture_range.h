#pragma once

#include <vector>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <limits>
#include <stdexcept>

namespace rtp_llm {

// Adaptive target verification is keyed by total packed rows, not B * ceil(R/B).
// Since every request contributes at least one row, min(R, concurrency) metadata
// slots cover every legal batch whose live rows fit the captured token bucket.
struct AdaptiveVerifyGraphShape {
    int rows;
    int batch_capacity;
};

inline int adaptiveVerifyMaxRows(int concurrency, int tokens_per_request) {
    if (concurrency <= 0 || tokens_per_request <= 1
        || concurrency > std::numeric_limits<int>::max() / tokens_per_request) {
        throw std::invalid_argument("invalid adaptive verification capture limits");
    }
    return concurrency * tokens_per_request;
}

inline AdaptiveVerifyGraphShape adaptiveVerifyGraphShape(int concurrency, int tokens_per_request, int rows) {
    const int max_rows = adaptiveVerifyMaxRows(concurrency, tokens_per_request);
    if (rows <= 0 || rows > max_rows) {
        throw std::invalid_argument("adaptive verification rows exceed capture limits");
    }
    return {rows, std::min(rows, concurrency)};
}

inline AdaptiveVerifyGraphShape
adaptiveVerifyReplayShape(int concurrency, int tokens_per_request, int batch, int rows) {
    auto shape = adaptiveVerifyGraphShape(concurrency, tokens_per_request, rows);
    if (batch <= 0 || batch > concurrency || rows < batch
        || static_cast<int64_t>(rows) > static_cast<int64_t>(batch) * tokens_per_request) {
        throw std::invalid_argument("invalid adaptive verification replay dimensions");
    }
    return shape;
}

inline std::vector<int> adaptiveVerifyCaptureRows(int concurrency, int tokens_per_request) {
    const int        max_rows = adaptiveVerifyMaxRows(concurrency, tokens_per_request);
    std::vector<int> rows;
    // vLLM varlen decode grid: a fine geometric head, then bounded strides.
    // int64_t loop counters also cover a final bucket near INT_MAX safely.
    for (int row : {1, 2, 4, 8}) {
        if (row < max_rows)
            rows.push_back(row);
    }
    for (int64_t low = 8; low < 1024; low *= 2) {
        for (int64_t row = low + low / 4; row <= low * 2 && row < max_rows; row += low / 4) {
            rows.push_back(static_cast<int>(row));
        }
    }
    for (int64_t row = 1152; row < 4096 && row < max_rows; row += 128) {
        rows.push_back(static_cast<int>(row));
    }
    for (int64_t row = 4096; row < max_rows; row += 256) {
        rows.push_back(static_cast<int>(row));
    }
    rows.push_back(max_rows);
    return rows;
}

// Shared resolver for target graph capture and the adaptive cost model.
// Empty/unset keeps the default grid. Explicit lists are normalized and always
// closed at the maximum so every legal verification batch has a replay bucket.
inline std::vector<int>
configuredAdaptiveVerifyCaptureRows(int concurrency, int tokens_per_request, const std::string& config) {
    if (config.empty()) {
        return adaptiveVerifyCaptureRows(concurrency, tokens_per_request);
    }
    const int        max_rows = adaptiveVerifyMaxRows(concurrency, tokens_per_request);
    std::vector<int> rows;
    size_t           begin = 0;
    while (true) {
        const auto end   = config.find(',', begin);
        const auto token = config.substr(begin, end == std::string::npos ? end : end - begin);
        const auto first = token.find_first_not_of(" \t\r\n");
        const auto last  = token.find_last_not_of(" \t\r\n");
        if (first == std::string::npos) {
            throw std::invalid_argument("RTP_LLM_ADAPTIVE_CAPTURE_ROWS contains an empty bucket");
        }
        int value = 0;
        for (size_t i = first; i <= last; ++i) {
            const char digit = token[i];
            if (digit < '0' || digit > '9' || value > (max_rows - (digit - '0')) / 10) {
                throw std::invalid_argument("RTP_LLM_ADAPTIVE_CAPTURE_ROWS requires positive integers within max rows");
            }
            value = value * 10 + (digit - '0');
        }
        if (value <= 0 || value > max_rows) {
            throw std::invalid_argument("RTP_LLM_ADAPTIVE_CAPTURE_ROWS bucket outside capture limits");
        }
        rows.push_back(value);
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
    rows.push_back(max_rows);
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    return rows;
}

inline std::vector<int> configuredAdaptiveVerifyCaptureRows(int concurrency, int tokens_per_request) {
    const char* config = std::getenv("RTP_LLM_ADAPTIVE_CAPTURE_ROWS");
    return configuredAdaptiveVerifyCaptureRows(concurrency, tokens_per_request, config ? config : "");
}

// Valid ragged capture metadata: every slot is nonempty, each length is within
// the configured verify limit, and the sum is exactly the graph's row count.
inline std::vector<int> adaptiveVerifyCaptureLengths(const AdaptiveVerifyGraphShape& shape) {
    std::vector<int> lengths(shape.batch_capacity, shape.rows / shape.batch_capacity);
    for (int i = 0; i < shape.rows % shape.batch_capacity; ++i) {
        ++lengths[i];
    }
    return lengths;
}

inline std::vector<int> draftPrefillCaptureSeqLens(const std::vector<int>& decode_batch_sizes, int num_tokens_per_bs) {
    std::vector<int> seq_lens;
    seq_lens.reserve(decode_batch_sizes.size());
    for (int batch_size : decode_batch_sizes) {
        seq_lens.push_back(batch_size * num_tokens_per_bs);
    }
    return seq_lens;
}

}  // namespace rtp_llm
