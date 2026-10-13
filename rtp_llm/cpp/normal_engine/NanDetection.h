#pragma once

#include <optional>
#include <tuple>
#include <torch/all.h>

#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"

namespace rtp_llm {

// Inspect only the committed token interval. In particular, callers must crop
// speculative verification rows to accept_len before selecting the first site.
inline std::optional<NumericNanInfo> firstNanInfo(const torch::Tensor& metadata,
                                                  int64_t              row_offset,
                                                  int64_t              row_count,
                                                  int64_t              token_base,
                                                  int64_t              tokens_per_sequence = 0,
                                                  int64_t              max_stage           = 4,
                                                  bool                 token_order_first   = false) {
    if (!metadata.defined() || row_count == 0) {
        return std::nullopt;
    }
    const auto*                                             data = metadata.data_ptr<int64_t>();
    std::optional<NumericNanInfo>                           first;
    std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t> first_key;
    for (int64_t row = 0; row < row_count; ++row) {
        const auto* value = data + (row_offset + row) * 4;
        if (value[3] == 0 || value[1] > max_stage) {
            continue;
        }
        // Tail checks use layer_id == num_layers and follow transformer blocks.
        auto key = std::make_tuple(token_order_first ? row : 0, value[0], value[1], row, value[2]);
        if (!first.has_value() || key < first_key) {
            first_key = key;
            first     = NumericNanInfo{value[0],
                                   value[1],
                                   token_base + (tokens_per_sequence > 0 ? row % tokens_per_sequence : row),
                                   value[2],
                                   value[3]};
        }
    }
    return first;
}

inline void keepFirstNanInfo(std::optional<NumericNanInfo>& first, const std::optional<NumericNanInfo>& candidate) {
    // Each call here represents a later forward. Its lower layer number must
    // not replace an event already observed in an earlier forward.
    if (!first.has_value()) {
        first = candidate;
    }
}

}  // namespace rtp_llm
