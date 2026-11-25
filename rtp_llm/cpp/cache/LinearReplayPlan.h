#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "rtp_llm/models_py/bindings/core/Types.h"

namespace rtp_llm {

// Resolved rank-local log shapes shared by memory budgeting and allocation.
struct LinearReplayPlan {
    std::vector<std::string> group_tags;
    std::array<int64_t, 4> k_shape{};
    std::array<int64_t, 4> u_shape{};
    std::array<int64_t, 4> g_shape{};
    std::array<int64_t, 3> conv_input_shape{};
    DataType               conv_dtype    = DataType::TYPE_INVALID;
    size_t                 reserve_bytes = 0;

    bool enabled() const {
        return !group_tags.empty();
    }
    bool containsGroup(std::string_view tag) const {
        return std::find(group_tags.begin(), group_tags.end(), tag) != group_tags.end();
    }

    int64_t slotCount() const {
        return u_shape[0];
    }
    int64_t maxSteps() const {
        return u_shape[1];
    }

    static LinearReplayPlan create(std::vector<std::string> tags,
                                   int64_t                  slots,
                                   int64_t                  steps,
                                   std::array<int64_t, 2>   key_dims,
                                   std::array<int64_t, 2>   value_dims,
                                   bool                     vector_gate,
                                   DataType                 conv_type,
                                   size_t                   layer_count) {
        if (tags.empty() || slots <= 0 || steps <= 0 || layer_count == 0) {
            throw std::invalid_argument("LINEAR replay requires tags, capacity and layers");
        }
        const auto channels = checkedAdd(checkedMultiply(elements(key_dims), 2), elements(value_dims));
        if (channels > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
            throw std::overflow_error("LINEAR replay channel count overflows int64");
        }
        if (vector_gate && slots > std::numeric_limits<int64_t>::max() / 2) {
            throw std::overflow_error("LINEAR replay banked slot count overflows int64");
        }
        // Fused KDA CTAs share K/G logs across tiles. Preserve their read bank
        // while they write the next verify window to the other bank.
        const int64_t    kg_slots = vector_gate ? 2 * slots : slots;
        LinearReplayPlan plan;
        plan.group_tags       = std::move(tags);
        plan.k_shape          = {kg_slots, steps, key_dims[0], key_dims[1]};
        plan.u_shape          = {slots, steps, value_dims[0], value_dims[1]};
        plan.g_shape          = {kg_slots, steps, value_dims[0], vector_gate ? key_dims[1] : 1};
        plan.conv_input_shape = {slots, steps, static_cast<int64_t>(channels)};
        plan.conv_dtype       = conv_type;
        plan.reserve_bytes    = checkedMultiply(plan.layerLogBytes(), layer_count);
        return plan;
    }

    size_t layerLogBytes() const {
        const auto conv_item_bytes = getTypeSize(conv_dtype);
        if (conv_item_bytes == 0) {
            throw std::invalid_argument("LINEAR replay requires a valid convolution log dtype");
        }
        const auto floats = checkedAdd(checkedAdd(elements(k_shape), elements(u_shape)), elements(g_shape));
        return checkedAdd(checkedMultiply(floats, sizeof(float)),
                          checkedMultiply(elements(conv_input_shape), conv_item_bytes));
    }

private:
    static size_t checkedMultiply(size_t value, size_t count) {
        if (count != 0 && value > std::numeric_limits<size_t>::max() / count) {
            throw std::overflow_error("LINEAR replay size overflows size_t");
        }
        return value * count;
    }

    static size_t checkedAdd(size_t lhs, size_t rhs) {
        if (rhs > std::numeric_limits<size_t>::max() - lhs) {
            throw std::overflow_error("LINEAR replay size overflows size_t");
        }
        return lhs + rhs;
    }

    template<size_t N>
    static size_t elements(const std::array<int64_t, N>& shape) {
        size_t count = 1;
        for (const auto dim : shape) {
            if (dim <= 0) {
                throw std::invalid_argument("LINEAR replay log dimensions must be positive");
            }
            count = checkedMultiply(count, static_cast<size_t>(dim));
        }
        return count;
    }
};

}  // namespace rtp_llm
