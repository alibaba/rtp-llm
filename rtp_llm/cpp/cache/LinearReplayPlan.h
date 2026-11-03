#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "rtp_llm/models_py/bindings/core/Types.h"

namespace rtp_llm {

// Resolved rank-local log shapes shared by memory budgeting and allocation.
struct LinearReplayPlan {
    std::vector<int>       group_ids;
    std::array<int64_t, 4> k_shape{};
    std::array<int64_t, 4> u_shape{};
    std::array<int64_t, 4> g_shape{};
    std::array<int64_t, 3> conv_input_shape{};
    DataType               conv_dtype    = DataType::TYPE_INVALID;
    size_t                 reserve_bytes = 0;

    bool enabled() const {
        return !group_ids.empty();
    }
    bool containsGroup(int group_id) const {
        return std::find(group_ids.begin(), group_ids.end(), group_id) != group_ids.end();
    }

    int64_t slotCount() const {
        return k_shape[0];
    }
    int64_t maxSteps() const {
        return k_shape[1];
    }

    static LinearReplayPlan create(std::vector<int>       groups,
                                   int64_t                slots,
                                   int64_t                steps,
                                   std::array<int64_t, 2> key_dims,
                                   std::array<int64_t, 2> value_dims,
                                   bool                   vector_gate,
                                   DataType               conv_type,
                                   size_t                 layer_count) {
        if (groups.empty() || slots <= 0 || steps <= 0 || layer_count == 0) {
            throw std::invalid_argument("LINEAR replay requires groups, capacity and layers");
        }
        const auto channels = checkedAdd(checkedMultiply(elements(key_dims), 2), elements(value_dims));
        if (channels > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
            throw std::overflow_error("LINEAR replay channel count overflows int64");
        }
        LinearReplayPlan plan;
        plan.group_ids        = std::move(groups);
        plan.k_shape          = {slots, steps, key_dims[0], key_dims[1]};
        plan.u_shape          = {slots, steps, value_dims[0], value_dims[1]};
        plan.g_shape          = {slots, steps, value_dims[0], vector_gate ? key_dims[1] : 1};
        plan.conv_input_shape = {slots, steps, static_cast<int64_t>(channels)};
        plan.conv_dtype       = conv_type;
        const size_t headers  = checkedMultiply(static_cast<size_t>(slots), 2 * sizeof(int64_t) + 2 * sizeof(int32_t));
        plan.reserve_bytes    = checkedAdd(headers, checkedMultiply(plan.layerLogBytes(), layer_count));
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
