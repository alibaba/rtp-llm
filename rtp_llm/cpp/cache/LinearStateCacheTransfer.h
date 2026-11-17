#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/CacheGroupType.h"

namespace rtp_llm {

inline bool isSplitLinearState(LinearStateLayout layout) {
    return layout == LinearStateLayout::DOUBLE_BUFFER_CONV || layout == LinearStateLayout::DOUBLE_BUFFER_SSM
           || layout == LinearStateLayout::DOUBLE_BUFFER_REPLAY || layout == LinearStateLayout::SINGLE_SSM;
}

// Wire aliases describe slices of P's existing combined LINEAR block. D's
// allocator group names remain independent of those aliases.
inline std::string linearStateCacheWireTag(LinearStateLayout layout, const std::string& tag) {
    switch (layout) {
        case LinearStateLayout::LEGACY:
            return tag;
        case LinearStateLayout::SPLIT_EXPORT:
        case LinearStateLayout::DOUBLE_BUFFER_CONV:
            if (tag != "linear") {
                throw std::invalid_argument("split linear cache requires tag=linear");
            }
            return layout == LinearStateLayout::SPLIT_EXPORT ? tag : "linear.conv";
        case LinearStateLayout::DOUBLE_BUFFER_SSM:
        case LinearStateLayout::SINGLE_SSM:
            if (tag != "linear_ssm") {
                throw std::invalid_argument("split SSM cache requires tag=linear_ssm");
            }
            return "linear.ssm";
        case LinearStateLayout::DOUBLE_BUFFER_REPLAY:
            throw std::invalid_argument("linear replay state has no PD wire key");
    }
    throw std::invalid_argument("unknown linear state layout");
}

struct LinearStateByteSlice {
    size_t offset;
    size_t bytes;
};

// Validate in integer space before forming a pointer. Null block sentinels must
// be handled by the caller, and cannot become huge unsigned block offsets.
inline LinearStateByteSlice checkedLinearStateCacheSlice(
    size_t allocation_bytes, size_t block_stride, int64_t block_id, size_t slice_offset, size_t slice_bytes) {
    if (block_id < 0 || block_stride == 0 || slice_bytes == 0 || slice_bytes > std::numeric_limits<uint32_t>::max()
        || slice_offset > block_stride || slice_bytes > block_stride - slice_offset) {
        throw std::invalid_argument("invalid linear state cache slice");
    }
    const auto block = static_cast<uint64_t>(block_id);
    if (block > std::numeric_limits<size_t>::max() / block_stride) {
        throw std::overflow_error("linear state cache block offset overflow");
    }
    const size_t block_offset = static_cast<size_t>(block) * block_stride;
    if (block_offset > allocation_bytes || block_stride > allocation_bytes - block_offset) {
        throw std::out_of_range("linear state cache block is outside its allocation");
    }
    return {block_offset + slice_offset, slice_bytes};
}

struct LinearStateWireSlice {
    std::string          tag;
    LinearStateByteSlice slice;
};

inline std::array<LinearStateWireSlice, 2> linearStateSplitExportSlices(
    size_t allocation_bytes, size_t block_stride, int64_t block_id, size_t ssm_bytes, size_t conv_bytes) {
    return {{{"linear.ssm", checkedLinearStateCacheSlice(allocation_bytes, block_stride, block_id, 0, ssm_bytes)},
             {"linear.conv",
              checkedLinearStateCacheSlice(allocation_bytes, block_stride, block_id, ssm_bytes, conv_bytes)}}};
}

inline void validateSplitLinearTransfer(
    size_t decode_tp_size, size_t peer_count, size_t prefill_cp_size, size_t partition_count, size_t partition_id) {
    // With P=TP2 and D=TP2, each D rank receives one whole local slice from
    // its matching P rank. Combined-state TP reassembly remains legacy-only.
    if (decode_tp_size != 2 || peer_count != 1 || prefill_cp_size != 1 || partition_count != 1 || partition_id != 0) {
        throw std::invalid_argument("split linear PD requires TP2->TP2 and CP=1");
    }
}

inline void validateSplitLinearGlobalTopology(size_t decode_tp_size,
                                              size_t decode_worker_count,
                                              size_t prefill_peer_count,
                                              size_t prefill_cp_size) {
    if (decode_tp_size != 2 || decode_worker_count != 2 || prefill_peer_count != 2 || prefill_cp_size != 1) {
        throw std::invalid_argument("split linear PD requires one TP2 prefill and one TP2 decode worker");
    }
}

inline void validateLinearStateDestinationBlock(LinearStateLayout layout, int64_t block_id) {
    if ((layout == LinearStateLayout::DOUBLE_BUFFER_CONV || layout == LinearStateLayout::DOUBLE_BUFFER_SSM
         || layout == LinearStateLayout::SINGLE_SSM)
        && block_id < 0) {
        throw std::invalid_argument("linear state handoff destination block is null or invalid");
    }
}

inline std::vector<CacheStoreBlockPair> buildLinearStateLoadPlan(const CacheGroupPolicy& policy,
                                                                 size_t                  local_block_num,
                                                                 size_t                  cache_key_count,
                                                                 size_t                  group_seq_size_per_block,
                                                                 size_t                  base_seq_size_per_block) {
    const auto layout = policy.linear_state_layout;
    if (layout == LinearStateLayout::DOUBLE_BUFFER_REPLAY) {
        return {};
    }
    if (layout != LinearStateLayout::DOUBLE_BUFFER_CONV && layout != LinearStateLayout::DOUBLE_BUFFER_SSM
        && layout != LinearStateLayout::SINGLE_SSM) {
        throw std::invalid_argument("linear state load projection requires a decode layout");
    }
    if (cache_key_count == 0) {
        return {};
    }
    if (base_seq_size_per_block == 0 || group_seq_size_per_block != base_seq_size_per_block
        || policy.cp_mapping != CpBlockMappingMode::NONE || policy.cp_slice != CpBlockSliceMode::NONE
        || cache_key_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("unsupported linear state key projection");
    }
    const int last_key = static_cast<int>(cache_key_count - 1);
    if (layout == LinearStateLayout::DOUBLE_BUFFER_SSM || layout == LinearStateLayout::SINGLE_SSM) {
        const uint32_t slots = layout == LinearStateLayout::SINGLE_SSM ? 1 : 2;
        if (policy.group_type != CacheGroupType::REQUEST_STATE || policy.request_state_slots != slots
            || local_block_num != slots) {
            throw std::invalid_argument("SSM handoff requires exactly the declared request slots");
        }
        // Request-state columns are banks, not token pages. Always restore
        // the last prompt state to committed column 0, including prefix reuse.
        return {{last_key, 0}};
    }
    if (policy.group_type != CacheGroupType::LINEAR || local_block_num < cache_key_count) {
        throw std::invalid_argument("conv handoff requires its last prompt block");
    }
    return {{last_key, last_key}};
}

}  // namespace rtp_llm
