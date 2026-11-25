#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace rtp_llm {

struct MlaHostCachePlan {
    uint32_t logical_blocks               = 0;
    uint32_t prefill_hbm_layers           = 0;
    uint32_t prefill_staging_blocks       = 0;
    uint32_t prefill_staging_layers       = 0;
    uint32_t decode_host_blocks           = 0;
    uint32_t decode_resident_blocks       = 0;
    uint32_t decode_hbm_share_denominator = 0;

    bool layeredPrefill() const {
        return prefill_staging_layers != 0;
    }

    bool pinnedDecode() const {
        return decode_host_blocks != 0;
    }

    uint32_t decodeDeviceBlocks(uint32_t confirmed_blocks) const {
        if (!pinnedDecode() || confirmed_blocks <= 2) {
            return confirmed_blocks;
        }
        return confirmed_blocks - std::min(decode_host_blocks, confirmed_blocks - 2);
    }
};

struct LayeredMlaPrefillBudget {
    size_t   device_bytes               = 0;
    size_t   host_bytes                 = 0;
    size_t   all_device_bytes_per_block = 0;
    size_t   layer_bytes_per_block      = 0;
    uint32_t main_layers                = 0;
    uint32_t max_logical_blocks         = 0;
    uint32_t max_staging_blocks         = 0;
    uint32_t requested_staging_layers   = 4;
};

namespace mla_cache_budget {

inline bool charge(size_t& budget, size_t bytes, size_t count) {
    if (count != 0 && bytes > budget / count) {
        return false;
    }
    budget -= bytes * count;
    return true;
}

template<class Fits>
uint32_t largestFittingBlockCount(uint32_t limit, Fits&& fits) {
    uint32_t low  = 0;
    uint32_t high = limit;
    while (low < high) {
        const uint32_t distance  = high - low;
        const uint32_t candidate = low + distance / 2 + distance % 2;
        if (fits(candidate)) {
            low = candidate;
        } else {
            high = candidate - 1;
        }
    }
    return low;
}

}  // namespace mla_cache_budget

inline MlaHostCachePlan planLayeredMlaPrefill(const LayeredMlaPrefillBudget& input) {
    MlaHostCachePlan best;
    if (input.main_layers < 2 || input.layer_bytes_per_block == 0 || input.max_staging_blocks == 0
        || input.requested_staging_layers == 0
        || input.main_layers > input.all_device_bytes_per_block / input.layer_bytes_per_block) {
        return best;
    }

    // Keep at least the first MLA layer in HBM. All other group storage (such
    // as Indexer KV and the draft model) is still charged to the device budget.
    for (uint32_t host_layers = 1; host_layers < input.main_layers; ++host_layers) {
        const size_t   host_block_bytes    = host_layers * input.layer_bytes_per_block;
        const size_t   device_block_bytes  = input.all_device_bytes_per_block - host_block_bytes;
        const uint32_t staging_layers      = std::min(host_layers, input.requested_staging_layers);
        const size_t   staging_block_bytes = staging_layers * input.layer_bytes_per_block;
        const auto     host_limit =
            static_cast<uint32_t>(std::min<size_t>(input.max_logical_blocks, input.host_bytes / host_block_bytes));
        const auto logical_blocks = mla_cache_budget::largestFittingBlockCount(host_limit, [&](uint32_t blocks) {
            size_t remaining = input.device_bytes;
            return mla_cache_budget::charge(remaining, device_block_bytes, blocks)
                   && mla_cache_budget::charge(
                       remaining, staging_block_bytes, std::min(blocks, input.max_staging_blocks));
        });
        // An equal capacity prefers the smaller host suffix. Block zero is
        // the sentinel and cannot be admitted for a request.
        if (logical_blocks > 1 && logical_blocks > best.logical_blocks) {
            best.logical_blocks         = logical_blocks;
            best.prefill_hbm_layers     = input.main_layers - host_layers;
            best.prefill_staging_blocks = std::min(input.max_staging_blocks, logical_blocks);
            best.prefill_staging_layers = staging_layers;
        }
    }
    return best;
}

// A zero layered candidate means the rank cannot use the alternate placement.
// Staging depth applies to either an automatic or an explicitly layered plan.
struct MlaPrefillCapacities {
    static constexpr size_t candidate_width = 3;

    int32_t ordinary_blocks;
    int32_t layered_blocks;
    int32_t staging_layers;

    bool useLayered() const {
        return layered_blocks > ordinary_blocks;
    }
};

inline MlaPrefillCapacities minimumMlaPrefillCapacities(const int32_t* candidates, size_t rank_count) {
    MlaPrefillCapacities capacities{candidates[0], candidates[1], candidates[2]};
    for (size_t rank = 1; rank < rank_count; ++rank) {
        const auto* candidate      = candidates + MlaPrefillCapacities::candidate_width * rank;
        capacities.ordinary_blocks = std::min(capacities.ordinary_blocks, candidate[0]);
        capacities.layered_blocks  = std::min(capacities.layered_blocks, candidate[1]);
        capacities.staging_layers  = std::min(capacities.staging_layers, candidate[2]);
    }
    return capacities;
}

inline size_t pinnedMlaRequiredTokenSlots(size_t target_topk, size_t draft_topk, size_t queries, size_t batch) {
    const auto topk  = std::max(target_topk, draft_topk);
    const auto limit = std::numeric_limits<size_t>::max();
    if (topk == 0 || queries == 0 || batch == 0 || topk > limit / queries || topk * queries > limit / batch) {
        return 0;
    }
    return topk * queries * batch;
}

struct PinnedMlaDecodeBudget {
    size_t   device_bytes                    = 0;
    size_t   host_bytes                      = 0;
    size_t   mla_bytes_per_block             = 0;
    size_t   device_metadata_bytes_per_block = 0;
    size_t   resident_workspace_bytes        = 0;
    uint32_t resident_blocks                 = 0;
    uint32_t max_logical_blocks              = 0;
    uint32_t max_device_blocks               = 0;
    uint32_t hbm_share_denominator           = 0;
    size_t   context_workspace_bytes         = 0;
};

inline MlaHostCachePlan planPinnedMlaDecode(const PinnedMlaDecodeBudget& input) {
    MlaHostCachePlan   plan;
    constexpr uint32_t min_hbm_blocks = 2;  // Sentinel and one usable HBM block.
    if (input.max_logical_blocks <= min_hbm_blocks || input.max_device_blocks < min_hbm_blocks
        || input.mla_bytes_per_block == 0 || input.device_metadata_bytes_per_block == 0 || input.resident_blocks == 0
        || input.mla_bytes_per_block > std::numeric_limits<size_t>::max() - input.device_metadata_bytes_per_block) {
        return plan;
    }
    size_t device_bytes = input.device_bytes;
    if (!mla_cache_budget::charge(device_bytes, input.resident_workspace_bytes, 1)
        || !mla_cache_budget::charge(device_bytes, input.context_workspace_bytes, 1)) {
        return plan;
    }
    const size_t hbm_bytes_per_block = input.mla_bytes_per_block + input.device_metadata_bytes_per_block;
    size_t       for_host_metadata   = device_bytes;
    if (!mla_cache_budget::charge(for_host_metadata, hbm_bytes_per_block, min_hbm_blocks)) {
        return plan;
    }

    const auto host_blocks =
        static_cast<uint32_t>(std::min({static_cast<size_t>(input.max_logical_blocks - min_hbm_blocks),
                                        input.host_bytes / input.mla_bytes_per_block,
                                        for_host_metadata / input.device_metadata_bytes_per_block}));
    if (host_blocks == 0) {
        return plan;
    }
    const size_t for_hbm  = device_bytes - host_blocks * input.device_metadata_bytes_per_block;
    const auto hbm_blocks = static_cast<uint32_t>(std::min({static_cast<size_t>(input.max_logical_blocks - host_blocks),
                                                            static_cast<size_t>(input.max_device_blocks),
                                                            for_hbm / hbm_bytes_per_block}));
    plan.logical_blocks   = hbm_blocks + host_blocks;
    plan.decode_host_blocks           = host_blocks;
    plan.decode_resident_blocks       = input.resident_blocks;
    plan.decode_hbm_share_denominator = input.hbm_share_denominator;
    return plan;
}

}  // namespace rtp_llm
