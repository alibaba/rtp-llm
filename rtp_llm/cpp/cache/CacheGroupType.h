#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace rtp_llm {

// M01-PR1/PR3: contract between DSV4CacheConfigHelper (M02) and the unified
// allocator (M01). bps[p] is the number of physical blocks pool p holds per
// super-block. For DSV4 today bps == 1 for all p ⇒ single physical block per
// pool per super-block. num_super_blocks = floor(paged_budget / Σ_p bps[p] *
// pool_block_size_bytes[p]).
//
// Invariants (enforced at config time by M02; PR-1 only declares the type):
//   bps.size() == cache_specs.size()
//   bps[p] >= 1
//   total bytes pool p == num_super_blocks * bps[p] * pool_block_size_bytes[p]
//
// Default-constructed instance has enabled=false / num_super_blocks=0 / empty
// bps, which means "legacy per-group path" — i.e. exactly today's behaviour.
//
// Lives in this lightweight header so widely-included consumers (e.g.
// KVCacheResource) can take ``const SuperBlockLayout&`` without dragging in
// the full CacheConfig transitive surface (KVCacheSpec / AttentionConfig /
// c10 headers).
struct SuperBlockLayout {
    std::vector<uint32_t> bps;
    uint32_t              num_super_blocks{0};
    bool                  enabled{false};

    bool isUnified() const {
        return enabled;
    }
};

// Cache group type for hybrid KV-cache:
// - LINEAR: linear attention group (PD cache-store transfer keeps the last block)
// - FULL: full attention group (all blocks are needed for cache-store transfer)
// - SWA: sliding-window attention group (PD cache-store transfer keeps the last two blocks)
enum class CacheGroupType : int8_t {
    LINEAR = 0,
    FULL   = 1,
    SWA    = 2,
};

// Cache identity for models where one logical layer owns multiple cache entries.
// CacheGroupType describes allocation/reuse policy, while KVCacheRegionName
// describes which cache object a layer wants to access.
enum class KVCacheRegionName : int8_t {
    DEFAULT       = 0,
    CSA_KV        = 1,
    HCA_KV        = 2,
    INDEXER_KV    = 3,
    INDEXER_STATE = 4,
    CSA_STATE     = 5,
    HCA_STATE     = 6,
    SWA_KV        = 7,
    REGION_COUNT  = 8,
};

inline bool isStateRegion(KVCacheRegionName region_name) {
    return region_name == KVCacheRegionName::INDEXER_STATE || region_name == KVCacheRegionName::CSA_STATE
           || region_name == KVCacheRegionName::HCA_STATE;
}

// M04-PR1: contiguous enumeration of every valid ``KVCacheRegionName`` value.
// Lets callers iterate regions without hard-coding the enum order. The unified
// planner caller uses this to flatten typed regions to per-pool ids; the
// length is exactly REGION_COUNT (the sentinel), so the array does NOT include
// REGION_COUNT itself.
inline constexpr std::array<KVCacheRegionName, static_cast<size_t>(KVCacheRegionName::REGION_COUNT)>
    kAllKvCacheRegions{KVCacheRegionName::DEFAULT,
                       KVCacheRegionName::CSA_KV,
                       KVCacheRegionName::HCA_KV,
                       KVCacheRegionName::INDEXER_KV,
                       KVCacheRegionName::INDEXER_STATE,
                       KVCacheRegionName::CSA_STATE,
                       KVCacheRegionName::HCA_STATE,
                       KVCacheRegionName::SWA_KV};

}  // namespace rtp_llm
