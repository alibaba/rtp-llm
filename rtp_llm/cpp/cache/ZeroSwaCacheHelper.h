#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "rtp_llm/cpp/cache/CacheConfig.h"

namespace rtp_llm {

inline bool isZeroSwaCachingEnabled(const CacheConfig& config) {
    return config.dsv4_zero_swa_caching;
}

inline bool isZeroSwaKvRegion(KVCacheRegionName region_name) {
    return region_name == KVCacheRegionName::SWA_KV;
}

inline bool isZeroSwaKvGroup(const CacheConfig& config, int group_id) {
    return group_id >= 0 && static_cast<size_t>(group_id) < config.group_region_names.size()
           && isZeroSwaKvRegion(config.group_region_names[static_cast<size_t>(group_id)]);
}

inline bool skipSwaKvForZeroSwaCaching(const CacheConfig& config, int group_id) {
    return isZeroSwaCachingEnabled(config) && isZeroSwaKvGroup(config, group_id);
}

inline bool skipSwaKvForZeroSwaCaching(const CacheConfig& config, KVCacheRegionName region_name) {
    return isZeroSwaCachingEnabled(config) && isZeroSwaKvRegion(region_name);
}

inline uint64_t zeroSwaRestoreWindowTokens(const CacheConfig& config) {
    if (!isZeroSwaCachingEnabled(config) || config.swa_window_size == 0 || config.layer_num == 0) {
        return 0;
    }
    return static_cast<uint64_t>(config.swa_window_size) * static_cast<uint64_t>(config.layer_num);
}

// Number of physical SWA blocks the runtime keeps live at the tail of a sequence.
// Must stay in sync with kSwaActiveTailBlocks in SWAKVCacheGroup.cc. These are the
// SWA_KV blocks that PD cache_store transfers to the decode node
// (KVCacheTransferPlanner::blockPositionsForCacheTransfer) and that runtime decode
// attention reads. Because zero-SWA never stores SWA_KV in any prefix/reuse cache,
// the prefill recompute window MUST cover this tail so those blocks are always
// freshly regenerated rather than reused-but-uninitialized.
constexpr int kZeroSwaRuntimeActiveTailBlocks = 2;

// Worst-case token span of the runtime SWA active tail.
inline uint64_t zeroSwaActiveTailTokens(const CacheConfig& config) {
    const uint64_t swa_block_tokens =
        config.kernel_seq_size_per_block > 0 ? config.kernel_seq_size_per_block : config.seq_size_per_block;
    return static_cast<uint64_t>(kZeroSwaRuntimeActiveTailBlocks) * swa_block_tokens;
}

// Central invariant: the recompute window (>= swa_window * layer_num tokens) must cover
// the runtime SWA active tail; otherwise a SWA block that decode reads / PD transfers
// could be neither cached (zero-SWA) nor recomputed, yielding stale runtime SWA. This
// ties the prefill-side reuse cap to the decode-side SWA consumption so a future change
// (e.g. per-region mixed reuse) cannot silently break the coupling.
inline bool zeroSwaRestoreWindowCoversActiveTail(const CacheConfig& config) {
    if (!isZeroSwaCachingEnabled(config)) {
        return true;
    }
    return zeroSwaRestoreWindowTokens(config) >= zeroSwaActiveTailTokens(config);
}

inline size_t zeroSwaRestoreWindowBlocks(const CacheConfig& config, size_t reuse_unit_tokens) {
    const uint64_t restore_tokens = zeroSwaRestoreWindowTokens(config);
    if (restore_tokens == 0 || reuse_unit_tokens == 0) {
        return 0;
    }
    return static_cast<size_t>((restore_tokens + static_cast<uint64_t>(reuse_unit_tokens) - 1)
                               / static_cast<uint64_t>(reuse_unit_tokens));
}

inline size_t zeroSwaRestoreBlocksToDrop(const CacheConfig& config,
                                         size_t             matched_blocks,
                                         size_t             reuse_unit_tokens,
                                         size_t             seq_len_tokens) {
    const uint64_t restore_tokens = zeroSwaRestoreWindowTokens(config);
    if (matched_blocks == 0 || restore_tokens == 0 || reuse_unit_tokens == 0) {
        return 0;
    }
    const uint64_t matched_tokens = static_cast<uint64_t>(matched_blocks) * static_cast<uint64_t>(reuse_unit_tokens);
    const uint64_t suffix_tokens =
        seq_len_tokens > matched_tokens ? static_cast<uint64_t>(seq_len_tokens) - matched_tokens : 0;
    if (suffix_tokens >= restore_tokens) {
        return 0;
    }
    const uint64_t missing_tokens = restore_tokens - suffix_tokens;
    const uint64_t drop_blocks =
        (missing_tokens + static_cast<uint64_t>(reuse_unit_tokens) - 1) / static_cast<uint64_t>(reuse_unit_tokens);
    return std::min<size_t>(matched_blocks, static_cast<size_t>(drop_blocks));
}

inline int capReuseBlocksForZeroSwaCaching(const CacheConfig& config, int matched_blocks, int reuse_unit_tokens) {
    if (matched_blocks <= 0 || reuse_unit_tokens <= 0) {
        return std::max(matched_blocks, 0);
    }
    const auto restore_blocks =
        static_cast<int>(zeroSwaRestoreWindowBlocks(config, static_cast<size_t>(reuse_unit_tokens)));
    return restore_blocks > 0 ? std::max(matched_blocks - restore_blocks, 0) : matched_blocks;
}

inline size_t capReuseBlocksForZeroSwaCaching(const CacheConfig& config,
                                              size_t             matched_blocks,
                                              size_t             reuse_unit_tokens) {
    const size_t restore_blocks = zeroSwaRestoreWindowBlocks(config, reuse_unit_tokens);
    if (restore_blocks == 0) {
        return matched_blocks;
    }
    return matched_blocks > restore_blocks ? matched_blocks - restore_blocks : 0;
}

inline int
capReuseBlocksForZeroSwaCaching(const CacheConfig& config, int matched_blocks, int reuse_unit_tokens, int seq_len_tokens) {
    if (matched_blocks <= 0 || reuse_unit_tokens <= 0 || seq_len_tokens <= 0) {
        return std::max(matched_blocks, 0);
    }
    const auto drop_blocks = static_cast<int>(zeroSwaRestoreBlocksToDrop(config,
                                                                         static_cast<size_t>(matched_blocks),
                                                                         static_cast<size_t>(reuse_unit_tokens),
                                                                         static_cast<size_t>(seq_len_tokens)));
    return std::max(matched_blocks - drop_blocks, 0);
}

inline size_t capReuseBlocksForZeroSwaCaching(const CacheConfig& config,
                                              size_t             matched_blocks,
                                              size_t             reuse_unit_tokens,
                                              size_t             seq_len_tokens) {
    const size_t drop_blocks = zeroSwaRestoreBlocksToDrop(config, matched_blocks, reuse_unit_tokens, seq_len_tokens);
    return matched_blocks > drop_blocks ? matched_blocks - drop_blocks : 0;
}

}  // namespace rtp_llm
