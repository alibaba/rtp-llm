#pragma once

#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

// ``SuperBlockLayout`` is defined in CacheGroupType.h (already included above).
// Kept as a separately-defined lightweight struct so this header does NOT
// transitively pull in CacheConfig.h / c10 headers.

using CacheKeyType = int64_t;
using BlockIdxType = int32_t;

constexpr BlockIdxType NULL_BLOCK_IDX = static_cast<BlockIdxType>(-1);

inline bool isNullBlockIdx(BlockIdxType block_idx) {
    return block_idx == NULL_BLOCK_IDX;
}

using CacheKeysType    = std::vector<CacheKeyType>;
using BlockIndicesType = std::vector<BlockIdxType>;

class BlockIds {
public:
    explicit BlockIds(size_t kernel_blocks_per_kv_block = 1):
        kernel_blocks_per_kv_block_(kernel_blocks_per_kv_block > 0 ? kernel_blocks_per_kv_block : 1) {}

    size_t blocksNum() const;

    const BlockIndicesType& blocks() const;

    const BlockIndicesType& kernelBlocks() const;

    size_t kernelBlocksPerKvBlock() const;

    // Remove and return the last physical block ID.
    BlockIdxType popBack();

    // Append new physical block IDs to the tail.
    void add(const BlockIndicesType& ids);
    void remove(const std::vector<size_t>& indices);

    // Swap the physical block IDs at positions pos_a and pos_b.
    // Corresponding kernel slots for both positions are updated incrementally.
    void swap(size_t pos_a, size_t pos_b);

    void assign(const BlockIndicesType& new_block_indices);
    void assign(BlockIndicesType&& new_block_indices);
    void setAt(size_t pos, BlockIdxType val);

    void resize(size_t new_size, BlockIdxType value = NULL_BLOCK_IDX);

private:
    // Update the kernel slots that correspond to physical block position `pos`.
    void updateKernelSlotAt(size_t pos, BlockIdxType val);
    // Update all kernel slots
    void syncKernelBlocks();

    BlockIndicesType block_indices;
    // Kernel-granularity block IDs, always maintained.
    // Size is always block_indices.size() * kernel_blocks_per_kv_block_.
    // When kernel_blocks_per_kv_block_ == 1, kernel_block_indices_ mirrors block_indices.
    BlockIndicesType kernel_block_indices_;
    size_t           kernel_blocks_per_kv_block_ = 1;
};

using GroupBlockIds     = std::vector<std::shared_ptr<BlockIds>>;
using LayerBlockIds     = std::vector<std::shared_ptr<BlockIds>>;
using LayerAttnBlockIds = std::vector<std::vector<std::shared_ptr<BlockIds>>>;

class KVCacheResource {
public:
    void initGroups(int                                  group_num,
                    int                                  layer_num,
                    const std::vector<int>&              layer_to_group_id          = {},
                    size_t                               kernel_blocks_per_kv_block = 1,
                    const std::vector<CacheGroupType>&   group_types                = {},
                    const std::vector<std::vector<int>>& layer_region_to_group_id   = {});
    void resizeBlocks(int reserver_blocks, int value = 0);

    int                     blocksNum(int group_id = 0) const;
    const BlockIndicesType& blocks(int group_id = 0) const;
    const BlockIndicesType& blocks(int layer_id, KVCacheRegionName region_name) const;
    const BlockIndicesType& kernelBlocks(int group_id = 0) const;
    const BlockIndicesType& kernelBlocks(int layer_id, KVCacheRegionName region_name) const;
    BlockIds&               mutableBlockIds(int group_id = 0) const;
    BlockIds&               mutableBlockIds(int layer_id, KVCacheRegionName region_name) const;

    int groupNums() const;

    GroupBlockIds&       groupBlocks();
    const GroupBlockIds& groupBlocks() const;

    const LayerBlockIds&     layerBlocks() const;
    const LayerAttnBlockIds& layerAttnBlocks() const;
    int                      groupId(int layer_id, KVCacheRegionName region_name) const;

    CacheKeysType&       cacheKeys();
    const CacheKeysType& cacheKeys() const;

    // ---------- M01-PR3 / M04 pattern: controlled cache-keys mutators ----------
    // Additive convenience wrappers that match the sanctioned construction-time
    // pattern documented in M01 §3.6 (Panel-A item 5 / C3). Phase-6 cleanup
    // will privatise ``cache_keys`` and force all call-sites through these
    // methods + ``BatchKVCacheResource``'s fan-out (``clearCacheKeys`` /
    // ``pushBackCacheKey``). Today they remain optional — the existing public
    // ``cacheKeys()`` accessor is retained for backward compatibility.
    void clearCacheKeys() {
        cache_keys.clear();
    }
    void appendCacheKey(CacheKeyType k) {
        cache_keys.push_back(k);
    }

    // Return rank-local cache keys: every cp_size-th key starting from cp_rank.
    // localCacheKeys(r, s)[i] == cacheKeys()[i * s + r]
    // Note: when cacheKeys().size() % cp_size != 0 (e.g. 1 real block, cp_size=2),
    // localCacheKeys may return fewer entries than blocks().size().  This is
    // intentional — padding blocks carry no real data and must NOT participate in
    // device cache insert, PD transfer, or connector operations.  Downstream code
    // (e.g. insertIntoCache) already uses min(keys, blocks) to handle this.
    CacheKeysType localCacheKeys(int cp_rank, int cp_size) const {
        CacheKeysType local;
        for (int i = cp_rank; i < static_cast<int>(cache_keys.size()); i += cp_size) {
            local.push_back(cache_keys[i]);
        }
        return local;
    }

    size_t reuseBlockNum() const;

    size_t deviceReuseBlockNum() const;
    void   setDeviceReuseBlockNum(size_t device_reuse_blocks_num);

    size_t memoryReuseBlockNum() const;
    void   setMemoryReuseBlockNum(size_t memory_reuse_blocks_num);

    size_t remoteReuseBlockNum() const;
    void   setRemoteReuseBlockNum(size_t remote_reuse_blocks_num);

    bool lastBlockAligned() const;
    void setLastBlockAligned(bool last_block_aligned);
    // M01-PR3 / M04 sanctioned naming alias (Panel-A item 5 / C3 closure).
    // Same semantic as ``setLastBlockAligned`` — present so call-sites
    // converging on the unified pattern can use the canonical name without
    // touching the legacy spelling. Phase-6 cleanup will retire the legacy
    // name in favour of this one.
    void setLastBlockAlignedAll(bool v) {
        setLastBlockAligned(v);
    }

    size_t remoteReuseBlocksNum() const;
    void   setRemoteReuseBlocksNum(size_t remote_reuse_blocks_num);

    void swapBlocks(size_t group_id, size_t rhs, size_t lhs);

    std::string debugString() const;

    // ---------- M01-PR3: unified super-block view (additive, dual-storage) ----------
    //
    // ``super_block_ids_`` is the canonical per-stream allocation list under the
    // unified path (``CacheConfig::super_block_layout.enabled == true``). It is
    // populated by ``HybridPoolKVCacheAllocator::unifiedMalloc`` alongside the
    // existing per-pool ``group_block_ids`` (which remains the per-pool view —
    // byte-identical under bps[p]==1 for DSV4 today) and drained by
    // ``unifiedFree``. Under the legacy per-pool path (default) this vector is
    // empty and ``isUnified()`` returns false; consumers continue using
    // ``group_block_ids`` with no behaviour change. The per-pool collapse to a
    // single source of truth happens in Phase-6 cleanup (M01-PR6).
    bool isUnified() const {
        return !super_block_ids_.empty();
    }

    const BlockIndicesType& superBlockIds() const {
        return super_block_ids_;
    }
    BlockIndicesType& superBlockIds() {
        return super_block_ids_;
    }

    // Materialise the per-pool view of ``super_block_ids_`` for pool ``p``
    // according to ``layout.bps``. Emits ``bps[p]`` entries per super-block:
    //   poolBlockIdsView(p)[i*bps[p] + k] = layout.poolBlockId(p, S_i, k)
    // Under bps[p]==1 (DSV4 today) the result is byte-equal to
    // ``superBlockIds()``; future bps>1 layouts expand it. Returns a freshly
    // constructed vector — callers that need a long-lived buffer should cache.
    BlockIndicesType poolBlockIdsView(int p, const SuperBlockLayout& layout) const;

private:
    // layer_id -> block_indices
    LayerBlockIds layer_block_ids;
    // layer_id -> region_name -> block_indices
    LayerAttnBlockIds layer_region_block_ids;
    // group_id -> block_indices
    GroupBlockIds group_block_ids;
    CacheKeysType cache_keys;

    // M01-PR3: canonical unified-path super-block id list. Empty under legacy
    // per-pool mode. Dual-storage with ``group_block_ids`` until Phase-6
    // cleanup removes ``group_block_ids`` entirely.
    BlockIndicesType super_block_ids_;

    size_t device_reuse_block_num_{0};
    size_t memory_reuse_block_num_{0};
    size_t remote_reuse_block_num_{0};
    bool   last_block_aligned_{false};
};

using KVCacheResourcePtr = std::shared_ptr<KVCacheResource>;

}  // namespace rtp_llm
