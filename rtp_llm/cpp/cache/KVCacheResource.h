#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

struct CacheConfig;

using CacheKeyType = int64_t;
using BlockIdxType = int32_t;

constexpr BlockIdxType NULL_BLOCK_IDX = static_cast<BlockIdxType>(-1);

inline bool isNullBlockIdx(BlockIdxType block_idx) {
    return block_idx == NULL_BLOCK_IDX;
}

// Legacy block tables and wire records use block 0 as their missing/default
// value. Keep NULL_BLOCK_IDX internal to request-owned sparse metadata.
inline BlockIdxType toLegacyBlockIdx(BlockIdxType block_idx) {
    RTP_LLM_CHECK_WITH_INFO(block_idx >= NULL_BLOCK_IDX, "invalid internal block id=%d", block_idx);
    return isNullBlockIdx(block_idx) ? 0 : block_idx;
}

using CacheKeysType    = std::vector<CacheKeyType>;
using BlockIndicesType = std::vector<BlockIdxType>;

struct BlockDependency {
    // Dependency metadata belongs to the request's global cache-key timeline. Filtered resource views preserve the
    // original ordinal and may retain a parent_key that is absent from the view so prefix-tree caches can attach it
    // when the parent becomes available.
    bool         has_parent{false};
    CacheKeyType parent_key{0};
    uint32_t     ordinal{0};
};

using BlockDependenciesType = std::vector<BlockDependency>;

// Request-owned physical block bindings. The vector index is the GroupBlockPosition
// for the owning cache group. Consumer projections are derived outside this type.
class PoolBlockIds {
public:
    PoolBlockIds() = default;

    size_t blocksNum() const;

    const std::vector<BlockIdxType>& blocks() const;

    // Remove and return the last physical block ID.
    BlockIdxType popBack();

    // Append new physical block IDs to the tail.
    void add(const std::vector<BlockIdxType>& ids);
    void remove(const std::vector<size_t>& indices);

    // Swap the physical block IDs at positions pos_a and pos_b.
    void swap(size_t pos_a, size_t pos_b);

    void assign(const std::vector<BlockIdxType>& new_block_indices);
    void assign(std::vector<BlockIdxType>&& new_block_indices);
    void setAt(size_t pos, BlockIdxType val);

    void resize(size_t new_size, BlockIdxType value = NULL_BLOCK_IDX);

private:
    std::vector<BlockIdxType> block_ids_;
};

// Derived view from physical pool IDs to a consumer's kernel block table. This
// object owns geometry only; project() writes caller-owned transient storage.
class PoolBlockToKernelBlockProjection {
public:
    explicit PoolBlockToKernelBlockProjection(size_t kernel_blocks_per_pool_block);

    size_t projectedSize(size_t pool_block_count) const;
    void   append(BlockIdxType source, std::vector<BlockIdxType>& destination) const;
    void   project(const std::vector<BlockIdxType>& source, std::vector<BlockIdxType>& destination) const;

private:
    size_t kernel_blocks_per_pool_block_;
};

class KVCacheResource {
public:
    void initGroups(const CacheConfig& config);
    void resizeBlocks(int reserver_blocks, int value = 0);

    int                     blocksNum(std::string_view tag) const;
    const BlockIndicesType& blocks(std::string_view tag) const;
    const BlockIndicesType& blocksForLayer(int layer_id, std::string_view tag) const;
    PoolBlockIds&           mutableBlockIds(std::string_view tag) const;
    PoolBlockIds&           mutableBlockIdsForLayer(int layer_id, std::string_view tag) const;

    const PoolBlockIds& blockIds(std::string_view tag) const;
    const PoolBlockIds& blockIdsForLayer(int layer_id, std::string_view tag) const;

    const std::vector<std::string>& groupTagsForLayer(int layer_id) const;
    const std::string&              soleGroupTagForLayer(int layer_id) const;

    int layerNum() const;
    int groupNums() const;

    // Group-owned physical bindings. The string key is the stable group tag and
    // scopes each numeric block ID.
    const std::map<std::string, PoolBlockIds>& blocksByGroup() const;

    bool layerOwnsTag(int layer_id, std::string_view tag) const;

    const CacheKeysType& cacheKeys() const;
    void                 setCacheKeysAndBlockDependencies(CacheKeysType keys, BlockDependenciesType dependencies);
    void                 setCacheKeys(CacheKeysType keys);
    bool                 cacheKeysAreCpCanonical() const;
    void                 setCacheKeysAreCpCanonical(bool cache_keys_are_cp_canonical);
    void                 appendCacheKey(CacheKeyType key);
    void                 popBackCacheKey();
    void                 clearCacheKeys();

    const BlockDependenciesType& blockDependencies() const;

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

    size_t remoteReuseBlocksNum() const;
    void   setRemoteReuseBlocksNum(size_t remote_reuse_blocks_num);

    void swapBlocks(std::string_view tag, size_t rhs, size_t lhs);

    std::string debugString() const;

private:
    bool layerContainsTag(int layer_id, std::string_view tag) const;

    std::vector<std::vector<std::string>>       layer_group_tags_;
    mutable std::map<std::string, PoolBlockIds> blocks_by_group_;
    CacheKeysType                               cache_keys;
    BlockDependenciesType                       block_dependencies;
    bool                                        cache_keys_are_cp_canonical_{false};

    size_t device_reuse_block_num_{0};
    size_t memory_reuse_block_num_{0};
    size_t remote_reuse_block_num_{0};
    bool   last_block_aligned_{false};
};

using KVCacheResourcePtr = std::shared_ptr<KVCacheResource>;

}  // namespace rtp_llm
