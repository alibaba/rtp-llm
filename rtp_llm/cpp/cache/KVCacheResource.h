#pragma once

#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

class CacheTopology;

using CacheKeyType = int64_t;
using BlockIdxType = int32_t;

constexpr BlockIdxType NULL_BLOCK_IDX = static_cast<BlockIdxType>(-1);

inline bool isNullBlockIdx(BlockIdxType block_idx) {
    return block_idx == NULL_BLOCK_IDX;
}

using CacheKeysType    = std::vector<CacheKeyType>;
using BlockIndicesType = std::vector<BlockIdxType>;

struct BlockDependency {
    bool         has_parent{false};
    CacheKeyType parent_key{0};
    uint32_t     ordinal{0};
};

using BlockDependenciesType = std::vector<BlockDependency>;

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

using GroupBlockIds = std::vector<std::shared_ptr<BlockIds>>;
// Legacy per-layer view. Valid only when each layer maps to exactly one group.
using LayerBlockIds     = std::vector<std::shared_ptr<BlockIds>>;
using LayerAttnBlockIds = std::vector<std::vector<std::shared_ptr<BlockIds>>>;

class KVCacheResource {
public:
    void initGroups(std::shared_ptr<const CacheTopology> topology);
    void resizeBlocks(int reserver_blocks, int value = 0);

    int                     blocksNum(std::string_view tag) const;
    const BlockIndicesType& blocks(std::string_view tag) const;
    const BlockIndicesType& blocksForLayer(int layer_id, std::string_view tag) const;
    const BlockIndicesType& kernelBlocks(std::string_view tag) const;
    const BlockIndicesType& kernelBlocksForLayer(int layer_id, std::string_view tag) const;
    BlockIds&               mutableBlockIds(std::string_view tag) const;
    BlockIds&               mutableBlockIdsForLayer(int layer_id, std::string_view tag) const;

    const BlockIds& blockIds(std::string_view tag) const;
    const BlockIds& blockIdsForLayer(int layer_id, std::string_view tag) const;

    const std::vector<std::string>& groupTagsForLayer(int layer_id) const;
    const std::string&              soleGroupTagForLayer(int layer_id) const;

    int layerNum() const;
    int groupNums() const;

    const std::vector<std::string>& groupTags() const;

    GroupBlockIds&       groupBlocks();
    const GroupBlockIds& groupBlocks() const;

    LayerBlockIds            layerBlocks() const;
    const LayerAttnBlockIds& layerGroupBlocks() const;

    CacheKeysType&       cacheKeys();
    const CacheKeysType& cacheKeys() const;
    void                 setCacheKeys(const CacheKeysType& keys);
    void                 setCacheKeys(CacheKeysType&& keys);
    bool                 cacheKeysAreCpCanonical() const;
    void                 setCacheKeysAreCpCanonical(bool cache_keys_are_cp_canonical);

    BlockDependenciesType&       blockDependencies();
    const BlockDependenciesType& blockDependencies() const;
    void                         setBlockDependencies(const BlockDependenciesType& dependencies);
    void                         setBlockDependencies(BlockDependenciesType&& dependencies);
    void                         rebuildLinearBlockDependencies();
    void                         ensureLinearBlockDependencies();

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
    size_t    groupIndex(std::string_view tag) const;
    BlockIds& mutableBlockIdsByIndex(size_t group_index) const;
    BlockIds& mutableBlockIdsForLayerByIndex(int layer_id, size_t group_index) const;
    bool      hasOneGroupPerLayer() const;

    std::vector<std::string>                group_tags_;
    std::unordered_map<std::string, size_t> tag_to_group_index_;
    std::vector<std::vector<std::string>>   layer_group_tags_;
    // layer_id -> topology group index -> block_indices
    LayerAttnBlockIds layer_group_block_ids;
    // topology group index -> block_indices
    GroupBlockIds         group_block_ids;
    CacheKeysType         cache_keys;
    BlockDependenciesType block_dependencies;
    bool                  cache_keys_are_cp_canonical_{false};

    size_t device_reuse_block_num_{0};
    size_t memory_reuse_block_num_{0};
    size_t remote_reuse_block_num_{0};
    bool   last_block_aligned_{false};
};

using KVCacheResourcePtr = std::shared_ptr<KVCacheResource>;

}  // namespace rtp_llm
