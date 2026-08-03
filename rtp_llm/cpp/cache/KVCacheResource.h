#pragma once

#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/cache/RequestPrefixResource.h"

namespace rtp_llm {

class CacheTopology;
struct GroupBase;

using CacheKeyType = int64_t;
using BlockIdxType = int32_t;

constexpr BlockIdxType NULL_BLOCK_IDX = static_cast<BlockIdxType>(-1);

inline bool isNullBlockIdx(BlockIdxType block_idx) {
    return block_idx == NULL_BLOCK_IDX;
}

using CacheKeysType    = std::vector<CacheKeyType>;
using CacheKeysByGroup = std::unordered_map<std::string, CacheKeysType>;
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

struct KVCacheGroupResource {
    std::string               tag;
    std::shared_ptr<BlockIds> block_ids;
    CacheKeysType             cache_keys;
    BlockDependenciesType     block_dependencies;
    bool                      cache_keys_are_cp_canonical{false};
    bool                      last_block_aligned{false};
};

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

    const BlockIds&                 blockIds(std::string_view tag) const;
    const BlockIds&                 blockIdsForLayer(int layer_id, std::string_view tag) const;
    const std::vector<std::string>& groupTagsForLayer(int layer_id) const;

    int              layerNum() const;
    int              groupNums() const;
    const GroupBase& groupSpec(std::string_view tag) const;
    size_t           physicalBlockSpan(std::string_view tag) const;

    RequestPrefixResource& requestPrefix() {
        return request_prefix_;
    }
    const RequestPrefixResource& requestPrefix() const {
        return request_prefix_;
    }

    const std::vector<KVCacheGroupResource>& groupResources() const;
    const std::vector<KVCacheGroupResource>& groupBlocks() const {
        return groupResources();
    }

    CacheKeysType&       cacheKeys(std::string_view tag);
    const CacheKeysType& cacheKeys(std::string_view tag) const;
    CacheKeysType&       cacheKeys();
    const CacheKeysType& cacheKeys() const;
    void                 setCacheKeys(std::string_view tag, const CacheKeysType& keys);
    void                 setCacheKeys(std::string_view tag, CacheKeysType&& keys);
    void                 setCacheKeys(const CacheKeysType& keys);
    void                 setCacheKeys(CacheKeysType&& keys);
    bool                 cacheKeysAreCpCanonical(std::string_view tag) const;
    bool                 cacheKeysAreCpCanonical() const {
        return cacheKeysAreCpCanonical(strictSingleGroupTag());
    }
    void setCacheKeysAreCpCanonical(std::string_view tag, bool cache_keys_are_cp_canonical);
    void setCacheKeysAreCpCanonical(bool value) {
        setCacheKeysAreCpCanonical(strictSingleGroupTag(), value);
    }

    BlockDependenciesType&       blockDependencies(std::string_view tag);
    const BlockDependenciesType& blockDependencies(std::string_view tag) const;
    BlockDependenciesType&       blockDependencies();
    const BlockDependenciesType& blockDependencies() const;
    void                         setBlockDependencies(std::string_view tag, const BlockDependenciesType& dependencies);
    void                         setBlockDependencies(std::string_view tag, BlockDependenciesType&& dependencies);
    void                         setBlockDependencies(BlockDependenciesType&& dependencies) {
        setBlockDependencies(strictSingleGroupTag(), std::move(dependencies));
    }
    void rebuildLinearBlockDependencies(std::string_view tag);
    void rebuildLinearBlockDependencies();
    void ensureLinearBlockDependencies(std::string_view tag);
    void ensureLinearBlockDependencies() {
        ensureLinearBlockDependencies(strictSingleGroupTag());
    }

    // Return rank-local cache keys: every cp_size-th key starting from cp_rank.
    // localCacheKeys(tag, r, s)[i] == cacheKeys(tag)[i * s + r]
    // Note: when cacheKeys(tag).size() % cp_size != 0 (e.g. 1 real block, cp_size=2),
    // localCacheKeys may return fewer entries than blocks().size().  This is
    // intentional — padding blocks carry no real data and must NOT participate in
    // device cache insert, PD transfer, or connector operations.  Downstream code
    // (e.g. insertIntoCache) already uses min(keys, blocks) to handle this.
    CacheKeysType localCacheKeys(std::string_view tag, int cp_rank, int cp_size) const {
        CacheKeysType local;
        const auto&   keys = cacheKeys(tag);
        for (int i = cp_rank; i < static_cast<int>(keys.size()); i += cp_size) {
            local.push_back(keys[i]);
        }
        return local;
    }

    size_t reuseTokenNum() const;
    size_t deviceReuseTokenNum() const;
    size_t memoryReuseTokenNum() const;
    size_t remoteReuseTokenNum() const;

    size_t reuseBlockNum() const;
    size_t deviceReuseBlockNum() const;
    size_t memoryReuseBlockNum() const;
    size_t remoteReuseBlockNum() const;

    void setDeviceReuseTokenNum(size_t tokens);
    void setMemoryReuseTokenNum(size_t tokens);
    void setRemoteReuseTokenNum(size_t tokens);

    void setDeviceReuseBlockNum(size_t blocks);
    void setMemoryReuseBlockNum(size_t blocks);
    void setRemoteReuseBlockNum(size_t blocks);

    bool lastBlockAligned(std::string_view tag) const;
    void setLastBlockAligned(std::string_view tag, bool last_block_aligned);
    bool lastBlockAligned() const;
    void setLastBlockAligned(bool last_block_aligned);

    void swapBlocks(std::string_view tag, size_t rhs, size_t lhs);

    std::string debugString() const;

private:
    size_t groupOffset(std::string_view tag) const;
    bool   layerContainsTag(int layer_id, std::string_view tag) const;

    std::shared_ptr<const CacheTopology> topology_;
    KVCacheGroupResource&                groupResource(std::string_view tag);
    const KVCacheGroupResource&          groupResource(std::string_view tag) const;
    const std::string&                   strictSingleGroupTag() const;

    std::vector<KVCacheGroupResource>       group_resources_;
    std::unordered_map<std::string, size_t> group_offset_by_tag_;
    RequestPrefixResource                   request_prefix_;
};

using KVCacheResourcePtr = std::shared_ptr<KVCacheResource>;

}  // namespace rtp_llm
