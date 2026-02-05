#pragma once

#include <mutex>
#include <memory>
#include <vector>
#include <sstream>
#include "rtp_llm/cpp/utils/LRUCache.h"
#include "rtp_llm/cpp/cache/Types.h"

namespace rtp_llm {

const size_t kCacheMaxCapacity = 10000000;

using CacheKeyGroupPair = std::pair<CacheKeyType, GroupIdType>;  // <cache_key, group_id>

class BlockCache {
public:
    struct CacheItem {
        CacheKeyType cache_key;
        GroupIdType  group_id;
        BlockIdxType block_index;
        bool         is_resident = false;
        int64_t      epoch       = 0;  // Epoch ID: 0 = global visible, >0 = batch-specific
        std::string  debugString() const {
            std::stringstream debug_string;
            debug_string << "CacheItem cache_key: " << cache_key << ", group_id: " << group_id
                         << ", block_index: " << block_index << ", is_resident: " << is_resident
                         << ", epoch: " << epoch;
            return debug_string.str();
        }
    };

    struct BatchMatchResult {
        std::vector<BlockIdxType> matched_indices;
    };

    struct MatchResult {
        BlockIdxType matched_index;
    };

    struct PutResult {
        enum class Action {
            INSERTED,  // 新插入的 item（需要增加新 block 引用计数）
            REPLACED,  // 替换了旧 item（需要减少旧 block，增加新 block 引用计数）
            SKIPPED    // 跳过（epoch 优先级，不需要更新引用计数）
        };

        Action       action;           // 操作类型
        BlockIdxType old_block_index;  // 旧 block index（如果替换，否则为 NULL_BLOCK_IDX）
    };

    using LRUCacheType  = LRUCache<CacheKeyGroupPair,
                                   CacheItem,
                                   PairFirstHash<CacheKeyType, GroupIdType>,
                                   PairBothEqual<CacheKeyType, GroupIdType>>;
    using CacheSnapshot = typename LRUCacheType::CacheSnapshot;

public:
    explicit BlockCache(): lru_cache_(kCacheMaxCapacity) {}

    PutResult put(CacheItem& cache_item);

    bool contains(CacheKeyType cache_key, int group_id = 0) const;

    MatchResult match(CacheKeyType cache_key, int group_id = 0, int64_t current_batch_epoch = -1);

    BlockIndicesType pop(int n);

    bool empty() const;

    size_t size() const;

    CacheSnapshot cacheSnapshot(int64_t latest_version) const;

    // Print current cache state for debugging
    void debugString() const;

private:
    size_t       seq_size_per_block_;
    LRUCacheType lru_cache_;
    // NOTE: BlockCache/LRUCache is accessed from multiple RPC/engine threads.
    // LRUCache is NOT thread-safe (unordered_map + list). Guard all accesses here.
    mutable std::mutex mu_;
};

using BlockCachePtr = std::shared_ptr<BlockCache>;

}  // namespace rtp_llm
