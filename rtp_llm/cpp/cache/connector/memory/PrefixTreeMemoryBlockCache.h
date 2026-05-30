#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/cache/connector/memory/CacheBlockKind.h"
#include "rtp_llm/cpp/cache/connector/memory/MemoryDiskBlockCache.h"

namespace rtp_llm {

class PrefixTreeMemoryBlockCache {
public:
    static constexpr size_t kKindCount = 2;

    struct KindState {
        bool             has_value{false};
        CacheBackingType backing_type{CacheBackingType::MEMORY};
        BlockIdxType     block_index{NULL_BLOCK_IDX};
        int32_t          disk_slot{-1};
        size_t           block_size{0};
        bool             is_resident{false};
        bool             detached{false};
        uint64_t         generation{0};
        uint64_t         last_access_seq{0};
        uint32_t         in_flight_ref{0};
        uint32_t         subtree_ref_count{0};
        std::vector<uint8_t> slot_valid_mask;
    };

    struct CacheItem {
        CacheKeyType     cache_key{0};
        CacheBlockKind   kind{CacheBlockKind::COMPRESSED_KV};
        CacheBackingType backing_type{CacheBackingType::MEMORY};
        BlockIdxType     block_index{NULL_BLOCK_IDX};
        int32_t          disk_slot{-1};
        size_t           block_size{0};
        bool             is_resident{false};
        uint64_t         generation{0};
        std::vector<uint8_t> slot_valid_mask;
    };

    struct MatchResult {
        bool             found{false};
        CacheBackingType backing_type{CacheBackingType::MEMORY};
        BlockIdxType     block_index{NULL_BLOCK_IDX};
        int32_t          disk_slot{-1};
        size_t           block_size{0};
        uint64_t         generation{0};
        std::vector<uint8_t> slot_valid_mask;
    };

    bool contains(CacheKeyType cache_key, CacheBlockKind kind) const;
    bool contains(CacheKeyType cache_key, CacheBlockKind kind, const std::vector<uint8_t>& required_slot_mask) const;
    MatchResult match(CacheKeyType cache_key, CacheBlockKind kind);
    MatchResult match(CacheKeyType cache_key, CacheBlockKind kind, const std::vector<uint8_t>& required_slot_mask);
    MatchResult matchAndMarkInFlight(CacheKeyType cache_key, CacheBlockKind kind);
    MatchResult matchAndMarkInFlight(CacheKeyType                 cache_key,
                                     CacheBlockKind               kind,
                                     const std::vector<uint8_t>& required_slot_mask);

    std::pair<bool, std::optional<CacheItem>>
    putCommitted(CacheKeyType cache_key, const BlockDependency& dependency, const CacheItem& item);
    std::optional<CacheItem> detachIfMatch(CacheKeyType     cache_key,
                                           CacheBlockKind   kind,
                                           CacheBackingType backing_type,
                                           BlockIdxType     expected_block_index,
                                           int32_t          expected_disk_slot,
                                           uint64_t         expected_generation);
    std::optional<CacheItem> releaseInFlight(CacheKeyType     cache_key,
                                             CacheBlockKind   kind,
                                             CacheBackingType backing_type,
                                             BlockIdxType     block_index,
                                             int32_t          disk_slot,
                                             uint64_t         generation);

    std::optional<CacheItem> popOldestEvictable(CacheBlockKind kind);
    std::vector<CacheKeyType> cacheKeys() const;
    size_t size() const;

private:
    struct RetiredItem {
        CacheItem item;
        uint32_t  in_flight_ref{0};
    };

    struct Node {
        CacheKeyType cache_key{0};
        CacheKeyType parent_key{0};
        bool         has_parent{false};
        uint32_t     ordinal{0};
        std::unordered_set<CacheKeyType> children;
        std::array<KindState, kKindCount> kinds;
        std::array<std::vector<RetiredItem>, kKindCount> retired_items;
    };

    struct EvictKey {
        uint64_t     last_access_seq{0};
        CacheKeyType cache_key{0};
        uint64_t     generation{0};

        bool operator<(const EvictKey& other) const {
            if (last_access_seq != other.last_access_seq) {
                return last_access_seq < other.last_access_seq;
            }
            if (cache_key != other.cache_key) {
                return cache_key < other.cache_key;
            }
            return generation < other.generation;
        }
    };

    static size_t kindIndex(CacheBlockKind kind);
    static bool   validKind(CacheBlockKind kind);
    static bool   slotMaskCovers(const std::vector<uint8_t>& stored, const std::vector<uint8_t>& required);

    Node& upsertNodeLocked(CacheKeyType cache_key, const BlockDependency& dependency);
    void  incrementAncestorsLocked(CacheKeyType cache_key, CacheBlockKind kind);
    void  decrementAncestorsLocked(CacheKeyType cache_key, CacheBlockKind kind);
    void  addSubtreeRefsToAncestorsLocked(CacheKeyType ancestor_key, const Node& child);
    void  subtractSubtreeRefsFromAncestorsLocked(CacheKeyType ancestor_key, const Node& child);
    void  detachPendingChildLocked(CacheKeyType parent_key, CacheKeyType child_key);
    void  attachPendingChildrenLocked(Node& node);
    void  touchLocked(Node& node, CacheBlockKind kind);
    void  insertEvictKeyLocked(const Node& node, CacheBlockKind kind);
    void  eraseEvictKeyLocked(const Node& node, CacheBlockKind kind);
    void  refreshEvictKeyLocked(const Node& node, CacheBlockKind kind);
    void  pruneLocked(CacheKeyType cache_key);
    std::optional<CacheItem> toItemLocked(const Node& node, CacheBlockKind kind) const;
    bool isKindLeafLocked(const Node& node, CacheBlockKind kind) const;

private:
    mutable std::shared_mutex mutex_;
    std::unordered_map<CacheKeyType, Node> nodes_;
    std::unordered_map<CacheKeyType, std::unordered_set<CacheKeyType>> pending_children_by_parent_;
    std::array<std::set<EvictKey>, kKindCount> leaf_lru_;
    uint64_t access_seq_{0};
    uint64_t generation_seq_{0};
};

using PrefixTreeMemoryBlockCachePtr = std::shared_ptr<PrefixTreeMemoryBlockCache>;

}  // namespace rtp_llm
