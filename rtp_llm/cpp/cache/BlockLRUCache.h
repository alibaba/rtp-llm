#pragma once

#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include <mutex>
#include <unordered_map>
#include <memory>
#include <functional>

#include "rtp_llm/cpp/utils/LRUCache.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/cache/BlockRefCounter.h"

namespace rtp_llm {

// LRU缓存中存储的数据结构
struct MemoryCacheValue {
    size_t             cache_key;           // 对应的cache_key
    int                block_id;            // 对应的内存block_id
    std::vector<float> losses;              // 完整的losses向量
    bool               is_resident;         // 标记是否为resident项，resident项不会被evict
    int                prev_block_id = -1;  // 前序的block_id记录ref

    MemoryCacheValue(const size_t&             cache_key,
                     const int&                block_id,
                     const std::vector<float>& losses,
                     bool                      is_resident,
                     const int&                prev_block_id = -1):
        cache_key(cache_key),
        block_id(block_id),
        losses(losses),
        is_resident(is_resident),
        prev_block_id(prev_block_id) {}
};

// 匹配结果结构
struct BlockLRUMatchResult {
    size_t             matched_len = 0;
    std::vector<int>   block_ids;
    std::vector<float> losses;
};

class BlockLRUCache {
public:
    explicit BlockLRUCache(size_t capacity, size_t seq_size_per_block):
        lru_cache_(capacity),
        seq_size_per_block_(seq_size_per_block),
        outer_block_ref_counter_(capacity),
        inner_block_ref_counter_(capacity) {}

    // 匹配缓存项, 匹配到的block会增加ref counter
    BlockLRUMatchResult match(const std::vector<size_t>& cache_keys);

    // 将数据放入缓存, 放入block会减少ref counter, 这里假设用了match的请求一定会在最后调用put
    // 返回没有缓存住的block_ids
    std::vector<int> put(const std::vector<size_t>& cache_keys,
                         const std::vector<int>&    block_ids,
                         const std::vector<float>&  losses,
                         bool                       is_resident = false);

    // 弹出指定数量的最久未使用的项，返回对应的block_id列表
    std::vector<int> pop(size_t num);

    // 检查是否为空
    bool empty() const;

    // 获取缓存大小
    size_t size() const;

    // 没有被引用的block数量
    uint32_t availableBlockNum() const;

    void incrBlockRefCounter(const std::vector<int>& block_ids);
    void decrBlockRefCounter(const std::vector<int>& block_ids);

private:
    // 根据block_index从losses中提取对应的losses片段
    std::vector<float> constructLosses(const std::vector<float>& losses, size_t block_index) const;

private:
    mutable LRUCache<size_t, std::shared_ptr<MemoryCacheValue>> lru_cache_;
    mutable std::mutex                                          mutex_;
    size_t                                                      seq_size_per_block_;  // 一个Block可以对应多少个loss

    BlockRefCounter outer_block_ref_counter_;  // 被外部ref
    BlockRefCounter inner_block_ref_counter_;  // 被其他block ref
};

}  // namespace rtp_llm
