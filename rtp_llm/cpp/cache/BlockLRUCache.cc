#include "rtp_llm/cpp/cache/BlockLRUCache.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

BlockLRUMatchResult BlockLRUCache::match(const std::vector<size_t>& cache_keys) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<int>   matched_block_ids;
    std::vector<float> aggregated_losses;
    size_t             matched_len = 0;

    // 从前往后检查，找到第一个不匹配的位置
    for (size_t i = 0; i < cache_keys.size(); ++i) {
        size_t cache_key                = cache_keys[i];
        auto [success, cache_value_ptr] = lru_cache_.get(cache_key);
        if (success) {
            matched_block_ids.push_back(cache_value_ptr->block_id);
            if (!cache_value_ptr->losses.empty()) {
                aggregated_losses.insert(
                    aggregated_losses.end(), cache_value_ptr->losses.begin(), cache_value_ptr->losses.end());
            }
            matched_len++;
        } else {
            break;
        }
    }
    return {matched_len, matched_block_ids, aggregated_losses};
}

std::vector<int> BlockLRUCache::put(const std::vector<size_t>& cache_keys,
                                    const std::vector<int>&    block_ids,
                                    const std::vector<float>&  losses,
                                    bool                       is_resident) {
    // 输入参数验证
    // assume block id is valid
    if (cache_keys.empty() || block_ids.empty() || cache_keys.size() != block_ids.size()) {
        return {};
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // 同一个cache key对应多个block id, 只需要存一个block id
    std::vector<int> duplicate_block_ids;
    int              prev_block_id = -1;

    // 逐个处理每个cache_key和block_id
    for (size_t i = 0; i < cache_keys.size(); ++i) {
        size_t cache_key = cache_keys[i];
        int    block_id  = block_ids[i];

        auto [exists, existing_value] = lru_cache_.get(cache_key);
        if (exists) {
            if (existing_value->block_id != block_id) {
                duplicate_block_ids.push_back(block_id);
                continue;
            }
            if (is_resident && !existing_value->is_resident) {
                existing_value->is_resident = is_resident;
            }
            prev_block_id = existing_value->block_id;
        } else {
            auto cache_value = std::make_shared<MemoryCacheValue>(
                cache_key, block_id, constructLosses(losses, i), is_resident, prev_block_id);
            // 不会触发evict, 所以用put而不是putWithCond
            lru_cache_.put(cache_key, cache_value);
            if (prev_block_id != -1) {
                inner_block_ref_counter_.incrementRefCounter({prev_block_id});
            }
            prev_block_id = block_id;
        }
    }
    return duplicate_block_ids;
}

bool BlockLRUCache::empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return lru_cache_.empty();
}

size_t BlockLRUCache::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return lru_cache_.size();
}

std::vector<int> BlockLRUCache::pop(size_t num) {
    std::vector<int> result;

    std::lock_guard<std::mutex> lock(mutex_);

    auto cond = [&](const size_t& key, const std::shared_ptr<MemoryCacheValue>& value) {
        return !value->is_resident && outer_block_ref_counter_.getRefCounter(value->block_id) == 0
               && inner_block_ref_counter_.getRefCounter(value->block_id) == 0;
    };

    // 从LRU缓存中弹出指定数量的项
    while (!lru_cache_.empty() && result.size() < num) {
        auto [success, cache_value_ptr] = lru_cache_.popWithCond(cond);
        if (success && cache_value_ptr) {
            result.push_back(cache_value_ptr->block_id);
            if (cache_value_ptr->prev_block_id != -1) {
                inner_block_ref_counter_.decrementRefCounter({cache_value_ptr->prev_block_id});
            }
        } else {
            break;
        }
    }
    return result;
}

std::vector<float> BlockLRUCache::constructLosses(const std::vector<float>& losses, size_t block_index) const {
    if (losses.empty()) {
        return {};
    }

    size_t loss_start_idx = block_index * seq_size_per_block_;
    if (loss_start_idx >= losses.size()) {
        return {};
    }

    std::vector<float> block_losses;
    size_t             max_losses = std::min(seq_size_per_block_, losses.size() - loss_start_idx);
    block_losses.reserve(max_losses);

    for (size_t j = 0; j < max_losses; ++j) {
        block_losses.push_back(losses[loss_start_idx + j]);
    }

    return block_losses;
}

uint32_t BlockLRUCache::availableBlockNum() const {
    std::lock_guard<std::mutex> lock(mutex_);
    RTP_LLM_CHECK_WITH_INFO(lru_cache_.size() >= outer_block_ref_counter_.busyBlockNum(),
                            "lru_cache_.size() < outer_block_ref_counter_.busyBlockNum()");
    return lru_cache_.size() - outer_block_ref_counter_.busyBlockNum();
}

void BlockLRUCache::incrBlockRefCounter(const std::vector<int>& block_ids) {
    std::lock_guard<std::mutex> lock(mutex_);
    outer_block_ref_counter_.incrementRefCounter(block_ids);
}

void BlockLRUCache::decrBlockRefCounter(const std::vector<int>& block_ids) {
    std::lock_guard<std::mutex> lock(mutex_);
    outer_block_ref_counter_.decrementRefCounter(block_ids);
}

}  // namespace rtp_llm
