#include "rtp_llm/cpp/cache/KVCacheGroup.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

bool KVCacheGroup::init() {
    auto layer_tensors = block_pool_->allLayerCacheBase();
    auto scale_tensors = block_pool_->allLayerScaleCacheBase();

    const auto& layer_ids = cache_group_.layer_ids;
    RTP_LLM_CHECK_WITH_INFO(layer_tensors.size() >= layer_ids.size(),
                            "layer_tensors size (%zu) is less than layer_ids size (%zu)",
                            layer_tensors.size(),
                            layer_ids.size());
    RTP_LLM_CHECK_WITH_INFO(scale_tensors.size() >= layer_ids.size(),
                            "scale_tensors size (%zu) is less than layer_ids size (%zu)",
                            scale_tensors.size(),
                            layer_ids.size());

    for (int i = 0; i < static_cast<int>(layer_ids.size()); ++i) {
        const int global_layer_id                   = layer_ids[static_cast<size_t>(i)];
        global_layer_to_kv_tensors[global_layer_id] = layer_tensors[static_cast<size_t>(i)];

        if (!scale_tensors.empty()) {
            global_layer_to_kv_scale_tensors[global_layer_id] = scale_tensors[static_cast<size_t>(i)];
        }
        global_layer_to_local_layer[layer_ids[static_cast<size_t>(i)]] = i;
    }

    return true;
}

bool KVCacheGroup::ensureFreeBlocks(int required_blocks) {
    if (required_blocks <= 0) {
        return true;
    }

    while (true) {
        const auto free_blocks = block_pool_->freeBlocksNum();
        if (free_blocks >= static_cast<size_t>(required_blocks)) {
            break;
        }

        if (!shared_cache_) {
            RTP_LLM_LOG_WARNING(
                "ensure free blocks failed, no shared cache, free blocks: %zu, need: %d", free_blocks, required_blocks);
            return false;
        }

        const size_t                  need_evict = static_cast<size_t>(required_blocks) - free_blocks;
        SharedBlockCache::EvictResult evict_result;
        size_t                        freed = shared_cache_->evictAndFreeForGroup(group_id_, need_evict, &evict_result);

        if (metrics_reporter_) {
            for (const auto& [cache_key, lifetime_ms] : evict_result.evicted_lifetime_ms) {
                RtpLLMCacheEvictionMetricsCollector collector;
                collector.lifetime_ms = lifetime_ms;
                kmonitor::MetricsTags tags("scope", "gpu");
                tags.AddTag("evict_policy",
                            evict_result.evicted_independent_group.count(cache_key) ? "independent" : "chain");
                tags.AddTag("backing", "device");
                metrics_reporter_->report<RtpLLMCacheEvictionMetrics, RtpLLMCacheEvictionMetricsCollector>(&tags,
                                                                                                           &collector);
            }
        }

        if (freed == 0) {
            RTP_LLM_LOG_WARNING("ensure free blocks failed, free blocks: %zu, need evict blocks: %zu",
                                block_pool_->freeBlocksNum(),
                                need_evict);
            return false;
        }
    }

    return true;
}

MatchResult KVCacheGroup::match(const CacheKeysType& cache_keys) {
    return matchPrefix(cache_keys);
}

MatchResult KVCacheGroup::matchPrefix(const CacheKeysType& /*cache_keys*/) const {
    RTP_LLM_FAIL("KVCacheGroup tag=%s does not support prefix matching", tag().c_str());
    return {};
}

MatchResult KVCacheGroup::matchSingleKey(CacheKeyType /*cache_key*/) const {
    RTP_LLM_FAIL("KVCacheGroup tag=%s does not support single-key matching", tag().c_str());
    return {};
}

void KVCacheGroup::insertIntoCache(const CacheKeysType&    cache_keys,
                                   const BlockIndicesType& block_indices,
                                   bool                    is_resident) {
    if (!shared_cache_) {
        return;
    }

    const size_t block_num = std::min(cache_keys.size(), block_indices.size());
    for (size_t i = 0; i < block_num; ++i) {
        if (isNullBlockIdx(block_indices[i])) {
            continue;
        }
        std::vector<BlockIdxType> group_slots(static_cast<size_t>(group_id_ + 1), NULL_BLOCK_IDX);
        group_slots[static_cast<size_t>(group_id_)] = block_indices[i];
        shared_cache_->put(cache_keys[i], group_slots, is_resident);
    }
}

size_t KVCacheGroup::freeBlocksNum() const {
    return block_pool_->freeBlocksNum();
}

int KVCacheGroup::seqSizePerBlock() const {
    return static_cast<int>(cache_group_.seq_size_per_block);
}

const std::string& KVCacheGroup::tag() const {
    return cache_group_.tag;
}

const GroupBase& KVCacheGroup::config() const {
    return cache_group_;
}

int KVCacheGroup::group_id() const {
    return group_id_;
}

const CacheGroupPolicy& KVCacheGroup::policy() const {
    return cache_group_.policy;
}

bool KVCacheGroup::prefixReuseEnabled() const {
    return policy().enable_prefix_reuse;
}

CacheEvictPolicy KVCacheGroup::evictPolicy() const {
    return policy().evict_policy;
}

uint32_t KVCacheGroup::explicitBlockNum() const {
    return policy().explicit_block_num;
}

size_t KVCacheGroup::activeTailBlocks() const {
    return policy().active_tail_blocks > 0 ? static_cast<size_t>(policy().active_tail_blocks) : 0;
}

std::unordered_map<int, torch::Tensor> KVCacheGroup::allLayerCacheBase() const {
    return global_layer_to_kv_tensors;
}

std::unordered_map<int, torch::Tensor> KVCacheGroup::allLayerScaleCacheBase() const {
    return global_layer_to_kv_scale_tensors;
}

BlockAddrInfo KVCacheGroup::convertIndexToAddr(int layer_id, int block_id) const {
    auto it = global_layer_to_local_layer.find(layer_id);
    RTP_LLM_CHECK_WITH_INFO(it != global_layer_to_local_layer.end(), "invalid layer_id: " + std::to_string(layer_id));
    int local_layer_id = it->second;
    return block_pool_->convertIndexToAddr(local_layer_id, block_id);
}

std::vector<BlockInfo> KVCacheGroup::convertIndexToBuffer(int layer_id, int block_id) const {
    auto it = global_layer_to_local_layer.find(layer_id);
    RTP_LLM_CHECK_WITH_INFO(it != global_layer_to_local_layer.end(), "invalid layer_id: " + std::to_string(layer_id));
    int local_layer_id = it->second;
    return block_pool_->convertIndexToBuffer(local_layer_id, block_id);
}

std::vector<BlockInfo>
KVCacheGroup::convertIndexToBuffer(int layer_id, int block_id, int partition_count, int partition_id) const {
    auto it = global_layer_to_local_layer.find(layer_id);
    RTP_LLM_CHECK_WITH_INFO(it != global_layer_to_local_layer.end(), "invalid layer_id: " + std::to_string(layer_id));
    int local_layer_id = it->second;
    return block_pool_->convertIndexToBuffer(local_layer_id, block_id, partition_count, partition_id);
}

void KVCacheGroup::reference(const BlockIndicesType& new_block_indices) {
    block_pool_->requestReference(new_block_indices);
}

bool KVCacheGroup::prefixReusable() const {
    return policy().enable_prefix_reuse;
}

bool KVCacheGroup::hasSparseSlots() const {
    return policy().group_type != CacheGroupType::FULL;
}

bool KVCacheGroup::hasKernelBlockSubdiv() const {
    return policy().group_type == CacheGroupType::FULL;
}

bool KVCacheGroup::transferTailBlocks() const {
    return activeTailBlocks() > 0;
}

bool KVCacheGroup::isReservable() const {
    return policy().reservable;
}

CacheMemoryPlacement KVCacheGroup::memoryPlacement() const {
    return policy().memory_placement;
}

}  // namespace rtp_llm
