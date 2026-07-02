#include "rtp_llm/cpp/cache/group/KVCacheGroup.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

bool KVCacheGroup::init() {
    auto layer_tensors = block_pool_->allLayerCacheBase();
    auto scale_tensors = block_pool_->allLayerScaleCacheBase();

    RTP_LLM_CHECK_WITH_INFO(layer_tensors.size() >= layer_ids_.size(),
                            "layer_tensors size (%zu) is less than layer_ids size (%zu)",
                            layer_tensors.size(),
                            layer_ids_.size());
    RTP_LLM_CHECK_WITH_INFO(scale_tensors.size() >= layer_ids_.size(),
                            "scale_tensors size (%zu) is less than layer_ids size (%zu)",
                            scale_tensors.size(),
                            layer_ids_.size());

    for (int i = 0; i < static_cast<int>(layer_ids_.size()); ++i) {
        const int global_layer_id = layer_ids_[i];
        global_layer_to_kv_tensors[global_layer_id] = layer_tensors[static_cast<size_t>(i)];

        if (!scale_tensors.empty()) {
            global_layer_to_kv_scale_tensors[global_layer_id] = scale_tensors[static_cast<size_t>(i)];
        }
        global_layer_to_local_layer[layer_ids_[i]] = i;
    }

    return true;
}

bool KVCacheGroup::ensureFreeBlocks(int required_blocks) {
    if (required_blocks <= 0) {
        return true;
    }

    if (!shared_cache_) {
        RTP_LLM_LOG_WARNING("ensureFreeBlocks called without shared_cache_, cannot evict");
        return false;
    }

    while (true) {
        const auto free_blocks = block_pool_->freeBlocksNum();
        if (free_blocks >= static_cast<size_t>(required_blocks)) {
            break;
        }

        const size_t need_evict = static_cast<size_t>(required_blocks) - free_blocks;
        SharedBlockCache::EvictResult evict_result;
        size_t freed = shared_cache_->evictAndFreeForGroup(group_id_, need_evict, &evict_result);

        if (freed == 0) {
            RTP_LLM_LOG_WARNING("ensure free blocks failed, free blocks : %zu, need evict blocks : %zu",
                                block_pool_->freeBlocksNum(),
                                need_evict);
            return false;
        }

        if (metrics_reporter_) {
            for (const auto& [cache_key, lifetime_ms] : evict_result.evicted_lifetime_ms) {
                RtpLLMCacheEvictionMetricsCollector collector;
                collector.lifetime_ms = lifetime_ms;
                kmonitor::MetricsTags tags("scope", "gpu");
                metrics_reporter_->report<RtpLLMCacheEvictionMetrics, RtpLLMCacheEvictionMetricsCollector>(
                    &tags, &collector);
            }
        }
    }

    return true;
}

MatchResult KVCacheGroup::match(const CacheKeysType& cache_keys) {
    return matchPrefix(cache_keys);
}

MatchResult KVCacheGroup::matchPrefix(const CacheKeysType& /*cache_keys*/) {
    RTP_LLM_FAIL("matchPrefix not implemented for this group type");
    return MatchResult{};
}

MatchResult KVCacheGroup::matchSingleKey(CacheKeyType /*cache_key*/) const {
    RTP_LLM_FAIL("matchSingleKey not implemented for this group type");
    return MatchResult{};
}

size_t KVCacheGroup::freeBlocksNum() const {
    return block_pool_->freeBlocksNum();
}

int KVCacheGroup::seqSizePerBlock() const {
    return seq_size_per_block_;
}

int KVCacheGroup::group_id() const {
    return group_id_;
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

const CacheGroupPolicy& KVCacheGroup::policy() const {
    return policy_;
}

CacheReusePolicy KVCacheGroup::reusePolicy() const {
    return policy_.reuse_policy;
}

CacheEvictPolicy KVCacheGroup::evictPolicy() const {
    return policy_.evict_policy;
}

int KVCacheGroup::explicitBlockNum() const {
    return policy_.explicit_block_num;
}

int KVCacheGroup::activeTailBlocks() const {
    return policy_.active_tail_blocks;
}

bool KVCacheGroup::isCpShardable() const {
    return policy_.is_cp_shardable;
}

bool KVCacheGroup::prefixReusable() const {
    return policy_.prefix_reusable;
}

bool KVCacheGroup::hasSparseSlots() const {
    return policy_.has_sparse_slots;
}

bool KVCacheGroup::hasKernelBlockSubdiv() const {
    return policy_.has_kernel_block_subdiv;
}

bool KVCacheGroup::transferTailBlocks() const {
    return activeTailBlocks() > 0;
}

bool KVCacheGroup::cpCompactTailBlocks() const {
    return policy_.cp_compact_tail_blocks;
}

bool KVCacheGroup::isReservable() const {
    return policy_.is_reservable;
}

bool KVCacheGroup::usesPinnedCpuBacking() const {
    return policy_.uses_pinned_cpu_backing;
}

}  // namespace rtp_llm
