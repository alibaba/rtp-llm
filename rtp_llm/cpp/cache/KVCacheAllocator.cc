#include <algorithm>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"

namespace rtp_llm {

bool KVCacheAllocator::init() {
    RTP_LLM_CHECK_WITH_INFO(doInit(), "init failed");

    // NOTE: `availableBlocksNum()` depends on `block_pool_` and must be queried after `doInit()`.
    const int64_t reserve_ratio = reserve_block_ratio_;
    if (reserve_ratio > 0) {
        const size_t available_blocks = availableBlocksNum();
        const size_t reserve_blocks = static_cast<size_t>(reserve_ratio) * available_blocks / static_cast<size_t>(100);
        reserve_block_num_          = reserve_blocks;
        RTP_LLM_LOG_INFO("KVCacheAllocator set reserve blocks: ratio=%ld%% reserve_blocks=%zu available_blocks=%zu",
                         reserve_ratio,
                         reserve_blocks,
                         available_blocks);
    } else {
        reserve_block_num_ = 0;
    }

    return true;
}

MallocResult KVCacheAllocator::initMalloc(const MallocInfo& malloc_info) {
    auto init_result = initMallocForCommonLen(malloc_info);
    if (!init_result.success) {
        FreeInfo free_info{malloc_info.batch_kv_cache_resource, malloc_info.complete_token_ids};
        free(free_info);
        return init_result;
    }

    auto incr_result = incrMalloc(malloc_info);
    if (!incr_result.success) {
        FreeInfo free_info{malloc_info.batch_kv_cache_resource, malloc_info.complete_token_ids};
        free(free_info);
        return incr_result;
    } else {
        if (metrics_reporter_ && malloc_info.enable_device_cache) {
            int64_t device_input_length = 0;
            if (malloc_info.batch_kv_cache_resource) {
                const auto& cache_keys      = malloc_info.batch_kv_cache_resource->cacheKeys(0);
                size_t      match_keys_size = cache_keys.size();
                device_input_length         = static_cast<int64_t>(match_keys_size) * config_.seq_size_per_block;
            }

            if (device_input_length > 0) {
                RtpLLMDeviceCacheReuseMetricsCollector collector;
                collector.match_cost_time_us    = init_result.match_cost_time_us;
                collector.device_input_length   = device_input_length;
                collector.device_reuse_length   = init_result.reuse_len;
                collector.device_cache_hit_rate = static_cast<float>(static_cast<int64_t>(collector.device_reuse_length)
                                                                     * 100 / collector.device_input_length);
                kmonitor::MetricsTags tags;
                metrics_reporter_->report<RtpLLMDeviceCacheReuseMetrics, RtpLLMDeviceCacheReuseMetricsCollector>(
                    &tags, &collector);
            }
        }
        return init_result;
    }
}

MallocResult KVCacheAllocator::malloc(const MallocInfo& malloc_info) {
    if (!malloc_info.batch_kv_cache_resource) {
        RTP_LLM_LOG_ERROR("BatchKVCacheResource is null");
        return {false, 0};
    }

    if (!malloc_info.complete_token_ids) {
        RTP_LLM_LOG_ERROR("CompleteTokenIds is null");
        return {false, 0};
    }

    if (malloc_info.batch_kv_cache_resource->curBlocksNum() == 0) {
        return initMalloc(malloc_info);
    } else {
        return incrMalloc(malloc_info);
    }
}

void KVCacheAllocator::blockCopy(int src_block_index, int dest_block_index) {
    BlockIdPair copy_mapping{src_block_index, dest_block_index};
    blockBatchCopy(&copy_mapping, &copy_mapping + 1);
}

void KVCacheAllocator::blockBatchCopy(const std::vector<BlockIdPair>& copy_mapping) {
    blockBatchCopy(copy_mapping.data(), copy_mapping.data() + copy_mapping.size());
}

void KVCacheAllocator::blockBatchCopy(const Buffer& copy_mapping) {
    RTP_LLM_CHECK(copy_mapping.dim() == 2 && copy_mapping.shape()[1] == 2);
    const auto* begin_ptr = (const BlockIdPair*)copy_mapping.data();
    size_t      copy_num  = copy_mapping.shape()[0];
    blockBatchCopy(begin_ptr, begin_ptr + copy_num);
}

void KVCacheAllocator::blockBatchCopy(const BlockIdPair* begin_ptr, const BlockIdPair* end_ptr) {
    using CopyType = BatchCopyParams::CopyType;

    if (end_ptr == begin_ptr) {
        return;
    }

    BatchCopyParams copy_params;

    const size_t copy_num = (end_ptr - begin_ptr) * config_.layer_num;

    size_t copy_nums[CopyType::TYPE_SIZE] = {};
    auto   copy_type                      = BatchCopyParams::get_copy_type(
        allocation_type_ == AllocationType::DEVICE ? rtp_llm::MEMORY_GPU : rtp_llm::MEMORY_CPU,
        allocation_type_ == AllocationType::DEVICE ? rtp_llm::MEMORY_GPU : rtp_llm::MEMORY_CPU);
    copy_nums[copy_type] += copy_num;  // for kv

    for (size_t i = 0; i < CopyType::TYPE_SIZE; ++i) {
        copy_params.reserve(static_cast<CopyType>(i), copy_nums[i]);
    }

    auto&  spec                = config_.cache_specs[0];
    size_t kv_block_size_bytes = spec->block_size_bytes();

    for (auto it = begin_ptr; it != end_ptr; ++it) {
        auto [src_block_index, dest_block_index] = *it;

        for (int layer_id = 0; layer_id < config_.layer_num; layer_id++) {
            auto src_addr_info = convertIndexToAddr(layer_id, src_block_index);
            auto dst_addr_info = convertIndexToAddr(layer_id, dest_block_index);

            if (!src_addr_info.kv_addr || !dst_addr_info.kv_addr) {
                RTP_LLM_LOG_ERROR("Failed to get block address for layer %d, src_block %d, dst_block %d",
                                  layer_id,
                                  src_block_index,
                                  dest_block_index);
                continue;
            }

            copy_params.add(dst_addr_info.kv_addr, src_addr_info.kv_addr, kv_block_size_bytes, copy_type);

            if (src_addr_info.kv_scale_addr && dst_addr_info.kv_scale_addr) {
                copy_params.add(dst_addr_info.kv_scale_addr,
                                src_addr_info.kv_scale_addr,
                                static_cast<size_t>(config_.kv_scale_stride_bytes),
                                copy_type);
            }
        }
    }

    device_->batchCopy(copy_params);
}

size_t KVCacheAllocator::freeBlocksNum() const {
    return block_pool_ ? block_pool_->freeBlocksNum() : 0;
}

int64_t KVCacheAllocator::getMrCostTimeMs() const {
    return block_pool_ ? block_pool_->getMrCostTimeMs() : 0;
}

size_t KVCacheAllocator::availableBlocksNum() const {
    return block_pool_ ? block_pool_->availableBlocksNum() : 0;
}

BatchKVCacheResourcePtr KVCacheAllocator::popBlocksFromCache(size_t min_blocks_to_free) {
    if (!block_pool_ || min_blocks_to_free == 0) {
        return nullptr;
    }

    auto block_cache = block_pool_->blockCache();
    if (!block_cache) {
        return nullptr;
    }

    const auto snapshot = block_cache->cacheSnapshot(std::numeric_limits<int64_t>::min());
    if (snapshot.values.empty()) {
        return nullptr;
    }

    std::unordered_map<CacheKeyType, std::vector<BlockCache::CacheItem>> grouped_items;
    std::unordered_set<CacheKeyType>                                     resident_keys;
    std::vector<CacheKeyType>                                            lru_keys;
    grouped_items.reserve(snapshot.values.size());
    resident_keys.reserve(snapshot.values.size());
    lru_keys.reserve(snapshot.values.size());

    for (const auto& item : snapshot.values) {
        if (item.is_resident) {
            resident_keys.insert(item.cache_key);
        }
    }
    for (auto it = snapshot.values.rbegin(); it != snapshot.values.rend(); ++it) {
        const auto& item = *it;
        if (item.is_resident) {
            continue;
        }
        auto [iter, inserted] = grouped_items.try_emplace(item.cache_key);
        if (inserted) {
            lru_keys.push_back(item.cache_key);
        }
        iter->second.push_back(item);
    }

    std::vector<CacheKeyType> selected_keys;
    size_t                    selected_blocks = 0;
    for (const auto cache_key : lru_keys) {
        if (resident_keys.find(cache_key) != resident_keys.end()) {
            continue;
        }
        const auto group_it = grouped_items.find(cache_key);
        if (group_it == grouped_items.end() || group_it->second.empty()) {
            continue;
        }
        selected_keys.push_back(cache_key);
        selected_blocks += group_it->second.size();
        if (selected_blocks >= min_blocks_to_free) {
            break;
        }
    }
    if (selected_keys.empty()) {
        return nullptr;
    }

    auto batch_resource = std::make_shared<BatchKVCacheResource>();
    batch_resource->resetBatchSize(1);
    batch_resource->initGroups(config_.groupNums(), static_cast<int>(config_.layer_all_num), config_.layer_to_group_id);
    batch_resource->setLastBlockAligned(true);

    for (int gid = 0; gid < config_.groupNums(); ++gid) {
        batch_resource->mutableBlocks(0, gid).reserve(selected_keys.size());
    }

    for (const auto cache_key : selected_keys) {
        batch_resource->pushBackCacheKey(0, cache_key);
        for (int gid = 0; gid < config_.groupNums(); ++gid) {
            batch_resource->mutableBlocks(0, gid).push_back(NULL_BLOCK_IDX);
        }
        auto& items = grouped_items.at(cache_key);
        for (const auto& item : items) {
            auto removed = block_cache->remove(item.cache_key, item.group_id);
            if (!removed.has_value()) {
                continue;
            }
            auto& group_blocks = batch_resource->mutableBlocks(0, item.group_id);
            RTP_LLM_CHECK_WITH_INFO(!group_blocks.empty(),
                                    "group blocks must be initialized before assigning removed cache item");
            group_blocks.back() = removed->block_index;
        }
    }
    return batch_resource;
}

void KVCacheAllocator::blockCacheFree(const BatchKVCacheResourcePtr& batch_kv_cache_resource) {
    if (!block_pool_ || !batch_kv_cache_resource) {
        return;
    }

    BlockIndicesType                 blocks_to_free;
    std::unordered_set<BlockIdxType> seen_blocks;
    for (int batch_id = 0; batch_id < batch_kv_cache_resource->batchSize(); ++batch_id) {
        for (int gid = 0; gid < batch_kv_cache_resource->groupNums(); ++gid) {
            for (const auto block_idx : batch_kv_cache_resource->blocks(batch_id, gid)) {
                if (isNullBlockIdx(block_idx) || !seen_blocks.insert(block_idx).second) {
                    continue;
                }
                blocks_to_free.push_back(block_idx);
            }
        }
    }
    if (!blocks_to_free.empty()) {
        block_pool_->blockCacheFree(blocks_to_free);
    }
}

size_t KVCacheAllocator::requestRefBlocksNum() const {
    return block_pool_->requestRefBlocksNum();
}

size_t KVCacheAllocator::connectorRefBlocksNum() const {
    return block_pool_->connectorRefBlocksNum();
}

size_t KVCacheAllocator::blockCacheRefBlocksNum() const {
    return block_pool_ ? block_pool_->blockCacheRefBlocksNum() : 0;
}

size_t KVCacheAllocator::notInUseBlocksNum() const {
    return block_pool_ ? block_pool_->notInUseBlocksNum() : 0;
}

size_t KVCacheAllocator::availableTokensNum() const {
    return block_pool_ ? (block_pool_->availableBlocksNum() * seqSizePerBlock()) : 0;
}

size_t KVCacheAllocator::totalBlocksNum() const {
    return block_pool_ ? block_pool_->totalBlocksNum() : 0;
}

size_t KVCacheAllocator::maxAvailableTokensNum() const {
    return block_pool_ ? (block_pool_->totalBlocksNum() * seqSizePerBlock()) : 0;
}

void KVCacheAllocator::regUserMr(size_t model_id) {
    if (block_pool_) {
        block_pool_->regUserMr(model_id);
    }
}

std::vector<std::pair<BufferPtr, size_t>> KVCacheAllocator::getAllBuffers() const {
    std::vector<std::pair<BufferPtr, size_t>> results;

    CacheLayerLayout layout = allLayerCacheBase();
    results.reserve(layout.layers_to_kv_buffer_ptrs.size());

    for (const auto& buf : layout.layers_to_kv_buffer_ptrs) {
        if (!buf || buf->sizeBytes() == 0) {
            continue;
        }
        const size_t kv_block_stride_bytes = config_.kv_block_stride_bytes;
        results.emplace_back(buf, kv_block_stride_bytes);
    }

    for (const auto& buf : layout.layers_to_scale_buffer_ptrs) {
        if (!buf || buf->sizeBytes() == 0) {
            continue;
        }
        const size_t kv_scale_stride_bytes = config_.kv_scale_stride_bytes;
        results.emplace_back(buf, kv_scale_stride_bytes);
    }

    return results;
}

}  // namespace rtp_llm
