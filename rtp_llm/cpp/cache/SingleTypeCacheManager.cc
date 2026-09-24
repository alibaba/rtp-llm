#include "rtp_llm/cpp/cache/SingleTypeCacheManager.h"

#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

bool SingleTypeCacheManager::init() {
    auto layer_tensors = block_pool_->allLayerCacheBase();
    auto scale_tensors = block_pool_->allLayerScaleCacheBase();

    const auto layer_count = global_layer_to_local_layer.size();
    RTP_LLM_CHECK_WITH_INFO(layer_tensors.size() >= layer_count,
                            "layer_tensors size (%zu) is less than layer_ids size (%zu)",
                            layer_tensors.size(),
                            layer_count);
    RTP_LLM_CHECK_WITH_INFO(scale_tensors.size() >= layer_count,
                            "scale_tensors size (%zu) is less than layer_ids size (%zu)",
                            scale_tensors.size(),
                            layer_count);

    for (const auto& [global_layer_id, i] : global_layer_to_local_layer) {
        global_layer_to_kv_tensors[global_layer_id] = layer_tensors[static_cast<size_t>(i)];

        if (!scale_tensors.empty()) {
            global_layer_to_kv_scale_tensors[global_layer_id] = scale_tensors[static_cast<size_t>(i)];
        }
    }

    return true;
}

void SingleTypeCacheManager::setEvictCallback(EvictCallback callback) {
    evict_callback_ = std::move(callback);
}

bool SingleTypeCacheManager::ensureFreeBlocks(int required_blocks) {
    if (required_blocks <= 0) {
        return true;
    }

    while (true) {
        const auto free_blocks = block_pool_->freeBlocksNum();
        if (free_blocks >= static_cast<size_t>(required_blocks)) {
            break;
        }

        const size_t need_evict = static_cast<size_t>(required_blocks) - free_blocks;
        if (evict_callback_) {
            if (evict_callback_(need_evict) == 0) {
                RTP_LLM_LOG_DEBUG("ensure free blocks failed, BTC reclaimed no blocks for tag=%s need=%zu",
                                  tag().c_str(),
                                  need_evict);
                return false;
            }
            continue;
        }

        RTP_LLM_LOG_WARNING("ensure free blocks failed, no BlockTree eviction callback for tag=%s, free=%zu, need=%d",
                            tag().c_str(),
                            free_blocks,
                            required_blocks);
        return false;
    }

    return true;
}

size_t SingleTypeCacheManager::freeBlocksNum() const {
    return block_pool_->freeBlocksNum();
}

int SingleTypeCacheManager::seqSizePerBlock() const {
    return static_cast<int>(cache_group_.seqSizePerBlock());
}

const std::string& SingleTypeCacheManager::tag() const {
    return cache_group_.tag;
}

const GroupBase& SingleTypeCacheManager::config() const {
    return cache_group_;
}

int SingleTypeCacheManager::group_id() const {
    return group_id_;
}

const CacheGroupPolicy& SingleTypeCacheManager::policy() const {
    return cache_group_.policy;
}

bool SingleTypeCacheManager::prefixReuseEnabled() const {
    return policy().enable_prefix_reuse;
}

uint32_t SingleTypeCacheManager::explicitBlockNum() const {
    return policy().explicit_block_num;
}

size_t SingleTypeCacheManager::activeTailBlocks() const {
    return policy().active_tail_blocks > 0 ? static_cast<size_t>(policy().active_tail_blocks) : 0;
}

std::unordered_map<int, torch::Tensor> SingleTypeCacheManager::allLayerCacheBase() const {
    return global_layer_to_kv_tensors;
}

std::unordered_map<int, torch::Tensor> SingleTypeCacheManager::allLayerScaleCacheBase() const {
    return global_layer_to_kv_scale_tensors;
}

BlockAddrInfo SingleTypeCacheManager::convertIndexToAddr(int layer_id, int block_id) const {
    auto it = global_layer_to_local_layer.find(layer_id);
    RTP_LLM_CHECK_WITH_INFO(it != global_layer_to_local_layer.end(), "invalid layer_id: " + std::to_string(layer_id));
    int local_layer_id = it->second;
    return block_pool_->convertIndexToAddr(local_layer_id, block_id);
}

std::vector<BlockInfo> SingleTypeCacheManager::convertIndexToBuffer(int layer_id, int block_id) const {
    auto it = global_layer_to_local_layer.find(layer_id);
    RTP_LLM_CHECK_WITH_INFO(it != global_layer_to_local_layer.end(), "invalid layer_id: " + std::to_string(layer_id));
    int local_layer_id = it->second;
    return block_pool_->convertIndexToBuffer(local_layer_id, block_id);
}

std::vector<BlockInfo>
SingleTypeCacheManager::convertIndexToBuffer(int layer_id, int block_id, int partition_count, int partition_id) const {
    auto it = global_layer_to_local_layer.find(layer_id);
    RTP_LLM_CHECK_WITH_INFO(it != global_layer_to_local_layer.end(), "invalid layer_id: " + std::to_string(layer_id));
    int local_layer_id = it->second;
    return block_pool_->convertIndexToBuffer(local_layer_id, block_id, partition_count, partition_id);
}

void SingleTypeCacheManager::reference(BlockIds& block_ids, const BlockIndicesType& new_block_indices) {
    block_ids.add(new_block_indices);
    BlockIndicesType valid_blocks;
    valid_blocks.reserve(new_block_indices.size());
    for (const BlockIdxType block : new_block_indices) {
        if (!isNullBlockIdx(block)) {
            valid_blocks.push_back(block);
        }
    }
    reference(valid_blocks);
}

void SingleTypeCacheManager::reference(const BlockIndicesType& block_indices) {
    if (!block_indices.empty()) {
        block_pool_->incRef(block_indices);
    }
}

void SingleTypeCacheManager::unreference(const BlockIndicesType& block_indices) {
    BlockIndicesType valid_blocks;
    valid_blocks.reserve(block_indices.size());
    for (const BlockIdxType block : block_indices) {
        if (!isNullBlockIdx(block)) {
            valid_blocks.push_back(block);
        }
    }
    if (!valid_blocks.empty()) {
        block_pool_->decRef(valid_blocks);
    }
}

bool SingleTypeCacheManager::prefixReusable() const {
    return policy().enable_prefix_reuse;
}

bool SingleTypeCacheManager::hasSparseSlots() const {
    return policy().group_type != CacheGroupType::FULL;
}

bool SingleTypeCacheManager::hasKernelBlockSubdiv() const {
    return policy().group_type == CacheGroupType::FULL;
}

bool SingleTypeCacheManager::transferTailBlocks() const {
    return activeTailBlocks() > 0;
}

bool SingleTypeCacheManager::isReservable() const {
    return policy().reservable;
}

CacheMemoryPlacement SingleTypeCacheManager::memoryPlacement() const {
    return policy().memory_placement;
}

}  // namespace rtp_llm
