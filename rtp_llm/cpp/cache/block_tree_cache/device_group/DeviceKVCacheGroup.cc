#include "rtp_llm/cpp/cache/block_tree_cache/device_group/DeviceKVCacheGroup.h"

#include <algorithm>

#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

bool DeviceKVCacheGroup::init() {
    auto layer_tensors = block_pool_->allLayerCacheBase();
    auto scale_tensors = block_pool_->allLayerScaleCacheBase();

    // 检查layer_tensors的大小是否足够
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
        // - For non-hybrid (single-model) layout, DeviceBlockPool exposes per-layer tensors indexed by global layer id,
        //   and typically global_layer_id == i.
        // - For hybrid layout, DeviceBlockPool exposes per-group "physical layer slot" tensors sized by
        //   CacheConfig.group_layer_num, while layer_ids_ still stores global model layer ids.
        //   In that case, we must bind global_layer_id -> layer_tensors[local_slot=i].

        global_layer_to_kv_tensors[global_layer_id] = layer_tensors[static_cast<size_t>(i)];

        if (!scale_tensors.empty()) {
            global_layer_to_kv_scale_tensors[global_layer_id] = scale_tensors[static_cast<size_t>(i)];
        }
        global_layer_to_local_layer[layer_ids_[i]] = i;
    }

    return true;
}

bool DeviceKVCacheGroup::ensureFreeBlocks(int required_blocks) {
    if (required_blocks <= 0) {
        return true;
    }

    while (true) {
        const auto free_blocks = block_pool_->freeBlocksNum();
        if (free_blocks >= static_cast<size_t>(required_blocks)) {
            break;
        }

        if (!eviction_fn_) {
            RTP_LLM_LOG_WARNING("ensure free blocks failed, no eviction callback, free blocks: %zu, need: %d",
                                free_blocks,
                                required_blocks);
            return false;
        }

        const size_t need_evict = static_cast<size_t>(required_blocks) - free_blocks;
        const int    freed      = eviction_fn_(group_id_, need_evict);
        if (freed == 0) {
            RTP_LLM_LOG_WARNING("ensure free blocks failed, free blocks: %zu, need evict blocks: %zu",
                                block_pool_->freeBlocksNum(),
                                need_evict);
            return false;
        }
    }

    return true;
}

bool DeviceKVCacheGroup::materializePositions(BlockIds& block_ids, const std::vector<size_t>& positions) {
    std::vector<size_t> missing_positions;
    missing_positions.reserve(positions.size());
    for (size_t position : positions) {
        RTP_LLM_CHECK_WITH_INFO(position < block_ids.blocksNum(),
                                "load-back position out of range, group=%d position=%zu blocks=%zu",
                                group_id_,
                                position,
                                block_ids.blocksNum());
        if (isNullBlockIdx(block_ids.blocks()[position])
            && std::find(missing_positions.begin(), missing_positions.end(), position) == missing_positions.end()) {
            missing_positions.push_back(position);
        }
    }
    if (missing_positions.empty()) {
        return true;
    }

    const int required_blocks = static_cast<int>(missing_positions.size());
    if (!ensureFreeBlocks(required_blocks)) {
        return false;
    }
    auto allocated = block_pool_->malloc(missing_positions.size());
    if (!allocated.has_value() || allocated->size() != missing_positions.size()) {
        return false;
    }
    block_pool_->incRef(*allocated);
    for (size_t i = 0; i < missing_positions.size(); ++i) {
        block_ids.setAt(missing_positions[i], (*allocated)[i]);
    }
    return true;
}

size_t DeviceKVCacheGroup::freeBlocksNum() const {
    return block_pool_->freeBlocksNum();
}

int DeviceKVCacheGroup::seqSizePerBlock() const {
    return seq_size_per_block_;
}

int DeviceKVCacheGroup::group_id() const {
    return group_id_;
}

const CacheGroupPolicy& DeviceKVCacheGroup::policy() const {
    return policy_;
}

CacheReusePolicy DeviceKVCacheGroup::reusePolicy() const {
    return policy_.reuse_policy;
}

CacheEvictPolicy DeviceKVCacheGroup::evictPolicy() const {
    return policy_.evict_policy;
}

uint32_t DeviceKVCacheGroup::explicitBlockNum() const {
    return policy_.explicit_block_num;
}

size_t DeviceKVCacheGroup::activeTailBlocks() const {
    return policy_.active_tail_blocks > 0 ? static_cast<size_t>(policy_.active_tail_blocks) : 0;
}

std::unordered_map<int, torch::Tensor> DeviceKVCacheGroup::allLayerCacheBase() const {
    return global_layer_to_kv_tensors;
}

std::unordered_map<int, torch::Tensor> DeviceKVCacheGroup::allLayerScaleCacheBase() const {
    return global_layer_to_kv_scale_tensors;
}

BlockAddrInfo DeviceKVCacheGroup::convertIndexToAddr(int layer_id, int block_id) const {
    auto it = global_layer_to_local_layer.find(layer_id);
    RTP_LLM_CHECK_WITH_INFO(it != global_layer_to_local_layer.end(), "invalid layer_id: " + std::to_string(layer_id));
    int local_layer_id = it->second;
    return block_pool_->convertIndexToAddr(local_layer_id, block_id);
}

std::vector<BlockInfo> DeviceKVCacheGroup::convertIndexToBuffer(int layer_id, int block_id) const {
    auto it = global_layer_to_local_layer.find(layer_id);
    RTP_LLM_CHECK_WITH_INFO(it != global_layer_to_local_layer.end(), "invalid layer_id: " + std::to_string(layer_id));
    int local_layer_id = it->second;
    return block_pool_->convertIndexToBuffer(local_layer_id, block_id);
}

std::vector<BlockInfo>
DeviceKVCacheGroup::convertIndexToBuffer(int layer_id, int block_id, int partition_count, int partition_id) const {
    auto it = global_layer_to_local_layer.find(layer_id);
    RTP_LLM_CHECK_WITH_INFO(it != global_layer_to_local_layer.end(), "invalid layer_id: " + std::to_string(layer_id));
    int local_layer_id = it->second;
    return block_pool_->convertIndexToBuffer(local_layer_id, block_id, partition_count, partition_id);
}

void DeviceKVCacheGroup::reference(const BlockIndicesType& new_block_indices) {
    block_pool_->incRef(new_block_indices);
}

bool DeviceKVCacheGroup::isCpShardable() const {
    return policy_.is_cp_shardable;
}

bool DeviceKVCacheGroup::prefixReusable() const {
    return policy_.prefix_reusable && policy_.reuse_policy == CacheReusePolicy::REUSABLE;
}

bool DeviceKVCacheGroup::hasSparseSlots() const {
    return policy_.has_sparse_slots;
}

bool DeviceKVCacheGroup::hasKernelBlockSubdiv() const {
    return policy_.has_kernel_block_subdiv;
}

bool DeviceKVCacheGroup::transferTailBlocks() const {
    return activeTailBlocks() > 0;
}

bool DeviceKVCacheGroup::cpCompactTailBlocks() const {
    return policy_.cp_compact_tail_blocks;
}

bool DeviceKVCacheGroup::isReservable() const {
    return policy_.is_reservable;
}

bool DeviceKVCacheGroup::usesPinnedCpuBacking() const {
    return policy_.uses_pinned_cpu_backing;
}

}  // namespace rtp_llm
