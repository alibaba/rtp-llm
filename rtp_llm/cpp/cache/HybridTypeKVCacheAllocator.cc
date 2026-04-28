#include "rtp_llm/cpp/cache/HybridTypeKVCacheAllocator.h"

#include <algorithm>

#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

HybridTypeKVCacheAllocator::HybridTypeKVCacheAllocator(const CacheConfig&                 config,
                                                       AllocationType                     allocation_type,
                                                       const kmonitor::MetricsReporterPtr metrics_reporter,
                                                       int64_t                            reserve_block_ratio):
    HybridKVCacheAllocator(config, allocation_type, metrics_reporter, reserve_block_ratio) {}

bool HybridTypeKVCacheAllocator::doInit() {
    RTP_LLM_CHECK_WITH_INFO(!config_.cache_specs.empty(), "no cache_specs found in CacheConfig");

    auto pool_config = BlockPoolConfigHelper::createConfig(config_);
    block_pool_      = std::make_shared<BlockPool>(pool_config, allocation_type_);
    RTP_LLM_CHECK_WITH_INFO(block_pool_->init(), "Failed to initialize block pool for HybridTypeKVCacheAllocator");

    const auto& layer_groups = config_.global_layer_ids;
    const int   group_nums   = static_cast<int>(layer_groups.size());
    kv_cache_groups_.reserve(group_nums);

    layer_to_group_id_ = config_.layer_to_group_id;

    for (int gid = 0; gid < group_nums; ++gid) {
        KVCacheSpecPtr spec = config_.cache_specs[static_cast<size_t>(gid)];
        const auto&    ids  = layer_groups[static_cast<size_t>(gid)];

        KVCacheGroupPtr group;
        if (spec && spec->type == KVCacheSpecType::LinearAttention) {
            group = std::make_shared<LinearKVCacheGroup>(ids, spec, block_pool_, gid, config_.linear_step);
            linear_group_ids_.push_back(gid);
        } else {
            group = std::make_shared<FullKVCacheGroup>(ids, spec, block_pool_, gid);
            full_group_ids_.push_back(gid);
        }

        RTP_LLM_CHECK_WITH_INFO(group->init(), "Failed to initialize KVCacheGroup gid %d", gid);
        kv_cache_groups_.push_back(group);
    }

    global_layer_to_local_id_.assign(static_cast<size_t>(config_.layer_all_num), -1);
    for (const auto& cur_group_layers : layer_groups) {
        for (size_t local_layer_idx = 0; local_layer_idx < cur_group_layers.size(); ++local_layer_idx) {
            const int global_layer_idx = cur_group_layers[local_layer_idx];
            if (global_layer_idx >= 0 && static_cast<size_t>(global_layer_idx) < global_layer_to_local_id_.size()) {
                global_layer_to_local_id_[static_cast<size_t>(global_layer_idx)] = static_cast<int>(local_layer_idx);
            }
        }
    }

    RTP_LLM_LOG_INFO("HybridTypeKVCacheAllocator init success");
    return true;
}

void HybridTypeKVCacheAllocator::referenceBlocksInGroup(int                     gid,
                                                        const BlockIndicesType& blocks,
                                                        bool                    is_connector) const {
    (void)gid;
    if (is_connector) {
        block_pool_->connectorReference(blocks);
    } else {
        block_pool_->requestReference(blocks);
    }
}

void HybridTypeKVCacheAllocator::freeBlocksInGroup(int gid, const BlockIndicesType& blocks, bool is_connector) {
    (void)gid;
    if (is_connector) {
        block_pool_->connectorFree(blocks);
    } else {
        block_pool_->requestFree(blocks);
    }
}

CacheLayerLayout HybridTypeKVCacheAllocator::allLayerCacheBase() const {
    CacheLayerLayout layout;
    const auto       layer_tensors = block_pool_->allLayerCacheBase();
    const auto       scale_tensors = block_pool_->allLayerScaleCacheBase();

    layout.layer_to_groups = layer_to_group_id_;
    layout.layers_to_kv_buffer_ptrs.resize(config_.layer_all_num);
    layout.layers_to_scale_buffer_ptrs.resize(config_.layer_all_num);

    for (size_t layer_id = 0; layer_id < static_cast<size_t>(config_.layer_all_num); ++layer_id) {
        int32_t      local     = global_layer_to_local_id_[layer_id];
        const size_t local_idx = static_cast<size_t>(local);

        if (local_idx < layer_tensors.size() && layer_tensors[local_idx].defined()
            && layer_tensors[local_idx].numel() > 0) {
            layout.layers_to_kv_buffer_ptrs[layer_id] = layer_tensors[local_idx];
        }

        if (!scale_tensors.empty() && local_idx < scale_tensors.size() && scale_tensors[local_idx].defined()
            && scale_tensors[local_idx].numel() > 0) {
            layout.layers_to_scale_buffer_ptrs[layer_id] = scale_tensors[local_idx];
        }
    }
    return layout;
}

BlockAddrInfo HybridTypeKVCacheAllocator::convertIndexToAddr(int layer_id, int block_id) const {
    if (layer_id < 0 || layer_id >= static_cast<int>(layer_to_group_id_.size())) {
        RTP_LLM_FAIL("convertIndexToAddr invalid layer_id=%d", layer_id);
    }
    const int gid = layer_to_group_id_[static_cast<size_t>(layer_id)];
    RTP_LLM_CHECK_WITH_INFO(gid >= 0 && gid < static_cast<int>(kv_cache_groups_.size()), "invalid group id mapping");
    return kv_cache_groups_[static_cast<size_t>(gid)]->convertIndexToAddr(layer_id, block_id);
}

std::vector<BlockInfo> HybridTypeKVCacheAllocator::convertIndexToBuffer(int layer_id, int block_id) const {
    if (layer_id < 0 || layer_id >= static_cast<int>(layer_to_group_id_.size())) {
        RTP_LLM_FAIL("convertIndexToBuffer invalid layer_id=%d", layer_id);
    }
    const int gid = layer_to_group_id_[static_cast<size_t>(layer_id)];
    RTP_LLM_CHECK_WITH_INFO(gid >= 0 && gid < static_cast<int>(kv_cache_groups_.size()), "invalid group id mapping");
    return kv_cache_groups_[static_cast<size_t>(gid)]->convertIndexToBuffer(layer_id, block_id);
}

std::vector<BlockInfo> HybridTypeKVCacheAllocator::convertIndexToBuffer(int layer_id,
                                                                        int block_id,
                                                                        int partition_count,
                                                                        int partition_id) const {
    if (layer_id < 0 || layer_id >= static_cast<int>(layer_to_group_id_.size())) {
        RTP_LLM_FAIL("convertIndexToBuffer(partition) invalid layer_id=%d", layer_id);
    }
    const int gid = layer_to_group_id_[static_cast<size_t>(layer_id)];
    RTP_LLM_CHECK_WITH_INFO(gid >= 0 && gid < static_cast<int>(kv_cache_groups_.size()), "invalid group id mapping");
    return kv_cache_groups_[static_cast<size_t>(gid)]->convertIndexToBuffer(
        layer_id, block_id, partition_count, partition_id);
}

}  // namespace rtp_llm
