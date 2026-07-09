#include "rtp_llm/cpp/cache/allocator/HybridTypeKVCacheAllocator.h"

#include <algorithm>
#include <utility>

#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/block_tree_cache/DeviceBlockPool.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

HybridTypeKVCacheAllocator::HybridTypeKVCacheAllocator(const CacheConfig&                 config,
                                                       AllocationType                     allocation_type,
                                                       const kmonitor::MetricsReporterPtr metrics_reporter,
                                                       int64_t                            reserve_block_ratio):
    HybridKVCacheAllocator(config, allocation_type, metrics_reporter, reserve_block_ratio) {}

bool HybridTypeKVCacheAllocator::doInit() {
    RTP_LLM_CHECK_WITH_INFO(config_.groupNums() > 0, "no cache groups found in CacheConfig");

    BlockPoolConfig pool_config = BlockPoolConfigHelper::createConfig(config_);

    auto device_config                     = std::make_shared<DeviceBlockPoolConfig>();
    device_config->pool_type               = BlockPoolType::DEVICE;
    device_config->pool_name               = pool_config.pool_name;
    device_config->physical_block_count    = pool_config.block_num;
    device_config->total_size_bytes        = pool_config.total_size_bytes;
    device_config->memory_layouts          = pool_config.memory_layouts;
    device_config->allocation_type         = allocation_type_;
    device_config->use_cuda_malloc_backing = use_cuda_malloc_block_pool_;

    std::shared_ptr<const DeviceBlockPoolConfig> const_config = device_config;
    block_pool_                                               = std::make_shared<DeviceBlockPool>(const_config);
    RTP_LLM_CHECK_WITH_INFO(block_pool_->init(), "Failed to initialize block pool for HybridTypeKVCacheAllocator");

    const int group_nums = config_.groupNums();
    kv_cache_groups_.reserve(group_nums);

    SharedBlockCache* shared_cache_raw = shared_block_cache_ ? shared_block_cache_.get() : nullptr;

    if (shared_block_cache_) {
        std::vector<DeviceBlockPoolPtr> group_pools(static_cast<size_t>(group_nums), block_pool_);
        shared_block_cache_->init(group_nums, group_pools);
    }

    for (int gid = 0; gid < group_nums; ++gid) {
        KVCacheSpecPtr spec = config_.specForGroup(static_cast<size_t>(gid));
        const auto&    ids  = config_.layerIdsForGroup(static_cast<size_t>(gid));

        KVCacheGroupPtr group;
        const auto      group_type = config_.typeForGroup(static_cast<size_t>(gid));
        const auto      policy     = config_.policyForGroup(static_cast<size_t>(gid));
        if (group_type == CacheGroupType::SWA) {
            group = std::make_shared<SWAKVCacheGroup>(
                ids, spec, block_pool_, gid, config_.linear_step, shared_cache_raw, nullptr, policy);
            swa_group_ids_.push_back(gid);
        } else if (group_type == CacheGroupType::LINEAR || (spec && spec->type == KVCacheSpecType::LinearAttention)) {
            group = std::make_shared<LinearKVCacheGroup>(
                ids, spec, block_pool_, gid, config_.linear_step, shared_cache_raw, nullptr, policy);
            linear_group_ids_.push_back(gid);
        } else {
            group = std::make_shared<FullKVCacheGroup>(ids, spec, block_pool_, gid, shared_cache_raw, nullptr, policy);
            full_group_ids_.push_back(gid);
        }

        RTP_LLM_CHECK_WITH_INFO(group->init(), "Failed to initialize KVCacheGroup gid %d", gid);
        kv_cache_groups_.push_back(group);
    }

    global_layer_to_local_id_.assign(static_cast<size_t>(config_.layer_all_num), -1);
    for (int gid = 0; gid < group_nums; ++gid) {
        const auto& cur_group_layers = config_.layerIdsForGroup(static_cast<size_t>(gid));
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
    // Single-count shared pool: request/connector holders share one reference category.
    (void)gid;
    (void)is_connector;
    block_pool_->incRef(blocks);
}

void HybridTypeKVCacheAllocator::freeBlocksInGroup(int gid, const BlockIndicesType& blocks, bool is_connector) {
    (void)gid;
    (void)is_connector;
    block_pool_->releaseRef(blocks);
}

CacheLayerLayout HybridTypeKVCacheAllocator::allLayerCacheBase() const {
    CacheLayerLayout layout;
    const auto       layer_tensors = block_pool_->allLayerCacheBase();
    const auto       scale_tensors = block_pool_->allLayerScaleCacheBase();

    layout.layer_to_group_ids = config_.layerGroupIdsSnapshot();
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

int HybridTypeKVCacheAllocator::defaultGroupIdForLayer(int layer_id) const {
    if (layer_id < 0 || static_cast<size_t>(layer_id) >= config_.layer_all_num) {
        RTP_LLM_FAIL("invalid layer_id=%d", layer_id);
    }
    const int gid = config_.groupIdFor(layer_id);
    RTP_LLM_CHECK_WITH_INFO(gid >= 0 && gid < static_cast<int>(kv_cache_groups_.size()), "invalid group id mapping");
    return gid;
}

int HybridTypeKVCacheAllocator::validateGroupIdForLayer(int layer_id, int group_id) const {
    RTP_LLM_CHECK_WITH_INFO(group_id >= 0 && group_id < static_cast<int>(kv_cache_groups_.size()),
                            "invalid group id %d for layer %d",
                            group_id,
                            layer_id);
    RTP_LLM_CHECK_WITH_INFO(layer_id >= 0 && static_cast<size_t>(layer_id) < config_.layer_all_num,
                            "invalid layer id %d for layer_all_num=%u",
                            layer_id,
                            config_.layer_all_num);
    const auto& group_ids = config_.groupIdsForLayer(layer_id);
    RTP_LLM_CHECK_WITH_INFO(std::find(group_ids.begin(), group_ids.end(), group_id) != group_ids.end(),
                            "layer %d does not own cache group %d",
                            layer_id,
                            group_id);
    return group_id;
}

BlockAddrInfo HybridTypeKVCacheAllocator::convertIndexToAddr(int layer_id, int block_id) const {
    if (layer_id < 0 || static_cast<size_t>(layer_id) >= config_.layer_all_num) {
        RTP_LLM_FAIL("convertIndexToAddr invalid layer_id=%d", layer_id);
    }
    const int gid = defaultGroupIdForLayer(layer_id);
    return kv_cache_groups_[static_cast<size_t>(gid)]->convertIndexToAddr(layer_id, block_id);
}

std::vector<BlockInfo> HybridTypeKVCacheAllocator::convertIndexToBuffer(int layer_id, int block_id) const {
    if (layer_id < 0 || static_cast<size_t>(layer_id) >= config_.layer_all_num) {
        RTP_LLM_FAIL("convertIndexToBuffer invalid layer_id=%d", layer_id);
    }
    const int gid = defaultGroupIdForLayer(layer_id);
    return kv_cache_groups_[static_cast<size_t>(gid)]->convertIndexToBuffer(layer_id, block_id);
}

std::vector<BlockInfo> HybridTypeKVCacheAllocator::convertIndexToBuffer(int layer_id,
                                                                        int block_id,
                                                                        int partition_count,
                                                                        int partition_id) const {
    if (layer_id < 0 || static_cast<size_t>(layer_id) >= config_.layer_all_num) {
        RTP_LLM_FAIL("convertIndexToBuffer(partition) invalid layer_id=%d", layer_id);
    }
    const int gid = defaultGroupIdForLayer(layer_id);
    return kv_cache_groups_[static_cast<size_t>(gid)]->convertIndexToBuffer(
        layer_id, block_id, partition_count, partition_id);
}

BlockAddrInfo HybridTypeKVCacheAllocator::convertIndexToAddr(int layer_id, int group_id, int block_id) const {
    const int gid = validateGroupIdForLayer(layer_id, group_id);
    return kv_cache_groups_[static_cast<size_t>(gid)]->convertIndexToAddr(layer_id, block_id);
}

std::vector<BlockInfo>
HybridTypeKVCacheAllocator::convertIndexToBuffer(int layer_id, int group_id, int block_id) const {
    const int gid = validateGroupIdForLayer(layer_id, group_id);
    return kv_cache_groups_[static_cast<size_t>(gid)]->convertIndexToBuffer(layer_id, block_id);
}

std::vector<BlockInfo> HybridTypeKVCacheAllocator::convertIndexToBuffer(
    int layer_id, int group_id, int block_id, int partition_count, int partition_id) const {
    const int gid = validateGroupIdForLayer(layer_id, group_id);
    return kv_cache_groups_[static_cast<size_t>(gid)]->convertIndexToBuffer(
        layer_id, block_id, partition_count, partition_id);
}

}  // namespace rtp_llm
