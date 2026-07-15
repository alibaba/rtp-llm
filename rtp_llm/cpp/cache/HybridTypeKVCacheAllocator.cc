#include "rtp_llm/cpp/cache/HybridTypeKVCacheAllocator.h"

#include <algorithm>
#include <utility>

#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

HybridTypeKVCacheAllocator::HybridTypeKVCacheAllocator(const CacheConfig&                 config,
                                                       AllocationType                     allocation_type,
                                                       const kmonitor::MetricsReporterPtr metrics_reporter,
                                                       int64_t                            reserve_block_ratio):
    HybridKVCacheAllocator(config, allocation_type, metrics_reporter, reserve_block_ratio) {}

bool HybridTypeKVCacheAllocator::doInit() {
    RTP_LLM_CHECK_WITH_INFO(config_.groupNums() > 0, "no cache groups found in CacheConfig");
    if (!config_.use_independent_block_pools) {
        const bool has_full_attention = std::any_of(
            config_.topology().groups().begin(), config_.topology().groups().end(), [](const GroupBase& group) {
                return group.policy.group_type == CacheGroupType::FULL && group.spec
                       && (group.spec->type == KVCacheSpecType::MultiHeadAttention
                           || group.spec->type == KVCacheSpecType::MultiHeadLatentAttention);
            });
        RTP_LLM_CHECK_WITH_INFO(has_full_attention,
                                "HybridTypeKVCacheAllocator requires at least one FULL MHA/MLA cache group");
    }

    auto pool_config = BlockPoolConfigHelper::createConfig(config_);
    block_pool_      = std::make_shared<BlockPool>(
        pool_config, allocation_type_, /*use_pinned_cpu_backing=*/false, use_cuda_malloc_block_pool_);
    RTP_LLM_CHECK_WITH_INFO(block_pool_->init(), "Failed to initialize block pool for HybridTypeKVCacheAllocator");

    const int group_nums = config_.groupNums();
    kv_cache_groups_.reserve(group_nums);

    SharedBlockCache* shared_cache_raw = shared_block_cache_ ? shared_block_cache_.get() : nullptr;

    if (shared_block_cache_) {
        std::vector<BlockPoolPtr> group_pools(static_cast<size_t>(group_nums), block_pool_);
        shared_block_cache_->init(group_nums, group_pools);
    }

    for (int gid = 0; gid < group_nums; ++gid) {
        const auto& cache_group = config_.topology().groupBySlot(static_cast<size_t>(gid));
        const auto& spec        = cache_group.spec;

        KVCacheGroupPtr group;
        const auto      group_type = cache_group.policy.group_type;
        if (group_type == CacheGroupType::SWA) {
            group = std::make_shared<SWAKVCacheGroup>(
                cache_group, block_pool_, gid, config_.linear_step, shared_cache_raw, nullptr);
            swa_group_ids_.push_back(gid);
        } else if (group_type == CacheGroupType::LINEAR || (spec && spec->type == KVCacheSpecType::LinearAttention)) {
            group = std::make_shared<LinearKVCacheGroup>(
                cache_group, block_pool_, gid, config_.linear_step, shared_cache_raw, nullptr);
            linear_group_ids_.push_back(gid);
        } else {
            group = std::make_shared<FullKVCacheGroup>(cache_group, block_pool_, gid, shared_cache_raw, nullptr);
            full_group_ids_.push_back(gid);
        }

        RTP_LLM_CHECK_WITH_INFO(group->init(), "Failed to initialize KVCacheGroup gid %d", gid);
        kv_cache_groups_.push_back(group);
    }

    global_layer_to_local_id_.assign(static_cast<size_t>(config_.layer_all_num), -1);
    for (int gid = 0; gid < group_nums; ++gid) {
        const auto& cur_group_layers = config_.topology().groupBySlot(static_cast<size_t>(gid)).layer_ids;
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

GroupedCacheLayerLayout HybridTypeKVCacheAllocator::allLayerCacheBase() const {
    const auto layer_tensors = block_pool_->allLayerCacheBase();
    const auto scale_tensors = block_pool_->allLayerScaleCacheBase();
    const auto topology      = config_.topologyPtr();

    GroupedCacheLayerLayout::GroupLayouts groups;
    for (const auto& group : topology->groups()) {
        std::vector<BlockBufferPtrInfo> layers(topology->layers().size());
        for (int layer_id : group.layer_ids) {
            const auto global = static_cast<size_t>(layer_id);
            RTP_LLM_CHECK_WITH_INFO(global < global_layer_to_local_id_.size(),
                                    "cache group tag=%s invalid global layer=%d",
                                    group.tag.c_str(),
                                    layer_id);
            const int32_t local = global_layer_to_local_id_[global];
            if (local < 0) {
                continue;
            }
            const auto local_idx = static_cast<size_t>(local);
            if (local_idx < layer_tensors.size() && layer_tensors[local_idx].defined()) {
                layers[global].kv_addr = layer_tensors[local_idx];
            }
            if (local_idx < scale_tensors.size() && scale_tensors[local_idx].defined()) {
                layers[global].kv_scale_addr = scale_tensors[local_idx];
            }
        }
        groups.emplace(group.tag, CacheLayerLayout(std::move(layers)));
    }
    return GroupedCacheLayerLayout(topology, std::move(groups));
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
    RTP_LLM_CHECK_WITH_INFO(group_id >= 0, "invalid cache topology slot=%d", group_id);
    return convertIndexToAddrByTag(
        layer_id, config_.topology().groupBySlot(static_cast<size_t>(group_id)).tag, block_id);
}

std::vector<BlockInfo>
HybridTypeKVCacheAllocator::convertIndexToBuffer(int layer_id, int group_id, int block_id) const {
    RTP_LLM_CHECK_WITH_INFO(group_id >= 0, "invalid cache topology slot=%d", group_id);
    return convertIndexToBufferByTag(
        layer_id, config_.topology().groupBySlot(static_cast<size_t>(group_id)).tag, block_id);
}

std::vector<BlockInfo> HybridTypeKVCacheAllocator::convertIndexToBuffer(
    int layer_id, int group_id, int block_id, int partition_count, int partition_id) const {
    RTP_LLM_CHECK_WITH_INFO(group_id >= 0, "invalid cache topology slot=%d", group_id);
    return convertIndexToBufferByTag(layer_id,
                                     config_.topology().groupBySlot(static_cast<size_t>(group_id)).tag,
                                     block_id,
                                     partition_count,
                                     partition_id);
}

BlockAddrInfo
HybridTypeKVCacheAllocator::convertIndexToAddrByTag(int layer_id, const std::string& tag, int block_id) const {
    const auto gid = static_cast<int>(config_.topology().slotForTag(tag));
    validateGroupIdForLayer(layer_id, gid);
    return kv_cache_groups_[static_cast<size_t>(gid)]->convertIndexToAddr(layer_id, block_id);
}

std::vector<BlockInfo>
HybridTypeKVCacheAllocator::convertIndexToBufferByTag(int layer_id, const std::string& tag, int block_id) const {
    const auto gid = static_cast<int>(config_.topology().slotForTag(tag));
    validateGroupIdForLayer(layer_id, gid);
    return kv_cache_groups_[static_cast<size_t>(gid)]->convertIndexToBuffer(layer_id, block_id);
}

std::vector<BlockInfo> HybridTypeKVCacheAllocator::convertIndexToBufferByTag(
    int layer_id, const std::string& tag, int block_id, int partition_count, int partition_id) const {
    const auto gid = static_cast<int>(config_.topology().slotForTag(tag));
    validateGroupIdForLayer(layer_id, gid);
    return kv_cache_groups_[static_cast<size_t>(gid)]->convertIndexToBuffer(
        layer_id, block_id, partition_count, partition_id);
}

}  // namespace rtp_llm
