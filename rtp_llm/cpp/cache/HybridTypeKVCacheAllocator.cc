#include "rtp_llm/cpp/cache/HybridTypeKVCacheAllocator.h"

#include <algorithm>
#include <utility>

#include "rtp_llm/cpp/cache/DeviceBlockPoolConfigHelper.h"
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

    auto pool_config = std::make_shared<DeviceBlockPoolConfig>(DeviceBlockPoolConfigHelper::createConfig(config_));
    pool_config->use_cuda_malloc_backing = use_cuda_malloc_block_pool_;
    block_pool_ = std::make_shared<DeviceBlockPool>(std::shared_ptr<const DeviceBlockPoolConfig>(pool_config));
    RTP_LLM_CHECK_WITH_INFO(block_pool_->init(), "Failed to initialize block pool for HybridTypeKVCacheAllocator");

    const int group_nums = config_.groupNums();
    kv_cache_groups_.reserve(group_nums);

    for (int gid = 0; gid < group_nums; ++gid) {
        const auto& cache_group = config_.topology().groupById(static_cast<size_t>(gid));
        const auto& spec        = cache_group.spec;

        KVCacheGroupPtr group;
        const auto      group_type = cache_group.policy.group_type;
        if (group_type == CacheGroupType::SWA) {
            group = std::make_shared<SWAKVCacheGroup>(cache_group, block_pool_, gid, config_.linear_step, nullptr);
            swa_group_ids_.push_back(gid);
        } else if (group_type == CacheGroupType::LINEAR || (spec && spec->type == KVCacheSpecType::LinearAttention)) {
            group = std::make_shared<LinearKVCacheGroup>(cache_group, block_pool_, gid, config_.linear_step, nullptr);
            linear_group_ids_.push_back(gid);
        } else {
            group = std::make_shared<FullKVCacheGroup>(cache_group, block_pool_, gid, nullptr);
            full_group_ids_.push_back(gid);
        }

        RTP_LLM_CHECK_WITH_INFO(group->init(), "Failed to initialize KVCacheGroup gid %d", gid);
        kv_cache_groups_.push_back(group);
    }

    global_layer_to_local_id_.assign(static_cast<size_t>(config_.layer_all_num), -1);
    for (int gid = 0; gid < group_nums; ++gid) {
        const auto& cur_group_layers = config_.topology().groupById(static_cast<size_t>(gid)).layer_ids;
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
                                                        BlockRefType            ref_type) const {
    (void)gid;
    block_pool_->incRef(blocks, ref_type);
}

void HybridTypeKVCacheAllocator::freeBlocksInGroup(int gid,
                                                   const BlockIndicesType& blocks,
                                                   BlockRefType            ref_type) {
    (void)gid;
    block_pool_->decRef(blocks, ref_type);
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
    const auto& group = config_.topology().soleGroupForLayer(layer_id);
    const int   gid   = static_cast<int>(config_.topology().groupIdForTag(group.tag));
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

std::vector<BlockInfo> HybridTypeKVCacheAllocator::logicalGroupBlockBuffers(int                    group_id,
                                                                            std::vector<BlockInfo> buffers) const {
    const auto& group = config_.topology().groupById(static_cast<size_t>(group_id));
    RTP_LLM_CHECK_WITH_INFO(!buffers.empty(), "cache group tag=%s returned no block buffers", group.tag.c_str());
    RTP_LLM_CHECK_WITH_INFO(
        buffers[0].size_bytes >= group.kv_block_stride_bytes,
        "cache group tag=%s physical kv block is smaller than logical block: physical=%zu logical=%zu",
        group.tag.c_str(),
        buffers[0].size_bytes,
        group.kv_block_stride_bytes);
    buffers[0].size_bytes = group.kv_block_stride_bytes;

    if (group.kv_scale_stride_bytes == 0) {
        buffers.resize(1);
        return buffers;
    }

    RTP_LLM_CHECK_WITH_INFO(
        buffers.size() >= 2, "cache group tag=%s is missing its scale block buffer", group.tag.c_str());
    RTP_LLM_CHECK_WITH_INFO(
        buffers[1].size_bytes >= group.kv_scale_stride_bytes,
        "cache group tag=%s physical scale block is smaller than logical block: physical=%zu logical=%zu",
        group.tag.c_str(),
        buffers[1].size_bytes,
        group.kv_scale_stride_bytes);
    buffers[1].size_bytes = group.kv_scale_stride_bytes;
    buffers.resize(2);
    return buffers;
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
    RTP_LLM_CHECK_WITH_INFO(group_id >= 0, "invalid cache topology group id=%d", group_id);
    return convertIndexToAddrByTag(layer_id, config_.topology().groupById(static_cast<size_t>(group_id)).tag, block_id);
}

std::vector<BlockInfo>
HybridTypeKVCacheAllocator::convertIndexToBuffer(int layer_id, int group_id, int block_id) const {
    RTP_LLM_CHECK_WITH_INFO(group_id >= 0, "invalid cache topology group id=%d", group_id);
    return convertIndexToBufferByTag(
        layer_id, config_.topology().groupById(static_cast<size_t>(group_id)).tag, block_id);
}

std::vector<BlockInfo> HybridTypeKVCacheAllocator::convertIndexToBuffer(
    int layer_id, int group_id, int block_id, int partition_count, int partition_id) const {
    RTP_LLM_CHECK_WITH_INFO(group_id >= 0, "invalid cache topology group id=%d", group_id);
    return convertIndexToBufferByTag(layer_id,
                                     config_.topology().groupById(static_cast<size_t>(group_id)).tag,
                                     block_id,
                                     partition_count,
                                     partition_id);
}

BlockAddrInfo
HybridTypeKVCacheAllocator::convertIndexToAddrByTag(int layer_id, const std::string& tag, int block_id) const {
    const auto gid = static_cast<int>(config_.topology().groupIdForTag(tag));
    validateGroupIdForLayer(layer_id, gid);
    return kv_cache_groups_[static_cast<size_t>(gid)]->convertIndexToAddr(layer_id, block_id);
}

std::vector<BlockInfo>
HybridTypeKVCacheAllocator::convertIndexToBufferByTag(int layer_id, const std::string& tag, int block_id) const {
    const auto gid = static_cast<int>(config_.topology().groupIdForTag(tag));
    validateGroupIdForLayer(layer_id, gid);
    return logicalGroupBlockBuffers(
        gid, kv_cache_groups_[static_cast<size_t>(gid)]->convertIndexToBuffer(layer_id, block_id));
}

std::vector<BlockInfo> HybridTypeKVCacheAllocator::convertIndexToBufferByTag(
    int layer_id, const std::string& tag, int block_id, int partition_count, int partition_id) const {
    const auto gid = static_cast<int>(config_.topology().groupIdForTag(tag));
    validateGroupIdForLayer(layer_id, gid);
    return kv_cache_groups_[static_cast<size_t>(gid)]->convertIndexToBuffer(
        layer_id, block_id, partition_count, partition_id);
}

}  // namespace rtp_llm
