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
        SharedBlockCache::TaggedBlockPools group_pools;
        for (const auto& group : config_.topology().groups()) {
            group_pools.emplace_back(group.tag, block_pool_);
        }
        shared_block_cache_->init(group_pools);
    }

    for (int group_index = 0; group_index < group_nums; ++group_index) {
        const auto& cache_group = config_.topology().groups().at(static_cast<size_t>(group_index));
        const auto& spec        = cache_group.spec;

        KVCacheGroupPtr group;
        const auto      group_type = cache_group.policy.group_type;
        if (group_type == CacheGroupType::SWA) {
            group = std::make_shared<SWAKVCacheGroup>(
                cache_group, block_pool_, config_.linear_step, shared_cache_raw, nullptr);
            swa_group_indices_.push_back(group_index);
        } else if (group_type == CacheGroupType::LINEAR || (spec && spec->type == KVCacheSpecType::LinearAttention)) {
            group = std::make_shared<LinearKVCacheGroup>(
                cache_group, block_pool_, config_.linear_step, shared_cache_raw, nullptr);
            linear_group_indices_.push_back(group_index);
        } else {
            group = std::make_shared<FullKVCacheGroup>(cache_group, block_pool_, shared_cache_raw, nullptr);
            full_group_indices_.push_back(group_index);
        }

        RTP_LLM_CHECK_WITH_INFO(group->init(), "Failed to initialize KVCacheGroup group_index %d", group_index);
        kv_cache_groups_.push_back(group);
    }

    global_layer_to_local_id_.assign(static_cast<size_t>(config_.layer_all_num), -1);
    for (int group_index = 0; group_index < group_nums; ++group_index) {
        const auto& cur_group_layers = config_.topology().groups().at(static_cast<size_t>(group_index)).layer_ids;
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

void HybridTypeKVCacheAllocator::referenceBlocksInGroup(int                     group_index,
                                                        const BlockIndicesType& blocks,
                                                        bool                    is_connector) const {
    (void)group_index;
    if (is_connector) {
        block_pool_->connectorReference(blocks);
    } else {
        block_pool_->requestReference(blocks);
    }
}

void HybridTypeKVCacheAllocator::freeBlocksInGroup(int group_index, const BlockIndicesType& blocks, bool is_connector) {
    (void)group_index;
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

std::vector<BlockInfo> HybridTypeKVCacheAllocator::logicalGroupBlockBuffers(const GroupBase&       group,
                                                                            std::vector<BlockInfo> buffers) const {
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
    const auto& tag = config_.topology().soleGroupForLayer(layer_id).tag;
    return cacheGroup(tag)->convertIndexToAddr(layer_id, block_id);
}

std::vector<BlockInfo> HybridTypeKVCacheAllocator::convertIndexToBuffer(int layer_id, int block_id) const {
    if (layer_id < 0 || static_cast<size_t>(layer_id) >= config_.layer_all_num) {
        RTP_LLM_FAIL("convertIndexToBuffer invalid layer_id=%d", layer_id);
    }
    const auto& tag = config_.topology().soleGroupForLayer(layer_id).tag;
    return cacheGroup(tag)->convertIndexToBuffer(layer_id, block_id);
}

std::vector<BlockInfo> HybridTypeKVCacheAllocator::convertIndexToBuffer(int layer_id,
                                                                        int block_id,
                                                                        int partition_count,
                                                                        int partition_id) const {
    if (layer_id < 0 || static_cast<size_t>(layer_id) >= config_.layer_all_num) {
        RTP_LLM_FAIL("convertIndexToBuffer(partition) invalid layer_id=%d", layer_id);
    }
    const auto& tag = config_.topology().soleGroupForLayer(layer_id).tag;
    return cacheGroup(tag)->convertIndexToBuffer(layer_id, block_id, partition_count, partition_id);
}

BlockAddrInfo
HybridTypeKVCacheAllocator::convertIndexToAddrByTag(int layer_id, const std::string& tag, int block_id) const {
    (void)config_.groupForLayer(layer_id, tag);
    return cacheGroup(tag)->convertIndexToAddr(layer_id, block_id);
}

std::vector<BlockInfo>
HybridTypeKVCacheAllocator::convertIndexToBufferByTag(int layer_id, const std::string& tag, int block_id) const {
    const auto& group = config_.groupForLayer(layer_id, tag);
    return logicalGroupBlockBuffers(group, cacheGroup(tag)->convertIndexToBuffer(layer_id, block_id));
}

std::vector<BlockInfo> HybridTypeKVCacheAllocator::convertIndexToBufferByTag(
    int layer_id, const std::string& tag, int block_id, int partition_count, int partition_id) const {
    (void)config_.groupForLayer(layer_id, tag);
    return cacheGroup(tag)->convertIndexToBuffer(layer_id, block_id, partition_count, partition_id);
}

}  // namespace rtp_llm
