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

    SharedBlockCache* shared_cache_raw = shared_block_cache_ ? shared_block_cache_.get() : nullptr;

    if (shared_block_cache_) {
        std::vector<SharedBlockCache::GroupPool> pools;
        pools.reserve(config_.topology().groups().size());
        for (const auto& cache_group : config_.topology().groups()) {
            pools.push_back({cache_group.tag, block_pool_});
        }
        shared_block_cache_->init(pools);
    }

    for (const auto& cache_group : config_.topology().groups()) {
        const auto& spec = cache_group.spec;

        KVCacheGroupPtr group;
        const auto      group_type = cache_group.policy.group_type;
        if (group_type == CacheGroupType::SWA) {
            group = std::make_shared<SWAKVCacheGroup>(
                cache_group, block_pool_, config_.linear_step, shared_cache_raw, nullptr);
            swa_group_tags_.push_back(cache_group.tag);
        } else if (group_type == CacheGroupType::LINEAR || (spec && spec->type == KVCacheSpecType::LinearAttention)) {
            group = std::make_shared<LinearKVCacheGroup>(
                cache_group, block_pool_, config_.linear_step, shared_cache_raw, nullptr);
            linear_group_tags_.push_back(cache_group.tag);
        } else {
            group = std::make_shared<FullKVCacheGroup>(cache_group, block_pool_, shared_cache_raw, nullptr);
            full_group_tags_.push_back(cache_group.tag);
        }

        RTP_LLM_CHECK_WITH_INFO(group->init(), "Failed to initialize KVCacheGroup tag=%s", cache_group.tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(kv_cache_groups_.emplace(cache_group.tag, std::move(group)).second,
                                "duplicate KVCacheGroup tag=%s",
                                cache_group.tag.c_str());

        auto& local_ids = group_global_layer_to_local_id_[cache_group.tag];
        local_ids.assign(static_cast<size_t>(config_.layer_all_num), -1);
        for (size_t local_layer_idx = 0; local_layer_idx < cache_group.layer_ids.size(); ++local_layer_idx) {
            const int global_layer_idx = cache_group.layer_ids[local_layer_idx];
            if (global_layer_idx >= 0 && static_cast<size_t>(global_layer_idx) < local_ids.size()) {
                local_ids[static_cast<size_t>(global_layer_idx)] = static_cast<int>(local_layer_idx);
            }
        }
    }

    RTP_LLM_LOG_INFO("HybridTypeKVCacheAllocator init success");
    return true;
}

void HybridTypeKVCacheAllocator::referenceBlocksInGroup(std::string_view        tag,
                                                        const BlockIndicesType& blocks,
                                                        bool                    is_connector) const {
    (void)tag;
    if (is_connector) {
        block_pool_->connectorReference(blocks);
    } else {
        block_pool_->requestReference(blocks);
    }
}

void HybridTypeKVCacheAllocator::freeBlocksInGroup(std::string_view        tag,
                                                   const BlockIndicesType& blocks,
                                                   bool                    is_connector) {
    (void)tag;
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
        const auto&                     local_ids = group_global_layer_to_local_id_.at(group.tag);
        std::vector<BlockBufferPtrInfo> layers(topology->layers().size());
        for (int layer_id : group.layer_ids) {
            const auto global = static_cast<size_t>(layer_id);
            RTP_LLM_CHECK_WITH_INFO(
                global < local_ids.size(), "cache group tag=%s invalid global layer=%d", group.tag.c_str(), layer_id);
            const int32_t local = local_ids[global];
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

namespace {
const GroupBase& validateGroupForLayer(const CacheConfig& config, int layer_id, std::string_view tag) {
    RTP_LLM_CHECK_WITH_INFO(layer_id >= 0 && static_cast<size_t>(layer_id) < config.layer_all_num,
                            "invalid layer id %d for layer_all_num=%u",
                            layer_id,
                            config.layer_all_num);
    const auto& group = config.topology().groupForLayer(layer_id, tag);
    RTP_LLM_CHECK_WITH_INFO(
        group.tag == tag, "layer %d does not own cache group tag=%s", layer_id, std::string(tag).c_str());
    return group;
}
}  // namespace

std::vector<BlockInfo> HybridTypeKVCacheAllocator::logicalGroupBlockBuffers(std::string_view       tag,
                                                                            std::vector<BlockInfo> buffers) const {
    const auto& group = config_.topology().group(tag);
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

BlockAddrInfo HybridTypeKVCacheAllocator::convertIndexToAddr(int layer_id, const std::string& tag, int block_id) const {
    validateGroupForLayer(config_, layer_id, tag);
    return kv_cache_groups_.at(tag)->convertIndexToAddr(layer_id, block_id);
}

std::vector<BlockInfo>
HybridTypeKVCacheAllocator::convertIndexToBuffer(int layer_id, const std::string& tag, int block_id) const {
    validateGroupForLayer(config_, layer_id, tag);
    return logicalGroupBlockBuffers(tag, kv_cache_groups_.at(tag)->convertIndexToBuffer(layer_id, block_id));
}

std::vector<BlockInfo> HybridTypeKVCacheAllocator::convertIndexToBuffer(
    int layer_id, const std::string& tag, int block_id, int partition_count, int partition_id) const {
    validateGroupForLayer(config_, layer_id, tag);
    return kv_cache_groups_.at(tag)->convertIndexToBuffer(layer_id, block_id, partition_count, partition_id);
}

}  // namespace rtp_llm
