#include "rtp_llm/cpp/cache/HybridPoolKVCacheAllocator.h"

#include <algorithm>
#include <array>
#include <limits>
#include <numeric>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/KVCachePhysicalMemoryController.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/core/OpData.h"

namespace rtp_llm {
namespace {

inline bool cpShardThisGroupForReserve(const std::shared_ptr<CPSlotMapper>& mapper, CacheGroupType group_type) {
    return mapper && mapper->isSharded() && group_type == CacheGroupType::FULL;
}

inline int
cpEffectiveSeqLenForReserve(const std::shared_ptr<CPSlotMapper>& mapper, CacheGroupType group_type, int seq_len) {
    return cpShardThisGroupForReserve(mapper, group_type) ? mapper->effectiveSeqLenForAlloc(seq_len) : seq_len;
}

const char* cacheGroupTypeName(CacheGroupType group_type) {
    switch (group_type) {
        case CacheGroupType::LINEAR:
            return "LINEAR";
        case CacheGroupType::FULL:
            return "FULL";
        case CacheGroupType::SWA:
            return "SWA";
    }
    return "UNKNOWN";
}

const char* cacheRegionName(KVCacheRegionName region_name) {
    switch (region_name) {
        case KVCacheRegionName::DEFAULT:
            return "DEFAULT";
        case KVCacheRegionName::CSA_KV:
            return "CSA_KV";
        case KVCacheRegionName::HCA_KV:
            return "HCA_KV";
        case KVCacheRegionName::INDEXER_KV:
            return "INDEXER_KV";
        case KVCacheRegionName::INDEXER_STATE:
            return "INDEXER_STATE";
        case KVCacheRegionName::CSA_STATE:
            return "CSA_STATE";
        case KVCacheRegionName::HCA_STATE:
            return "HCA_STATE";
        case KVCacheRegionName::SWA_KV:
            return "SWA_KV";
        case KVCacheRegionName::REGION_COUNT:
            return "REGION_COUNT";
    }
    return "UNKNOWN";
}

}  // namespace

HybridPoolKVCacheAllocator::HybridPoolKVCacheAllocator(const CacheConfig&                 config,
                                                       AllocationType                     allocation_type,
                                                       const kmonitor::MetricsReporterPtr metrics_reporter,
                                                       int64_t                            reserve_block_ratio,
                                                       RoleType                           role_type):
    HybridKVCacheAllocator(config, allocation_type, metrics_reporter, reserve_block_ratio), role_type_(role_type) {}

bool HybridPoolKVCacheAllocator::doInit() {
    RTP_LLM_CHECK_WITH_INFO(!config_.cache_specs.empty(), "no cache_specs found in CacheConfig");
    RTP_LLM_CHECK_WITH_INFO(config_.cache_specs.size() == config_.global_layer_ids.size(),
                            "cache_specs size %zu != global_layer_ids size %zu",
                            config_.cache_specs.size(),
                            config_.global_layer_ids.size());

    const int group_nums = static_cast<int>(config_.cache_specs.size());
    group_block_pools_.reserve(static_cast<size_t>(group_nums));
    kv_cache_groups_.reserve(static_cast<size_t>(group_nums));

    SharedBlockCache*       shared_cache_raw = shared_block_cache_ ? shared_block_cache_.get() : nullptr;
    static constexpr double kBytesPerMB      = 1024.0 * 1024.0;
    std::array<size_t, 3>   dsv4_paged_pool_group_bytes{0, 0, 0};
    std::array<uint32_t, 3> dsv4_paged_pool_group_blocks{0, 0, 0};
    size_t                  dsv4_paged_pool_total_bytes = 0;
    bool                    has_dsv4_paged_pool         = false;
    std::array<size_t, 4>   dsv4_fixed_pool_group_bytes{0, 0, 0, 0};
    std::array<uint32_t, 4> dsv4_fixed_pool_group_blocks{0, 0, 0, 0};
    size_t                  dsv4_fixed_pool_total_bytes = 0;
    bool                    has_dsv4_fixed_pool         = false;

    std::vector<BlockPoolConfig> group_pool_configs;
    group_pool_configs.reserve(static_cast<size_t>(group_nums));
    const auto usesPinnedCpuBacking = [&](int gid) {
        return allocation_type_ == AllocationType::DEVICE && config_.fixed_pool_uses_pinned_cpu
               && static_cast<size_t>(gid) < config_.group_region_names.size()
               && isDsv4FixedRegion(config_.group_region_names[static_cast<size_t>(gid)]);
    };

    for (int gid = 0; gid < group_nums; ++gid) {
        auto pool_config = BlockPoolConfigHelper::createConfigForGroup(config_, static_cast<size_t>(gid));
        if (gid >= 0 && gid <= 2) {
            const size_t paged_idx                  = static_cast<size_t>(gid);
            has_dsv4_paged_pool                     = true;
            dsv4_paged_pool_group_bytes[paged_idx]  = pool_config.total_size_bytes;
            dsv4_paged_pool_group_blocks[paged_idx] = pool_config.block_num;
            dsv4_paged_pool_total_bytes += pool_config.total_size_bytes;
        }
        if (static_cast<size_t>(gid) < config_.group_region_names.size()
            && isDsv4FixedRegion(config_.group_region_names[static_cast<size_t>(gid)])) {
            has_dsv4_fixed_pool = true;
            dsv4_fixed_pool_total_bytes += pool_config.total_size_bytes;
            if (gid >= 3 && gid <= 6) {
                const size_t fixed_idx                  = static_cast<size_t>(gid - 3);
                dsv4_fixed_pool_group_bytes[fixed_idx]  = pool_config.total_size_bytes;
                dsv4_fixed_pool_group_blocks[fixed_idx] = pool_config.block_num;
            }
        }
        group_pool_configs.push_back(std::move(pool_config));
    }

    if (has_dsv4_paged_pool) {
        RTP_LLM_LOG_INFO("DSV4 paged pool summary: group_0=%zu bytes(%.2f MB, blocks=%u), "
                         "group_1=%zu bytes(%.2f MB, blocks=%u), group_2=%zu bytes(%.2f MB, blocks=%u), "
                         "total_size=%zu bytes total_size_mb=%.2f",
                         dsv4_paged_pool_group_bytes[0],
                         static_cast<double>(dsv4_paged_pool_group_bytes[0]) / kBytesPerMB,
                         dsv4_paged_pool_group_blocks[0],
                         dsv4_paged_pool_group_bytes[1],
                         static_cast<double>(dsv4_paged_pool_group_bytes[1]) / kBytesPerMB,
                         dsv4_paged_pool_group_blocks[1],
                         dsv4_paged_pool_group_bytes[2],
                         static_cast<double>(dsv4_paged_pool_group_bytes[2]) / kBytesPerMB,
                         dsv4_paged_pool_group_blocks[2],
                         dsv4_paged_pool_total_bytes,
                         static_cast<double>(dsv4_paged_pool_total_bytes) / kBytesPerMB);
    }

    if (has_dsv4_fixed_pool) {
        RTP_LLM_LOG_INFO("DSV4 fixed pool summary: group_3=%zu bytes(%.2f MB, blocks=%u), "
                         "group_4=%zu bytes(%.2f MB, blocks=%u), group_5=%zu bytes(%.2f MB, blocks=%u), "
                         "group_6=%zu bytes(%.2f MB, blocks=%u), total_size=%zu bytes total_size_mb=%.2f",
                         dsv4_fixed_pool_group_bytes[0],
                         static_cast<double>(dsv4_fixed_pool_group_bytes[0]) / kBytesPerMB,
                         dsv4_fixed_pool_group_blocks[0],
                         dsv4_fixed_pool_group_bytes[1],
                         static_cast<double>(dsv4_fixed_pool_group_bytes[1]) / kBytesPerMB,
                         dsv4_fixed_pool_group_blocks[1],
                         dsv4_fixed_pool_group_bytes[2],
                         static_cast<double>(dsv4_fixed_pool_group_bytes[2]) / kBytesPerMB,
                         dsv4_fixed_pool_group_blocks[2],
                         dsv4_fixed_pool_group_bytes[3],
                         static_cast<double>(dsv4_fixed_pool_group_bytes[3]) / kBytesPerMB,
                         dsv4_fixed_pool_group_blocks[3],
                         dsv4_fixed_pool_total_bytes,
                         static_cast<double>(dsv4_fixed_pool_total_bytes) / kBytesPerMB);
    }

    const bool is_warmup = std::all_of(
        group_pool_configs.begin(), group_pool_configs.end(), [](const auto& config) { return config.block_num == 1; });

    std::vector<size_t> arena_offsets(static_cast<size_t>(group_nums), 0);
    VmmBackend          vmm_backend;
    if (allocation_type_ == AllocationType::DEVICE && use_cuda_malloc_block_pool_ && !is_warmup
        && vmm_backend.isAvailable()) {
        constexpr size_t kBackingAlignment = 256;
        size_t           arena_size        = 0;
        for (int gid = 0; gid < group_nums; ++gid) {
            if (usesPinnedCpuBacking(gid)) {
                continue;
            }
            arena_size = (arena_size + kBackingAlignment - 1) / kBackingAlignment * kBackingAlignment;
            arena_offsets[static_cast<size_t>(gid)] = arena_size;
            arena_size += group_pool_configs[static_cast<size_t>(gid)].total_size_bytes;
        }
        if (arena_size > 0) {
            kv_buffer_arena_ = BlockPool::allocatePausableDeviceBacking(arena_size);
            RTP_LLM_LOG_INFO("HybridPool KV buffer arena: size=%zu bytes (%.2f MB), groups=%d",
                             arena_size,
                             static_cast<double>(arena_size) / kBytesPerMB,
                             group_nums);
        }
    }

    for (int gid = 0; gid < group_nums; ++gid) {
        RTP_LLM_CHECK_WITH_INFO(gid < static_cast<int>(config_.group_types.size()),
                                "missing group type for group %d in HybridPoolKVCacheAllocator",
                                gid);
        const auto group_type = config_.group_types[static_cast<size_t>(gid)];

        const auto&  pool_config            = group_pool_configs[static_cast<size_t>(gid)];
        const bool   use_pinned_cpu_backing = usesPinnedCpuBacking(gid);
        BlockPoolPtr group_pool;
        if (kv_buffer_arena_.defined() && !use_pinned_cpu_backing) {
            auto backing = kv_buffer_arena_.narrow(0,
                                                   static_cast<int64_t>(arena_offsets[static_cast<size_t>(gid)]),
                                                   static_cast<int64_t>(pool_config.total_size_bytes));
            group_pool   = std::make_shared<BlockPool>(pool_config, std::move(backing));
        } else if (is_warmup && allocation_type_ == AllocationType::DEVICE && vmm_backend.isAvailable()
                   && !use_pinned_cpu_backing) {
            // The throwaway warmup pools must not claim the persistent kv_cache
            // tag before the real arena is allocated.
            auto backing = torch::empty({static_cast<int64_t>(pool_config.total_size_bytes)},
                                        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
            group_pool   = std::make_shared<BlockPool>(pool_config, std::move(backing));
        } else {
            group_pool = std::make_shared<BlockPool>(pool_config,
                                                     allocation_type_,
                                                     use_pinned_cpu_backing,
                                                     use_cuda_malloc_block_pool_ && !use_pinned_cpu_backing);
        }
        RTP_LLM_CHECK_WITH_INFO(group_pool->init(), "Failed to initialize block pool for group %d", gid);

        const auto& ids  = config_.global_layer_ids[static_cast<size_t>(gid)];
        auto        spec = config_.cache_specs[static_cast<size_t>(gid)];

        KVCacheGroupPtr group;
        if (group_type == CacheGroupType::LINEAR) {
            group = std::make_shared<LinearKVCacheGroup>(
                ids, spec, group_pool, gid, config_.linear_step, shared_cache_raw, metrics_reporter_);
            linear_group_ids_.push_back(gid);
        } else if (group_type == CacheGroupType::SWA) {
            group = std::make_shared<SWAKVCacheGroup>(
                ids, spec, group_pool, gid, config_.linear_step, shared_cache_raw, metrics_reporter_);
            swa_group_ids_.push_back(gid);
        } else {
            group = std::make_shared<FullKVCacheGroup>(ids, spec, group_pool, gid, shared_cache_raw, metrics_reporter_);
            full_group_ids_.push_back(gid);
        }

        RTP_LLM_CHECK_WITH_INFO(group->init(), "Failed to initialize KVCacheGroup gid %d", gid);
        group_block_pools_.push_back(group_pool);
        kv_cache_groups_.push_back(group);
    }

    // HybridPool owns one BlockPool per group; do not read pool stats from block_pool_ in HybridPool mode.
    block_pool_ = group_block_pools_.empty() ? nullptr : group_block_pools_[0];

    if (shared_block_cache_) {
        shared_block_cache_->init(group_nums, group_block_pools_);
    }

    RTP_LLM_LOG_INFO("HybridPoolKVCacheAllocator init success, group pools=%zu", group_block_pools_.size());
    return true;
}

int HybridPoolKVCacheAllocator::defaultGroupIdForLayer(int layer_id) const {
    if (layer_id < 0 || static_cast<size_t>(layer_id) >= config_.layer_to_group_id.size()) {
        RTP_LLM_FAIL("invalid layer_id=%d", layer_id);
    }
    const int gid = config_.layer_to_group_id[static_cast<size_t>(layer_id)];
    RTP_LLM_CHECK_WITH_INFO(gid >= 0 && gid < static_cast<int>(kv_cache_groups_.size()),
                            "invalid default group id %d for layer %d",
                            gid,
                            layer_id);
    return gid;
}

int HybridPoolKVCacheAllocator::groupIdForLayerRegion(int layer_id, KVCacheRegionName region_name) const {
    const size_t attn_id = static_cast<size_t>(region_name);
    if (layer_id >= 0 && static_cast<size_t>(layer_id) < config_.layer_region_to_group_id.size()) {
        const auto& dense = config_.layer_region_to_group_id[static_cast<size_t>(layer_id)];
        if (attn_id < dense.size() && dense[attn_id] >= 0) {
            const int gid = dense[attn_id];
            RTP_LLM_CHECK_WITH_INFO(gid < static_cast<int>(kv_cache_groups_.size()),
                                    "invalid group id %d for layer %d region %zu",
                                    gid,
                                    layer_id,
                                    attn_id);
            return gid;
        }
    }
    if (region_name == KVCacheRegionName::DEFAULT) {
        return defaultGroupIdForLayer(layer_id);
    }
    RTP_LLM_FAIL("missing group mapping for layer_id=%d region=%zu", layer_id, attn_id);
}

void HybridPoolKVCacheAllocator::referenceBlocksInGroup(int                     gid,
                                                        const BlockIndicesType& blocks,
                                                        bool                    is_connector) const {
    if (is_connector) {
        group_block_pools_[static_cast<size_t>(gid)]->connectorReference(blocks);
    } else {
        group_block_pools_[static_cast<size_t>(gid)]->requestReference(blocks);
    }
}

void HybridPoolKVCacheAllocator::freeBlocksInGroup(int gid, const BlockIndicesType& blocks, bool is_connector) {
    if (is_connector) {
        group_block_pools_[static_cast<size_t>(gid)]->connectorFree(blocks);
    } else {
        group_block_pools_[static_cast<size_t>(gid)]->requestFree(blocks);
    }
}

CacheLayerLayout HybridPoolKVCacheAllocator::allLayerCacheBase() const {
    CacheLayerLayout layout;
    layout.layer_to_groups          = config_.layer_to_group_id;
    layout.layer_to_group_ids       = config_.layer_to_group_ids;
    layout.layer_region_to_group_id = config_.layer_region_to_group_id;
    layout.group_types              = config_.group_types;
    layout.group_region_names       = config_.group_region_names;
    layout.group_seq_size_per_block = config_.group_seq_size_per_block;
    layout.layer_group_types        = config_.layer_group_types;

    const bool has_typed_mapping = !config_.layer_region_to_group_id.empty();
    if (has_typed_mapping) {
        RTP_LLM_CHECK_WITH_INFO(config_.group_region_names.size() == kv_cache_groups_.size(),
                                "group_region_names size %zu != group num %zu for typed layer-region mapping",
                                config_.group_region_names.size(),
                                kv_cache_groups_.size());
    }

    layout.layers_to_kv_buffer_ptrs.resize(config_.layer_all_num);
    layout.layers_to_scale_buffer_ptrs.resize(config_.layer_all_num);
    const size_t region_name_count = static_cast<size_t>(KVCacheRegionName::REGION_COUNT);
    layout.layers_to_kv_buffer_ptrs_by_attn.resize(config_.layer_all_num);
    layout.layers_to_scale_buffer_ptrs_by_attn.resize(config_.layer_all_num);
    for (size_t layer_id = 0; layer_id < static_cast<size_t>(config_.layer_all_num); ++layer_id) {
        layout.layers_to_kv_buffer_ptrs_by_attn[layer_id].resize(region_name_count);
        layout.layers_to_scale_buffer_ptrs_by_attn[layer_id].resize(region_name_count);
    }

    for (size_t layer_id = 0; layer_id < static_cast<size_t>(config_.layer_all_num); ++layer_id) {
        const int  gid           = defaultGroupIdForLayer(static_cast<int>(layer_id));
        const auto layer_tensors = kv_cache_groups_[static_cast<size_t>(gid)]->allLayerCacheBase();
        const auto scale_tensors = kv_cache_groups_[static_cast<size_t>(gid)]->allLayerScaleCacheBase();
        auto       it            = layer_tensors.find(static_cast<int>(layer_id));
        if (it != layer_tensors.end()) {
            layout.layers_to_kv_buffer_ptrs[layer_id] = it->second;
        }
        auto scale_it = scale_tensors.find(static_cast<int>(layer_id));
        if (scale_it != scale_tensors.end()) {
            layout.layers_to_scale_buffer_ptrs[layer_id] = scale_it->second;
        }
    }

    for (int gid = 0; gid < static_cast<int>(kv_cache_groups_.size()); ++gid) {
        const auto layer_tensors = kv_cache_groups_[static_cast<size_t>(gid)]->allLayerCacheBase();
        const auto scale_tensors = kv_cache_groups_[static_cast<size_t>(gid)]->allLayerScaleCacheBase();
        const auto region_name   = static_cast<size_t>(gid < static_cast<int>(config_.group_region_names.size()) ?
                                                         config_.group_region_names[static_cast<size_t>(gid)] :
                                                         KVCacheRegionName::DEFAULT);
        RTP_LLM_CHECK_WITH_INFO(
            region_name < region_name_count, "group %d has invalid region id %zu", gid, region_name);
        for (const auto& [layer_id, tensor] : layer_tensors) {
            RTP_LLM_CHECK_WITH_INFO(
                layer_id >= 0 && static_cast<size_t>(layer_id) < layout.layers_to_kv_buffer_ptrs_by_attn.size(),
                "layer_id %d out of typed kv layout range %zu",
                layer_id,
                layout.layers_to_kv_buffer_ptrs_by_attn.size());
            layout.layers_to_kv_buffer_ptrs_by_attn[static_cast<size_t>(layer_id)][region_name] = tensor;
        }
        for (const auto& [layer_id, tensor] : scale_tensors) {
            RTP_LLM_CHECK_WITH_INFO(
                layer_id >= 0 && static_cast<size_t>(layer_id) < layout.layers_to_scale_buffer_ptrs_by_attn.size(),
                "layer_id %d out of typed scale layout range %zu",
                layer_id,
                layout.layers_to_scale_buffer_ptrs_by_attn.size());
            layout.layers_to_scale_buffer_ptrs_by_attn[static_cast<size_t>(layer_id)][region_name] = tensor;
        }
    }
    return layout;
}

BlockAddrInfo HybridPoolKVCacheAllocator::convertIndexToAddr(int layer_id, int block_id) const {
    const int gid = defaultGroupIdForLayer(layer_id);
    return kv_cache_groups_[static_cast<size_t>(gid)]->convertIndexToAddr(layer_id, block_id);
}

std::vector<BlockInfo> HybridPoolKVCacheAllocator::convertIndexToBuffer(int layer_id, int block_id) const {
    const int gid = defaultGroupIdForLayer(layer_id);
    return kv_cache_groups_[static_cast<size_t>(gid)]->convertIndexToBuffer(layer_id, block_id);
}

std::vector<BlockInfo> HybridPoolKVCacheAllocator::convertIndexToBuffer(int layer_id,
                                                                        int block_id,
                                                                        int partition_count,
                                                                        int partition_id) const {
    const int gid = defaultGroupIdForLayer(layer_id);
    return kv_cache_groups_[static_cast<size_t>(gid)]->convertIndexToBuffer(
        layer_id, block_id, partition_count, partition_id);
}

BlockAddrInfo
HybridPoolKVCacheAllocator::convertIndexToAddr(int layer_id, KVCacheRegionName region_name, int block_id) const {
    const int gid = groupIdForLayerRegion(layer_id, region_name);
    return kv_cache_groups_[static_cast<size_t>(gid)]->convertIndexToAddr(layer_id, block_id);
}

std::vector<BlockInfo>
HybridPoolKVCacheAllocator::convertIndexToBuffer(int layer_id, KVCacheRegionName region_name, int block_id) const {
    const int gid = groupIdForLayerRegion(layer_id, region_name);
    return kv_cache_groups_[static_cast<size_t>(gid)]->convertIndexToBuffer(layer_id, block_id);
}

std::vector<BlockInfo> HybridPoolKVCacheAllocator::convertIndexToBuffer(
    int layer_id, KVCacheRegionName region_name, int block_id, int partition_count, int partition_id) const {
    const int gid = groupIdForLayerRegion(layer_id, region_name);
    return kv_cache_groups_[static_cast<size_t>(gid)]->convertIndexToBuffer(
        layer_id, block_id, partition_count, partition_id);
}

void HybridPoolKVCacheAllocator::blockBatchCopy(const BlockIdPair* begin_ptr, const BlockIdPair* end_ptr) {
    if (end_ptr == begin_ptr) {
        return;
    }

    size_t copy_nums[BatchCopyParams::TYPE_SIZE] = {};
    for (int gid = 0; gid < static_cast<int>(kv_cache_groups_.size()); ++gid) {
        RTP_LLM_CHECK_WITH_INFO(
            static_cast<size_t>(gid) < group_block_pools_.size(), "missing block pool for group %d", gid);
        RTP_LLM_CHECK_WITH_INFO(
            static_cast<size_t>(gid) < config_.cache_specs.size(), "missing cache spec for group %d", gid);
        RTP_LLM_CHECK_WITH_INFO(
            static_cast<size_t>(gid) < config_.global_layer_ids.size(), "missing layer ids for group %d", gid);
        const auto   copy_type = BatchCopyParams::get_copy_type(group_block_pools_[static_cast<size_t>(gid)]->where(),
                                                              group_block_pools_[static_cast<size_t>(gid)]->where());
        const auto&  spec      = config_.cache_specs[static_cast<size_t>(gid)];
        const size_t buffers_per_layer = spec->scale_block_size_bytes() > 0 ? 2 : 1;
        copy_nums[copy_type] += config_.global_layer_ids[static_cast<size_t>(gid)].size()
                                * static_cast<size_t>(end_ptr - begin_ptr) * buffers_per_layer;
    }

    BatchCopyParams copy_params;
    for (size_t i = 0; i < BatchCopyParams::TYPE_SIZE; ++i) {
        copy_params.reserve(static_cast<BatchCopyParams::CopyType>(i), copy_nums[i]);
    }

    for (auto it = begin_ptr; it != end_ptr; ++it) {
        auto [src_block_index, dest_block_index] = *it;

        for (int gid = 0; gid < static_cast<int>(kv_cache_groups_.size()); ++gid) {
            RTP_LLM_CHECK_WITH_INFO(
                static_cast<size_t>(gid) < config_.cache_specs.size(), "missing cache spec for group %d", gid);
            RTP_LLM_CHECK_WITH_INFO(
                static_cast<size_t>(gid) < config_.global_layer_ids.size(), "missing layer ids for group %d", gid);

            const size_t kv_block_size_bytes = config_.cache_specs[static_cast<size_t>(gid)]->block_size_bytes();
            const size_t scale_block_bytes   = config_.cache_specs[static_cast<size_t>(gid)]->scale_block_size_bytes();
            const auto   copy_type =
                BatchCopyParams::get_copy_type(group_block_pools_[static_cast<size_t>(gid)]->where(),
                                               group_block_pools_[static_cast<size_t>(gid)]->where());

            for (int layer_id : config_.global_layer_ids[static_cast<size_t>(gid)]) {
                auto src_addr_info =
                    kv_cache_groups_[static_cast<size_t>(gid)]->convertIndexToAddr(layer_id, src_block_index);
                auto dst_addr_info =
                    kv_cache_groups_[static_cast<size_t>(gid)]->convertIndexToAddr(layer_id, dest_block_index);

                if (!src_addr_info.kv_addr || !dst_addr_info.kv_addr) {
                    RTP_LLM_LOG_ERROR("Failed to get block address for group %d layer %d, src_block %d, dst_block %d",
                                      gid,
                                      layer_id,
                                      src_block_index,
                                      dest_block_index);
                    continue;
                }

                copy_params.add(dst_addr_info.kv_addr, src_addr_info.kv_addr, kv_block_size_bytes, copy_type);

                if (scale_block_bytes > 0 && src_addr_info.kv_scale_addr && dst_addr_info.kv_scale_addr) {
                    copy_params.add(
                        dst_addr_info.kv_scale_addr, src_addr_info.kv_scale_addr, scale_block_bytes, copy_type);
                }
            }
        }
    }

    execBatchCopy(copy_params);
}

size_t HybridPoolKVCacheAllocator::freeBlocksNum() const {
    size_t total = 0;
    for (const auto& pool : group_block_pools_) {
        total += pool->freeBlocksNum();
    }
    return total;
}

size_t HybridPoolKVCacheAllocator::availableBlocksNum() const {
    size_t total = 0;
    for (const auto& pool : group_block_pools_) {
        total += pool->availableBlocksNum();
    }
    return total;
}

BatchKVCacheResourcePtr HybridPoolKVCacheAllocator::popBlocksFromCache(size_t min_blocks_to_free) {
    if (min_blocks_to_free == 0 || !shared_block_cache_) {
        return nullptr;
    }

    auto evict_result = shared_block_cache_->selectAndEvict(min_blocks_to_free);
    if (evict_result.evicted_keys.empty()) {
        return nullptr;
    }
    if (metrics_reporter_) {
        for (const auto& [cache_key, lifetime_ms] : evict_result.evicted_lifetime_ms) {
            RtpLLMCacheEvictionMetricsCollector collector;
            collector.lifetime_ms = lifetime_ms;
            kmonitor::MetricsTags tags("scope", "gpu");
            tags.AddTag("kind", evict_result.evicted_state_only_group.count(cache_key) ? "state_swa_kv" : "chain");
            tags.AddTag("backing", "device");
            metrics_reporter_->report<RtpLLMCacheEvictionMetrics, RtpLLMCacheEvictionMetricsCollector>(&tags,
                                                                                                       &collector);
        }
    }

    auto batch_resource = std::make_shared<BatchKVCacheResource>();
    batch_resource->resetBatchSize(1);
    batch_resource->initGroups(config_.groupNums(),
                               static_cast<int>(config_.layer_all_num),
                               config_.layer_to_group_id,
                               config_.kernelBlocksPerKvBlock(),
                               config_.group_types,
                               config_.layer_region_to_group_id);
    batch_resource->setLastBlockAligned(true);

    for (int gid = 0; gid < config_.groupNums(); ++gid) {
        batch_resource->mutableBlockIds(0, gid).resize(evict_result.evicted_keys.size(), NULL_BLOCK_IDX);
    }

    CacheKeysType         evicted_keys;
    BlockDependenciesType evicted_dependencies;
    evicted_keys.reserve(evict_result.evicted_keys.size());
    evicted_dependencies.reserve(evict_result.evicted_keys.size());
    for (size_t evicted_idx = 0; evicted_idx < evict_result.evicted_keys.size(); ++evicted_idx) {
        const auto  cache_key = evict_result.evicted_keys[evicted_idx];
        const auto& slots     = evict_result.evicted_slots.at(cache_key);
        evicted_keys.push_back(cache_key);
        auto dep_it = evict_result.evicted_dependencies.find(cache_key);
        if (dep_it != evict_result.evicted_dependencies.end()) {
            evicted_dependencies.push_back(dep_it->second);
        } else {
            BlockDependency dependency;
            dependency.ordinal = static_cast<uint32_t>(evicted_idx);
            if (evicted_idx > 0) {
                dependency.has_parent = true;
                dependency.parent_key = evict_result.evicted_keys[evicted_idx - 1];
            }
            evicted_dependencies.push_back(dependency);
        }
        for (int gid = 0; gid < static_cast<int>(slots.size()) && gid < config_.groupNums(); ++gid) {
            if (!isNullBlockIdx(slots[gid])) {
                batch_resource->mutableBlockIds(0, gid).setAt(evicted_idx, slots[gid]);
            }
        }
    }
    batch_resource->cacheResource(0).setCacheKeys(std::move(evicted_keys));
    batch_resource->cacheResource(0).setBlockDependencies(std::move(evicted_dependencies));
    // Evicted keys already come from the GPU cache's actual key namespace.
    // Under CP this can be a mixed batch of canonical paged keys and logical
    // state/SWA keys, so coordinator must not remap the whole batch again.
    batch_resource->cacheResource(0).setCacheKeysAreCpCanonical(true);
    return batch_resource;
}

void HybridPoolKVCacheAllocator::blockCacheFree(const BatchKVCacheResourcePtr& batch_kv_cache_resource) {
    if (!batch_kv_cache_resource) {
        return;
    }
    for (int batch_id = 0; batch_id < batch_kv_cache_resource->batchSize(); ++batch_id) {
        for (int gid = 0; gid < batch_kv_cache_resource->groupNums(); ++gid) {
            BlockIndicesType                 blocks_to_free;
            std::unordered_set<BlockIdxType> seen_blocks;
            for (auto block_idx : batch_kv_cache_resource->blocks(batch_id, gid)) {
                if (isNullBlockIdx(block_idx) || !seen_blocks.insert(block_idx).second) {
                    continue;
                }
                blocks_to_free.push_back(block_idx);
            }
            if (!blocks_to_free.empty()) {
                group_block_pools_[static_cast<size_t>(gid)]->blockCacheFree(blocks_to_free);
            }
        }
    }
}

size_t HybridPoolKVCacheAllocator::requestRefBlocksNum() const {
    size_t total = 0;
    for (const auto& pool : group_block_pools_) {
        total += pool->requestRefBlocksNum();
    }
    return total;
}

size_t HybridPoolKVCacheAllocator::connectorRefBlocksNum() const {
    size_t total = 0;
    for (const auto& pool : group_block_pools_) {
        total += pool->connectorRefBlocksNum();
    }
    return total;
}

size_t HybridPoolKVCacheAllocator::blockCacheRefBlocksNum() const {
    size_t total = 0;
    for (const auto& pool : group_block_pools_) {
        total += pool->blockCacheRefBlocksNum();
    }
    return total;
}

size_t HybridPoolKVCacheAllocator::notInUseBlocksNum() const {
    size_t total = 0;
    for (const auto& pool : group_block_pools_) {
        total += pool->notInUseBlocksNum();
    }
    return total;
}

size_t HybridPoolKVCacheAllocator::minTokenCapacity(bool use_available_blocks, bool full_groups_only) const {
    if (group_block_pools_.empty()) {
        return 0;
    }

    auto calculate = [&](bool only_full_groups) {
        size_t min_tokens = std::numeric_limits<size_t>::max();
        bool   saw_group  = false;
        for (size_t gid = 0; gid < group_block_pools_.size(); ++gid) {
            if (only_full_groups && gid < config_.group_types.size()
                && config_.group_types[gid] != CacheGroupType::FULL) {
                continue;
            }
            if (!group_block_pools_[gid]) {
                continue;
            }
            saw_group        = true;
            const auto block = use_available_blocks ? group_block_pools_[gid]->availableBlocksNum() :
                                                      group_block_pools_[gid]->totalBlocksNum();
            min_tokens       = std::min(min_tokens, block * logicalSeqSizePerBlockForCapacity(gid));
        }
        return std::make_pair(saw_group, min_tokens);
    };

    if (full_groups_only) {
        const auto [saw_full_group, min_tokens] = calculate(/*only_full_groups=*/true);
        if (saw_full_group) {
            return min_tokens;
        }
    }

    const auto [saw_group, min_tokens] = calculate(/*only_full_groups=*/false);
    return saw_group ? min_tokens : 0;
}

size_t HybridPoolKVCacheAllocator::availableTokensNum() const {
    return minTokenCapacity(/*use_available_blocks=*/true, /*full_groups_only=*/true);
}

size_t HybridPoolKVCacheAllocator::totalTokensNum() const {
    return minTokenCapacity(/*use_available_blocks=*/false, /*full_groups_only=*/true);
}

size_t HybridPoolKVCacheAllocator::totalBlocksNum() const {
    size_t total = 0;
    for (const auto& pool : group_block_pools_) {
        total += pool->totalBlocksNum();
    }
    return total;
}

size_t HybridPoolKVCacheAllocator::maxAvailableTokensNum() const {
    return minTokenCapacity(/*use_available_blocks=*/false, /*full_groups_only=*/true);
}

void HybridPoolKVCacheAllocator::regUserMr(size_t model_id, std::shared_ptr<CacheStore> cache_store) {
    for (auto& pool : group_block_pools_) {
        pool->regUserMr(model_id, cache_store);
    }
}

void HybridPoolKVCacheAllocator::deregUserMr() {
    // Must mirror regUserMr and deregister EVERY group pool. The base
    // KVCacheAllocator::deregUserMr only touches the single block_pool_, which for the
    // hybrid allocator leaves the other groups' RDMA user MRs registered. Those MRs pin
    // the KV arena's GPU pages (nvidia-peermem / GPUDirect), so on engine sleep the
    // subsequent tms_pause("kv_cache") cannot unmap the still-pinned physical pages and
    // the whole arena stays resident. Deregistering all groups first lets pause reclaim it.
    for (auto& pool : group_block_pools_) {
        pool->deregUserMr();
    }
}

int64_t HybridPoolKVCacheAllocator::getMrCostTimeMs() const {
    int64_t total = 0;
    for (const auto& pool : group_block_pools_) {
        total += pool->getMrCostTimeMs();
    }
    return total;
}

MallocStatus HybridPoolKVCacheAllocator::evaluateInitCapacity(const MallocInfo& malloc_info,
                                                              size_t            reserve_blocks,
                                                              InitCapacityMode  mode) const {
    if (!malloc_info.batch_kv_cache_resource || !malloc_info.complete_token_ids) {
        return MallocStatus::NONE;
    }
    const auto& cp_mapper          = malloc_info.cp_slot_mapper;
    const int   batch_size         = malloc_info.batch_kv_cache_resource->batchSize();
    const int   total_seq_len      = malloc_info.complete_token_ids->totalSeqLength();
    const int   raw_common_seq_len = std::min(malloc_info.complete_token_ids->commonSeqLength(), total_seq_len);
    const int   raw_seq_len        = malloc_info.complete_token_ids->seqLength();
    const int   reserve_step       = malloc_info.complete_token_ids->getReserveStep();
    const bool  reuse_enabled      = malloc_info.reuse_cache;

    size_t total_reservable_blocks = 0;
    for (size_t gid = 0; gid < group_block_pools_.size(); ++gid) {
        const auto region =
            gid < config_.group_region_names.size() ? config_.group_region_names[gid] : KVCacheRegionName::DEFAULT;
        if (isDsv4FixedRegion(region)) {
            continue;
        }
        total_reservable_blocks += group_block_pools_[gid]->totalBlocksNum();
    }

    int    retryable_group          = -1;
    int    retryable_need_blocks    = 0;
    size_t retryable_capacity       = 0;
    size_t retryable_reserve_blocks = 0;
    for (int gid = 0; gid < static_cast<int>(kv_cache_groups_.size()); ++gid) {
        const auto group_type             = static_cast<size_t>(gid) < config_.group_types.size() ?
                                                config_.group_types[static_cast<size_t>(gid)] :
                                                CacheGroupType::FULL;
        const int  group_common_seq       = cpEffectiveSeqLenForReserve(cp_mapper, group_type, raw_common_seq_len);
        const int  group_seq_len          = cpEffectiveSeqLenForReserve(cp_mapper, group_type, raw_seq_len);
        const int  group_reuse_blocks_len = reuse_enabled ? malloc_info.batch_kv_cache_resource->blocksNum(0, gid) : 0;
        const auto need                   = kv_cache_groups_[static_cast<size_t>(gid)]->getNeedBlocks(
            group_common_seq, group_seq_len, reserve_step, group_reuse_blocks_len, reuse_enabled);
        const int need_blocks = need.common_blocks + batch_size * need.extra_blocks;
        if (need_blocks <= 0) {
            continue;
        }
        const size_t group_index  = static_cast<size_t>(gid);
        const size_t total_blocks = group_block_pools_[group_index]->totalBlocksNum();
        const auto region = group_index < config_.group_region_names.size() ? config_.group_region_names[group_index] :
                                                                              KVCacheRegionName::DEFAULT;
        const size_t group_reserve_blocks = isDsv4FixedRegion(region) || total_reservable_blocks == 0 ?
                                                0 :
                                                reserve_blocks * total_blocks / total_reservable_blocks;
        const size_t required_blocks      = static_cast<size_t>(need_blocks);
        auto         fits                 = [required_blocks, group_reserve_blocks](size_t capacity) {
            return required_blocks <= capacity && group_reserve_blocks <= capacity - required_blocks;
        };

        if (!fits(total_blocks)) {
            if (malloc_info.verbose) {
                RTP_LLM_LOG_INFO("HybridPool initMalloc permanently rejected: request_id=%ld group=%d need_blocks=%d "
                                 "capacity=%zu reserve_blocks=%zu group_reserve_blocks=%zu",
                                 malloc_info.request_id,
                                 gid,
                                 need_blocks,
                                 total_blocks,
                                 reserve_blocks,
                                 group_reserve_blocks);
            }
            return MallocStatus::PERMANENT_RESOURCE_EXHAUSTED;
        }

        if (mode == InitCapacityMode::TOTAL_AND_AVAILABLE && retryable_group < 0) {
            const size_t available_blocks = group_block_pools_[group_index]->availableBlocksNum();
            if (!fits(available_blocks)) {
                retryable_group          = gid;
                retryable_need_blocks    = need_blocks;
                retryable_capacity       = available_blocks;
                retryable_reserve_blocks = group_reserve_blocks;
            }
        }
    }

    if (retryable_group >= 0) {
        if (malloc_info.verbose) {
            RTP_LLM_LOG_INFO("HybridPool initMalloc deferred: request_id=%ld group=%d need_blocks=%d "
                             "capacity=%zu reserve_blocks=%zu group_reserve_blocks=%zu",
                             malloc_info.request_id,
                             retryable_group,
                             retryable_need_blocks,
                             retryable_capacity,
                             reserve_blocks,
                             retryable_reserve_blocks);
        }
        return MallocStatus::RETRYABLE_RESOURCE_EXHAUSTED;
    }
    return MallocStatus::NONE;
}

bool HybridPoolKVCacheAllocator::hasAvailableBlocksForReserve(const MallocInfo& malloc_info,
                                                              size_t            reserve_blocks) const {
    return evaluateInitCapacity(malloc_info, reserve_blocks, InitCapacityMode::TOTAL_AND_AVAILABLE)
           == MallocStatus::NONE;
}

void HybridPoolKVCacheAllocator::logMallocFailure(const MallocInfo& malloc_info,
                                                  const char*       phase,
                                                  int               failed_batch,
                                                  int               failed_group,
                                                  bool              incremental,
                                                  int               failed_need_blocks) const {
    if (!malloc_info.verbose || !malloc_info.batch_kv_cache_resource || !malloc_info.complete_token_ids) {
        return;
    }

    const auto& resource       = malloc_info.batch_kv_cache_resource;
    const auto& cp_mapper      = malloc_info.cp_slot_mapper;
    const int   batch_size     = resource->batchSize();
    const int   raw_seq_len    = incremental ? malloc_info.incrSeqLen() : malloc_info.complete_token_ids->seqLength();
    const int   raw_common_len = std::min(malloc_info.complete_token_ids->commonSeqLength(), raw_seq_len);
    const int   total_seq_len  = malloc_info.complete_token_ids->totalSeqLength();
    const int   request_reserve_step = malloc_info.complete_token_ids->getReserveStep();
    const bool  reserve_admission    = !incremental && failed_group < 0;
    const int   reserve_step         = incremental || reserve_admission ? request_reserve_step : 0;
    const int   planning_raw_seq_len = !incremental && !reserve_admission ? raw_common_len : raw_seq_len;
    const auto  reserve_blocks       = reserveBlockNum();

    size_t total_reservable_available_blocks = 0;
    for (size_t gid = 0; gid < group_block_pools_.size(); ++gid) {
        const auto region =
            gid < config_.group_region_names.size() ? config_.group_region_names[gid] : KVCacheRegionName::DEFAULT;
        if (!isDsv4FixedRegion(region)) {
            total_reservable_available_blocks += group_block_pools_[gid]->availableBlocksNum();
        }
    }

    RTP_LLM_LOG_WARNING("HybridPool malloc failure: error_code=602 request_id=%ld phase=%s failed_batch=%d "
                        "failed_group=%d incremental=%d batch_size=%d seq_len=%d common_seq_len=%d total_seq_len=%d "
                        "planning_seq_len=%d request_reserve_step=%d planning_reserve_step=%d "
                        "failed_need_blocks=%d reserve_blocks=%zu snapshot=best_effort_at_failure",
                        malloc_info.request_id,
                        phase,
                        failed_batch,
                        failed_group,
                        incremental,
                        batch_size,
                        raw_seq_len,
                        raw_common_len,
                        total_seq_len,
                        planning_raw_seq_len,
                        request_reserve_step,
                        reserve_step,
                        failed_need_blocks,
                        reserve_blocks);

    for (int gid = 0; gid < static_cast<int>(kv_cache_groups_.size()); ++gid) {
        const size_t group_index = static_cast<size_t>(gid);
        const auto   group_type =
            group_index < config_.group_types.size() ? config_.group_types[group_index] : CacheGroupType::FULL;
        const auto region = group_index < config_.group_region_names.size() ? config_.group_region_names[group_index] :
                                                                              KVCacheRegionName::DEFAULT;
        const int  group_seq_len = cpEffectiveSeqLenForReserve(cp_mapper, group_type, planning_raw_seq_len);

        int    need_blocks          = 0;
        int    need_slots           = 0;
        size_t current_slots        = 0;
        size_t current_valid_blocks = 0;
        for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
            const auto& blocks = resource->blocks(batch_id, gid);
            current_slots += blocks.size();
            current_valid_blocks += static_cast<size_t>(std::count_if(
                blocks.begin(), blocks.end(), [](auto block) { return !isNullBlockIdx(block) && block > 0; }));
            need_slots += kv_cache_groups_[group_index]->needBlocksNum(
                group_seq_len, static_cast<int>(blocks.size()), reserve_step);
        }
        if (incremental) {
            // FULL groups materialize every logical slot. LINEAR/SWA groups
            // sparsify slots, so their exact physical request is emitted by
            // the group allocator immediately before this snapshot.
            need_blocks = group_type == CacheGroupType::FULL ? need_slots : -1;
        } else if (!reserve_admission && gid < failed_group) {
            // These groups already completed their initial allocation before
            // a later group failed.
            need_blocks = 0;
            need_slots  = 0;
        } else {
            const int  group_common_len = cpEffectiveSeqLenForReserve(cp_mapper, group_type, raw_common_len);
            const int  reuse_blocks_len = malloc_info.reuse_cache ? resource->blocksNum(0, gid) : 0;
            const auto need             = kv_cache_groups_[group_index]->getNeedBlocks(
                group_common_len, group_seq_len, reserve_step, reuse_blocks_len, malloc_info.reuse_cache);
            need_blocks = need.common_blocks + batch_size * need.extra_blocks;
        }
        if (gid == failed_group && failed_need_blocks >= 0) {
            need_blocks = failed_need_blocks;
        }

        const auto&  pool      = group_block_pools_[group_index];
        const size_t available = pool->availableBlocksNum();
        const bool   has_reserve =
            reserve_admission && !isDsv4FixedRegion(region) && total_reservable_available_blocks > 0;
        const size_t group_reserve = has_reserve ? reserve_blocks * available / total_reservable_available_blocks : 0;
        const long long required_available = need_blocks < 0 ? -1 : static_cast<long long>(need_blocks + group_reserve);
        const long long shortfall =
            required_available < 0 ? -1 : std::max(required_available - static_cast<long long>(available), 0LL);
        const size_t layer_count =
            group_index < config_.global_layer_ids.size() ? config_.global_layer_ids[group_index].size() : 0;
        const size_t block_bytes =
            group_index < config_.group_block_size_bytes.size() ? config_.group_block_size_bytes[group_index] : 0;
        const size_t seq_per_block = group_index < config_.group_seq_size_per_block.size() ?
                                         config_.group_seq_size_per_block[group_index] :
                                         config_.seq_size_per_block;

        RTP_LLM_LOG_WARNING("HybridPool malloc failure pool: error_code=602 request_id=%ld gid=%d "
                            "group_type=%s region=%s failed=%d need_blocks=%d need_slots=%d "
                            "group_reserve_blocks=%zu required_available_blocks=%lld shortfall_blocks=%lld "
                            "current_slots=%zu "
                            "current_valid_blocks=%zu total_blocks=%zu available_blocks=%zu free_blocks=%zu "
                            "request_ref_blocks=%zu connector_ref_blocks=%zu block_cache_ref_blocks=%zu "
                            "layer_count=%zu block_bytes=%zu seq_size_per_block=%zu",
                            malloc_info.request_id,
                            gid,
                            cacheGroupTypeName(group_type),
                            cacheRegionName(region),
                            gid == failed_group,
                            need_blocks,
                            need_slots,
                            group_reserve,
                            required_available,
                            shortfall,
                            current_slots,
                            current_valid_blocks,
                            pool->totalBlocksNum(),
                            available,
                            pool->freeBlocksNum(),
                            pool->requestRefBlocksNum(),
                            pool->connectorRefBlocksNum(),
                            pool->blockCacheRefBlocksNum(),
                            layer_count,
                            block_bytes,
                            seq_per_block);
    }
}

}  // namespace rtp_llm
