#include "rtp_llm/cpp/cache/HybridPoolKVCacheAllocator.h"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

#include "rtp_llm/cpp/cache/DeviceBlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/core/OpData.h"

namespace rtp_llm {
namespace {

inline int cpEffectiveSeqLenForReserve(const std::shared_ptr<CPSlotMapper>& mapper,
                                       const CacheConfig&                   config,
                                       size_t                               group_id,
                                       int                                  seq_len) {
    return (mapper && mapper->isSharded()) ? mapper->effectiveSeqLenForAlloc(config, group_id, seq_len) : seq_len;
}

void appendPoolSummary(std::ostringstream&          os,
                       bool&                        has_any,
                       int                          group_id,
                       const std::string&           tag,
                       CacheGroupType               group_type,
                       const DeviceBlockPoolConfig& pool_config) {
    static constexpr double kBytesPerMB = 1024.0 * 1024.0;
    if (has_any) {
        os << "; ";
    }
    has_any = true;
    os << "pool_name=" << pool_config.pool_name << ", group_id=" << group_id << ", tag=" << tag
       << ", type=" << cacheGroupTypeName(group_type) << ", size=" << pool_config.total_size_bytes << " bytes("
       << std::fixed << std::setprecision(2) << static_cast<double>(pool_config.total_size_bytes) / kBytesPerMB
       << " MB)"
       << ", blocks=" << pool_config.physical_block_count;
}

}  // namespace

HybridPoolKVCacheAllocator::HybridPoolKVCacheAllocator(const CacheConfig&                 config,
                                                       AllocationType                     allocation_type,
                                                       const kmonitor::MetricsReporterPtr metrics_reporter,
                                                       int64_t                            reserve_block_ratio,
                                                       RoleType                           role_type):
    HybridKVCacheAllocator(config, allocation_type, metrics_reporter, reserve_block_ratio), role_type_(role_type) {}

bool HybridPoolKVCacheAllocator::doInit() {
    RTP_LLM_CHECK_WITH_INFO(config_.groupNums() > 0, "no cache groups found in CacheConfig");

    const int group_nums = config_.groupNums();
    group_block_pools_.reserve(static_cast<size_t>(group_nums));
    kv_cache_groups_.reserve(static_cast<size_t>(group_nums));

    static constexpr double kBytesPerMB = 1024.0 * 1024.0;
    std::ostringstream      pool_summary;
    size_t                  pool_total_bytes  = 0;
    size_t                  pool_total_blocks = 0;
    bool                    has_pool          = false;

    std::vector<DeviceBlockPoolConfig> group_pool_configs;
    group_pool_configs.reserve(static_cast<size_t>(group_nums));
    for (int group_id = 0; group_id < group_nums; ++group_id) {
        auto pool_config = DeviceBlockPoolConfigHelper::createConfigForGroup(config_, static_cast<size_t>(group_id));
        const auto policy = config_.policyForGroup(static_cast<size_t>(group_id));
        pool_config.use_pinned_cpu_backing = policy.memory_placement == CacheMemoryPlacement::HOST_PINNED;
        pool_config.use_cuda_malloc_backing = use_cuda_malloc_block_pool_ && !pool_config.use_pinned_cpu_backing;
        const auto tag                      = config_.tagForGroup(static_cast<size_t>(group_id));
        const auto group_type               = config_.typeForGroup(static_cast<size_t>(group_id));
        appendPoolSummary(pool_summary, has_pool, group_id, tag, group_type, pool_config);
        pool_total_bytes += pool_config.total_size_bytes;
        pool_total_blocks += pool_config.physical_block_count;
        group_pool_configs.push_back(std::move(pool_config));
    }

    if (has_pool) {
        const auto summary = pool_summary.str();
        RTP_LLM_LOG_INFO("HybridPool pool summary: pools=[%s], total_size=%zu bytes total_size_mb=%.2f "
                         "total_blocks=%zu",
                         summary.c_str(),
                         pool_total_bytes,
                         static_cast<double>(pool_total_bytes) / kBytesPerMB,
                         pool_total_blocks);
    }

    for (int group_id = 0; group_id < group_nums; ++group_id) {
        const auto& pool_config = group_pool_configs[static_cast<size_t>(group_id)];
        const auto  group_type  = config_.typeForGroup(static_cast<size_t>(group_id));
        auto group_pool = std::make_shared<DeviceBlockPool>(std::make_shared<const DeviceBlockPoolConfig>(pool_config));
        RTP_LLM_CHECK_WITH_INFO(group_pool->init(),
                                "Failed to initialize block pool %s(group %d)",
                                pool_config.pool_name.c_str(),
                                group_id);

        const auto& cache_group = config_.topology().groupById(static_cast<size_t>(group_id));

        KVCacheGroupPtr group;
        if (group_type == CacheGroupType::LINEAR) {
            group = std::make_shared<LinearKVCacheGroup>(cache_group, group_pool, group_id, config_.linear_step);
            linear_group_ids_.push_back(group_id);
        } else if (group_type == CacheGroupType::SWA) {
            group = std::make_shared<SWAKVCacheGroup>(cache_group, group_pool, group_id, config_.linear_step);
            swa_group_ids_.push_back(group_id);
        } else {
            group = std::make_shared<FullKVCacheGroup>(cache_group, group_pool, group_id);
            full_group_ids_.push_back(group_id);
        }

        RTP_LLM_CHECK_WITH_INFO(group->init(),
                                "Failed to initialize KVCacheGroup %s(group_id %d)",
                                pool_config.pool_name.c_str(),
                                group_id);
        group_block_pools_.push_back(group_pool);
        kv_cache_groups_.push_back(group);
    }

    RTP_LLM_LOG_INFO("HybridPoolKVCacheAllocator init success, group pools=%zu", group_block_pools_.size());
    return true;
}

int HybridPoolKVCacheAllocator::defaultGroupIdForLayer(int layer_id) const {
    if (layer_id < 0 || static_cast<size_t>(layer_id) >= config_.layer_all_num) {
        RTP_LLM_FAIL("invalid layer_id=%d", layer_id);
    }
    const auto& group    = config_.topology().soleGroupForLayer(layer_id);
    const int   group_id = static_cast<int>(config_.topology().groupIdForTag(group.tag));
    RTP_LLM_CHECK_WITH_INFO(group_id >= 0 && group_id < static_cast<int>(kv_cache_groups_.size()),
                            "invalid default group id %d for layer %d",
                            group_id,
                            layer_id);
    return group_id;
}

int HybridPoolKVCacheAllocator::validateGroupIdForLayer(int layer_id, int group_id) const {
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

GroupedCacheLayerLayout HybridPoolKVCacheAllocator::allLayerCacheBase() const {
    const auto topology = config_.topologyPtr();
    RTP_LLM_CHECK_WITH_INFO(kv_cache_groups_.size() == topology->groups().size(),
                            "cache group count=%zu topology count=%zu",
                            kv_cache_groups_.size(),
                            topology->groups().size());

    GroupedCacheLayerLayout::GroupLayouts groups;
    for (size_t group_id = 0; group_id < kv_cache_groups_.size(); ++group_id) {
        std::vector<BlockBufferPtrInfo> layers(topology->layers().size());
        const auto                      layer_tensors = kv_cache_groups_[group_id]->allLayerCacheBase();
        const auto                      scale_tensors = kv_cache_groups_[group_id]->allLayerScaleCacheBase();
        for (const auto& [layer_id, tensor] : layer_tensors) {
            RTP_LLM_CHECK_WITH_INFO(layer_id >= 0 && static_cast<size_t>(layer_id) < layers.size(),
                                    "layer_id %d out of group kv layout range %zu",
                                    layer_id,
                                    layers.size());
            layers[static_cast<size_t>(layer_id)].kv_addr = tensor;
        }
        for (const auto& [layer_id, tensor] : scale_tensors) {
            RTP_LLM_CHECK_WITH_INFO(layer_id >= 0 && static_cast<size_t>(layer_id) < layers.size(),
                                    "layer_id %d out of group scale layout range %zu",
                                    layer_id,
                                    layers.size());
            layers[static_cast<size_t>(layer_id)].kv_scale_addr = tensor;
        }
        groups.emplace(topology->groupById(group_id).tag, CacheLayerLayout(std::move(layers)));
    }
    return GroupedCacheLayerLayout(topology, std::move(groups));
}

BlockAddrInfo HybridPoolKVCacheAllocator::convertIndexToAddr(int layer_id, int block_id) const {
    const int group_id = defaultGroupIdForLayer(layer_id);
    return kv_cache_groups_[static_cast<size_t>(group_id)]->convertIndexToAddr(layer_id, block_id);
}

std::vector<BlockInfo> HybridPoolKVCacheAllocator::convertIndexToBuffer(int layer_id, int block_id) const {
    const int group_id = defaultGroupIdForLayer(layer_id);
    return kv_cache_groups_[static_cast<size_t>(group_id)]->convertIndexToBuffer(layer_id, block_id);
}

std::vector<BlockInfo> HybridPoolKVCacheAllocator::convertIndexToBuffer(int layer_id,
                                                                        int block_id,
                                                                        int partition_count,
                                                                        int partition_id) const {
    const int group_id = defaultGroupIdForLayer(layer_id);
    return kv_cache_groups_[static_cast<size_t>(group_id)]->convertIndexToBuffer(
        layer_id, block_id, partition_count, partition_id);
}

BlockAddrInfo HybridPoolKVCacheAllocator::convertIndexToAddr(int layer_id, int group_id, int block_id) const {
    RTP_LLM_CHECK_WITH_INFO(group_id >= 0, "invalid cache topology group id=%d", group_id);
    return convertIndexToAddrByTag(layer_id, config_.topology().groupById(static_cast<size_t>(group_id)).tag, block_id);
}

std::vector<BlockInfo>
HybridPoolKVCacheAllocator::convertIndexToBuffer(int layer_id, int group_id, int block_id) const {
    RTP_LLM_CHECK_WITH_INFO(group_id >= 0, "invalid cache topology group id=%d", group_id);
    return convertIndexToBufferByTag(
        layer_id, config_.topology().groupById(static_cast<size_t>(group_id)).tag, block_id);
}

std::vector<BlockInfo> HybridPoolKVCacheAllocator::convertIndexToBuffer(
    int layer_id, int group_id, int block_id, int partition_count, int partition_id) const {
    RTP_LLM_CHECK_WITH_INFO(group_id >= 0, "invalid cache topology group id=%d", group_id);
    return convertIndexToBufferByTag(layer_id,
                                     config_.topology().groupById(static_cast<size_t>(group_id)).tag,
                                     block_id,
                                     partition_count,
                                     partition_id);
}

BlockAddrInfo
HybridPoolKVCacheAllocator::convertIndexToAddrByTag(int layer_id, const std::string& tag, int block_id) const {
    const auto group_id = static_cast<int>(config_.topology().groupIdForTag(tag));
    validateGroupIdForLayer(layer_id, group_id);
    return kv_cache_groups_[static_cast<size_t>(group_id)]->convertIndexToAddr(layer_id, block_id);
}

std::vector<BlockInfo>
HybridPoolKVCacheAllocator::convertIndexToBufferByTag(int layer_id, const std::string& tag, int block_id) const {
    const auto group_id = static_cast<int>(config_.topology().groupIdForTag(tag));
    validateGroupIdForLayer(layer_id, group_id);
    return kv_cache_groups_[static_cast<size_t>(group_id)]->convertIndexToBuffer(layer_id, block_id);
}

std::vector<BlockInfo> HybridPoolKVCacheAllocator::convertIndexToBufferByTag(
    int layer_id, const std::string& tag, int block_id, int partition_count, int partition_id) const {
    const auto group_id = static_cast<int>(config_.topology().groupIdForTag(tag));
    validateGroupIdForLayer(layer_id, group_id);
    return kv_cache_groups_[static_cast<size_t>(group_id)]->convertIndexToBuffer(
        layer_id, block_id, partition_count, partition_id);
}

void HybridPoolKVCacheAllocator::blockBatchCopy(const BlockIdPair* begin_ptr, const BlockIdPair* end_ptr) {
    if (end_ptr == begin_ptr) {
        return;
    }

    RTP_LLM_CHECK_WITH_INFO(config_.topology().hasOneGroupPerLayer(),
                            "legacy layer-only block copy requires exactly one cache group per layer");
    std::vector<TaggedBlockIdPair> tagged_mappings;
    tagged_mappings.reserve(static_cast<size_t>(end_ptr - begin_ptr) * config_.topology().groups().size());
    for (const auto& group : config_.topology().groups()) {
        for (auto it = begin_ptr; it != end_ptr; ++it) {
            tagged_mappings.push_back({group.tag, it->src, it->dst});
        }
    }
    blockBatchCopyByTag(tagged_mappings);
}

void HybridPoolKVCacheAllocator::blockBatchCopyByTag(const std::vector<TaggedBlockIdPair>& copy_mapping) {
    if (copy_mapping.empty()) {
        return;
    }

    size_t copy_nums[BatchCopyParams::TYPE_SIZE] = {};
    for (const auto& mapping : copy_mapping) {
        const auto group_id = static_cast<int>(config_.topology().groupIdForTag(mapping.tag));
        RTP_LLM_CHECK_WITH_INFO(
            static_cast<size_t>(group_id) < group_block_pools_.size(), "missing block pool for group %d", group_id);
        const auto copy_type =
            BatchCopyParams::get_copy_type(group_block_pools_[static_cast<size_t>(group_id)]->where(),
                                           group_block_pools_[static_cast<size_t>(group_id)]->where());
        const auto&  group             = config_.topology().groupById(static_cast<size_t>(group_id));
        const size_t buffers_per_layer = group.kv_scale_stride_bytes > 0 ? 2 : 1;
        copy_nums[copy_type] += config_.layerIdsForGroup(static_cast<size_t>(group_id)).size() * buffers_per_layer;
    }

    BatchCopyParams copy_params;
    for (size_t i = 0; i < BatchCopyParams::TYPE_SIZE; ++i) {
        copy_params.reserve(static_cast<BatchCopyParams::CopyType>(i), copy_nums[i]);
    }

    for (const auto& mapping : copy_mapping) {
        const auto group_id = static_cast<int>(config_.topology().groupIdForTag(mapping.tag));
        RTP_LLM_CHECK_WITH_INFO(
            static_cast<size_t>(group_id) < group_block_pools_.size(), "missing block pool for group %d", group_id);
        const auto&  group               = config_.topology().groupById(static_cast<size_t>(group_id));
        const size_t kv_block_size_bytes = group.kv_block_stride_bytes;
        const size_t scale_block_bytes   = group.kv_scale_stride_bytes;
        const auto   copy_type =
            BatchCopyParams::get_copy_type(group_block_pools_[static_cast<size_t>(group_id)]->where(),
                                           group_block_pools_[static_cast<size_t>(group_id)]->where());

        for (int layer_id : config_.layerIdsForGroup(static_cast<size_t>(group_id))) {
            auto src_addr_info =
                kv_cache_groups_[static_cast<size_t>(group_id)]->convertIndexToAddr(layer_id, mapping.src);
            auto dst_addr_info =
                kv_cache_groups_[static_cast<size_t>(group_id)]->convertIndexToAddr(layer_id, mapping.dst);

            if (!src_addr_info.kv_addr || !dst_addr_info.kv_addr) {
                RTP_LLM_LOG_ERROR("Failed to get block address for pool %s(group %d) layer %d, src_block %d, "
                                  "dst_block %d",
                                  group_block_pools_[static_cast<size_t>(group_id)]->poolName().c_str(),
                                  group_id,
                                  layer_id,
                                  mapping.src,
                                  mapping.dst);
                continue;
            }

            copy_params.add(dst_addr_info.kv_addr, src_addr_info.kv_addr, kv_block_size_bytes, copy_type);

            if (scale_block_bytes > 0 && src_addr_info.kv_scale_addr && dst_addr_info.kv_scale_addr) {
                copy_params.add(dst_addr_info.kv_scale_addr, src_addr_info.kv_scale_addr, scale_block_bytes, copy_type);
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

size_t HybridPoolKVCacheAllocator::activeTreeCachedBlocksNum() const {
    size_t total = 0;
    for (const auto& pool : group_block_pools_) {
        if (pool) {
            total += pool->activeTreeCachedBlocksNum();
        }
    }
    return total;
}

size_t HybridPoolKVCacheAllocator::minTokenCapacity(bool use_free_blocks, bool full_groups_only) const {
    if (group_block_pools_.empty()) {
        return 0;
    }

    auto calculate = [&](bool only_full_groups) {
        size_t min_tokens = std::numeric_limits<size_t>::max();
        bool   saw_group  = false;
        for (size_t group_id = 0; group_id < group_block_pools_.size(); ++group_id) {
            if (only_full_groups && config_.typeForGroup(group_id) != CacheGroupType::FULL) {
                continue;
            }
            if (!group_block_pools_[group_id]) {
                continue;
            }
            saw_group        = true;
            const auto block = use_free_blocks ? group_block_pools_[group_id]->freeBlocksNum() :
                                                 group_block_pools_[group_id]->totalBlocksNum();
            min_tokens       = std::min(min_tokens, block * logicalSeqSizePerBlockForCapacity(group_id));
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
    return minTokenCapacity(/*use_free_blocks=*/true, /*full_groups_only=*/true);
}

size_t HybridPoolKVCacheAllocator::totalTokensNum() const {
    return minTokenCapacity(/*use_free_blocks=*/false, /*full_groups_only=*/true);
}

size_t HybridPoolKVCacheAllocator::totalBlocksNum() const {
    size_t total = 0;
    for (const auto& pool : group_block_pools_) {
        total += pool->totalBlocksNum();
    }
    return total;
}

size_t HybridPoolKVCacheAllocator::maxAvailableTokensNum() const {
    return minTokenCapacity(/*use_free_blocks=*/false, /*full_groups_only=*/true);
}

KVCacheTokenCapacity HybridPoolKVCacheAllocator::tokenCapacity(size_t default_seq_size_per_block) const {
    (void)default_seq_size_per_block;
    if (group_block_pools_.empty()) {
        return {};
    }
    size_t total_tokens     = std::numeric_limits<size_t>::max();
    size_t available_tokens = std::numeric_limits<size_t>::max();
    bool   has_pool         = false;
    for (size_t group_id = 0; group_id < group_block_pools_.size(); ++group_id) {
        const auto& pool = group_block_pools_[group_id];
        if (!pool) {
            continue;
        }
        const size_t seq_size = config_.seqSizePerBlockForGroup(group_id);
        total_tokens          = std::min(total_tokens, pool->totalBlocksNum() * seq_size);
        available_tokens      = std::min(available_tokens, pool->freeBlocksNum() * seq_size);
        has_pool              = true;
    }
    return has_pool ? KVCacheTokenCapacity{total_tokens, available_tokens} : KVCacheTokenCapacity{};
}

size_t HybridPoolKVCacheAllocator::reserveBlocksForPoolMetrics(size_t pool_index) const {
    return reserveBlocksForPool(pool_index);
}

void HybridPoolKVCacheAllocator::regUserMr(size_t model_id, std::shared_ptr<CacheStore> cache_store) {
    for (auto& pool : group_block_pools_) {
        pool->regUserMr(model_id, cache_store);
    }
}

int64_t HybridPoolKVCacheAllocator::getMrCostTimeMs() const {
    int64_t total = 0;
    for (const auto& pool : group_block_pools_) {
        total += pool->getMrCostTimeMs();
    }
    return total;
}

size_t HybridPoolKVCacheAllocator::totalReservableFreeBlocks() const {
    size_t total = 0;
    for (size_t group_id = 0; group_id < group_block_pools_.size(); ++group_id) {
        if (!group_block_pools_[group_id] || group_id >= kv_cache_groups_.size() || !kv_cache_groups_[group_id]
            || !kv_cache_groups_[group_id]->isReservable() || config_.usesExplicitIndependentBlocks(group_id)) {
            continue;
        }
        total += group_block_pools_[group_id]->freeBlocksNum();
    }
    return total;
}

size_t HybridPoolKVCacheAllocator::reservableFreeBlocksNum() const {
    return totalReservableFreeBlocks();
}

size_t HybridPoolKVCacheAllocator::reserveBlocksForPool(size_t group_id) const {
    if (group_id >= group_block_pools_.size() || group_id >= kv_cache_groups_.size()
        || !group_block_pools_[group_id] || !kv_cache_groups_[group_id]
        || !kv_cache_groups_[group_id]->isReservable() || config_.usesExplicitIndependentBlocks(group_id)) {
        return 0;
    }

    size_t total_reservable_blocks = 0;
    for (size_t current_group_id = 0; current_group_id < group_block_pools_.size(); ++current_group_id) {
        if (!group_block_pools_[current_group_id] || current_group_id >= kv_cache_groups_.size()
            || !kv_cache_groups_[current_group_id] || !kv_cache_groups_[current_group_id]->isReservable()
            || config_.usesExplicitIndependentBlocks(current_group_id)) {
            continue;
        }
        total_reservable_blocks += group_block_pools_[current_group_id]->totalBlocksNum();
    }
    return total_reservable_blocks == 0 ?
               0 :
               reserveBlocksNum() * group_block_pools_[group_id]->totalBlocksNum() / total_reservable_blocks;
}

MallocStatus HybridPoolKVCacheAllocator::evaluateInitCapacity(const MallocInfo& malloc_info,
                                                              size_t            reserve_blocks,
                                                              InitCapacityMode  mode) const {
    return evaluateInitCapacityImpl(malloc_info, reserve_blocks, mode, nullptr);
}

MallocStatus HybridPoolKVCacheAllocator::evaluateInitCapacityImpl(
    const MallocInfo&                     malloc_info,
    size_t                                reserve_blocks,
    InitCapacityMode                      mode,
    const std::vector<RequiredPositions>* required_positions) const {
    if (!malloc_info.batch_kv_cache_resource || !malloc_info.complete_token_ids) {
        return MallocStatus::NONE;
    }
    RTP_LLM_CHECK_WITH_INFO(required_positions == nullptr || required_positions->size() == kv_cache_groups_.size(),
                            "prepared load group count mismatch: required_positions=%zu groups=%zu",
                            required_positions == nullptr ? 0 : required_positions->size(),
                            kv_cache_groups_.size());
    const auto& cp_mapper          = cp_slot_mapper_;
    const int   batch_size         = malloc_info.batch_kv_cache_resource->batchSize();
    const int   total_seq_len      = malloc_info.complete_token_ids->totalSeqLength();
    const int   raw_common_seq_len = std::min(malloc_info.complete_token_ids->commonSeqLength(), total_seq_len);
    const int   raw_seq_len        = malloc_info.complete_token_ids->seqLength();
    const int   reserve_step       = malloc_info.complete_token_ids->getReserveStep();
    const bool  reuse_enabled      = malloc_info.reuse_cache;

    size_t total_reservable_blocks = 0;
    for (size_t group_id = 0; group_id < group_block_pools_.size(); ++group_id) {
        if (!group_block_pools_[group_id] || group_id >= kv_cache_groups_.size() || !kv_cache_groups_[group_id]
            || !kv_cache_groups_[group_id]->isReservable() || config_.usesExplicitIndependentBlocks(group_id)) {
            continue;
        }
        total_reservable_blocks += group_block_pools_[group_id]->totalBlocksNum();
    }

    MallocStatus            status = MallocStatus::NONE;
    const RequiredPositions no_required_positions;
    for (int group_id = 0; group_id < static_cast<int>(kv_cache_groups_.size()); ++group_id) {
        const size_t group_index = static_cast<size_t>(group_id);
        const int    group_common_seq =
            cpEffectiveSeqLenForReserve(cp_mapper, config_, group_index, raw_common_seq_len);
        const int group_seq_len = cpEffectiveSeqLenForReserve(cp_mapper, config_, group_index, raw_seq_len);
        const int group_reuse_blocks_len = malloc_info.batch_kv_cache_resource->blocksNum(0, group_id);
        const auto& group_required_positions =
            required_positions == nullptr ? no_required_positions : (*required_positions)[group_index];
        const auto need = kv_cache_groups_[group_index]->getNeedBlocks(
            group_common_seq,
            group_seq_len,
            reserve_step,
            group_reuse_blocks_len,
            reuse_enabled,
            group_required_positions);
        const int need_blocks = need.common_blocks + batch_size * need.extra_blocks;
        const size_t planned_blocks = static_cast<size_t>(std::max(need_blocks, 0));

        const auto&  pool            = group_block_pools_[group_index];
        const size_t total_blocks    = pool->totalBlocksNum();
        const auto   demand          = initBlockDemand(malloc_info, planned_blocks, group_id);
        const size_t group_reserve_blocks =
            (!kv_cache_groups_[group_index]->isReservable() || config_.usesExplicitIndependentBlocks(group_index)
             || total_reservable_blocks == 0) ?
                0 :
                reserve_blocks * total_blocks / total_reservable_blocks;

        if (demand.retained_blocks > total_blocks || planned_blocks > total_blocks - demand.retained_blocks
            || group_reserve_blocks > total_blocks - demand.retained_blocks - planned_blocks) {
            if (malloc_info.verbose) {
                RTP_LLM_LOG_INFO("HybridPool initMalloc permanently rejected: request_id=%ld pool_name=%s "
                                 "group=%d tag=%s retained_blocks=%zu planned_blocks=%zu total_blocks=%zu "
                                 "reserve_blocks=%zu group_reserve_blocks=%zu",
                                 malloc_info.request_id,
                                 pool->poolName().c_str(),
                                 group_id,
                                 config_.tagForGroup(group_index).c_str(),
                                 demand.retained_blocks,
                                 planned_blocks,
                                 total_blocks,
                                 reserve_blocks,
                                 group_reserve_blocks);
            }
            return MallocStatus::PERMANENT_RESOURCE_EXHAUSTED;
        }

        if (mode != InitCapacityMode::TOTAL_AND_AVAILABLE || status != MallocStatus::NONE) {
            continue;
        }

        const size_t required_free_blocks = demand.additional_blocks + group_reserve_blocks;
        size_t       free_blocks          = pool->freeBlocksNum();
        if (free_blocks < required_free_blocks
            && required_free_blocks <= static_cast<size_t>(std::numeric_limits<int>::max())) {
            (void)kv_cache_groups_[group_index]->ensureFreeBlocks(static_cast<int>(required_free_blocks));
            free_blocks = pool->freeBlocksNum();
        }
        if (free_blocks < required_free_blocks) {
            if (malloc_info.verbose) {
                RTP_LLM_LOG_INFO("HybridPool initMalloc rejected by reserve blocks: request_id=%ld pool_name=%s "
                                 "group=%d need_blocks=%zu total_blocks=%zu free_blocks=%zu "
                                 "reserve_blocks=%zu group_reserve_blocks=%zu",
                                 malloc_info.request_id,
                                 pool->poolName().c_str(),
                                 group_id,
                                 demand.additional_blocks,
                                 total_blocks,
                                 free_blocks,
                                 reserve_blocks,
                                 group_reserve_blocks);
            }
            // Keep scanning: a later pool may turn this into a permanent verdict.
            status = MallocStatus::RETRYABLE_RESOURCE_EXHAUSTED;
        }
    }
    return status;
}

bool HybridPoolKVCacheAllocator::hasAvailableBlocksForReserve(const MallocInfo& malloc_info,
                                                              size_t            reserve_blocks) const {
    return evaluateInitCapacity(malloc_info, reserve_blocks, InitCapacityMode::TOTAL_AND_AVAILABLE)
           == MallocStatus::NONE;
}

MallocStatus HybridPoolKVCacheAllocator::evaluatePreparedInitCapacity(const MallocInfo&       malloc_info,
                                                                      size_t                  reserve_blocks,
                                                                      const PreparedKVCache& prepared,
                                                                      bool has_load_context) const {
    if (reserve_blocks == 0 && !has_load_context) {
        return MallocStatus::NONE;
    }
    if (!has_load_context) {
        return evaluateInitCapacity(malloc_info, reserve_blocks, InitCapacityMode::TOTAL_AND_AVAILABLE);
    }
    return evaluateInitCapacityImpl(malloc_info,
                                    reserve_blocks,
                                    InitCapacityMode::TOTAL_AND_AVAILABLE,
                                    &prepared.required_positions);
}

// Per-pool KV-exhaustion record. This is the primary field-debug tool for
// KV-exhaustion incidents: one aggregate line plus one line per pool carrying
// the demand, the reserve share, the shortfall and the pool's ref-count split.
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
    const auto& cp_mapper      = cp_slot_mapper_;
    const int   batch_size     = resource->batchSize();
    const int   raw_seq_len    = incremental ? malloc_info.incrSeqLen() : malloc_info.complete_token_ids->seqLength();
    const int   raw_common_len = std::min(malloc_info.complete_token_ids->commonSeqLength(), raw_seq_len);
    const int   total_seq_len  = malloc_info.complete_token_ids->totalSeqLength();
    const int   request_reserve_step = malloc_info.complete_token_ids->getReserveStep();
    const bool  reserve_admission    = !incremental && failed_group < 0;
    const int   reserve_step         = incremental || reserve_admission ? request_reserve_step : 0;
    const int   planning_raw_seq_len = !incremental && !reserve_admission ? raw_common_len : raw_seq_len;
    const auto  reserve_blocks       = reserveBlocksNum();

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
        const size_t group_index   = static_cast<size_t>(gid);
        const auto   group_type    = config_.typeForGroup(group_index);
        const int    group_seq_len = cpEffectiveSeqLenForReserve(cp_mapper, config_, group_index, planning_raw_seq_len);

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
            // Dense groups materialize every logical slot. Sparse groups
            // (LINEAR / SWA) skip slots, so their exact physical request is the
            // value the group allocator reported immediately before this snapshot.
            need_blocks = kv_cache_groups_[group_index]->hasSparseSlots() ? -1 : need_slots;
        } else if (!reserve_admission && gid < failed_group) {
            // These groups already completed their initial allocation before
            // a later group failed.
            need_blocks = 0;
            need_slots  = 0;
        } else {
            const int  group_common_len = cpEffectiveSeqLenForReserve(cp_mapper, config_, group_index, raw_common_len);
            const int  reuse_blocks_len = malloc_info.reuse_cache ? resource->blocksNum(0, gid) : 0;
            const auto need             = kv_cache_groups_[group_index]->getNeedBlocks(
                group_common_len, group_seq_len, reserve_step, reuse_blocks_len, malloc_info.reuse_cache);
            need_blocks = need.common_blocks + batch_size * need.extra_blocks;
        }
        if (gid == failed_group && failed_need_blocks >= 0) {
            need_blocks = failed_need_blocks;
        }

        const auto&  pool          = group_block_pools_[group_index];
        const size_t free_blocks   = pool->freeBlocksNum();
        const size_t group_reserve = reserve_admission ? reserveBlocksForPool(group_index) : 0;
        const long long required_available = need_blocks < 0 ? -1 : static_cast<long long>(need_blocks + group_reserve);
        const long long shortfall =
            required_available < 0 ? -1 : std::max(required_available - static_cast<long long>(free_blocks), 0LL);

        RTP_LLM_LOG_WARNING("HybridPool malloc failure pool: error_code=602 request_id=%ld gid=%d pool_name=%s "
                            "group_type=%s tag=%s failed=%d need_blocks=%d need_slots=%d "
                            "group_reserve_blocks=%zu required_available_blocks=%lld shortfall_blocks=%lld "
                            "current_slots=%zu "
                            "current_valid_blocks=%zu total_blocks=%zu free_blocks=%zu "
                            "request_ref_blocks=%zu connector_ref_blocks=%zu block_cache_ref_blocks=%zu "
                            "layer_count=%zu block_bytes=%zu seq_size_per_block=%zu",
                            malloc_info.request_id,
                            gid,
                            pool->poolName().c_str(),
                            cacheGroupTypeName(group_type),
                            config_.tagForGroup(group_index).c_str(),
                            gid == failed_group,
                            need_blocks,
                            need_slots,
                            group_reserve,
                            required_available,
                            shortfall,
                            current_slots,
                            current_valid_blocks,
                            pool->totalBlocksNum(),
                            free_blocks,
                            pool->referencedBlocksNum(BlockRefType::REQUEST),
                            pool->referencedBlocksNum(BlockRefType::STORAGE_BACKEND),
                            pool->referencedBlocksNum(BlockRefType::BLOCK_CACHE),
                            config_.layerIdsForGroup(group_index).size(),
                            config_.blockSizeBytesForGroup(group_index),
                            config_.seqSizePerBlockForGroup(group_index));
    }
}

}  // namespace rtp_llm
