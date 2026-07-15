#include "rtp_llm/cpp/cache/allocator/HybridPoolKVCacheAllocator.h"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "rtp_llm/cpp/cache/DeviceBlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/cache/block_tree_cache/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
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

void appendPoolSummary(std::ostringstream&          os,
                       bool&                        has_any,
                       int                          gid,
                       const std::string&           tag,
                       CacheGroupType               group_type,
                       const DeviceBlockPoolConfig& pool_config) {
    static constexpr double kBytesPerMB = 1024.0 * 1024.0;
    if (has_any) {
        os << "; ";
    }
    has_any = true;
    os << "pool_name=" << pool_config.pool_name << ", gid=" << gid << ", tag=" << tag
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

    static constexpr double kBytesPerMB = 1024.0 * 1024.0;
    std::ostringstream      pool_summary;
    size_t                  pool_total_bytes  = 0;
    size_t                  pool_total_blocks = 0;
    bool                    has_pool          = false;

    std::vector<DeviceBlockPoolConfig> group_pool_configs;
    group_pool_configs.reserve(static_cast<size_t>(group_nums));
    for (int gid = 0; gid < group_nums; ++gid) {
        auto       pool_config = DeviceBlockPoolConfigHelper::createConfigForGroup(config_, static_cast<size_t>(gid));
        const auto tag         = config_.tagForGroup(static_cast<size_t>(gid));
        const auto group_type  = config_.typeForGroup(static_cast<size_t>(gid));
        appendPoolSummary(pool_summary, has_pool, gid, tag, group_type, pool_config);
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

    for (int gid = 0; gid < group_nums; ++gid) {
        const auto& pool_config = group_pool_configs[static_cast<size_t>(gid)];
        const auto  group_type  = config_.typeForGroup(static_cast<size_t>(gid));

        auto device_config                     = std::make_shared<DeviceBlockPoolConfig>(pool_config);
        device_config->allocation_type         = allocation_type_;
        device_config->use_cuda_malloc_backing = use_cuda_malloc_block_pool_;

        std::shared_ptr<const DeviceBlockPoolConfig> const_config = device_config;
        auto                                         group_pool   = std::make_shared<DeviceBlockPool>(const_config);
        RTP_LLM_CHECK_WITH_INFO(
            group_pool->init(), "Failed to initialize block pool %s(group %d)", pool_config.pool_name.c_str(), gid);

        // Classify group ids by type (used by the base class); the DeviceKVCacheGroup
        // objects are created and owned by BlockTreeCache (see BlockTreeCacheFactory).
        if (group_type == CacheGroupType::LINEAR) {
            linear_group_ids_.push_back(gid);
        } else if (group_type == CacheGroupType::SWA) {
            swa_group_ids_.push_back(gid);
        } else {
            full_group_ids_.push_back(gid);
        }

        group_block_pools_.push_back(group_pool);
    }

    RTP_LLM_LOG_INFO("HybridPoolKVCacheAllocator init success, group pools=%zu", group_block_pools_.size());
    return true;
}

int HybridPoolKVCacheAllocator::defaultGroupIdForLayer(int layer_id) const {
    if (layer_id < 0 || static_cast<size_t>(layer_id) >= config_.layer_all_num) {
        RTP_LLM_FAIL("invalid layer_id=%d", layer_id);
    }
    const int gid = config_.groupIdFor(layer_id);
    RTP_LLM_CHECK_WITH_INFO(
        gid >= 0 && gid < config_.groupNums(), "invalid default group id %d for layer %d", gid, layer_id);
    return gid;
}

int HybridPoolKVCacheAllocator::validateGroupIdForLayer(int layer_id, int group_id) const {
    RTP_LLM_CHECK_WITH_INFO(
        group_id >= 0 && group_id < config_.groupNums(), "invalid group id %d for layer %d", group_id, layer_id);
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

void HybridPoolKVCacheAllocator::referenceBlocksInGroup(int                     gid,
                                                        const BlockIndicesType& blocks,
                                                        bool                    is_connector) const {
    // Single-count pool: request and connector holders are the same reference category.
    // is_connector is retained in the signature for call-site parity but no longer selects
    // an independent counter.
    (void)is_connector;
    group_block_pools_[static_cast<size_t>(gid)]->incRef(blocks);
}

void HybridPoolKVCacheAllocator::freeBlocksInGroup(int gid, const BlockIndicesType& blocks, bool is_connector) {
    (void)is_connector;
    group_block_pools_[static_cast<size_t>(gid)]->decRef(blocks);
}

CacheLayerLayout HybridPoolKVCacheAllocator::allLayerCacheBase() const {
    CacheLayerLayout layout;
    const auto       layer_group_ids = config_.layerGroupIdsSnapshot();
    layout.layer_to_group_ids        = layer_group_ids;
    layout.group_types               = config_.groupTypesSnapshot();
    layout.group_tags                = config_.groupTagsSnapshot();
    layout.layer_tag_to_group_id     = config_.layerTagToGroupIdSnapshot();
    layout.group_seq_size_per_block  = config_.group_seq_size_per_block;
    layout.layer_group_types.resize(config_.layer_all_num, CacheGroupType::FULL);
    for (size_t layer_id = 0; layer_id < layer_group_ids.size() && layer_id < layout.layer_group_types.size();
         ++layer_id) {
        if (!layer_group_ids[layer_id].empty()) {
            layout.layer_group_types[layer_id] =
                config_.typeForGroup(static_cast<size_t>(layer_group_ids[layer_id].front()));
        }
    }

    layout.layers_to_kv_buffer_ptrs.resize(config_.layer_all_num);
    layout.layers_to_scale_buffer_ptrs.resize(config_.layer_all_num);
    const size_t group_count = static_cast<size_t>(config_.groupNums());
    layout.layers_to_kv_buffer_ptrs_by_group.resize(config_.layer_all_num);
    layout.layers_to_scale_buffer_ptrs_by_group.resize(config_.layer_all_num);
    for (size_t layer_id = 0; layer_id < static_cast<size_t>(config_.layer_all_num); ++layer_id) {
        layout.layers_to_kv_buffer_ptrs_by_group[layer_id].resize(group_count);
        layout.layers_to_scale_buffer_ptrs_by_group[layer_id].resize(group_count);
    }

    for (size_t layer_id = 0; layer_id < static_cast<size_t>(config_.layer_all_num); ++layer_id) {
        if (layer_id >= layer_group_ids.size() || layer_group_ids[layer_id].size() != 1) {
            continue;
        }
        const int gid = layer_group_ids[layer_id][0];
        RTP_LLM_CHECK_WITH_INFO(
            gid >= 0 && gid < config_.groupNums(), "invalid single-tag group id %d for layer %zu", gid, layer_id);
        const auto layer_tensors = group(gid)->allLayerCacheBase();
        const auto scale_tensors = group(gid)->allLayerScaleCacheBase();
        auto       it            = layer_tensors.find(static_cast<int>(layer_id));
        if (it != layer_tensors.end()) {
            layout.layers_to_kv_buffer_ptrs[layer_id] = it->second;
        }
        auto scale_it = scale_tensors.find(static_cast<int>(layer_id));
        if (scale_it != scale_tensors.end()) {
            layout.layers_to_scale_buffer_ptrs[layer_id] = scale_it->second;
        }
    }

    for (int gid = 0; gid < config_.groupNums(); ++gid) {
        const auto layer_tensors = group(gid)->allLayerCacheBase();
        const auto scale_tensors = group(gid)->allLayerScaleCacheBase();
        for (const auto& [layer_id, tensor] : layer_tensors) {
            RTP_LLM_CHECK_WITH_INFO(
                layer_id >= 0 && static_cast<size_t>(layer_id) < layout.layers_to_kv_buffer_ptrs_by_group.size(),
                "layer_id %d out of by-group kv layout range %zu",
                layer_id,
                layout.layers_to_kv_buffer_ptrs_by_group.size());
            layout.layers_to_kv_buffer_ptrs_by_group[static_cast<size_t>(layer_id)][static_cast<size_t>(gid)] = tensor;
        }
        for (const auto& [layer_id, tensor] : scale_tensors) {
            RTP_LLM_CHECK_WITH_INFO(
                layer_id >= 0 && static_cast<size_t>(layer_id) < layout.layers_to_scale_buffer_ptrs_by_group.size(),
                "layer_id %d out of by-group scale layout range %zu",
                layer_id,
                layout.layers_to_scale_buffer_ptrs_by_group.size());
            layout.layers_to_scale_buffer_ptrs_by_group[static_cast<size_t>(layer_id)][static_cast<size_t>(gid)] =
                tensor;
        }
    }
    return layout;
}

BlockAddrInfo HybridPoolKVCacheAllocator::convertIndexToAddr(int layer_id, int block_id) const {
    const int gid = defaultGroupIdForLayer(layer_id);
    return group(gid)->convertIndexToAddr(layer_id, block_id);
}

std::vector<BlockInfo> HybridPoolKVCacheAllocator::convertIndexToBuffer(int layer_id, int block_id) const {
    const int gid = defaultGroupIdForLayer(layer_id);
    return group(gid)->convertIndexToBuffer(layer_id, block_id);
}

std::vector<BlockInfo> HybridPoolKVCacheAllocator::convertIndexToBuffer(int layer_id,
                                                                        int block_id,
                                                                        int partition_count,
                                                                        int partition_id) const {
    const int gid = defaultGroupIdForLayer(layer_id);
    return group(gid)->convertIndexToBuffer(layer_id, block_id, partition_count, partition_id);
}

BlockAddrInfo HybridPoolKVCacheAllocator::convertIndexToAddr(int layer_id, int group_id, int block_id) const {
    const int gid = validateGroupIdForLayer(layer_id, group_id);
    return group(gid)->convertIndexToAddr(layer_id, block_id);
}

std::vector<BlockInfo>
HybridPoolKVCacheAllocator::convertIndexToBuffer(int layer_id, int group_id, int block_id) const {
    const int gid = validateGroupIdForLayer(layer_id, group_id);
    return group(gid)->convertIndexToBuffer(layer_id, block_id);
}

std::vector<BlockInfo> HybridPoolKVCacheAllocator::convertIndexToBuffer(
    int layer_id, int group_id, int block_id, int partition_count, int partition_id) const {
    const int gid = validateGroupIdForLayer(layer_id, group_id);
    return group(gid)->convertIndexToBuffer(layer_id, block_id, partition_count, partition_id);
}

void HybridPoolKVCacheAllocator::blockBatchCopy(const BlockIdPair* begin_ptr, const BlockIdPair* end_ptr) {
    if (end_ptr == begin_ptr) {
        return;
    }

    size_t copy_nums[BatchCopyParams::TYPE_SIZE] = {};
    for (int gid = 0; gid < config_.groupNums(); ++gid) {
        RTP_LLM_CHECK_WITH_INFO(
            static_cast<size_t>(gid) < group_block_pools_.size(), "missing block pool for group %d", gid);
        const auto   copy_type = BatchCopyParams::get_copy_type(group_block_pools_[static_cast<size_t>(gid)]->where(),
                                                              group_block_pools_[static_cast<size_t>(gid)]->where());
        const auto&  spec      = config_.specForGroup(static_cast<size_t>(gid));
        const size_t buffers_per_layer = spec->scale_block_size_bytes() > 0 ? 2 : 1;
        copy_nums[copy_type] += config_.layerIdsForGroup(static_cast<size_t>(gid)).size()
                                * static_cast<size_t>(end_ptr - begin_ptr) * buffers_per_layer;
    }

    BatchCopyParams copy_params;
    for (size_t i = 0; i < BatchCopyParams::TYPE_SIZE; ++i) {
        copy_params.reserve(static_cast<BatchCopyParams::CopyType>(i), copy_nums[i]);
    }

    for (auto it = begin_ptr; it != end_ptr; ++it) {
        auto [src_block_index, dest_block_index] = *it;

        for (int gid = 0; gid < config_.groupNums(); ++gid) {
            const auto&  spec                = config_.specForGroup(static_cast<size_t>(gid));
            const size_t kv_block_size_bytes = spec->block_size_bytes();
            const size_t scale_block_bytes   = spec->scale_block_size_bytes();
            const auto   copy_type =
                BatchCopyParams::get_copy_type(group_block_pools_[static_cast<size_t>(gid)]->where(),
                                               group_block_pools_[static_cast<size_t>(gid)]->where());

            for (int layer_id : config_.layerIdsForGroup(static_cast<size_t>(gid))) {
                auto src_addr_info = group(gid)->convertIndexToAddr(layer_id, src_block_index);
                auto dst_addr_info = group(gid)->convertIndexToAddr(layer_id, dest_block_index);

                if (!src_addr_info.kv_addr || !dst_addr_info.kv_addr) {
                    RTP_LLM_LOG_ERROR("Failed to get block address for pool %s(group %d) layer %d, src_block %d, "
                                      "dst_block %d",
                                      group_block_pools_[static_cast<size_t>(gid)]->poolName().c_str(),
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

size_t HybridPoolKVCacheAllocator::perPoolAvailableBlocks(int gid) const {
    if (gid < 0 || static_cast<size_t>(gid) >= group_block_pools_.size()
        || !group_block_pools_[static_cast<size_t>(gid)]) {
        return 0;
    }
    const size_t free_blocks = group_block_pools_[static_cast<size_t>(gid)]->freeBlocksNum();
    const size_t evictable   = block_tree_cache_ ? block_tree_cache_->evictableBlocksNum(gid) : 0;
    return free_blocks + evictable;
}

size_t HybridPoolKVCacheAllocator::availableBlocksNum() const {
    size_t total = 0;
    for (int gid = 0; gid < static_cast<int>(group_block_pools_.size()); ++gid) {
        total += perPoolAvailableBlocks(gid);
    }
    return total;
}

BatchKVCacheResourcePtr HybridPoolKVCacheAllocator::popBlocksFromCache(size_t min_blocks_to_free) {
    if (min_blocks_to_free == 0 || !block_tree_cache_) {
        return nullptr;
    }
    // BlockTreeCache reclaims (drops) device blocks in place and returns them to the
    // owning pools; it does not surface evicted keys/slots. Callers only use the
    // return value to decide whether space was freed, so return nullptr once the
    // reclaim is done.
    const int reclaimed = block_tree_cache_->reclaimBlocks(min_blocks_to_free, Tier::DEVICE);
    if (reclaimed > 0 && metrics_reporter_) {
        RtpLLMCacheEvictionMetricsCollector collector;
        collector.lifetime_ms = 0;
        kmonitor::MetricsTags tags("scope", "gpu");
        tags.AddTag("evict_policy", "chain");
        tags.AddTag("backing", "device");
        metrics_reporter_->report<RtpLLMCacheEvictionMetrics, RtpLLMCacheEvictionMetricsCollector>(&tags, &collector);
    }
    return nullptr;
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
                group_block_pools_[static_cast<size_t>(gid)]->decRef(blocks_to_free);
            }
        }
    }
}

size_t HybridPoolKVCacheAllocator::requestRefBlocksNum() const {
    return 0;  // single-count pool: holder-type split is not recoverable
}

size_t HybridPoolKVCacheAllocator::connectorRefBlocksNum() const {
    return 0;  // single-count pool: holder-type split is not recoverable
}

size_t HybridPoolKVCacheAllocator::blockCacheRefBlocksNum() const {
    return 0;  // single-count pool: holder-type split is not recoverable
}

size_t HybridPoolKVCacheAllocator::notInUseBlocksNum() const {
    size_t total = 0;
    for (const auto& pool : group_block_pools_) {
        if (pool) {
            total += pool->freeBlocksNum();  // conservative: excludes connector in-flight
        }
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
            if (only_full_groups && config_.typeForGroup(gid) != CacheGroupType::FULL) {
                continue;
            }
            if (!group_block_pools_[gid]) {
                continue;
            }
            saw_group        = true;
            const auto block = use_available_blocks ? perPoolAvailableBlocks(static_cast<int>(gid)) :
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

KVCacheTokenCapacity HybridPoolKVCacheAllocator::tokenCapacity(size_t default_seq_size_per_block) const {
    if (group_block_pools_.empty()) {
        return {};
    }
    size_t total_tokens     = std::numeric_limits<size_t>::max();
    size_t available_tokens = std::numeric_limits<size_t>::max();
    bool   has_pool         = false;
    for (size_t gid = 0; gid < group_block_pools_.size(); ++gid) {
        const auto& pool = group_block_pools_[gid];
        if (!pool) {
            continue;
        }
        const size_t seq_size =
            (gid < config_.group_seq_size_per_block.size() && config_.group_seq_size_per_block[gid] > 0) ?
                config_.group_seq_size_per_block[gid] :
                default_seq_size_per_block;
        total_tokens     = std::min(total_tokens, pool->totalBlocksNum() * seq_size);
        available_tokens = std::min(available_tokens, perPoolAvailableBlocks(static_cast<int>(gid)) * seq_size);
        has_pool         = true;
    }
    return has_pool ? KVCacheTokenCapacity{total_tokens, available_tokens} : KVCacheTokenCapacity{};
}

std::vector<KVCachePoolMetricsSnapshot> HybridPoolKVCacheAllocator::poolMetricsSnapshots() const {
    std::vector<KVCachePoolMetricsSnapshot> snapshots;
    snapshots.reserve(group_block_pools_.size());
    const size_t reserve_blocks                    = reserveBlockNum();
    const size_t total_reservable_available_blocks = totalReservableAvailableBlocks();
    for (size_t gid = 0; gid < group_block_pools_.size(); ++gid) {
        const auto& pool = group_block_pools_[gid];
        if (!pool) {
            continue;
        }
        KVCachePoolMetricsSnapshot snapshot;
        snapshot.pool_index           = gid;
        snapshot.pool_name            = pool->poolName();
        snapshot.total_blocks         = pool->totalBlocksNum();
        snapshot.available_blocks     = perPoolAvailableBlocks(static_cast<int>(gid));
        snapshot.free_blocks          = pool->freeBlocksNum();
        snapshot.request_ref_blocks   = 0;  // single-count pool: holder-type split is not recoverable
        snapshot.connector_ref_blocks = 0;  // single-count pool: holder-type split is not recoverable
        snapshot.reserve_blocks       = reserveBlocksForPool(gid, reserve_blocks, total_reservable_available_blocks);
        snapshot.used_ratio           = (snapshot.total_blocks == 0) ?
                                            0.0f :
                                            static_cast<float>(100.0 * (snapshot.total_blocks - snapshot.available_blocks)
                                                     / static_cast<double>(snapshot.total_blocks));
        snapshots.push_back(snapshot);
    }
    return snapshots;
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

size_t HybridPoolKVCacheAllocator::totalReservableAvailableBlocks() const {
    size_t total = 0;
    for (size_t gid = 0; gid < group_block_pools_.size(); ++gid) {
        if (!group_block_pools_[gid] || config_.usesExplicitIndependentBlocks(gid)) {
            continue;
        }
        total += perPoolAvailableBlocks(static_cast<int>(gid));
    }
    return total;
}

size_t HybridPoolKVCacheAllocator::reserveBlocksForPool(size_t gid,
                                                        size_t reserve_blocks,
                                                        size_t total_reservable_available_blocks) const {
    if (gid >= group_block_pools_.size() || !group_block_pools_[gid] || config_.usesExplicitIndependentBlocks(gid)
        || total_reservable_available_blocks == 0) {
        return 0;
    }
    return reserve_blocks * perPoolAvailableBlocks(static_cast<int>(gid)) / total_reservable_available_blocks;
}

bool HybridPoolKVCacheAllocator::hasAvailableBlocksForReserve(const MallocInfo& malloc_info,
                                                              size_t            reserve_blocks) const {
    if (!malloc_info.batch_kv_cache_resource || !malloc_info.complete_token_ids) {
        return true;
    }
    const auto& cp_mapper          = cp_slot_mapper_;
    const int   batch_size         = malloc_info.batch_kv_cache_resource->batchSize();
    const int   total_seq_len      = malloc_info.complete_token_ids->totalSeqLength();
    const int   raw_common_seq_len = std::min(malloc_info.complete_token_ids->commonSeqLength(), total_seq_len);
    const int   raw_seq_len        = malloc_info.complete_token_ids->seqLength();
    const int   reserve_step       = malloc_info.complete_token_ids->getReserveStep();
    const bool  reuse_enabled      = malloc_info.reuse_cache;

    const size_t total_reservable_available_blocks = totalReservableAvailableBlocks();

    for (int gid = 0; gid < config_.groupNums(); ++gid) {
        const auto group_type             = config_.typeForGroup(static_cast<size_t>(gid));
        const int  group_common_seq       = cpEffectiveSeqLenForReserve(cp_mapper, group_type, raw_common_seq_len);
        const int  group_seq_len          = cpEffectiveSeqLenForReserve(cp_mapper, group_type, raw_seq_len);
        const int  group_reuse_blocks_len = reuse_enabled ? malloc_info.batch_kv_cache_resource->blocksNum(0, gid) : 0;
        const auto need                   = group(gid)->getNeedBlocks(
            group_common_seq, group_seq_len, reserve_step, group_reuse_blocks_len, reuse_enabled);
        const int need_blocks = need.common_blocks + batch_size * need.extra_blocks;
        if (need_blocks <= 0) {
            continue;
        }
        const auto&  pool             = group_block_pools_[static_cast<size_t>(gid)];
        const size_t available_blocks = perPoolAvailableBlocks(gid);
        const size_t total_blocks     = pool->totalBlocksNum();
        const size_t group_reserve_blocks =
            reserveBlocksForPool(static_cast<size_t>(gid), reserve_blocks, total_reservable_available_blocks);
        if (available_blocks < static_cast<size_t>(need_blocks) + group_reserve_blocks) {
            if (malloc_info.verbose) {
                RTP_LLM_LOG_INFO("HybridPool initMalloc rejected by reserve blocks: request_id=%ld pool_name=%s "
                                 "group=%d need_blocks=%d total_blocks=%zu available_blocks=%zu "
                                 "reserve_blocks=%zu group_reserve_blocks=%zu",
                                 malloc_info.request_id,
                                 pool->poolName().c_str(),
                                 gid,
                                 need_blocks,
                                 total_blocks,
                                 available_blocks,
                                 reserve_blocks,
                                 group_reserve_blocks);
            }
            return false;
        }
    }
    return true;
}

}  // namespace rtp_llm
