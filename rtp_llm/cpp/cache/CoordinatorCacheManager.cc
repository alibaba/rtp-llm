#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/CoordinatorCacheManager.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/core/OpData.h"

namespace rtp_llm {

bool CoordinatorCacheManager::init() {
    RTP_LLM_CHECK_WITH_INFO(doInit(), "init failed");

    // NOTE: the reservable block count depends on initialized block pools and must be queried after `doInit()`.
    const int64_t reserve_ratio = reserve_block_ratio_;
    if (reserve_ratio > 0) {
        const size_t available_blocks = reservableAvailableBlocksNum();
        const size_t reserve_blocks = static_cast<size_t>(reserve_ratio) * available_blocks / static_cast<size_t>(100);
        reserve_block_num_          = reserve_blocks;
        RTP_LLM_LOG_INFO(
            "CoordinatorCacheManager set reserve blocks: ratio=%ld%% reserve_blocks=%zu available_blocks=%zu",
            reserve_ratio,
            reserve_blocks,
            available_blocks);
    } else {
        reserve_block_num_ = 0;
    }

    return true;
}

MallocResult CoordinatorCacheManager::initMalloc(const MallocInfo& malloc_info) {
    // Gross demand decides whether this request can ever fit. Current
    // availability is checked after device-cache matching, when the coordinator_cache_manager
    // knows how many new physical blocks are actually required.
    const auto capacity_status = evaluateInitCapacity(malloc_info, reserveBlocksNum(), InitCapacityMode::TOTAL_ONLY);
    if (capacity_status != MallocStatus::NONE) {
        return {false, 0, 0, capacity_status};
    }

    auto finalize_init_failure = [this, &malloc_info](MallocResult result) {
        // Classify against the failure-time snapshot. Rolling back first can
        // make capacity look sufficient and turn a retryable race into an
        // internal error.
        if (result.status == MallocStatus::NONE || result.status == MallocStatus::INTERNAL_ERROR) {
            const auto status =
                evaluateInitCapacity(malloc_info, reserveBlocksNum(), InitCapacityMode::TOTAL_AND_AVAILABLE);
            result.status = status == MallocStatus::NONE ? MallocStatus::INTERNAL_ERROR : status;
        }
        FreeInfo free_info{malloc_info.batch_kv_cache_resource, malloc_info.complete_token_ids};
        free(free_info);
        return result;
    };

    auto init_result = initMallocForCommonLen(malloc_info);
    if (!init_result.success) {
        return finalize_init_failure(init_result);
    }

    auto incr_result = incrMalloc(malloc_info);
    if (!incr_result.success) {
        return finalize_init_failure(incr_result);
    } else {
        if (metrics_reporter_ && malloc_info.enable_device_cache) {
            int64_t device_input_length = 0;
            if (malloc_info.batch_kv_cache_resource) {
                const auto& cache_keys      = malloc_info.batch_kv_cache_resource->cacheKeys(0);
                size_t      match_keys_size = cache_keys.size();
                device_input_length         = static_cast<int64_t>(match_keys_size) * deviceCacheMetricTokensPerBlock();
            }

            if (device_input_length > 0) {
                RtpLLMDeviceCacheReuseMetricsCollector collector;
                collector.match_cost_time_us    = init_result.match_cost_time_us;
                collector.device_input_length   = device_input_length;
                collector.device_reuse_length   = init_result.reuse_len;
                collector.device_cache_hit_rate = static_cast<float>(static_cast<int64_t>(collector.device_reuse_length)
                                                                     * 100 / collector.device_input_length);
                kmonitor::MetricsTags tags;
                metrics_reporter_->report<RtpLLMDeviceCacheReuseMetrics, RtpLLMDeviceCacheReuseMetricsCollector>(
                    &tags, &collector);
            }
        }
        return init_result;
    }
}

MallocResult CoordinatorCacheManager::malloc(const MallocInfo& malloc_info) {
    if (!malloc_info.batch_kv_cache_resource) {
        RTP_LLM_LOG_ERROR("BatchKVCacheResource is null");
        return {false, 0};
    }

    if (!malloc_info.complete_token_ids) {
        RTP_LLM_LOG_ERROR("CompleteTokenIds is null");
        return {false, 0};
    }

    if (malloc_info.batch_kv_cache_resource->curBlocksNum() == 0) {
        return initMalloc(malloc_info);
    } else {
        return incrMalloc(malloc_info);
    }
}

int CoordinatorCacheManager::estimateBatchPeakNeedBlocks(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                                         int                            seq_len,
                                                         int                            common_seq_len,
                                                         int                            remaining_tokens,
                                                         int                            reserve_step,
                                                         bool                           enable_reuse_cache,
                                                         int                            target_batch_size) const {
    if (!batch_kv_cache_resource || batch_kv_cache_resource->batchSize() == 0) {
        return 0;
    }

    const int current_batch_size = batch_kv_cache_resource->batchSize();
    const int target_width       = std::max(current_batch_size, target_batch_size);
    const int clamped_common_len = std::clamp(common_seq_len, 0, seq_len);

    // A fresh resource follows initMalloc's two phases. Each group estimates that exact sequence so Linear groups can
    // distinguish the shared common tail from every sequence's private suffix tail.
    if (batch_kv_cache_resource->curBlocksNum() == 0) {
        return estimateInitialBatchPeakNeedBlocks(
            seq_len, clamped_common_len, remaining_tokens, reserve_step, enable_reuse_cache, target_width);
    }

    // Initialized sequences have the same layout, and all subsequent growth is private per sequence.
    const int per_sequence_growth = estimatePeakNeedBlocks(
        batch_kv_cache_resource->cacheResource(0), seq_len, remaining_tokens, reserve_step, enable_reuse_cache);

    // Full blocks remain shared when the batch expands, but every additional sequence needs a physical copy of the
    // current partial tail before it can diverge.
    const int expanded_sequences = target_width - current_batch_size;
    const int tail_copy_blocks   = expanded_sequences > 0 && seq_len % seqSizePerBlock() != 0 ? expanded_sequences : 0;
    return target_width * per_sequence_growth + tail_copy_blocks;
}

uint32_t CoordinatorCacheManager::convertToGlobalLayerId(size_t model_id, int local_layer_id) const {
    if (model_id == 0) {
        // main model: local_layer_id is the global layer id
        if (local_layer_id >= 0 && static_cast<size_t>(local_layer_id) < config_.layer_num) {
            return static_cast<uint32_t>(local_layer_id);
        }
        RTP_LLM_LOG_ERROR("convertToGlobalLayerId: local_layer_id=%d is invalid", local_layer_id);
        return std::numeric_limits<uint32_t>::max();
    }

    if (model_id > config_.mtp_sub_configs.size()) {
        RTP_LLM_LOG_ERROR("convertToGlobalLayerId: model_id=%zu out of range (mtp_sub_configs=%zu)",
                          model_id,
                          config_.mtp_sub_configs.size());
        return std::numeric_limits<uint32_t>::max();
    }

    const auto& sub = config_.mtp_sub_configs[model_id - 1];
    if (!sub) {
        RTP_LLM_LOG_ERROR("convertToGlobalLayerId: mtp_sub_configs[%zu] is null", model_id - 1);
        return std::numeric_limits<uint32_t>::max();
    }
    if (local_layer_id < 0 || static_cast<size_t>(local_layer_id) >= sub->layer_num) {
        RTP_LLM_LOG_ERROR("convertToGlobalLayerId: local_layer_id=%d is invalid", local_layer_id);
        return std::numeric_limits<uint32_t>::max();
    }

    return CacheConfig::mtpGlobalLayerId(
        config_.layer_num, static_cast<int>(model_id - 1), sub->layer_num, local_layer_id);
}
bool CoordinatorCacheManager::cpShardThisGroupForCapacity(std::string_view tag) const {
    return cp_slot_mapper_ && cp_slot_mapper_->isSharded() && cp_slot_mapper_->blockRoundRobinGroup(config_, tag);
}

size_t CoordinatorCacheManager::logicalSeqSizePerBlockForCapacity(std::string_view tag) const {
    if (cp_slot_mapper_ && cp_slot_mapper_->isSharded()) {
        return cp_slot_mapper_->logicalSeqSizePerBlock(config_, tag);
    }
    return config_.group(tag).seqSizePerBlock();
}

int CoordinatorCacheManager::cpEffectiveSeqLenForAlloc(std::string_view tag, int seq_len) const {
    return (cp_slot_mapper_ && cp_slot_mapper_->isSharded()) ?
               cp_slot_mapper_->effectiveSeqLenForAlloc(config_, tag, seq_len) :
               seq_len;
}

int CoordinatorCacheManager::deviceCacheMetricTokensPerBlock() const {
    if (cp_slot_mapper_ && cp_slot_mapper_->isSharded()) {
        return cp_slot_mapper_->virtualBlockSize();
    }
    return seqSizePerBlock();
}

namespace {

inline int cpEffectiveSeqLenForReserve(const std::shared_ptr<CPSlotMapper>& mapper,
                                       const CacheConfig&                   config,
                                       std::string_view                     tag,
                                       int                                  seq_len) {
    return (mapper && mapper->isSharded()) ? mapper->effectiveSeqLenForAlloc(config, tag, seq_len) : seq_len;
}

void appendPoolSummary(std::ostringstream&    os,
                       bool&                  has_any,
                       size_t                 idx,
                       const std::string&     tag,
                       CacheGroupType         group_type,
                       const BlockPoolConfig& pool_config) {
    static constexpr double kBytesPerMB = 1024.0 * 1024.0;
    if (has_any) {
        os << "; ";
    }
    has_any = true;
    os << "pool_name=" << pool_config.pool_name << ", idx=" << idx << ", tag=" << tag
       << ", type=" << cacheGroupTypeName(group_type) << ", size=" << pool_config.total_size_bytes << " bytes("
       << std::fixed << std::setprecision(2) << static_cast<double>(pool_config.total_size_bytes) / kBytesPerMB
       << " MB)"
       << ", blocks=" << pool_config.block_num;
}

}  // namespace

CoordinatorCacheManager::CoordinatorCacheManager(const CacheConfig&                 config,
                                                 AllocationType                     allocation_type,
                                                 const kmonitor::MetricsReporterPtr metrics_reporter,
                                                 int64_t                            reserve_block_ratio,
                                                 RoleType                           role_type):
    config_(config),
    allocation_type_(allocation_type),
    metrics_reporter_(metrics_reporter),
    reserve_block_ratio_(reserve_block_ratio),
    role_type_(role_type) {}

BlockPoolPtr CoordinatorCacheManager::soleGroupBlockPool() const {
    RTP_LLM_CHECK_WITH_INFO(group_block_pools_.size() == 1,
                            "sole group block pool requires exactly one initialized group pool, got %zu",
                            group_block_pools_.size());
    return group_block_pools_[0];
}

size_t CoordinatorCacheManager::storageIdxForTag(std::string_view tag) const {
    const auto it = tag_to_idx_.find(std::string(tag));
    RTP_LLM_CHECK_WITH_INFO(
        it != tag_to_idx_.end(), "missing coordinator_cache_manager cache group tag=%s", std::string(tag).c_str());
    return it->second;
}

const SingleTypeCacheManagerPtr& CoordinatorCacheManager::singleTypeManager(std::string_view tag) const {
    const auto idx = storageIdxForTag(tag);
    RTP_LLM_CHECK_WITH_INFO(
        idx < single_type_managers_.size(), "missing single-type cache manager for tag=%s", std::string(tag).c_str());
    return single_type_managers_[idx];
}

bool CoordinatorCacheManager::initSingleTypeManager(const SingleTypeCacheManagerPtr& manager) {
    return manager->init();
}

BlockPoolPtr CoordinatorCacheManager::blockPool(std::string_view tag) const {
    const auto idx = storageIdxForTag(tag);
    RTP_LLM_CHECK_WITH_INFO(idx < group_block_pools_.size(), "missing block pool for tag=%s", std::string(tag).c_str());
    return group_block_pools_[idx];
}

bool CoordinatorCacheManager::doInit() {
    RTP_LLM_CHECK_WITH_INFO(config_.groupNums() > 0, "no cache groups found in CacheConfig");

    const int                               group_nums = config_.groupNums();
    std::vector<BlockPoolPtr>               staged_group_block_pools;
    std::vector<SingleTypeCacheManagerPtr>  staged_single_type_managers;
    std::unordered_map<std::string, size_t> staged_tag_to_idx;
    std::vector<std::string>                staged_full_group_tags;
    std::vector<std::string>                staged_linear_group_tags;
    std::vector<std::string>                staged_swa_group_tags;
    staged_group_block_pools.reserve(static_cast<size_t>(group_nums));
    staged_single_type_managers.reserve(static_cast<size_t>(group_nums));
    staged_tag_to_idx.reserve(static_cast<size_t>(group_nums));
    staged_full_group_tags.reserve(static_cast<size_t>(group_nums));
    staged_linear_group_tags.reserve(static_cast<size_t>(group_nums));
    staged_swa_group_tags.reserve(static_cast<size_t>(group_nums));

    SharedBlockCache*       shared_cache_raw = shared_block_cache_ ? shared_block_cache_.get() : nullptr;
    static constexpr double kBytesPerMB      = 1024.0 * 1024.0;
    std::ostringstream      pool_summary;
    size_t                  pool_total_bytes  = 0;
    size_t                  pool_total_blocks = 0;
    bool                    has_pool          = false;

    std::vector<BlockPoolConfig> group_pool_configs;
    group_pool_configs.reserve(static_cast<size_t>(group_nums));
    for (size_t idx = 0; idx < config_.groups().size(); ++idx) {
        const auto& group       = config_.groups()[idx];
        auto        pool_config = BlockPoolConfigHelper::createConfigForGroup(config_, group.tag);
        appendPoolSummary(pool_summary, has_pool, idx, group.tag, group.policy.group_type, pool_config);
        pool_total_bytes += pool_config.total_size_bytes;
        pool_total_blocks += pool_config.block_num;
        group_pool_configs.push_back(std::move(pool_config));
    }

    if (has_pool) {
        const auto summary = pool_summary.str();
        RTP_LLM_LOG_INFO("Coordinator pool summary: pools=[%s], total_size=%zu bytes total_size_mb=%.2f "
                         "total_blocks=%zu",
                         summary.c_str(),
                         pool_total_bytes,
                         static_cast<double>(pool_total_bytes) / kBytesPerMB,
                         pool_total_blocks);
    }

    for (size_t idx = 0; idx < config_.groups().size(); ++idx) {
        const auto& pool_config = group_pool_configs[idx];
        const auto& cache_group = config_.groups()[idx];
        const auto  group_type  = cache_group.policy.group_type;
        auto        group_pool =
            std::make_shared<BlockPool>(pool_config, allocation_type_, false, use_cuda_malloc_block_pool_);
        RTP_LLM_CHECK_WITH_INFO(
            group_pool->init(), "Failed to initialize block pool %s", pool_config.pool_name.c_str());

        RTP_LLM_CHECK_WITH_INFO(staged_tag_to_idx.emplace(cache_group.tag, idx).second,
                                "duplicate coordinator_cache_manager cache group tag=%s",
                                cache_group.tag.c_str());

        SingleTypeCacheManagerPtr manager;
        if (group_type == CacheGroupType::LINEAR) {
            manager = std::make_shared<LinearCacheManager>(cache_group,
                                                           config_.groupLayerIds(cache_group.tag),
                                                           group_pool,
                                                           config_.linear_step,
                                                           shared_cache_raw,
                                                           metrics_reporter_);
            staged_linear_group_tags.push_back(cache_group.tag);
        } else if (group_type == CacheGroupType::SWA) {
            manager = std::make_shared<SWACacheManager>(cache_group,
                                                        config_.groupLayerIds(cache_group.tag),
                                                        group_pool,
                                                        config_.linear_step,
                                                        shared_cache_raw,
                                                        metrics_reporter_);
            staged_swa_group_tags.push_back(cache_group.tag);
        } else {
            manager = std::make_shared<FullCacheManager>(
                cache_group, config_.groupLayerIds(cache_group.tag), group_pool, shared_cache_raw, metrics_reporter_);
            staged_full_group_tags.push_back(cache_group.tag);
        }

        RTP_LLM_CHECK_WITH_INFO(initSingleTypeManager(manager),
                                "Failed to initialize SingleTypeCacheManager %s",
                                pool_config.pool_name.c_str());
        staged_group_block_pools.push_back(group_pool);
        staged_single_type_managers.push_back(manager);
    }

    group_block_pools_.swap(staged_group_block_pools);
    single_type_managers_.swap(staged_single_type_managers);
    tag_to_idx_.swap(staged_tag_to_idx);
    full_group_tags_.swap(staged_full_group_tags);
    linear_group_tags_.swap(staged_linear_group_tags);
    swa_group_tags_.swap(staged_swa_group_tags);

    if (shared_block_cache_) {
        std::map<std::string, BlockPoolPtr> group_pools_by_tag;
        for (size_t idx = 0; idx < config_.groups().size(); ++idx) {
            group_pools_by_tag.emplace(config_.groups()[idx].tag, group_block_pools_[idx]);
        }
        shared_block_cache_->init(config_, group_pools_by_tag);
    }

    RTP_LLM_LOG_INFO("CoordinatorCacheManager init success, group pools=%zu", group_block_pools_.size());
    return true;
}

const CacheGroup& CoordinatorCacheManager::defaultGroupForLayer(int layer_id) const {
    if (layer_id < 0 || static_cast<size_t>(layer_id) >= config_.layer_all_num) {
        RTP_LLM_FAIL("invalid layer_id=%d", layer_id);
    }
    const auto& group = config_.soleGroupForLayer(layer_id);
    (void)singleTypeManager(group.tag);
    return group;
}

const CacheGroup& CoordinatorCacheManager::validateGroupForLayer(int layer_id, std::string_view tag) const {
    RTP_LLM_CHECK_WITH_INFO(layer_id >= 0 && static_cast<size_t>(layer_id) < config_.layer_all_num,
                            "invalid layer id %d for layer_all_num=%u",
                            layer_id,
                            config_.layer_all_num);
    const auto& group = config_.groupForLayer(layer_id, tag);
    (void)singleTypeManager(group.tag);
    return group;
}

void CoordinatorCacheManager::referenceBlocks(std::string_view        tag,
                                              const BlockIndicesType& blocks,
                                              bool                    is_connector) const {
    if (is_connector) {
        blockPool(tag)->connectorReference(blocks);
    } else {
        blockPool(tag)->requestReference(blocks);
    }
}

void CoordinatorCacheManager::freeBlocks(std::string_view tag, const BlockIndicesType& blocks, bool is_connector) {
    if (is_connector) {
        blockPool(tag)->connectorFree(blocks);
    } else {
        blockPool(tag)->requestFree(blocks);
    }
}

GroupedCacheLayerLayout CoordinatorCacheManager::allLayerCacheBase() const {
    const auto& topology = config_;
    RTP_LLM_CHECK_WITH_INFO(single_type_managers_.size() == topology.groups().size(),
                            "cache group count=%zu topology count=%zu",
                            single_type_managers_.size(),
                            topology.groups().size());

    GroupedCacheLayerLayout::GroupLayouts groups;
    for (const auto& group_config : topology.groups()) {
        const auto&                     manager = singleTypeManager(group_config.tag);
        std::vector<BlockBufferPtrInfo> layers(topology.layers().size());
        const auto                      layer_tensors = manager->allLayerCacheBase();
        const auto                      scale_tensors = manager->allLayerScaleCacheBase();
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
        groups.emplace(group_config.tag, CacheLayerLayout(std::move(layers)));
    }
    return GroupedCacheLayerLayout(topology, std::move(groups));
}

BlockAddrInfo CoordinatorCacheManager::convertIndexToAddr(int layer_id, int block_id) const {
    const auto& group = defaultGroupForLayer(layer_id);
    return singleTypeManager(group.tag)->convertIndexToAddr(layer_id, block_id);
}

std::vector<BlockInfo> CoordinatorCacheManager::convertIndexToBuffer(int layer_id, int block_id) const {
    const auto& group = defaultGroupForLayer(layer_id);
    return singleTypeManager(group.tag)->convertIndexToBuffer(layer_id, block_id);
}

std::vector<BlockInfo>
CoordinatorCacheManager::convertIndexToBuffer(int layer_id, int block_id, int partition_count, int partition_id) const {
    const auto& group = defaultGroupForLayer(layer_id);
    return singleTypeManager(group.tag)->convertIndexToBuffer(layer_id, block_id, partition_count, partition_id);
}

BlockAddrInfo CoordinatorCacheManager::convertIndexToAddr(int layer_id, std::string_view tag, int block_id) const {
    const auto& group = validateGroupForLayer(layer_id, tag);
    return singleTypeManager(group.tag)->convertIndexToAddr(layer_id, block_id);
}

std::vector<BlockInfo>
CoordinatorCacheManager::convertIndexToBuffer(int layer_id, std::string_view tag, int block_id) const {
    const auto& group = validateGroupForLayer(layer_id, tag);
    return singleTypeManager(group.tag)->convertIndexToBuffer(layer_id, block_id);
}

std::vector<BlockInfo> CoordinatorCacheManager::convertIndexToBuffer(
    int layer_id, std::string_view tag, int block_id, int partition_count, int partition_id) const {
    const auto& group = validateGroupForLayer(layer_id, tag);
    return singleTypeManager(group.tag)->convertIndexToBuffer(layer_id, block_id, partition_count, partition_id);
}

void CoordinatorCacheManager::blockBatchCopy(const std::vector<TaggedBlockIdPair>& copy_mapping) {
    if (copy_mapping.empty()) {
        return;
    }

    size_t copy_nums[BatchCopyParams::TYPE_SIZE] = {};
    for (const auto& mapping : copy_mapping) {
        const auto&  pool              = blockPool(mapping.tag);
        const auto   copy_type         = BatchCopyParams::get_copy_type(pool->where(), pool->where());
        const auto&  group             = config_.group(mapping.tag);
        const size_t buffers_per_layer = group.kv_scale_stride_bytes > 0 ? 2 : 1;
        copy_nums[copy_type] += config_.groupLayerIds(group.tag).size() * buffers_per_layer;
    }

    BatchCopyParams copy_params;
    for (size_t i = 0; i < BatchCopyParams::TYPE_SIZE; ++i) {
        copy_params.reserve(static_cast<BatchCopyParams::CopyType>(i), copy_nums[i]);
    }

    for (const auto& mapping : copy_mapping) {
        const auto&  group               = config_.group(mapping.tag);
        const size_t kv_block_size_bytes = group.kv_block_stride_bytes;
        const size_t scale_block_bytes   = group.kv_scale_stride_bytes;
        const auto&  pool                = blockPool(mapping.tag);
        const auto   copy_type           = BatchCopyParams::get_copy_type(pool->where(), pool->where());

        for (int layer_id : config_.groupLayerIds(group.tag)) {
            auto src_addr_info = singleTypeManager(mapping.tag)->convertIndexToAddr(layer_id, mapping.src);
            auto dst_addr_info = singleTypeManager(mapping.tag)->convertIndexToAddr(layer_id, mapping.dst);

            if (!src_addr_info.kv_addr || !dst_addr_info.kv_addr) {
                RTP_LLM_LOG_ERROR("Failed to get block address for pool %s(tag %s) layer %d, src_block %d, "
                                  "dst_block %d",
                                  pool->poolName().c_str(),
                                  mapping.tag.c_str(),
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

size_t CoordinatorCacheManager::freeBlocksNum() const {
    size_t total = 0;
    for (const auto& pool : group_block_pools_) {
        total += pool->freeBlocksNum();
    }
    return total;
}

size_t CoordinatorCacheManager::availableBlocksNum() const {
    size_t total = 0;
    for (const auto& pool : group_block_pools_) {
        total += pool->availableBlocksNum();
    }
    return total;
}

BatchKVCacheResourcePtr CoordinatorCacheManager::popBlocksFromCache(size_t min_blocks_to_free) {
    if (min_blocks_to_free == 0 || !shared_block_cache_) {
        return nullptr;
    }

    auto evict_result = shared_block_cache_->selectAndEvict(min_blocks_to_free);
    if (evict_result.evictions.empty()) {
        return nullptr;
    }
    if (metrics_reporter_) {
        for (const auto& eviction : evict_result.evictions) {
            RtpLLMCacheEvictionMetricsCollector collector;
            collector.lifetime_ms = eviction.lifetime_ms;
            kmonitor::MetricsTags tags("scope", "gpu");
            tags.AddTag("evict_policy", eviction.kind == EvictionKind::IndependentGroup ? "independent" : "chain");
            tags.AddTag("backing", "device");
            metrics_reporter_->report<RtpLLMCacheEvictionMetrics, RtpLLMCacheEvictionMetricsCollector>(&tags,
                                                                                                       &collector);
        }
    }

    auto batch_resource = std::make_shared<BatchKVCacheResource>();
    batch_resource->resetBatchSize(1);
    batch_resource->initGroups(config_);
    batch_resource->setLastBlockAligned(true);

    for (const auto& group : config_.groups()) {
        batch_resource->mutableBlockIds(0, group.tag).resize(evict_result.evictions.size(), NULL_BLOCK_IDX);
    }

    CacheKeysType         evicted_keys;
    BlockDependenciesType evicted_dependencies;
    evicted_keys.reserve(evict_result.evictions.size());
    evicted_dependencies.reserve(evict_result.evictions.size());
    for (size_t evicted_idx = 0; evicted_idx < evict_result.evictions.size(); ++evicted_idx) {
        const auto& eviction = evict_result.evictions[evicted_idx];
        evicted_keys.push_back(eviction.cache_key);
        if (eviction.has_dependency) {
            evicted_dependencies.push_back(eviction.dependency);
        } else {
            BlockDependency dependency;
            dependency.ordinal = static_cast<uint32_t>(evicted_idx);
            if (evicted_idx > 0) {
                dependency.has_parent = true;
                dependency.parent_key = evict_result.evictions[evicted_idx - 1].cache_key;
            }
            evicted_dependencies.push_back(dependency);
        }
        for (const auto& [tag, block_id] : eviction.blocks_by_group) {
            batch_resource->mutableBlockIds(0, tag).setAt(evicted_idx, block_id);
        }
    }
    batch_resource->cacheResource(0).setCacheKeysAndBlockDependencies(std::move(evicted_keys),
                                                                      std::move(evicted_dependencies));
    // Evicted keys already come from the GPU cache's actual key namespace.
    // Under CP this can be a mixed batch of canonical paged keys and logical
    // state/SWA keys, so coordinator must not remap the whole batch again.
    batch_resource->cacheResource(0).setCacheKeysAreCpCanonical(true);
    return batch_resource;
}

void CoordinatorCacheManager::blockCacheFree(const BatchKVCacheResourcePtr& batch_kv_cache_resource) {
    if (!batch_kv_cache_resource) {
        return;
    }
    for (int batch_id = 0; batch_id < batch_kv_cache_resource->batchSize(); ++batch_id) {
        const auto& resource = batch_kv_cache_resource->cacheResource(batch_id);
        for (const auto& group : config_.groups()) {
            const auto&                      tag       = group.tag;
            const auto&                      block_ids = resource.blockIds(tag);
            BlockIndicesType                 blocks_to_free;
            std::unordered_set<BlockIdxType> seen_blocks;
            for (auto block_idx : block_ids.blocks()) {
                if (isNullBlockIdx(block_idx) || !seen_blocks.insert(block_idx).second) {
                    continue;
                }
                blocks_to_free.push_back(block_idx);
            }
            if (!blocks_to_free.empty()) {
                blockPool(tag)->blockCacheFree(blocks_to_free);
            }
        }
    }
}

size_t CoordinatorCacheManager::requestRefBlocksNum() const {
    size_t total = 0;
    for (const auto& pool : group_block_pools_) {
        total += pool->requestRefBlocksNum();
    }
    return total;
}

size_t CoordinatorCacheManager::connectorRefBlocksNum() const {
    size_t total = 0;
    for (const auto& pool : group_block_pools_) {
        total += pool->connectorRefBlocksNum();
    }
    return total;
}

size_t CoordinatorCacheManager::blockCacheRefBlocksNum() const {
    size_t total = 0;
    for (const auto& pool : group_block_pools_) {
        total += pool->blockCacheRefBlocksNum();
    }
    return total;
}

size_t CoordinatorCacheManager::notInUseBlocksNum() const {
    size_t total = 0;
    for (const auto& pool : group_block_pools_) {
        total += pool->notInUseBlocksNum();
    }
    return total;
}

size_t CoordinatorCacheManager::minTokenCapacity(bool use_available_blocks, bool full_groups_only) const {
    if (group_block_pools_.empty()) {
        return 0;
    }

    auto calculate = [&](bool only_full_groups) {
        size_t min_tokens = std::numeric_limits<size_t>::max();
        bool   saw_group  = false;
        for (const auto& group : config_.groups()) {
            if (only_full_groups && group.policy.group_type != CacheGroupType::FULL) {
                continue;
            }
            const auto& pool = blockPool(group.tag);
            if (!pool) {
                continue;
            }
            saw_group        = true;
            const auto block = use_available_blocks ? pool->availableBlocksNum() : pool->totalBlocksNum();
            min_tokens       = std::min(min_tokens, block * logicalSeqSizePerBlockForCapacity(group.tag));
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

size_t CoordinatorCacheManager::availableTokensNum() const {
    return minTokenCapacity(/*use_available_blocks=*/true, /*full_groups_only=*/true);
}

size_t CoordinatorCacheManager::totalTokensNum() const {
    return minTokenCapacity(/*use_available_blocks=*/false, /*full_groups_only=*/true);
}

size_t CoordinatorCacheManager::totalBlocksNum() const {
    size_t total = 0;
    for (const auto& pool : group_block_pools_) {
        total += pool->totalBlocksNum();
    }
    return total;
}

size_t CoordinatorCacheManager::maxAvailableTokensNum() const {
    return minTokenCapacity(/*use_available_blocks=*/false, /*full_groups_only=*/true);
}

KVCacheTokenCapacity CoordinatorCacheManager::tokenCapacity(size_t default_seq_size_per_block) const {
    (void)default_seq_size_per_block;
    if (group_block_pools_.empty()) {
        return {};
    }
    size_t total_tokens     = std::numeric_limits<size_t>::max();
    size_t available_tokens = std::numeric_limits<size_t>::max();
    bool   has_pool         = false;
    for (const auto& group : config_.groups()) {
        const auto& pool = blockPool(group.tag);
        if (!pool) {
            continue;
        }
        const size_t seq_size = logicalSeqSizePerBlockForCapacity(group.tag);
        total_tokens          = std::min(total_tokens, pool->totalBlocksNum() * seq_size);
        available_tokens      = std::min(available_tokens, pool->availableBlocksNum() * seq_size);
        has_pool              = true;
    }
    return has_pool ? KVCacheTokenCapacity{total_tokens, available_tokens} : KVCacheTokenCapacity{};
}

std::vector<KVCachePoolMetricsSnapshot> CoordinatorCacheManager::poolMetricsSnapshots() const {
    std::vector<KVCachePoolMetricsSnapshot> snapshots;
    snapshots.reserve(group_block_pools_.size());
    const size_t reserve_blocks                    = reserveBlocksNum();
    const size_t total_reservable_available_blocks = totalReservableAvailableBlocks();
    for (const auto& group : config_.groups()) {
        const auto& pool = blockPool(group.tag);
        if (!pool) {
            continue;
        }
        KVCachePoolMetricsSnapshot snapshot;
        snapshot.tag                  = group.tag;
        snapshot.pool_name            = pool->poolName();
        snapshot.total_blocks         = pool->totalBlocksNum();
        snapshot.available_blocks     = pool->availableBlocksNum();
        snapshot.free_blocks          = pool->freeBlocksNum();
        snapshot.request_ref_blocks   = pool->requestRefBlocksNum();
        snapshot.connector_ref_blocks = pool->connectorRefBlocksNum();
        snapshot.reserve_blocks = reserveBlocksForPool(group.tag, reserve_blocks, total_reservable_available_blocks);
        snapshot.used_ratio     = (snapshot.total_blocks == 0) ?
                                      0.0f :
                                      static_cast<float>(100.0 * (snapshot.total_blocks - snapshot.available_blocks)
                                                     / static_cast<double>(snapshot.total_blocks));
        snapshots.push_back(snapshot);
    }
    return snapshots;
}

void CoordinatorCacheManager::regUserMr(size_t model_id, std::shared_ptr<CacheStore> cache_store) {
    for (auto& pool : group_block_pools_) {
        pool->regUserMr(model_id, cache_store);
    }
}

int64_t CoordinatorCacheManager::getMrCostTimeMs() const {
    int64_t total = 0;
    for (const auto& pool : group_block_pools_) {
        total += pool->getMrCostTimeMs();
    }
    return total;
}

size_t CoordinatorCacheManager::totalReservableAvailableBlocks() const {
    size_t total = 0;
    for (const auto& group : config_.groups()) {
        const auto& pool = blockPool(group.tag);
        if (!pool || config_.usesExplicitIndependentBlocks(group.tag)) {
            continue;
        }
        total += pool->availableBlocksNum();
    }
    return total;
}

size_t CoordinatorCacheManager::reservableAvailableBlocksNum() const {
    return totalReservableAvailableBlocks();
}

size_t CoordinatorCacheManager::reserveBlocksForPool(std::string_view tag,
                                                     size_t           reserve_blocks,
                                                     size_t           total_reservable_available_blocks) const {
    const auto& pool = blockPool(tag);
    if (!pool || config_.usesExplicitIndependentBlocks(tag) || total_reservable_available_blocks == 0) {
        return 0;
    }
    return reserve_blocks * pool->availableBlocksNum() / total_reservable_available_blocks;
}

MallocStatus CoordinatorCacheManager::evaluateInitCapacity(const MallocInfo& malloc_info,
                                                           size_t            reserve_blocks,
                                                           InitCapacityMode  mode) const {
    if (!malloc_info.batch_kv_cache_resource || !malloc_info.complete_token_ids) {
        return MallocStatus::NONE;
    }
    const auto& cp_mapper          = cp_slot_mapper_;
    const int   batch_size         = malloc_info.batch_kv_cache_resource->batchSize();
    const int   total_seq_len      = malloc_info.complete_token_ids->totalSeqLength();
    const int   raw_common_seq_len = std::min(malloc_info.complete_token_ids->commonSeqLength(), total_seq_len);
    const int   raw_seq_len        = malloc_info.complete_token_ids->seqLength();
    const int   reserve_step       = malloc_info.complete_token_ids->getReserveStep();
    const bool  reuse_enabled      = malloc_info.reuse_cache;

    const size_t total_reservable_available_blocks = totalReservableAvailableBlocks();
    // The "can this ever fit" verdict must not depend on transient availability,
    // so the reserve share for the total-capacity test is prorated by each pool's
    // total size instead.
    size_t total_reservable_blocks = 0;
    for (const auto& group : config_.groups()) {
        const auto& reservable_pool = blockPool(group.tag);
        if (!reservable_pool || config_.usesExplicitIndependentBlocks(group.tag)) {
            continue;
        }
        total_reservable_blocks += reservable_pool->totalBlocksNum();
    }

    MallocStatus status = MallocStatus::NONE;
    const auto&  groups = config_.groups();
    for (size_t idx = 0; idx < groups.size(); ++idx) {
        const auto& group = groups[idx];
        // Diagnostic-only positional column in the rejection logs; never an identity.
        const int group_common_seq = cpEffectiveSeqLenForReserve(cp_mapper, config_, group.tag, raw_common_seq_len);
        const int group_seq_len    = cpEffectiveSeqLenForReserve(cp_mapper, config_, group.tag, raw_seq_len);
        const int group_reuse_blocks_len =
            reuse_enabled ? malloc_info.batch_kv_cache_resource->blocksNum(0, group.tag) : 0;
        const auto need = singleTypeManager(group.tag)->getNeedBlocks(
            group_common_seq, group_seq_len, reserve_step, group_reuse_blocks_len, reuse_enabled);
        const int need_blocks = need.common_blocks + batch_size * need.extra_blocks;
        if (need_blocks <= 0) {
            continue;
        }
        const auto&  pool             = blockPool(group.tag);
        const size_t available_blocks = pool->availableBlocksNum();
        const size_t total_blocks     = pool->totalBlocksNum();
        const size_t required_blocks  = static_cast<size_t>(need_blocks);

        const size_t total_reserve_blocks =
            (config_.usesExplicitIndependentBlocks(group.tag) || total_reservable_blocks == 0) ?
                0 :
                reserve_blocks * total_blocks / total_reservable_blocks;
        if (required_blocks > total_blocks || total_reserve_blocks > total_blocks - required_blocks) {
            if (malloc_info.verbose) {
                RTP_LLM_LOG_INFO("Coordinator initMalloc permanently rejected: request_id=%ld pool_name=%s "
                                 "idx=%zu tag=%s need_blocks=%d total_blocks=%zu "
                                 "reserve_blocks=%zu group_reserve_blocks=%zu",
                                 malloc_info.request_id,
                                 pool->poolName().c_str(),
                                 idx,
                                 group.tag.c_str(),
                                 need_blocks,
                                 total_blocks,
                                 reserve_blocks,
                                 total_reserve_blocks);
            }
            return MallocStatus::PERMANENT_RESOURCE_EXHAUSTED;
        }

        if (mode != InitCapacityMode::TOTAL_AND_AVAILABLE || status != MallocStatus::NONE) {
            continue;
        }
        const size_t group_reserve_blocks =
            reserveBlocksForPool(group.tag, reserve_blocks, total_reservable_available_blocks);
        if (available_blocks < required_blocks + group_reserve_blocks) {
            if (malloc_info.verbose) {
                RTP_LLM_LOG_INFO("Coordinator initMalloc rejected by reserve blocks: request_id=%ld pool_name=%s "
                                 "idx=%zu tag=%s need_blocks=%d total_blocks=%zu available_blocks=%zu "
                                 "reserve_blocks=%zu group_reserve_blocks=%zu",
                                 malloc_info.request_id,
                                 pool->poolName().c_str(),
                                 idx,
                                 group.tag.c_str(),
                                 need_blocks,
                                 total_blocks,
                                 available_blocks,
                                 reserve_blocks,
                                 group_reserve_blocks);
            }
            // Keep scanning: a later pool may turn this into a permanent verdict.
            status = MallocStatus::RETRYABLE_RESOURCE_EXHAUSTED;
        }
    }
    return status;
}

// Per-pool KV-exhaustion record. This is the primary field-debug tool for
// KV-exhaustion incidents: one aggregate line plus one line per pool carrying
// the demand, the reserve share, the shortfall and the pool's ref-count split.
void CoordinatorCacheManager::logMallocFailure(const MallocInfo& malloc_info,
                                               const char*       phase,
                                               int               failed_batch,
                                               std::string_view  failed_tag,
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
    const bool  reserve_admission    = !incremental && failed_tag.empty();
    const int   reserve_step         = incremental || reserve_admission ? request_reserve_step : 0;
    const int   planning_raw_seq_len = !incremental && !reserve_admission ? raw_common_len : raw_seq_len;
    const auto  reserve_blocks       = reserveBlocksNum();

    const size_t total_reservable_available_blocks = totalReservableAvailableBlocks();

    RTP_LLM_LOG_WARNING("Coordinator malloc failure: error_code=602 request_id=%ld phase=%s failed_batch=%d "
                        "failed_tag=%.*s incremental=%d batch_size=%d seq_len=%d common_seq_len=%d total_seq_len=%d "
                        "planning_seq_len=%d request_reserve_step=%d planning_reserve_step=%d "
                        "failed_need_blocks=%d reserve_blocks=%zu snapshot=best_effort_at_failure",
                        malloc_info.request_id,
                        phase,
                        failed_batch,
                        static_cast<int>(failed_tag.size()),
                        failed_tag.data(),
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

    size_t idx               = 0;
    bool   before_failed_tag = !failed_tag.empty();
    for (const auto& group_config : config_.groups()) {
        const auto& tag        = group_config.tag;
        const auto  group_type = group_config.policy.group_type;
        if (tag == failed_tag) {
            before_failed_tag = false;
        }
        // Diagnostic-only positional column in the failure log; never an identity.
        const int group_seq_len = cpEffectiveSeqLenForReserve(cp_mapper, config_, tag, planning_raw_seq_len);

        int    need_blocks          = 0;
        int    need_slots           = 0;
        size_t current_slots        = 0;
        size_t current_valid_blocks = 0;
        for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
            const auto& blocks = resource->blocks(batch_id, tag);
            current_slots += blocks.size();
            current_valid_blocks += static_cast<size_t>(std::count_if(
                blocks.begin(), blocks.end(), [](auto block) { return !isNullBlockIdx(block) && block > 0; }));
            need_slots +=
                singleTypeManager(tag)->needBlocksNum(group_seq_len, static_cast<int>(blocks.size()), reserve_step);
        }
        if (incremental) {
            // Dense groups materialize every logical slot. Sparse groups
            // (LINEAR / SWA) skip slots, so their exact physical request is the
            // value the group coordinator_cache_manager reported immediately before this snapshot.
            need_blocks = singleTypeManager(tag)->hasSparseSlots() ? -1 : need_slots;
        } else if (!reserve_admission && before_failed_tag) {
            // These groups already completed their initial allocation before
            // a later group failed.
            need_blocks = 0;
            need_slots  = 0;
        } else {
            const int  group_common_len = cpEffectiveSeqLenForReserve(cp_mapper, config_, tag, raw_common_len);
            const int  reuse_blocks_len = malloc_info.reuse_cache ? resource->blocksNum(0, tag) : 0;
            const auto need             = singleTypeManager(tag)->getNeedBlocks(
                group_common_len, group_seq_len, reserve_step, reuse_blocks_len, malloc_info.reuse_cache);
            need_blocks = need.common_blocks + batch_size * need.extra_blocks;
        }
        if (tag == failed_tag && failed_need_blocks >= 0) {
            need_blocks = failed_need_blocks;
        }

        const auto&  pool      = blockPool(tag);
        const size_t available = pool->availableBlocksNum();
        const size_t group_reserve =
            reserve_admission ? reserveBlocksForPool(tag, reserve_blocks, total_reservable_available_blocks) : 0;
        const long long required_available = need_blocks < 0 ? -1 : static_cast<long long>(need_blocks + group_reserve);
        const long long shortfall =
            required_available < 0 ? -1 : std::max(required_available - static_cast<long long>(available), 0LL);

        RTP_LLM_LOG_WARNING("Coordinator malloc failure pool: error_code=602 request_id=%ld idx=%zu "
                            "pool_name=%s "
                            "group_type=%s tag=%s failed=%d need_blocks=%d need_slots=%d "
                            "group_reserve_blocks=%zu required_available_blocks=%lld shortfall_blocks=%lld "
                            "current_slots=%zu "
                            "current_valid_blocks=%zu total_blocks=%zu available_blocks=%zu free_blocks=%zu "
                            "request_ref_blocks=%zu connector_ref_blocks=%zu block_cache_ref_blocks=%zu "
                            "layer_count=%zu block_bytes=%zu seq_size_per_block=%zu",
                            malloc_info.request_id,
                            idx,
                            pool->poolName().c_str(),
                            cacheGroupTypeName(group_type),
                            tag.c_str(),
                            tag == failed_tag,
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
                            config_.groupLayerIds(tag).size(),
                            config_.blockSizeBytes(tag),
                            group_config.seqSizePerBlock());
        ++idx;
    }
}

namespace {

// CP shard helpers: when mapper is null/passthrough, all helpers no-op.
inline CacheKeysType cpCanonicalCacheKeys(const std::shared_ptr<CPSlotMapper>& mapper, const CacheKeysType& full) {
    return (mapper && mapper->isSharded()) ? mapper->canonicalCacheKeys(full) : full;
}

inline bool
cpBlockRoundRobinGroup(const std::shared_ptr<CPSlotMapper>& mapper, const CacheConfig& config, std::string_view tag) {
    return mapper && mapper->isSharded() && mapper->blockRoundRobinGroup(config, tag);
}

inline int cpEffectiveSeqLenForGroup(const std::shared_ptr<CPSlotMapper>& mapper,
                                     const CacheConfig&                   config,
                                     std::string_view                     tag,
                                     int                                  seq_len) {
    return cpBlockRoundRobinGroup(mapper, config, tag) ? mapper->effectiveSeqLenForAlloc(config, tag, seq_len) :
                                                         seq_len;
}

inline int cpLogicalSeqSizeForGroup(const std::shared_ptr<CPSlotMapper>& mapper,
                                    const CacheConfig&                   config,
                                    std::string_view                     tag,
                                    int                                  fallback) {
    return (mapper && mapper->isSharded()) ? static_cast<int>(mapper->logicalSeqSizePerBlock(config, tag)) : fallback;
}

BlockIndicesType validBlocksAfter(const BlockIndicesType& blocks, size_t begin) {
    BlockIndicesType valid;
    if (begin >= blocks.size()) {
        return valid;
    }
    valid.reserve(blocks.size() - begin);
    for (size_t i = begin; i < blocks.size(); ++i) {
        if (!isNullBlockIdx(blocks[i])) {
            valid.push_back(blocks[i]);
        }
    }
    return valid;
}

}  // namespace

bool CoordinatorCacheManager::skipReuseCacheGroup(std::string_view tag) const {
    return !singleTypeManager(tag)->prefixReuseEnabled();
}

std::vector<std::string> CoordinatorCacheManager::independentEvictionGroupTags() const {
    std::vector<std::string> tags;
    for (const auto& group : config_.groups()) {
        if (singleTypeManager(group.tag)->evictPolicy() == CacheEvictPolicy::INDEPENDENT) {
            tags.push_back(group.tag);
        }
    }
    return tags;
}

bool CoordinatorCacheManager::cpCompactSwaGroup(std::string_view                     tag,
                                                const std::shared_ptr<CPSlotMapper>& mapper) const {
    return mapper && mapper->isSharded() && mapper->compactLastRankGroup(config_, tag);
}

int CoordinatorCacheManager::reuseCache(const CacheKeysType&                 cache_keys,
                                        BatchKVCacheResource&                kv_resource,
                                        const std::shared_ptr<CPSlotMapper>& cp_mapper) {
    const bool no_reusable_group = std::all_of(config_.groups().begin(),
                                               config_.groups().end(),
                                               [this](const auto& group) { return skipReuseCacheGroup(group.tag); });
    if (no_reusable_group) {
        return 0;
    }

    // Under cp shard, FULL groups index block_ids by cp-virtual-block units
    // (one entry covers cp_size physical blocks). LINEAR/SWA groups index by
    // raw block_size logical blocks. So when populating tail blocks for
    // LINEAR/SWA we need to scale the array length and matched-block position
    // back to the logical-block coordinate system.
    const int cp_scale              = (cp_mapper && cp_mapper->isSharded()) ? cp_mapper->cpSize() : 1;
    int       min_full_reuse_blocks = static_cast<int>(cache_keys.size());
    std::unordered_map<std::string, BlockIndicesType> full_matched_blocks;

    for (const auto& tag : full_group_tags_) {
        auto match_result     = singleTypeManager(tag)->match(cache_keys);
        min_full_reuse_blocks = std::min(min_full_reuse_blocks, static_cast<int>(match_result.reuse_blocks));
        full_matched_blocks.emplace(tag, std::move(match_result.block_indices));
    }

    int                           pos = min_full_reuse_blocks - 1;
    std::vector<BlockIdxType>     linear_tail_blocks(linear_group_tags_.size(), NULL_BLOCK_IDX);
    std::vector<BlockIndicesType> swa_tail_blocks(swa_group_tags_.size());
    const bool                    has_tail_groups = !linear_group_tags_.empty() || !swa_group_tags_.empty();
    for (; pos >= 0 && has_tail_groups; --pos) {
        bool                          all_tail_groups_matched = true;
        std::vector<BlockIdxType>     candidate_linear_tail_blocks(linear_group_tags_.size(), NULL_BLOCK_IDX);
        std::vector<BlockIndicesType> candidate_swa_tail_blocks(swa_group_tags_.size());
        for (size_t i = 0; i < linear_group_tags_.size(); ++i) {
            auto result =
                singleTypeManager(linear_group_tags_[i])->matchSingleKey(cache_keys[static_cast<size_t>(pos)]);
            if (result.block_indices.empty()) {
                all_tail_groups_matched = false;
                break;
            }
            candidate_linear_tail_blocks[i] = result.block_indices[0];
        }
        if (!all_tail_groups_matched) {
            continue;
        }
        for (size_t i = 0; i < swa_group_tags_.size(); ++i) {
            const auto& tag = swa_group_tags_[i];
            if (skipReuseCacheGroup(tag)) {
                continue;
            }
            auto result = singleTypeManager(tag)->matchSingleKey(cache_keys[static_cast<size_t>(pos)]);
            if (result.block_indices.empty()) {
                all_tail_groups_matched = false;
                break;
            }
            candidate_swa_tail_blocks[i].push_back(result.block_indices[0]);
        }
        if (all_tail_groups_matched) {
            linear_tail_blocks = std::move(candidate_linear_tail_blocks);
            swa_tail_blocks    = std::move(candidate_swa_tail_blocks);
            break;
        }
    }

    const int reuse_blocks_len = has_tail_groups ? std::max(pos + 1, 0) : std::max(min_full_reuse_blocks, 0);
    if (reuse_blocks_len <= 0) {
        return 0;
    }

    for (const auto& tag : full_group_tags_) {
        BlockIndicesType full_blocks = full_matched_blocks.at(tag);
        if (static_cast<int>(full_blocks.size()) > reuse_blocks_len) {
            full_blocks.resize(static_cast<size_t>(reuse_blocks_len));
        }
        kv_resource.mutableBlockIds(0, tag).assign(std::move(full_blocks));
    }

    // LINEAR/SWA arrays are sized in logical-block units (cp_size× larger
    // than the FULL groups' cp-virtual-block units). The matched tail block
    // corresponds to the LAST logical block in the canonical (last-rank)
    // namespace, so its index is `(reuse_blocks_len * cp_size) - 1` in
    // logical units, NOT `reuse_blocks_len - 1`.
    const int logical_reuse_len = reuse_blocks_len * cp_scale;
    for (size_t i = 0; i < linear_group_tags_.size(); ++i) {
        const auto& tag = linear_group_tags_[i];
        kv_resource.mutableBlockIds(0, tag).assign(
            BlockIndicesType(static_cast<size_t>(logical_reuse_len), NULL_BLOCK_IDX));
        kv_resource.mutableBlockIds(0, tag).setAt(static_cast<size_t>(logical_reuse_len - 1), linear_tail_blocks[i]);
    }
    for (size_t i = 0; i < swa_group_tags_.size(); ++i) {
        const auto& tag             = swa_group_tags_[i];
        const int   group_reuse_len = cpCompactSwaGroup(tag, cp_mapper) ? reuse_blocks_len : logical_reuse_len;
        kv_resource.mutableBlockIds(0, tag).assign(
            BlockIndicesType(static_cast<size_t>(group_reuse_len), NULL_BLOCK_IDX));
        if (skipReuseCacheGroup(tag)) {
            continue;
        }
        const size_t tail_begin =
            static_cast<size_t>(std::max(group_reuse_len - static_cast<int>(swa_tail_blocks[i].size()), 0));
        for (size_t j = 0; j < swa_tail_blocks[i].size(); ++j) {
            kv_resource.mutableBlockIds(0, tag).setAt(tail_begin + j, swa_tail_blocks[i][j]);
        }
    }
    return reuse_blocks_len;
}

MallocResult CoordinatorCacheManager::initMallocForCommonLen(const MallocInfo& malloc_info) {
    auto&     kv_resource = malloc_info.batch_kv_cache_resource;
    const int batch_size  = kv_resource->batchSize();

    const int   seq_len        = malloc_info.complete_token_ids->seqLength();
    const int   common_seq_len = std::min(malloc_info.complete_token_ids->commonSeqLength(), seq_len);
    const auto& cp_mapper      = cp_slot_mapper_;
    // A CP-sharded FULL group defines the canonical reuse unit. Topologies without
    // FULL groups use the logical cache-key block size for LINEAR/SWA matching.
    const SingleTypeCacheManagerPtr reuse_group =
        full_group_tags_.empty() ? SingleTypeCacheManagerPtr{} : singleTypeManager(full_group_tags_.front());
    const int reuse_unit_tokens =
        (reuse_group ? cpLogicalSeqSizeForGroup(cp_mapper, config_, full_group_tags_.front(), seqSizePerBlock()) :
                       seqSizePerBlock());

    const auto&                                cache_keys         = kv_resource->cacheKeys(0);
    int64_t                                    match_cost_time_us = 0;
    const size_t                               reserve_blocks     = reserveBlocksNum();
    int                                        reuse_blocks       = 0;
    std::map<std::string, BlockIndicesType>    referenced_blocks;
    std::map<std::string, size_t>              original_sizes;
    std::map<std::string, std::vector<size_t>> backfilled_positions;

    const bool match_device_cache = malloc_info.enable_device_cache;
    if (match_device_cache) {
        // CP-sharded: subsample to last-rank canonical key namespace before matching.
        CacheKeysType cp_keys = cpCanonicalCacheKeys(cp_mapper, cache_keys);
        // Always drop the last match key, CP-sharded or not. It may be a partial
        // tail; and even when it is a full block, fully reusing the input leaves
        // no prefill tokens to compute. Keeping every canonical key under CP
        // sharding is exactly that degenerate case: zero prefill tokens left.
        CacheKeysType match_keys(cp_keys.begin(), cp_keys.empty() ? cp_keys.end() : cp_keys.end() - 1);
        auto          begin_us = currentTimeUs();
        reuse_blocks           = reuseCache(match_keys, *kv_resource, cp_mapper);
        match_cost_time_us     = currentTimeUs() - begin_us;

        for (const auto& group : config_.groups()) {
            const auto&      blocks = kv_resource->blocks(0, group.tag);
            BlockIndicesType valid;
            valid.reserve(blocks.size());
            for (auto b : blocks) {
                if (!isNullBlockIdx(b)) {
                    valid.push_back(b);
                }
            }
            if (!valid.empty()) {
                referenceBlocks(group.tag, valid);
                referenced_blocks[group.tag] = std::move(valid);
            }
        }
        kv_resource->cacheResource(0).setDeviceReuseBlockNum(reuse_blocks);
    }

    // The resource shape after reuse is the transaction's original shape.
    // Capture it before either capacity preflight so rollback never mistakes
    // reused references for newly appended allocations.
    for (const auto& group : config_.groups()) {
        original_sizes[group.tag]       = kv_resource->blocksNum(0, group.tag);
        backfilled_positions[group.tag] = {};
    }

    // Post-match capacity preflight. Device-cache matching has already run, so
    // the coordinator_cache_manager now knows how many *new* physical blocks are required and can
    // separate "pools are momentarily full" (RETRYABLE, keeps the stream WAITING)
    // from "this request can never fit" (PERMANENT).
    const auto capacity_status =
        evaluateInitCapacity(malloc_info, reserve_blocks, InitCapacityMode::TOTAL_AND_AVAILABLE);
    if (capacity_status != MallocStatus::NONE) {
        logMallocFailure(malloc_info, "init_reserve", 0, {}, false, -1);
        rollbackInitMalloc(*kv_resource, referenced_blocks, original_sizes, backfilled_positions);
        return {false, 0, match_cost_time_us, capacity_status};
    }

    for (const auto& group_config : config_.groups()) {
        const auto& tag           = group_config.tag;
        auto&       block_ids_0   = kv_resource->mutableBlockIds(0, tag);
        const int   group_seq_len = cpEffectiveSeqLenForGroup(cp_mapper, config_, tag, common_seq_len);
        const auto& manager       = singleTypeManager(tag);
        // Snapshot the slot count before the call so a failure can report this
        // group's exact physical request in the error_code=602 record.
        const int blocks_before = static_cast<int>(block_ids_0.blocksNum());
        if (!manager->malloc(block_ids_0, group_seq_len, malloc_info.reuse_cache, 0, &backfilled_positions[tag])) {
            logMallocFailure(malloc_info,
                             "init_group_malloc",
                             0,
                             tag,
                             false,
                             manager->needBlocksNum(group_seq_len, blocks_before, 0));
            rollbackInitMalloc(*kv_resource, referenced_blocks, original_sizes, backfilled_positions);
            return {false, 0};
        }
    }

    for (int b = 1; b < batch_size; ++b) {
        for (const auto& group : config_.groups()) {
            singleTypeManager(group.tag)->reference(kv_resource->mutableBlockIds(b, group.tag),
                                                    kv_resource->blocks(0, group.tag));
        }
    }
    return {true, reuse_blocks * reuse_unit_tokens, match_cost_time_us};
}

MallocResult CoordinatorCacheManager::incrMalloc(const MallocInfo& malloc_info) {
    auto&       kv_resource  = malloc_info.batch_kv_cache_resource;
    const auto& cp_mapper    = cp_slot_mapper_;
    const int   batch_size   = kv_resource->batchSize();
    const int   raw_seq_len  = malloc_info.incrSeqLen();
    const int   reserve_step = malloc_info.complete_token_ids->getReserveStep();

    std::vector<std::map<std::string, size_t>>              batch_original_sizes(static_cast<size_t>(batch_size));
    std::vector<std::map<std::string, std::vector<size_t>>> batch_backfilled_positions(static_cast<size_t>(batch_size));
    for (int b = 0; b < batch_size; ++b) {
        for (const auto& group : config_.groups()) {
            batch_original_sizes[static_cast<size_t>(b)][group.tag]       = kv_resource->blocksNum(b, group.tag);
            batch_backfilled_positions[static_cast<size_t>(b)][group.tag] = {};
        }
    }

    bool        all_success  = true;
    int         failed_batch = -1;
    std::string failed_tag;
    int         failed_need_blocks = -1;
    for (int b = 0; b < batch_size; ++b) {
        for (const auto& group : config_.groups()) {
            const auto& tag           = group.tag;
            auto&       block_ids     = kv_resource->mutableBlockIds(b, tag);
            const int   group_seq_len = cpEffectiveSeqLenForGroup(cp_mapper, config_, tag, raw_seq_len);
            // Snapshot the slot count before the call so a failure can report this
            // group's exact physical request in the error_code=602 record.
            const int  blocks_before = static_cast<int>(block_ids.blocksNum());
            const bool injected_failure =
                shouldInjectGroupAllocationFailureForTest(*kv_resource, b, tag, /*incremental=*/true);
            if (injected_failure
                || !singleTypeManager(tag)->malloc(block_ids,
                                                   group_seq_len,
                                                   malloc_info.reuse_cache,
                                                   reserve_step,
                                                   &batch_backfilled_positions[static_cast<size_t>(b)][tag])) {
                all_success        = false;
                failed_batch       = b;
                failed_tag         = tag;
                failed_need_blocks = singleTypeManager(tag)->needBlocksNum(group_seq_len, blocks_before, reserve_step);
                break;
            }
        }
        if (!all_success) {
            break;
        }
    }

    if (all_success) {
        if (!malloc_info.enable_remove_skipped_blocks) {
            return {true, 0};
        }
        for (int b = 0; b < batch_size; ++b) {
            for (const auto& group : config_.groups()) {
                singleTypeManager(group.tag)->removeSkippedBlocks(
                    kv_resource->mutableBlockIds(b, group.tag), malloc_info.reuse_cache, reserve_step);
            }
        }
        return {true, 0};
    }

    // Emit the pool snapshot before rolling back: once the partially allocated
    // blocks go back to the pools, available_blocks no longer reflects the state
    // that caused the failure.
    logMallocFailure(malloc_info, "incremental_group_malloc", failed_batch, failed_tag, true, failed_need_blocks);
    rollbackIncrMalloc(
        *kv_resource, batch_original_sizes, batch_backfilled_positions, static_cast<size_t>(failed_batch));
    RTP_LLM_LOG_WARNING("Hybrid incrMalloc failed at batch=%d tag=%s", failed_batch, failed_tag.c_str());
    return {false, 0};
}

void CoordinatorCacheManager::free(const FreeInfo& free_info) {
    auto& kv_cache_resource = free_info.batch_kv_cache_resource;
    if (kv_cache_resource->curBlocksNum() == 0) {
        return;
    }
    for (int batch_id = 0; batch_id < kv_cache_resource->batchSize(); ++batch_id) {
        for (const auto& group : config_.groups()) {
            singleTypeManager(group.tag)->free(kv_cache_resource->blocks(batch_id, group.tag));
        }
    }
    kv_cache_resource->clearBlocks();
}

void CoordinatorCacheManager::insertIntoCache(const InsertInfo& insert_info) {
    auto& kv_cache_resource = insert_info.batch_kv_cache_resource;
    RTP_LLM_CHECK(kv_cache_resource != nullptr);
    if (!shared_block_cache_) {
        return;
    }

    const auto& cp_mapper  = cp_slot_mapper_;
    const bool  cp_active  = cp_mapper && cp_mapper->isSharded();
    const int   batch_size = kv_cache_resource->batchSize();

    const CacheGroup* sole_full_group =
        (config_.groupNums() == 1 && full_group_tags_.size() == 1) ? &config_.group(full_group_tags_.front()) : nullptr;
    const auto sole_spec            = sole_full_group != nullptr ? sole_full_group->spec : nullptr;
    const bool ordinary_single_full = !cp_active && sole_full_group != nullptr
                                      && sole_full_group->policy.group_type == CacheGroupType::FULL && sole_spec
                                      && (sole_spec->type == KVCacheSpecType::MultiHeadAttention
                                          || sole_spec->type == KVCacheSpecType::MultiHeadLatentAttention);
    if (ordinary_single_full) {
        if (!single_type_managers_[0]->prefixReuseEnabled() || batch_size == 0) {
            return;
        }
        const auto& cache_keys = kv_cache_resource->cacheKeys(/*batch_id=*/0);
        const auto& blocks     = kv_cache_resource->blocks(/*batch_id=*/0, full_group_tags_.front());
        const auto  block_num  = std::min(cache_keys.size(), blocks.size());
        if (block_num > 0) {
            single_type_managers_[0]->insertIntoCache(CacheKeysType(cache_keys.begin(), cache_keys.begin() + block_num),
                                                      BlockIndicesType(blocks.begin(), blocks.begin() + block_num),
                                                      insert_info.is_resident);
        }
        return;
    }

    for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
        const auto& full_keys = kv_cache_resource->cacheKeys(batch_id);
        if (full_keys.empty()) {
            continue;
        }
        const auto& full_dependencies = kv_cache_resource->cacheResource(batch_id).blockDependencies();

        if (!cp_active) {
            // Preserve the legacy non-CP GPU reuse surface: aggregate all groups
            // under one key. The prefix tree only receives extra dependency
            // metadata here.
            const size_t max_keys = full_keys.size();
            for (size_t pos = max_keys; pos > 0; --pos) {
                const size_t                        i = pos - 1;
                std::map<std::string, BlockIdxType> groups;
                bool                                has_valid = false;
                for (const auto& group : config_.groups()) {
                    const auto&  tag      = group.tag;
                    BlockIdxType block_id = NULL_BLOCK_IDX;
                    if (skipReuseCacheGroup(tag)) {
                        groups.emplace(tag, block_id);
                        continue;
                    }
                    const auto& blocks = kv_cache_resource->blocks(batch_id, tag);
                    if (i >= blocks.size()) {
                        groups.emplace(tag, block_id);
                        continue;
                    }
                    if (!isNullBlockIdx(blocks[i])) {
                        block_id  = blocks[i];
                        has_valid = true;
                    }
                    groups.emplace(tag, block_id);
                }
                if (has_valid) {
                    const auto dependency = i < full_dependencies.size() ?
                                                full_dependencies[i] :
                                                BlockDependency{false, 0, static_cast<uint32_t>(i)};
                    shared_block_cache_->put(full_keys[i],
                                             groups,
                                             {},
                                             insert_info.is_resident,
                                             SharedBlockCache::kGpuLogicalNamespace,
                                             dependency);
                }
            }
            continue;
        }

        // Per-group key namespace, per-(key, group) put. SharedBlockCache::put
        // merges multiple puts on the same key into one item with each group's block id
        // populated independently (NULL_BLOCK_IDX entries are skipped by the merge path).
        //
        // CP per-group key namespace: paged FULL groups use cp-subsampled (last-rank) keys
        // to align 1:1 with rank-local blocks; non-paged groups (SWA / LINEAR) keep the
        // full key sequence so their tail blocks (real entries at positions >= length-2)
        // get inserted alongside the keys that the reuseCache tail-loop later queries.
        CacheKeysType         cp_keys         = cpCanonicalCacheKeys(cp_mapper, full_keys);
        BlockDependenciesType cp_dependencies = cp_mapper->canonicalBlockDependencies(full_dependencies);
        auto                  token_ids       = insert_info.complete_token_ids->completeTokenIdsVec(batch_id);
        if (token_ids.size() <= 1) {
            continue;
        }
        const size_t token_len = token_ids.size() - 1;

        for (const auto& group : config_.groups()) {
            const auto& tag = group.tag;
            if (skipReuseCacheGroup(tag)) {
                continue;
            }
            const int            raw_group_seq = singleTypeManager(tag)->seqSizePerBlock();
            const bool           gp_sharded    = cpBlockRoundRobinGroup(cp_mapper, config_, tag);
            const bool           compact_swa   = cpCompactSwaGroup(tag, cp_mapper);
            const bool           use_cp_keys   = cp_active && (gp_sharded || compact_swa);
            const CacheKeysType& src_keys      = use_cp_keys ? cp_keys : full_keys;
            const auto&          dependencies  = use_cp_keys ? cp_dependencies : full_dependencies;
            const auto           namespace_id =
                use_cp_keys ? SharedBlockCache::kGpuCpCanonicalNamespace : SharedBlockCache::kGpuLogicalNamespace;
            if (src_keys.empty()) {
                continue;
            }
            const int    group_seq_size  = cpLogicalSeqSizeForGroup(cp_mapper, config_, tag, raw_group_seq);
            const size_t full_blocks_num = token_len / static_cast<size_t>(group_seq_size);
            const size_t n               = std::min(src_keys.size(), full_blocks_num);
            const auto&  blocks          = kv_cache_resource->blocks(batch_id, tag);
            const size_t loop_end        = std::min(n, blocks.size());

            // Reverse iterate so prefix-base keys land at MRU end (matches non-CP path).
            for (size_t pos = loop_end; pos > 0; --pos) {
                const size_t i = pos - 1;
                if (isNullBlockIdx(blocks[i])) {
                    continue;
                }
                std::map<std::string, BlockIdxType> groups;
                for (const auto& other_group : config_.groups()) {
                    groups.emplace(other_group.tag, other_group.tag == tag ? blocks[i] : NULL_BLOCK_IDX);
                }
                const auto dependency =
                    i < dependencies.size() ? dependencies[i] : BlockDependency{false, 0, static_cast<uint32_t>(i)};
                shared_block_cache_->put(src_keys[i], groups, {}, insert_info.is_resident, namespace_id, dependency);
            }
        }
    }
}

std::shared_ptr<KVCacheResource> CoordinatorCacheManager::incrKVCacheRef(const KVCacheResource& kvcache_resource,
                                                                         const CacheKeysType&   cache_keys,
                                                                         bool                   is_connector) {
    if (cache_keys.empty() || kvcache_resource.groupNums() <= 0) {
        return nullptr;
    }

    std::unordered_map<CacheKeyType, size_t> key_to_pos;
    const auto&                              resource_keys       = kvcache_resource.cacheKeys();
    const auto&                              source_dependencies = kvcache_resource.blockDependencies();
    RTP_LLM_CHECK_WITH_INFO(resource_keys.size() == source_dependencies.size(),
                            "incrKVCacheRef source timeline mismatch: keys=%zu dependencies=%zu",
                            resource_keys.size(),
                            source_dependencies.size());
    for (size_t i = 0; i < resource_keys.size(); ++i) {
        key_to_pos.emplace(resource_keys[i], i);
    }

    auto selected_resource_ptr = new KVCacheResource(kvcache_resource);
    auto deleter               = [self = shared_from_this(), is_connector](KVCacheResource* resource) {
        self->decrKVCacheRef(*resource, is_connector);
        delete resource;
    };
    std::shared_ptr<KVCacheResource> selected_resource(selected_resource_ptr, deleter);
    selected_resource->initGroups(config_);

    CacheKeysType                                     selected_keys;
    BlockDependenciesType                             selected_dependencies;
    std::unordered_map<std::string, BlockIndicesType> selected_blocks;
    for (const auto& group : config_.groups()) {
        selected_blocks.emplace(group.tag, BlockIndicesType{});
    }

    selected_dependencies.reserve(cache_keys.size());
    selected_keys.reserve(cache_keys.size());
    for (auto key : cache_keys) {
        auto it = key_to_pos.find(key);
        if (it == key_to_pos.end()) {
            continue;
        }
        const size_t                                  pos             = it->second;
        bool                                          any_valid_block = false;
        std::unordered_map<std::string, BlockIdxType> blocks_for_key;
        for (const auto& group : config_.groups()) {
            const auto& src_blocks = kvcache_resource.blocks(group.tag);
            const auto  block      = pos < src_blocks.size() ? src_blocks[pos] : NULL_BLOCK_IDX;
            blocks_for_key.emplace(group.tag, block);
            any_valid_block = any_valid_block || (!isNullBlockIdx(block) && block > 0);
        }
        const bool preserve_connector_tail = is_connector && !kvcache_resource.lastBlockAligned()
                                             && pos + 1 == resource_keys.size() && !selected_keys.empty();
        if (!any_valid_block && !preserve_connector_tail) {
            continue;
        }
        selected_keys.push_back(key);
        selected_dependencies.push_back(source_dependencies[pos]);
        for (const auto& group : config_.groups()) {
            selected_blocks.at(group.tag).push_back(blocks_for_key.at(group.tag));
        }
    }

    if (selected_keys.empty()) {
        return nullptr;
    }

    selected_resource->setCacheKeysAndBlockDependencies(std::move(selected_keys), std::move(selected_dependencies));
    selected_resource->setCacheKeysAreCpCanonical(kvcache_resource.cacheKeysAreCpCanonical());
    for (const auto& group : config_.groups()) {
        BlockIndicesType valid;
        for (auto b : selected_blocks.at(group.tag)) {
            if (!isNullBlockIdx(b) && b > 0) {
                valid.push_back(b);
            }
        }
        if (!valid.empty()) {
            referenceBlocks(group.tag, valid, is_connector);
        }
        selected_resource->mutableBlockIds(group.tag).assign(std::move(selected_blocks.at(group.tag)));
    }
    return selected_resource;
}

void CoordinatorCacheManager::decrKVCacheRef(const KVCacheResource& kvcache_resource, bool is_connector) {
    for (const auto& group : config_.groups()) {
        const auto&      tag       = group.tag;
        const auto&      block_ids = kvcache_resource.blockIds(tag);
        BlockIndicesType valid;
        for (auto b : block_ids.blocks()) {
            if (!isNullBlockIdx(b) && b > 0) {
                valid.push_back(b);
            }
        }
        if (!valid.empty()) {
            freeBlocks(tag, valid, is_connector);
        }
    }
}

bool CoordinatorCacheManager::updateKVBlock(const BatchKVCacheResourcePtr&  batch_kv_cache_resource,
                                            const std::vector<int>&         block_src_batch,
                                            bool                            copy_last_block,
                                            std::vector<TaggedBlockIdPair>& block_update_mapping) {
    block_update_mapping.clear();
    if (block_src_batch.empty()) {
        return true;
    }

    const int old_batch_size = batch_kv_cache_resource->batchSize();
    const int new_batch_size = static_cast<int>(block_src_batch.size());

    std::vector<int> batch_fork_count(old_batch_size, 0);
    for (const int old_batch_idx : block_src_batch) {
        RTP_LLM_CHECK_WITH_INFO(old_batch_idx >= 0 && old_batch_idx < old_batch_size,
                                "try to reuse an old batch %d that out of range %d",
                                old_batch_idx,
                                old_batch_size);
        ++batch_fork_count[old_batch_idx];
    }

    std::map<std::string, int> new_blocks_num;
    for (const auto& group : config_.groups()) {
        new_blocks_num.emplace(group.tag, 0);
    }
    for (int old_batch_idx = 0; old_batch_idx < old_batch_size; ++old_batch_idx) {
        const int fork_count = batch_fork_count[old_batch_idx];
        if (fork_count > 1 && copy_last_block) {
            for (const auto& group : config_.groups()) {
                const auto& tag = group.tag;
                if (!batch_kv_cache_resource->blocks(old_batch_idx, tag).empty()) {
                    new_blocks_num.at(tag) += fork_count - 1;
                }
            }
        }
    }

    // Transfer request ownership from dropped batches before allocating new
    // blocks. This keeps the operation transactional while allowing net-feasible
    // drop-and-fork updates to succeed when the pool is otherwise full.
    std::map<std::string, BlockIndicesType>                      replacement_blocks;
    std::map<std::string, BlockIndicesType>                      allocated_replacements;
    std::map<std::string, std::unordered_map<BlockIdxType, int>> transferred_ref_counts;
    for (const auto& group : config_.groups()) {
        const auto&                           tag = group.tag;
        std::unordered_set<BlockIdxType>      retained_blocks;
        std::unordered_map<BlockIdxType, int> dropped_block_counts;
        for (int old_batch_idx = 0; old_batch_idx < old_batch_size; ++old_batch_idx) {
            for (const auto block : batch_kv_cache_resource->blocks(old_batch_idx, tag)) {
                if (isNullBlockIdx(block) || block <= 0) {
                    continue;
                }
                if (batch_fork_count[old_batch_idx] == 0) {
                    ++dropped_block_counts[block];
                } else {
                    retained_blocks.insert(block);
                }
            }
        }

        auto&     replacements = replacement_blocks[tag];
        auto&     transferred  = transferred_ref_counts[tag];
        const int need         = new_blocks_num.at(tag);
        for (int old_batch_idx = 0; old_batch_idx < old_batch_size && static_cast<int>(replacements.size()) < need;
             ++old_batch_idx) {
            if (batch_fork_count[old_batch_idx] != 0) {
                continue;
            }
            const auto& dropped = batch_kv_cache_resource->blocks(old_batch_idx, tag);
            if (dropped.empty()) {
                continue;
            }
            const auto block = dropped.back();
            if (!isNullBlockIdx(block) && block > 0 && dropped_block_counts[block] == 1 && !retained_blocks.count(block)
                && !transferred.count(block)) {
                replacements.push_back(block);
                transferred[block] = 1;
            }
        }
    }

    auto rollback_replacements = [&]() {
        for (const auto& group : config_.groups()) {
            auto& blocks = allocated_replacements[group.tag];
            if (!blocks.empty()) {
                singleTypeManager(group.tag)->free(blocks);
                blocks.clear();
            }
        }
    };
    for (const auto& group : config_.groups()) {
        const auto& tag         = group.tag;
        const int   need_blocks = new_blocks_num.at(tag);
        auto&       reserved    = replacement_blocks[tag];
        reserved.reserve(static_cast<size_t>(need_blocks));
        for (int i = static_cast<int>(reserved.size()); i < need_blocks; ++i) {
            BlockIds    one_block(group.storedKernelBlocksPerKvBlock());
            const bool  ok     = singleTypeManager(tag)->malloc(one_block, singleTypeManager(tag)->seqSizePerBlock());
            const auto& blocks = one_block.blocks();
            if (ok && blocks.size() == 1 && !isNullBlockIdx(blocks.front())) {
                reserved.push_back(blocks.front());
                allocated_replacements[tag].push_back(blocks.front());
                continue;
            }
            if (!blocks.empty()) {
                allocated_replacements[tag].insert(allocated_replacements[tag].end(), blocks.begin(), blocks.end());
            }
            RTP_LLM_LOG_WARNING(
                "reserve replacement block failed for hybrid kv cache update, tag=%s need=%d reserved=%zu",
                tag.c_str(),
                need_blocks,
                reserved.size());
            rollback_replacements();
            return false;
        }
    }

    for (int old_batch_idx = 0; old_batch_idx < old_batch_size; ++old_batch_idx) {
        if (batch_fork_count[old_batch_idx] != 0) {
            continue;
        }
        for (const auto& group : config_.groups()) {
            const auto&      tag = group.tag;
            BlockIndicesType to_free;
            auto&            transferred = transferred_ref_counts[tag];
            for (const auto block : batch_kv_cache_resource->blocks(old_batch_idx, tag)) {
                if (isNullBlockIdx(block) || block <= 0) {
                    continue;
                }
                auto it = transferred.find(block);
                if (it != transferred.end() && it->second > 0) {
                    --it->second;
                } else {
                    to_free.push_back(block);
                }
            }
            if (!to_free.empty()) {
                singleTypeManager(tag)->free(to_free);
            }
        }
    }

    std::vector<KVCacheResource> old_resources;
    batch_kv_cache_resource->resetAndReturnOldResources(new_batch_size, old_resources);
    batch_kv_cache_resource->initGroups(config_);
    std::map<std::string, size_t> next_replacement;

    for (int new_batch_idx = 0; new_batch_idx < new_batch_size; ++new_batch_idx) {
        const int old_batch_idx = block_src_batch[new_batch_idx];
        auto&     fork_count    = batch_fork_count[old_batch_idx];
        RTP_LLM_CHECK_WITH_INFO(fork_count > 0, "old batch %d has been forked too many times", old_batch_idx);

        if (fork_count == 1) {
            batch_kv_cache_resource->moveBatchResource(new_batch_idx, std::move(old_resources[old_batch_idx]));
        } else {
            const auto& source_resource = old_resources[old_batch_idx];
            auto&       fork_resource   = batch_kv_cache_resource->cacheResource(new_batch_idx);
            fork_resource.setCacheKeysAndBlockDependencies(source_resource.cacheKeys(),
                                                           source_resource.blockDependencies());
            fork_resource.setCacheKeysAreCpCanonical(source_resource.cacheKeysAreCpCanonical());
            for (const auto& group : config_.groups()) {
                const auto& tag       = group.tag;
                auto&       block_ids = batch_kv_cache_resource->mutableBlockIds(new_batch_idx, tag);
                singleTypeManager(tag)->reference(block_ids, old_resources[old_batch_idx].blocks(tag));

                if (copy_last_block && !block_ids.blocks().empty()) {
                    const int  old_block       = block_ids.popBack();
                    const bool old_block_valid = !isNullBlockIdx(old_block) && old_block > 0;
                    if (old_block_valid) {
                        singleTypeManager(tag)->free({old_block});
                    }

                    auto&      reserved     = replacement_blocks[tag];
                    const auto reserved_idx = next_replacement[tag]++;
                    RTP_LLM_CHECK_WITH_INFO(reserved_idx < reserved.size(),
                                            "missing reserved replacement block for hybrid kv cache update, tag=%s",
                                            tag.c_str());
                    const int new_block = reserved[reserved_idx];
                    block_ids.add({new_block});
                    if (old_block_valid && !isNullBlockIdx(new_block) && new_block > 0) {
                        block_update_mapping.push_back({tag, old_block, new_block});
                    }
                }
            }
        }
        --fork_count;
    }
    for (const auto& group : config_.groups()) {
        const auto& tag = group.tag;
        RTP_LLM_CHECK_WITH_INFO(next_replacement[tag] == replacement_blocks[tag].size(),
                                "unused replacement blocks after hybrid kv cache update, tag=%s used=%zu reserved=%zu",
                                tag.c_str(),
                                next_replacement[tag],
                                replacement_blocks[tag].size());
    }
    return true;
}

int CoordinatorCacheManager::seqSizePerBlock() const {
    return static_cast<int>(config_.seq_size_per_block);
}

void CoordinatorCacheManager::rollbackBlockIdsToSize(std::string_view           tag,
                                                     BlockIds&                  block_ids,
                                                     size_t                     original_size,
                                                     const std::vector<size_t>& backfilled_positions) {
    const auto&      blocks = block_ids.blocks();
    BlockIndicesType blocks_to_free;
    blocks_to_free.reserve(backfilled_positions.size() + blocks.size() - std::min(original_size, blocks.size()));
    for (size_t pos : backfilled_positions) {
        RTP_LLM_CHECK_WITH_INFO(pos < original_size && pos < blocks.size(),
                                "invalid hybrid rollback tag=%s backfill position=%zu original_size=%zu size=%zu",
                                std::string(tag).c_str(),
                                pos,
                                original_size,
                                blocks.size());
        if (!isNullBlockIdx(blocks[pos])) {
            blocks_to_free.push_back(blocks[pos]);
        }
    }
    const auto appended_blocks = validBlocksAfter(blocks, original_size);
    blocks_to_free.insert(blocks_to_free.end(), appended_blocks.begin(), appended_blocks.end());
    if (!blocks_to_free.empty()) {
        freeBlocks(tag, blocks_to_free);
    }
    for (size_t pos : backfilled_positions) {
        block_ids.setAt(pos, NULL_BLOCK_IDX);
    }
    block_ids.resize(original_size);
}

void CoordinatorCacheManager::rollbackInitMalloc(
    BatchKVCacheResource&                             kv_resource,
    const std::map<std::string, BlockIndicesType>&    referenced_blocks,
    const std::map<std::string, size_t>&              original_sizes,
    const std::map<std::string, std::vector<size_t>>& backfilled_positions) {
    for (const auto& group : config_.groups()) {
        const auto& tag       = group.tag;
        auto&       block_ids = kv_resource.mutableBlockIds(0, tag);
        rollbackBlockIdsToSize(tag, block_ids, original_sizes.at(tag), backfilled_positions.at(tag));
        const auto referenced_it = referenced_blocks.find(tag);
        if (referenced_it != referenced_blocks.end() && !referenced_it->second.empty()) {
            freeBlocks(tag, referenced_it->second);
        }
        block_ids.resize(0);
    }
    kv_resource.cacheResource(0).setDeviceReuseBlockNum(0);
}

void CoordinatorCacheManager::rollbackIncrMalloc(
    BatchKVCacheResource&                                          kv_resource,
    const std::vector<std::map<std::string, size_t>>&              batch_original_sizes,
    const std::vector<std::map<std::string, std::vector<size_t>>>& batch_backfilled_positions,
    size_t                                                         last_touched_batch) {
    const size_t rollback_end = std::min(last_touched_batch + 1, batch_original_sizes.size());
    for (size_t batch_idx = 0; batch_idx < rollback_end; ++batch_idx) {
        for (const auto& group : config_.groups()) {
            const auto& tag       = group.tag;
            auto&       block_ids = kv_resource.mutableBlockIds(static_cast<int>(batch_idx), tag);
            rollbackBlockIdsToSize(
                tag, block_ids, batch_original_sizes[batch_idx].at(tag), batch_backfilled_positions[batch_idx].at(tag));
        }
    }
}

MemoryType CoordinatorCacheManager::memoryTypeForGroup(std::string_view tag) const {
    (void)config_.group(tag);
    return allocation_type_ == AllocationType::DEVICE ? MemoryType::MEMORY_GPU : MemoryType::MEMORY_CPU;
}

void CoordinatorCacheManager::copyBlockMappingForGroup(std::string_view                tag,
                                                       const std::vector<BlockIdPair>& block_update_mapping) const {
    if (block_update_mapping.empty()) {
        return;
    }

    const auto   memory_type         = memoryTypeForGroup(tag);
    const auto   copy_type           = BatchCopyParams::get_copy_type(memory_type, memory_type);
    const auto&  group               = config_.group(tag);
    const auto&  spec                = group.spec;
    const size_t kv_block_size_bytes = spec->block_size_bytes();
    const size_t scale_block_bytes   = spec->scale_block_size_bytes();
    const size_t buffers_per_layer   = scale_block_bytes > 0 ? 2 : 1;

    BatchCopyParams copy_params;
    copy_params.reserve(copy_type, config_.groupLayerIds(tag).size() * block_update_mapping.size() * buffers_per_layer);

    for (const auto& [src_block_index, dest_block_index] : block_update_mapping) {
        for (int layer_id : config_.groupLayerIds(tag)) {
            auto src_addr_info = singleTypeManager(tag)->convertIndexToAddr(layer_id, src_block_index);
            auto dst_addr_info = singleTypeManager(tag)->convertIndexToAddr(layer_id, dest_block_index);

            RTP_LLM_CHECK_WITH_INFO(src_addr_info.kv_addr && dst_addr_info.kv_addr,
                                    "failed to get block address for tag=%s layer %d src_block %d dst_block %d",
                                    std::string(tag).c_str(),
                                    layer_id,
                                    src_block_index,
                                    dest_block_index);

            copy_params.add(dst_addr_info.kv_addr, src_addr_info.kv_addr, kv_block_size_bytes, copy_type);

            if (scale_block_bytes > 0 && src_addr_info.kv_scale_addr && dst_addr_info.kv_scale_addr) {
                copy_params.add(dst_addr_info.kv_scale_addr, src_addr_info.kv_scale_addr, scale_block_bytes, copy_type);
            }
        }
    }

    execBatchCopy(copy_params);
}

int CoordinatorCacheManager::getNeedBlocks(const MallocInfo& malloc_info) const {
    if (!malloc_info.batch_kv_cache_resource || !malloc_info.complete_token_ids) {
        return 0;
    }
    const auto& cp_mapper          = cp_slot_mapper_;
    const int   batch_size         = malloc_info.batch_kv_cache_resource->batchSize();
    const int   total_seq_len      = malloc_info.complete_token_ids->totalSeqLength();
    const int   raw_common_seq_len = std::min(malloc_info.complete_token_ids->commonSeqLength(), total_seq_len);
    const int   raw_seq_len        = malloc_info.complete_token_ids->seqLength();
    const int   reserve_step       = malloc_info.complete_token_ids->getReserveStep();
    const bool  reuse_enabled      = malloc_info.reuse_cache;
    const int   reuse_blocks_len   = reuse_enabled ? malloc_info.batch_kv_cache_resource->curBlocksNum() : 0;

    int common_blocks_total = 0;
    int extra_blocks_total  = 0;
    for (const auto& group : config_.groups()) {
        const auto& tag              = group.tag;
        const int   group_common_seq = cpEffectiveSeqLenForGroup(cp_mapper, config_, tag, raw_common_seq_len);
        const int   group_seq_len    = cpEffectiveSeqLenForGroup(cp_mapper, config_, tag, raw_seq_len);
        const auto  need             = singleTypeManager(tag)->getNeedBlocks(
            group_common_seq, group_seq_len, reserve_step, reuse_blocks_len, reuse_enabled);
        common_blocks_total += need.common_blocks;
        extra_blocks_total += need.extra_blocks;
    }
    return common_blocks_total + batch_size * extra_blocks_total;
}

int CoordinatorCacheManager::estimatePeakNeedBlocks(const KVCacheResource& kv_cache_resource,
                                                    int                    seq_len,
                                                    int                    remaining_tokens,
                                                    int                    reserve_step,
                                                    bool                   enable_reuse_cache) const {
    int need_blocks = 0;
    for (const auto& group : config_.groups()) {
        need_blocks += singleTypeManager(group.tag)->estimatePeakNeedBlocks(
            seq_len, kv_cache_resource.blocks(group.tag), remaining_tokens, reserve_step, enable_reuse_cache);
    }
    return need_blocks;
}

int CoordinatorCacheManager::estimateInitialBatchPeakNeedBlocks(int  seq_len,
                                                                int  common_seq_len,
                                                                int  remaining_tokens,
                                                                int  reserve_step,
                                                                bool enable_reuse_cache,
                                                                int  target_batch_size) const {
    int peak_blocks = 0;
    for (const auto& manager : single_type_managers_) {
        peak_blocks += manager->estimateInitialBatchPeakNeedBlocks(
            seq_len, common_seq_len, remaining_tokens, reserve_step, enable_reuse_cache, target_batch_size);
    }
    return peak_blocks;
}

int CoordinatorCacheManager::singleBatchNeedBlocks(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                                   int                            seq_len,
                                                   int                            reserve_step) const {
    int need_blocks = 0;
    for (const auto& group : config_.groups()) {
        const auto& tag               = group.tag;
        const int   effective_seq_len = cpEffectiveSeqLenForGroup(cp_slot_mapper_, config_, tag, seq_len);
        const int   cur_blocks        = batch_kv_cache_resource->blocksNum(0, tag);
        need_blocks += singleTypeManager(tag)->needBlocksNum(effective_seq_len, cur_blocks, reserve_step);
    }
    return need_blocks;
}

}  // namespace rtp_llm
