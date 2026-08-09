#include "rtp_llm/cpp/cache/HybridPoolKVCacheAllocator.h"

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/core/OpData.h"

namespace rtp_llm {
namespace {
inline int cpEffectiveSeqLenForReserve(const std::shared_ptr<CPSlotMapper>& mapper,
                                       const CacheConfig&                   config,
                                       std::string_view                     tag,
                                       int                                  seq_len) {
    return (mapper && mapper->isSharded()) ? mapper->effectiveSeqLenForAlloc(config, tag, seq_len) : seq_len;
}

void appendPoolSummary(std::ostringstream&    os,
                       bool&                  has_any,
                       const std::string&     tag,
                       CacheGroupType         group_type,
                       const BlockPoolConfig& pool_config) {
    static constexpr double kBytesPerMB = 1024.0 * 1024.0;
    if (has_any) {
        os << "; ";
    }
    has_any = true;
    os << "pool_name=" << pool_config.pool_name << ", tag=" << tag << ", type=" << cacheGroupTypeName(group_type)
       << ", size=" << pool_config.total_size_bytes << " bytes(" << std::fixed << std::setprecision(2)
       << static_cast<double>(pool_config.total_size_bytes) / kBytesPerMB << " MB)"
       << ", blocks=" << pool_config.block_num;
}
const GroupTopology& validatePoolGroupForLayer(const CacheConfig& config, int layer_id, std::string_view tag) {
    RTP_LLM_CHECK_WITH_INFO(layer_id >= 0 && static_cast<size_t>(layer_id) < config.totalLayerNum(),
                            "invalid layer id %d for layer_all_num=%u",
                            layer_id,
                            config.totalLayerNum());
    return config.topology().groupForLayer(layer_id, tag);
}
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

}  // namespace

// CoordinatorKVCacheManager common implementation.

bool CoordinatorKVCacheManager::init() {
    RTP_LLM_CHECK_WITH_INFO(doInit(), "init failed");

    // NOTE: the reservable block count depends on initialized block pools and must be queried after `doInit()`.
    const int64_t reserve_ratio = reserve_block_ratio_;
    if (reserve_ratio > 0) {
        const size_t available_blocks = reservableAvailableBlocksNum();
        const size_t reserve_blocks = static_cast<size_t>(reserve_ratio) * available_blocks / static_cast<size_t>(100);
        reserve_block_num_          = reserve_blocks;
        RTP_LLM_LOG_INFO(
            "CoordinatorKVCacheManager set reserve blocks: ratio=%ld%% reserve_blocks=%zu available_blocks=%zu",
            reserve_ratio,
            reserve_blocks,
            available_blocks);
    } else {
        reserve_block_num_ = 0;
    }

    return true;
}

size_t CoordinatorKVCacheManager::reservableAvailableBlocksNum() const {
    return availableBlocksNum();
}

MallocResult CoordinatorKVCacheManager::initMalloc(const MallocInfo& malloc_info) {
    auto init_result = initMallocForCommonLen(malloc_info);
    if (!init_result.success) {
        FreeInfo free_info{malloc_info.batch_kv_cache_resource, malloc_info.complete_token_ids};
        free(free_info);
        return init_result;
    }

    auto incr_result = incrMalloc(malloc_info);
    if (!incr_result.success) {
        FreeInfo free_info{malloc_info.batch_kv_cache_resource, malloc_info.complete_token_ids};
        free(free_info);
        return incr_result;
    } else {
        if (metrics_reporter_ && malloc_info.enable_device_cache) {
            int64_t device_input_length = 0;
            if (malloc_info.batch_kv_cache_resource) {
                const auto& cache_keys      = malloc_info.batch_kv_cache_resource->cacheKeys(0);
                size_t      match_keys_size = cache_keys.size();
                device_input_length         = static_cast<int64_t>(match_keys_size) * config_.seq_size_per_block;
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

MallocResult CoordinatorKVCacheManager::malloc(const MallocInfo& malloc_info) {
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

int CoordinatorKVCacheManager::estimateBatchPeakNeedBlocks(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
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

uint32_t CoordinatorKVCacheManager::convertToGlobalLayerId(size_t model_id, int local_layer_id) const {
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

void CoordinatorKVCacheManager::blockBatchCopy(const std::vector<GroupBlockIdPair>& copy_mapping) {
    if (copy_mapping.empty()) {
        return;
    }

    const auto memory_type = allocation_type_ == AllocationType::DEVICE ? rtp_llm::MEMORY_GPU : rtp_llm::MEMORY_CPU;
    const auto copy_type   = BatchCopyParams::get_copy_type(memory_type, memory_type);
    size_t     copy_count  = 0;
    for (const auto& mapping : copy_mapping) {
        const auto& group = config_.topology().group(mapping.tag);
        copy_count += group.layer_ids.size() * (group.kv_scale_stride_bytes > 0 ? 2 : 1);
    }

    BatchCopyParams copy_params;
    copy_params.reserve(copy_type, copy_count);
    for (const auto& mapping : copy_mapping) {
        const auto& group = config_.topology().group(mapping.tag);
        for (int layer_id : group.layer_ids) {
            const auto src_addr = convertIndexToAddr(layer_id, mapping.tag, mapping.src);
            const auto dst_addr = convertIndexToAddr(layer_id, mapping.tag, mapping.dst);
            RTP_LLM_CHECK_WITH_INFO(src_addr.kv_addr && dst_addr.kv_addr,
                                    "cache block copy failed for tag=%s layer=%d src=%d dst=%d",
                                    mapping.tag.c_str(),
                                    layer_id,
                                    mapping.src,
                                    mapping.dst);
            copy_params.add(dst_addr.kv_addr, src_addr.kv_addr, group.kv_block_stride_bytes, copy_type);
            if (group.kv_scale_stride_bytes > 0 && src_addr.kv_scale_addr && dst_addr.kv_scale_addr) {
                copy_params.add(dst_addr.kv_scale_addr, src_addr.kv_scale_addr, group.kv_scale_stride_bytes, copy_type);
            }
        }
    }
    execBatchCopy(copy_params);
}

size_t CoordinatorKVCacheManager::freeBlocksNum() const {
    size_t blocks = 0;
    for (const auto& group : config_.topology().groups()) {
        if (const auto pool = getBlockPool(group.tag)) {
            blocks += pool->freeBlocksNum();
        }
    }
    return blocks;
}

int64_t CoordinatorKVCacheManager::getMrCostTimeMs() const {
    int64_t cost_ms = 0;
    for (const auto& group : config_.topology().groups()) {
        if (const auto pool = getBlockPool(group.tag)) {
            cost_ms += pool->getMrCostTimeMs();
        }
    }
    return cost_ms;
}

size_t CoordinatorKVCacheManager::availableBlocksNum() const {
    size_t blocks = 0;
    for (const auto& group : config_.topology().groups()) {
        if (const auto pool = getBlockPool(group.tag)) {
            blocks += pool->availableBlocksNum();
        }
    }
    return blocks;
}

BatchKVCacheResourcePtr CoordinatorKVCacheManager::popBlocksFromCache(size_t min_blocks_to_free) {
    if (!shared_block_cache_ || min_blocks_to_free == 0) {
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
            tags.AddTag("evict_policy",
                        evict_result.evicted_independent_tag.count(cache_key) ? "independent" : "chain");
            tags.AddTag("backing", "device");
            metrics_reporter_->report<RtpLLMCacheEvictionMetrics, RtpLLMCacheEvictionMetricsCollector>(&tags,
                                                                                                       &collector);
        }
    }

    auto batch_resource = std::make_shared<BatchKVCacheResource>();
    batch_resource->resetBatchSize(1);
    batch_resource->initGroups(config_.topologyPtr());
    batch_resource->setLastBlockAligned(true);

    for (const auto& group : config_.topology().groups()) {
        batch_resource->mutableBlockIds(0, group.tag).resize(evict_result.evicted_keys.size(), NULL_BLOCK_IDX);
    }

    CacheKeysType         evicted_keys;
    BlockDependenciesType evicted_dependencies;
    evicted_keys.reserve(evict_result.evicted_keys.size());
    evicted_dependencies.reserve(evict_result.evicted_keys.size());
    for (size_t evicted_idx = 0; evicted_idx < evict_result.evicted_keys.size(); ++evicted_idx) {
        const auto  cache_key       = evict_result.evicted_keys[evicted_idx];
        const auto& group_block_ids = evict_result.evicted_group_block_ids.at(cache_key);
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
        for (const auto& [tag, block_id] : group_block_ids) {
            if (!isNullBlockIdx(block_id)) {
                batch_resource->mutableBlockIds(0, tag).setAt(evicted_idx, block_id);
            }
        }
    }
    batch_resource->cacheResource(0).setCacheKeysAndBlockDependencies(std::move(evicted_keys),
                                                                      std::move(evicted_dependencies));
    batch_resource->cacheResource(0).setCacheKeysAreCpCanonical(true);
    return batch_resource;
}

void CoordinatorKVCacheManager::blockCacheFree(const BatchKVCacheResourcePtr& batch_kv_cache_resource) {
    if (!batch_kv_cache_resource) {
        return;
    }

    for (const auto& group : config_.topology().groups()) {
        BlockIndicesType                 blocks_to_free;
        std::unordered_set<BlockIdxType> seen_blocks;
        for (int batch_id = 0; batch_id < batch_kv_cache_resource->batchSize(); ++batch_id) {
            for (const auto block_idx : batch_kv_cache_resource->blocks(batch_id, group.tag)) {
                if (isNullBlockIdx(block_idx) || !seen_blocks.insert(block_idx).second) {
                    continue;
                }
                blocks_to_free.push_back(block_idx);
            }
        }
        if (!blocks_to_free.empty()) {
            if (const auto pool = getBlockPool(group.tag)) {
                pool->blockCacheFree(blocks_to_free);
            }
        }
    }
}

size_t CoordinatorKVCacheManager::requestRefBlocksNum() const {
    size_t blocks = 0;
    for (const auto& group : config_.topology().groups()) {
        if (const auto pool = getBlockPool(group.tag)) {
            blocks += pool->requestRefBlocksNum();
        }
    }
    return blocks;
}

size_t CoordinatorKVCacheManager::connectorRefBlocksNum() const {
    size_t blocks = 0;
    for (const auto& group : config_.topology().groups()) {
        if (const auto pool = getBlockPool(group.tag)) {
            blocks += pool->connectorRefBlocksNum();
        }
    }
    return blocks;
}

size_t CoordinatorKVCacheManager::blockCacheRefBlocksNum() const {
    size_t blocks = 0;
    for (const auto& group : config_.topology().groups()) {
        if (const auto pool = getBlockPool(group.tag)) {
            blocks += pool->blockCacheRefBlocksNum();
        }
    }
    return blocks;
}

size_t CoordinatorKVCacheManager::notInUseBlocksNum() const {
    size_t blocks = 0;
    for (const auto& group : config_.topology().groups()) {
        if (const auto pool = getBlockPool(group.tag)) {
            blocks += pool->notInUseBlocksNum();
        }
    }
    return blocks;
}

size_t CoordinatorKVCacheManager::maxSequenceLength() const {
    size_t min_tokens = std::numeric_limits<size_t>::max();
    for (const auto& group : config_.topology().groups()) {
        if (const auto pool = getBlockPool(group.tag)) {
            const size_t seq_size = logicalSeqSizePerBlockForCapacity(group.tag);
            min_tokens            = std::min(min_tokens, pool->totalBlocksNum() * seq_size);
        }
    }
    return min_tokens != std::numeric_limits<size_t>::max() ? min_tokens : 0;
}

size_t CoordinatorKVCacheManager::availableTokensNum() const {
    return availableBlocksNum() * static_cast<size_t>(seqSizePerBlock());
}

size_t CoordinatorKVCacheManager::totalBlocksNum() const {
    size_t blocks = 0;
    for (const auto& group : config_.topology().groups()) {
        if (const auto pool = getBlockPool(group.tag)) {
            blocks += pool->totalBlocksNum();
        }
    }
    return blocks;
}

size_t CoordinatorKVCacheManager::logicalSeqSizePerBlockForCapacity(std::string_view tag) const {
    if (cp_slot_mapper_ && cp_slot_mapper_->isSharded()) {
        return cp_slot_mapper_->logicalSeqSizePerBlock(config_, tag);
    }
    return config_.seqSizePerBlockForGroup(tag);
}

size_t CoordinatorKVCacheManager::totalTokensNum() const {
    return totalBlocksNum() * static_cast<size_t>(seqSizePerBlock());
}

std::vector<KVCachePoolMetricsSnapshot> CoordinatorKVCacheManager::poolMetricsSnapshots() const {
    return {};
}

std::vector<std::string> CoordinatorKVCacheManager::independentEvictionTags() const {
    return {};
}

void CoordinatorKVCacheManager::regUserMr(size_t model_id, std::shared_ptr<CacheStore> cache_store) {
    for (const auto& group : config_.topology().groups()) {
        if (const auto pool = getBlockPool(group.tag)) {
            pool->regUserMr(model_id, cache_store);
        }
    }
}

// HybridPool initialization and pool lookup.

HybridPoolKVCacheAllocator::HybridPoolKVCacheAllocator(
    const CacheConfig&                 config,
    AllocationType                     allocation_type,
    const kmonitor::MetricsReporterPtr metrics_reporter,
    int64_t                            reserve_block_ratio,
    RoleType                           role_type):
    CoordinatorKVCacheManager(config, allocation_type, metrics_reporter, reserve_block_ratio), role_type_(role_type) {}

bool HybridPoolKVCacheAllocator::doInit() {
    RTP_LLM_CHECK_WITH_INFO(config_.groupNums() > 0, "no cache groups found in CacheConfig");

    SharedBlockCache*       shared_cache_raw = shared_block_cache_ ? shared_block_cache_.get() : nullptr;
    static constexpr double kBytesPerMB      = 1024.0 * 1024.0;
    std::ostringstream      pool_summary;
    size_t                  pool_total_bytes  = 0;
    size_t                  pool_total_blocks = 0;
    bool                    has_pool          = false;

    struct PoolPlan {
        const GroupTopology* group;
        BlockPoolConfig      pool_config;
    };
    std::vector<PoolPlan> pool_plans;
    pool_plans.reserve(config_.topology().groups().size());

    for (const auto& group_topology : config_.topology().groups()) {
        auto pool_config = BlockPoolConfigHelper::createConfigForGroup(config_, group_topology.tag);
        appendPoolSummary(pool_summary, has_pool, group_topology.tag, group_topology.policy.group_type, pool_config);
        pool_total_bytes += pool_config.total_size_bytes;
        pool_total_blocks += pool_config.block_num;
        pool_plans.push_back(PoolPlan{&group_topology, std::move(pool_config)});
    }

    if (has_pool) {
        const auto summary = pool_summary.str();
        RTP_LLM_LOG_INFO("HybridPool pool plan: pools=[%s], total_size=%zu bytes total_size_mb=%.2f "
                         "total_blocks=%zu",
                         summary.c_str(),
                         pool_total_bytes,
                         static_cast<double>(pool_total_bytes) / kBytesPerMB,
                         pool_total_blocks);
    }

    for (const auto& plan : pool_plans) {
        const auto& group_topology = *plan.group;
        const auto& pool_config    = plan.pool_config;
        const auto  group_type     = group_topology.policy.group_type;

        auto group_pool = std::make_shared<BlockPool>(pool_config, allocation_type_, use_cuda_malloc_block_pool_);
        try {
            RTP_LLM_CHECK_WITH_INFO(group_pool->init(), "BlockPool::init returned false");
        } catch (const std::exception& e) {
            RTP_LLM_FAIL("Failed to initialize block pool: tag=%s type=%s bytes=%zu blocks=%u layers=%zu error=%s",
                         group_topology.tag.c_str(),
                         cacheGroupTypeName(group_type),
                         pool_config.total_size_bytes,
                         pool_config.block_num,
                         group_topology.layer_ids.size(),
                         e.what());
        }

        SingleTypeKVCacheManagerPtr group;
        if (group_type == CacheGroupType::LINEAR) {
            group = std::make_shared<LinearKVCacheManager>(
                group_topology, group_pool, config_.linear_step, shared_cache_raw, metrics_reporter_);
            linear_group_tags_.push_back(group_topology.tag);
        } else if (group_type == CacheGroupType::SWA) {
            group = std::make_shared<SWAKVCacheManager>(
                group_topology, group_pool, config_.linear_step, shared_cache_raw, metrics_reporter_);
            swa_group_tags_.push_back(group_topology.tag);
        } else {
            group =
                std::make_shared<FullKVCacheManager>(group_topology, group_pool, shared_cache_raw, metrics_reporter_);
            full_group_tags_.push_back(group_topology.tag);
        }

        RTP_LLM_CHECK_WITH_INFO(
            group->init(), "Failed to initialize single-type KV cache manager %s", pool_config.pool_name.c_str());
        RTP_LLM_CHECK(group_block_pools_.emplace(group_topology.tag, group_pool).second);
        RTP_LLM_CHECK(single_type_managers_.emplace(group_topology.tag, std::move(group)).second);
    }

    if (shared_block_cache_) {
        std::vector<std::pair<std::string, BlockPoolPtr>> pools;
        pools.reserve(group_block_pools_.size());
        for (const auto& [tag, pool] : group_block_pools_) {
            pools.push_back({tag, pool});
        }
        shared_block_cache_->init(pools);
    }

    RTP_LLM_LOG_INFO("HybridPoolKVCacheAllocator init success, group pools=%zu", group_block_pools_.size());
    return true;
}

const SingleTypeKVCacheManagerPtr&
HybridPoolKVCacheAllocator::cacheGroupForTag(std::string_view tag, const char* context) const {
    const auto value = std::string(tag);
    const auto it    = single_type_managers_.find(value);
    RTP_LLM_CHECK_WITH_INFO(it != single_type_managers_.end(),
                            "%s: missing single-type KV cache manager for tag=%s",
                            context,
                            value.c_str());
    RTP_LLM_CHECK_WITH_INFO(it->second != nullptr, "%s: null KV cache group for tag=%s", context, value.c_str());
    return it->second;
}

const BlockPoolPtr& HybridPoolKVCacheAllocator::blockPoolForTag(std::string_view tag,
                                                                         const char*      context) const {
    const auto value = std::string(tag);
    const auto it    = group_block_pools_.find(value);
    RTP_LLM_CHECK_WITH_INFO(
        it != group_block_pools_.end(), "%s: missing block pool for tag=%s", context, value.c_str());
    RTP_LLM_CHECK_WITH_INFO(it->second != nullptr, "%s: null block pool for tag=%s", context, value.c_str());
    return it->second;
}

// Allocation, reuse, and rollback.

bool HybridPoolKVCacheAllocator::skipReuseCacheGroup(std::string_view tag) const {
    const auto it = single_type_managers_.find(std::string(tag));
    return it != single_type_managers_.end() && !it->second->prefixReuseEnabled();
}

bool HybridPoolKVCacheAllocator::cpCompactSwaGroup(std::string_view                     tag,
                                                            const std::shared_ptr<CPSlotMapper>& mapper) const {
    return mapper && mapper->isSharded() && single_type_managers_.find(std::string(tag)) != single_type_managers_.end()
           && mapper->compactLastRankGroup(config_, tag);
}

int HybridPoolKVCacheAllocator::reuseCache(const CacheKeysType&                 cache_keys,
                                                    BatchKVCacheResource&                kv_resource,
                                                    const std::shared_ptr<CPSlotMapper>& cp_mapper) {
    // Under cp shard, FULL groups index block_ids by cp-virtual-block units
    // (one entry covers cp_size physical blocks). LINEAR/SWA groups index by
    // raw block_size logical blocks. So when populating tail blocks for
    // LINEAR/SWA we need to scale the array length and matched-block position
    // back to the logical-block coordinate system.
    const int cp_scale              = (cp_mapper && cp_mapper->isSharded()) ? cp_mapper->cpSize() : 1;
    int       min_full_reuse_blocks = static_cast<int>(cache_keys.size());
    std::unordered_map<std::string, BlockIndicesType> full_matched_blocks;

    for (const auto& tag : full_group_tags_) {
        auto match_result     = cacheGroupForTag(tag, "reuseCache full match")->match(cache_keys);
        min_full_reuse_blocks = std::min(min_full_reuse_blocks, static_cast<int>(match_result.reuse_blocks));
        full_matched_blocks.emplace(tag, std::move(match_result.block_indices));
    }

    int                                               pos = min_full_reuse_blocks - 1;
    std::unordered_map<std::string, BlockIdxType>     linear_tail_blocks;
    std::unordered_map<std::string, BlockIndicesType> swa_tail_blocks;
    const bool has_tail_groups = !linear_group_tags_.empty() || !swa_group_tags_.empty();
    for (; pos >= 0 && has_tail_groups; --pos) {
        bool                                              all_tail_groups_matched = true;
        std::unordered_map<std::string, BlockIdxType>     candidate_linear_tail_blocks;
        std::unordered_map<std::string, BlockIndicesType> candidate_swa_tail_blocks;
        for (const auto& tag : linear_group_tags_) {
            auto result = cacheGroupForTag(tag, "reuseCache linear match")
                              ->matchSingleKey(cache_keys[static_cast<size_t>(pos)]);
            if (result.block_indices.empty()) {
                all_tail_groups_matched = false;
                break;
            }
            candidate_linear_tail_blocks.emplace(tag, result.block_indices[0]);
        }
        if (!all_tail_groups_matched) {
            continue;
        }
        for (const auto& tag : swa_group_tags_) {
            if (skipReuseCacheGroup(tag)) {
                continue;
            }
            auto result = cacheGroupForTag(tag, "reuseCache SWA match")
                              ->matchSingleKey(cache_keys[static_cast<size_t>(pos)]);
            if (result.block_indices.empty()) {
                all_tail_groups_matched = false;
                break;
            }
            candidate_swa_tail_blocks[tag].push_back(result.block_indices[0]);
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
        BlockIndicesType full_blocks = std::move(full_matched_blocks.at(tag));
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
    for (const auto& tag : linear_group_tags_) {
        kv_resource.mutableBlockIds(0, tag).assign(
            BlockIndicesType(static_cast<size_t>(logical_reuse_len), NULL_BLOCK_IDX));
        kv_resource.mutableBlockIds(0, tag).setAt(static_cast<size_t>(logical_reuse_len - 1),
                                                  linear_tail_blocks.at(tag));
    }
    for (const auto& tag : swa_group_tags_) {
        const int group_reuse_len = cpCompactSwaGroup(tag, cp_mapper) ? reuse_blocks_len : logical_reuse_len;
        kv_resource.mutableBlockIds(0, tag).assign(
            BlockIndicesType(static_cast<size_t>(group_reuse_len), NULL_BLOCK_IDX));
        if (skipReuseCacheGroup(tag)) {
            continue;
        }
        const auto&  tail_blocks = swa_tail_blocks.at(tag);
        const size_t tail_begin =
            static_cast<size_t>(std::max(group_reuse_len - static_cast<int>(tail_blocks.size()), 0));
        for (size_t j = 0; j < tail_blocks.size(); ++j) {
            kv_resource.mutableBlockIds(0, tag).setAt(tail_begin + j, tail_blocks[j]);
        }
    }
    return reuse_blocks_len;
}

MallocResult HybridPoolKVCacheAllocator::initMallocForCommonLen(const MallocInfo& malloc_info) {
    auto&     kv_resource = malloc_info.batch_kv_cache_resource;
    const int batch_size  = kv_resource->batchSize();

    const int   seq_len        = malloc_info.complete_token_ids->seqLength();
    const int   common_seq_len = std::min(malloc_info.complete_token_ids->commonSeqLength(), seq_len);
    const auto& cp_mapper      = cp_slot_mapper_;
    // All paged FULL groups share one cache-key namespace, so their logical
    // reuse unit must agree. Validate that invariant by tag instead of choosing
    // a group by registry order.
    std::optional<int> full_reuse_unit_tokens;
    for (const auto& tag : full_group_tags_) {
        const int group_reuse_unit =
            cpLogicalSeqSizeForGroup(cp_mapper, config_, tag, static_cast<int>(config_.seqSizePerBlockForGroup(tag)));
        RTP_LLM_CHECK_WITH_INFO(!full_reuse_unit_tokens.has_value() || *full_reuse_unit_tokens == group_reuse_unit,
                                "FULL cache groups must share one logical reuse unit: expected=%d tag=%s actual=%d",
                                full_reuse_unit_tokens.value_or(group_reuse_unit),
                                tag.c_str(),
                                group_reuse_unit);
        full_reuse_unit_tokens = group_reuse_unit;
    }
    const int reuse_unit_tokens = full_reuse_unit_tokens.value_or(seqSizePerBlock());

    const auto&                                       cache_keys         = kv_resource->cacheKeys(0);
    int64_t                                           match_cost_time_us = 0;
    const size_t                                      reserve_blocks     = reserveBlocksNum();
    int                                               reuse_blocks       = 0;
    std::unordered_map<std::string, BlockIndicesType> referenced_blocks;

    const auto& reuse_anchor_tags = !full_group_tags_.empty()   ? full_group_tags_ :
                                    !linear_group_tags_.empty() ? linear_group_tags_ :
                                                                  swa_group_tags_;
    const bool  prefix_reuse_enabled =
        !reuse_anchor_tags.empty()
        && std::all_of(reuse_anchor_tags.begin(), reuse_anchor_tags.end(), [this](const std::string& tag) {
               return !skipReuseCacheGroup(tag);
           });
    if (malloc_info.enable_device_cache && prefix_reuse_enabled) {
        // CP-sharded: subsample to last-rank canonical key namespace before matching.
        CacheKeysType cp_keys = cpCanonicalCacheKeys(cp_mapper, cache_keys);
        // Off mode drops the last key to skip the partial trailing block. Under
        // CP sharding canonicalCacheKeys already excludes the partial block
        // (last-rank stride lands inside completed full blocks only), so the
        // extra drop would discard a valid full-block key — costing the SWA
        // tail-loop its only matchable key (full_keys[cp_size-1 + (n-1)*cp_size]
        // is exactly what the non-sharded SWA group caches).
        const bool    cp_active = cp_mapper && cp_mapper->isSharded();
        CacheKeysType match_keys(cp_keys.begin(),
                                 cp_active ? cp_keys.end() : (cp_keys.empty() ? cp_keys.end() : cp_keys.end() - 1));
        auto          begin_us = currentTimeUs();
        reuse_blocks           = reuseCache(match_keys, *kv_resource, cp_mapper);
        match_cost_time_us     = currentTimeUs() - begin_us;

        for (const auto& [tag, block_ids] : kv_resource->groupBlocks()) {
            const auto&      blocks = block_ids->blocks();
            BlockIndicesType valid;
            valid.reserve(blocks.size());
            for (auto b : blocks) {
                if (!isNullBlockIdx(b)) {
                    valid.push_back(b);
                }
            }
            if (!valid.empty()) {
                referenceBlocksInGroup(tag, valid);
                referenced_blocks.emplace(tag, std::move(valid));
            }
        }
        kv_resource->cacheResource(0).setDeviceReuseBlockNum(reuse_blocks);
    }

    if (reserve_blocks > 0 && !hasAvailableBlocksForReserve(malloc_info, reserve_blocks)) {
        rollbackInitMalloc(*kv_resource, referenced_blocks, {}, {});
        return {false, 0};
    }

    std::unordered_map<std::string, size_t> original_sizes;
    for (const auto& [tag, block_ids] : kv_resource->groupBlocks()) {
        original_sizes.emplace(tag, block_ids->blocksNum());
    }
    std::unordered_map<std::string, std::vector<size_t>> backfilled_positions;
    for (const auto& [tag, unused_block_ids] : kv_resource->groupBlocks()) {
        auto&     block_ids_0   = kv_resource->mutableBlockIds(0, tag);
        const int group_seq_len = cpEffectiveSeqLenForGroup(cp_mapper, config_, tag, common_seq_len);
        if (!cacheGroupForTag(tag, "initMalloc")
                 ->malloc(block_ids_0, group_seq_len, malloc_info.reuse_cache, 0, &backfilled_positions[tag])) {
            rollbackInitMalloc(*kv_resource, referenced_blocks, original_sizes, backfilled_positions);
            return {false, 0};
        }
    }

    for (int b = 1; b < batch_size; ++b) {
        for (const auto& [tag, unused_block_ids] : kv_resource->groupBlocks()) {
            cacheGroupForTag(tag, "initMalloc reference")
                ->reference(kv_resource->mutableBlockIds(b, tag), kv_resource->blocks(0, tag));
        }
    }
    return {true, reuse_blocks * reuse_unit_tokens, match_cost_time_us};
}

MallocResult HybridPoolKVCacheAllocator::incrMalloc(const MallocInfo& malloc_info) {
    auto&       kv_resource  = malloc_info.batch_kv_cache_resource;
    const auto& cp_mapper    = cp_slot_mapper_;
    const int   batch_size   = kv_resource->batchSize();
    const int   raw_seq_len  = malloc_info.incrSeqLen();
    const int   reserve_step = malloc_info.complete_token_ids->getReserveStep();

    std::vector<std::unordered_map<std::string, size_t>>              original_sizes(static_cast<size_t>(batch_size));
    std::vector<std::unordered_map<std::string, std::vector<size_t>>> backfilled_positions(
        static_cast<size_t>(batch_size));
    for (int b = 0; b < batch_size; ++b) {
        for (const auto& [tag, block_ids] : kv_resource->groupBlocks(b)) {
            original_sizes[static_cast<size_t>(b)].emplace(tag, block_ids->blocksNum());
        }
    }

    bool        all_success  = true;
    int         failed_batch = -1;
    std::string failed_tag;
    for (int b = 0; b < batch_size; ++b) {
        for (const auto& [tag, unused_block_ids] : kv_resource->groupBlocks(b)) {
            auto&     block_ids        = kv_resource->mutableBlockIds(b, tag);
            const int group_seq_len    = cpEffectiveSeqLenForGroup(cp_mapper, config_, tag, raw_seq_len);
            auto&     filled_positions = backfilled_positions[static_cast<size_t>(b)][tag];
            if (!cacheGroupForTag(tag, "incrMalloc")
                     ->malloc(block_ids, group_seq_len, malloc_info.reuse_cache, reserve_step, &filled_positions)) {
                all_success  = false;
                failed_batch = b;
                failed_tag   = tag;
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
            for (const auto& [tag, unused_block_ids] : kv_resource->groupBlocks(b)) {
                cacheGroupForTag(tag, "incrMalloc remove skipped blocks")
                    ->removeSkippedBlocks(kv_resource->mutableBlockIds(b, tag), malloc_info.reuse_cache, reserve_step);
            }
        }
        return {true, 0};
    }

    for (int b = 0; b <= failed_batch && b < batch_size; ++b) {
        for (const auto& [tag, unused_block_ids] : kv_resource->groupBlocks(b)) {
            rollbackGroupMalloc(tag,
                                kv_resource->mutableBlockIds(b, tag),
                                original_sizes[static_cast<size_t>(b)].at(tag),
                                backfilled_positions[static_cast<size_t>(b)][tag]);
        }
    }
    RTP_LLM_LOG_WARNING("Hybrid incrMalloc failed at batch=%d tag=%s", failed_batch, failed_tag.c_str());
    return {false, 0};
}

int HybridPoolKVCacheAllocator::seqSizePerBlock() const {
    return static_cast<int>(config_.seq_size_per_block);
}

void HybridPoolKVCacheAllocator::rollbackGroupMalloc(std::string_view           tag,
                                                              BlockIds&                  block_ids,
                                                              size_t                     original_size,
                                                              const std::vector<size_t>& filled_positions) {
    const auto&      blocks = block_ids.blocks();
    BlockIndicesType blocks_to_free;
    blocks_to_free.reserve(filled_positions.size() + blocks.size() - std::min(original_size, blocks.size()));
    for (size_t pos : filled_positions) {
        RTP_LLM_CHECK_WITH_INFO(pos < original_size && pos < blocks.size(),
                                "invalid hybrid rollback backfill position=%zu original_size=%zu size=%zu",
                                pos,
                                original_size,
                                blocks.size());
        if (!isNullBlockIdx(blocks[pos])) {
            blocks_to_free.push_back(blocks[pos]);
        }
    }
    for (size_t pos = original_size; pos < blocks.size(); ++pos) {
        const auto block = blocks[pos];
        if (!isNullBlockIdx(block)) {
            blocks_to_free.push_back(block);
        }
    }
    if (!blocks_to_free.empty()) {
        freeBlocksInGroup(tag, blocks_to_free);
    }
    for (size_t pos : filled_positions) {
        block_ids.setAt(pos, NULL_BLOCK_IDX);
    }
    block_ids.resize(original_size);
}

void HybridPoolKVCacheAllocator::rollbackInitMalloc(
    BatchKVCacheResource&                                       kv_resource,
    const std::unordered_map<std::string, BlockIndicesType>&    referenced_blocks,
    const std::unordered_map<std::string, size_t>&              original_sizes,
    const std::unordered_map<std::string, std::vector<size_t>>& backfilled_positions) {
    static const std::vector<size_t> kNoBackfill;
    for (const auto& [tag, unused_block_ids] : kv_resource.groupBlocks()) {
        auto&      block_ids = kv_resource.mutableBlockIds(0, tag);
        const auto original  = original_sizes.find(tag);
        if (original != original_sizes.end()) {
            const auto backfilled = backfilled_positions.find(tag);
            rollbackGroupMalloc(tag,
                                block_ids,
                                original->second,
                                backfilled != backfilled_positions.end() ? backfilled->second : kNoBackfill);
        }
        const auto referenced = referenced_blocks.find(tag);
        if (referenced != referenced_blocks.end() && !referenced->second.empty()) {
            freeBlocksInGroup(tag, referenced->second);
        }
        block_ids.resize(0);
    }
    kv_resource.cacheResource(0).setDeviceReuseBlockNum(0);
}

int HybridPoolKVCacheAllocator::estimatePeakNeedBlocks(const KVCacheResource& kv_cache_resource,
                                                                int                    seq_len,
                                                                int                    remaining_tokens,
                                                                int                    reserve_step,
                                                                bool                   enable_reuse_cache) const {
    int need_blocks = 0;
    for (const auto& [tag, block_ids] : kv_cache_resource.groupBlocks()) {
        need_blocks += cacheGroupForTag(tag, "getNeedBlocks")
                           ->estimatePeakNeedBlocks(
                               seq_len, block_ids->blocks(), remaining_tokens, reserve_step, enable_reuse_cache);
    }
    return need_blocks;
}

int HybridPoolKVCacheAllocator::estimateInitialBatchPeakNeedBlocks(int  seq_len,
                                                                            int  common_seq_len,
                                                                            int  remaining_tokens,
                                                                            int  reserve_step,
                                                                            bool enable_reuse_cache,
                                                                            int  target_batch_size) const {
    int peak_blocks = 0;
    for (const auto& [tag, group] : single_type_managers_) {
        peak_blocks += group->estimateInitialBatchPeakNeedBlocks(
            seq_len, common_seq_len, remaining_tokens, reserve_step, enable_reuse_cache, target_batch_size);
    }
    return peak_blocks;
}

int HybridPoolKVCacheAllocator::singleBatchNeedBlocks(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                                               int                            seq_len,
                                                               int                            reserve_step) const {
    int need_blocks = 0;
    for (const auto& [tag, unused_block_ids] : batch_kv_cache_resource->groupBlocks()) {
        const int effective_seq_len = cpEffectiveSeqLenForGroup(cp_slot_mapper_, config_, tag, seq_len);
        const int cur_blocks        = batch_kv_cache_resource->blocksNum(0, tag);
        need_blocks += cacheGroupForTag(tag, "singleBatchNeedBlocks")
                           ->needBlocksNum(effective_seq_len, cur_blocks, reserve_step);
    }
    return need_blocks;
}

// Cache resource ownership and block updates.

void HybridPoolKVCacheAllocator::referenceBlocksInGroup(std::string_view        tag,
                                                                 const BlockIndicesType& blocks,
                                                                 bool                    is_connector) const {
    if (is_connector) {
        blockPoolForTag(tag, "referenceBlocksInGroup")->connectorReference(blocks);
    } else {
        blockPoolForTag(tag, "referenceBlocksInGroup")->requestReference(blocks);
    }
}

void HybridPoolKVCacheAllocator::freeBlocksInGroup(std::string_view        tag,
                                                            const BlockIndicesType& blocks,
                                                            bool                    is_connector) {
    if (is_connector) {
        blockPoolForTag(tag, "freeBlocksInGroup")->connectorFree(blocks);
    } else {
        blockPoolForTag(tag, "freeBlocksInGroup")->requestFree(blocks);
    }
}

void HybridPoolKVCacheAllocator::free(const FreeInfo& free_info) {
    auto& kv_cache_resource = free_info.batch_kv_cache_resource;
    if (kv_cache_resource->curBlocksNum() == 0) {
        return;
    }
    for (int batch_id = 0; batch_id < kv_cache_resource->batchSize(); ++batch_id) {
        for (const auto& [tag, block_ids] : kv_cache_resource->groupBlocks(batch_id)) {
            cacheGroupForTag(tag, "free")->free(block_ids->blocks());
        }
    }
    kv_cache_resource->clearBlocks();
}

void HybridPoolKVCacheAllocator::insertIntoCache(const InsertInfo& insert_info) {
    auto& kv_cache_resource = insert_info.batch_kv_cache_resource;
    RTP_LLM_CHECK(kv_cache_resource != nullptr);
    if (!shared_block_cache_) {
        return;
    }

    const auto& cp_mapper  = cp_slot_mapper_;
    const bool  cp_active  = cp_mapper && cp_mapper->isSharded();
    const int   batch_size = kv_cache_resource->batchSize();

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
                const size_t                                      i = pos - 1;
                std::vector<std::pair<std::string, BlockIdxType>> group_block_ids;
                for (const auto& [tag, block_ids] : kv_cache_resource->groupBlocks(batch_id)) {
                    if (skipReuseCacheGroup(tag)) {
                        continue;
                    }
                    const auto& blocks = block_ids->blocks();
                    if (i >= blocks.size()) {
                        continue;
                    }
                    if (!isNullBlockIdx(blocks[i])) {
                        group_block_ids.push_back({tag, blocks[i]});
                    }
                }
                if (!group_block_ids.empty()) {
                    const auto dependency = i < full_dependencies.size() ?
                                                full_dependencies[i] :
                                                BlockDependency{false, 0, static_cast<uint32_t>(i)};
                    shared_block_cache_->put(full_keys[i],
                                             group_block_ids,
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
        CacheKeysType         cp_keys = cpCanonicalCacheKeys(cp_mapper, full_keys);
        BlockDependenciesType cp_dependencies;
        cp_dependencies.reserve(cp_keys.size());
        for (size_t i = 0; i < cp_keys.size(); ++i) {
            BlockDependency dependency;
            dependency.ordinal = static_cast<uint32_t>(i);
            if (i > 0) {
                dependency.has_parent = true;
                dependency.parent_key = cp_keys[i - 1];
            }
            cp_dependencies.push_back(dependency);
        }
        auto token_ids = insert_info.complete_token_ids->completeTokenIdsVec(batch_id);
        if (token_ids.size() <= 1) {
            continue;
        }
        const size_t token_len = token_ids.size() - 1;

        for (const auto& [tag, block_ids] : kv_cache_resource->groupBlocks(batch_id)) {
            if (skipReuseCacheGroup(tag)) {
                continue;
            }
            const int            raw_group_seq = cacheGroupForTag(tag, "insertIntoCache")->seqSizePerBlock();
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
            const auto&  blocks          = block_ids->blocks();
            const size_t loop_end        = std::min(n, blocks.size());

            // Reverse iterate so prefix-base keys land at MRU end (matches non-CP path).
            for (size_t pos = loop_end; pos > 0; --pos) {
                const size_t i = pos - 1;
                if (isNullBlockIdx(blocks[i])) {
                    continue;
                }
                std::vector<std::pair<std::string, BlockIdxType>> group_block_ids{{tag, blocks[i]}};
                const auto                                        dependency =
                    i < dependencies.size() ? dependencies[i] : BlockDependency{false, 0, static_cast<uint32_t>(i)};
                shared_block_cache_->put(
                    src_keys[i], group_block_ids, insert_info.is_resident, namespace_id, dependency);
            }
        }
    }
}

std::shared_ptr<KVCacheResource> HybridPoolKVCacheAllocator::incrKVCacheRef(
    const KVCacheResource& kvcache_resource, const CacheKeysType& cache_keys, bool is_connector) {
    if (cache_keys.empty() || kvcache_resource.groupNums() <= 0) {
        return nullptr;
    }

    std::unordered_map<CacheKeyType, size_t> key_to_pos;
    const auto&                              resource_keys = kvcache_resource.cacheKeys();
    for (size_t i = 0; i < resource_keys.size(); ++i) {
        key_to_pos.emplace(resource_keys[i], i);
    }

    auto selected_resource_ptr = new KVCacheResource(kvcache_resource);
    auto deleter               = [self = shared_from_this(), is_connector](KVCacheResource* resource) {
        self->decrKVCacheRef(*resource, is_connector);
        delete resource;
    };
    std::shared_ptr<KVCacheResource> selected_resource(selected_resource_ptr, deleter);
    selected_resource->initGroups(config_.topologyPtr());

    CacheKeysType                                     selected_keys;
    BlockDependenciesType                             selected_dependencies;
    std::unordered_map<std::string, BlockIndicesType> selected_blocks;
    const auto&                                       source_dependencies = kvcache_resource.blockDependencies();
    RTP_LLM_CHECK_WITH_INFO(source_dependencies.size() == resource_keys.size(),
                            "cache timeline size mismatch before reference selection: keys=%zu dependencies=%zu",
                            resource_keys.size(),
                            source_dependencies.size());

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
        for (const auto& [tag, block_ids] : kvcache_resource.groupBlocks()) {
            const auto& src_blocks = block_ids->blocks();
            const auto  block      = pos < src_blocks.size() ? src_blocks[pos] : NULL_BLOCK_IDX;
            blocks_for_key.emplace(tag, block);
            any_valid_block = any_valid_block || (!isNullBlockIdx(block) && block > 0);
        }
        const bool preserve_connector_tail = is_connector && !kvcache_resource.lastBlockAligned()
                                             && pos + 1 == resource_keys.size() && !selected_keys.empty();
        if (!any_valid_block && !preserve_connector_tail) {
            continue;
        }
        selected_keys.push_back(key);
        selected_dependencies.push_back(source_dependencies[pos]);
        for (const auto& [tag, block] : blocks_for_key) {
            selected_blocks[tag].push_back(block);
        }
    }

    if (selected_keys.empty()) {
        return nullptr;
    }

    selected_resource->setCacheKeysAndBlockDependencies(std::move(selected_keys), std::move(selected_dependencies));
    for (auto& [tag, blocks] : selected_blocks) {
        BlockIndicesType valid;
        for (auto b : blocks) {
            if (!isNullBlockIdx(b) && b > 0) {
                valid.push_back(b);
            }
        }
        if (!valid.empty()) {
            referenceBlocksInGroup(tag, valid, is_connector);
        }
        selected_resource->mutableBlockIds(tag).assign(std::move(blocks));
    }
    return selected_resource;
}

void HybridPoolKVCacheAllocator::decrKVCacheRef(const KVCacheResource& kvcache_resource, bool is_connector) {
    for (const auto& [tag, block_ids] : kvcache_resource.groupBlocks()) {
        BlockIndicesType valid;
        for (auto b : block_ids->blocks()) {
            if (!isNullBlockIdx(b) && b > 0) {
                valid.push_back(b);
            }
        }
        if (!valid.empty()) {
            freeBlocksInGroup(tag, valid, is_connector);
        }
    }
}

bool HybridPoolKVCacheAllocator::updateKVBlock(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                                        const std::vector<int>&        block_src_batch,
                                                        bool                           copy_last_block,
                                                        std::vector<GroupBlockIdPair>& block_update_mapping) {
    block_update_mapping.clear();
    if (block_src_batch.empty()) {
        return true;
    }

    const int        old_batch_size = batch_kv_cache_resource->batchSize();
    const int        new_batch_size = static_cast<int>(block_src_batch.size());
    std::vector<int> batch_fork_count(old_batch_size, 0);
    for (const int old_batch_idx : block_src_batch) {
        RTP_LLM_CHECK_WITH_INFO(old_batch_idx >= 0 && old_batch_idx < old_batch_size,
                                "try to reuse an old batch %d that out of range %d",
                                old_batch_idx,
                                old_batch_size);
        ++batch_fork_count[old_batch_idx];
    }

    std::unordered_map<std::string, int> new_blocks_num;
    for (int old_batch_idx = 0; old_batch_idx < old_batch_size; ++old_batch_idx) {
        const int fork_count = batch_fork_count[old_batch_idx];
        if (fork_count > 1 && copy_last_block) {
            for (const auto& [tag, block_ids] : batch_kv_cache_resource->groupBlocks(old_batch_idx)) {
                if (!block_ids->blocks().empty()) {
                    new_blocks_num[tag] += fork_count - 1;
                }
            }
        }
    }

    // Transfer request ownership from dropped batches before allocating new
    // blocks. This keeps the operation transactional while allowing net-feasible
    // drop-and-fork updates to succeed when the pool is otherwise full.
    std::unordered_map<std::string, BlockIndicesType>                      replacement_blocks;
    std::unordered_map<std::string, BlockIndicesType>                      allocated_replacements;
    std::unordered_map<std::string, std::unordered_map<BlockIdxType, int>> transferred_ref_counts;
    for (const auto& [tag, unused_block_ids] : batch_kv_cache_resource->groupBlocks()) {
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
        const int need         = new_blocks_num[tag];
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
        for (auto& [tag, blocks] : allocated_replacements) {
            if (!blocks.empty()) {
                cacheGroupForTag(tag, "updateKVBlock release source")->free(blocks);
                blocks.clear();
            }
        }
    };
    for (const auto& [tag, unused_block_ids] : batch_kv_cache_resource->groupBlocks()) {
        const int need_blocks = new_blocks_num[tag];
        auto&     reserved    = replacement_blocks[tag];
        reserved.reserve(static_cast<size_t>(need_blocks));
        for (int i = static_cast<int>(reserved.size()); i < need_blocks; ++i) {
            BlockIds    one_block;
            const auto& group  = cacheGroupForTag(tag, "updateKVBlock allocate destination");
            const bool  ok     = group->malloc(one_block, group->seqSizePerBlock());
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
        for (const auto& [tag, block_ids] : batch_kv_cache_resource->groupBlocks(old_batch_idx)) {
            BlockIndicesType to_free;
            auto&            transferred = transferred_ref_counts[tag];
            for (const auto block : block_ids->blocks()) {
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
                cacheGroupForTag(tag, "updateKVBlock rollback")->free(to_free);
            }
        }
    }

    std::vector<KVCacheResource> old_resources;
    batch_kv_cache_resource->resetAndReturnOldResources(new_batch_size, old_resources);
    batch_kv_cache_resource->initGroups(config_.topologyPtr());
    std::unordered_map<std::string, size_t> next_replacement;

    for (int new_batch_idx = 0; new_batch_idx < new_batch_size; ++new_batch_idx) {
        const int old_batch_idx = block_src_batch[new_batch_idx];
        auto&     fork_count    = batch_fork_count[old_batch_idx];
        RTP_LLM_CHECK_WITH_INFO(fork_count > 0, "old batch %d has been forked too many times", old_batch_idx);

        if (fork_count == 1) {
            batch_kv_cache_resource->moveBatchResource(new_batch_idx, std::move(old_resources[old_batch_idx]));
        } else {
            batch_kv_cache_resource->setBatchCacheKeys(new_batch_idx, old_resources[old_batch_idx].cacheKeys());
            for (const auto& [tag, source_block_ids] : old_resources[old_batch_idx].groupBlocks()) {
                auto& block_ids = batch_kv_cache_resource->mutableBlockIds(new_batch_idx, tag);
                cacheGroupForTag(tag, "updateKVBlock reference source")
                    ->reference(block_ids, source_block_ids->blocks());

                if (copy_last_block && !block_ids.blocks().empty()) {
                    const int  old_block       = block_ids.popBack();
                    const bool old_block_valid = !isNullBlockIdx(old_block) && old_block > 0;
                    if (old_block_valid) {
                        cacheGroupForTag(tag, "updateKVBlock replace tail")->free({old_block});
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
    for (const auto& [tag, blocks] : replacement_blocks) {
        RTP_LLM_CHECK_WITH_INFO(next_replacement[tag] == blocks.size(),
                                "unused replacement blocks after hybrid kv cache update, tag=%s used=%zu reserved=%zu",
                                tag.c_str(),
                                next_replacement[tag],
                                blocks.size());
    }
    return true;
}

// Cache layout, address conversion, and block copy.

GroupedCacheLayerLayout HybridPoolKVCacheAllocator::allLayerCacheBase() const {
    const auto topology = config_.topologyPtr();
    RTP_LLM_CHECK_WITH_INFO(single_type_managers_.size() == topology->groups().size(),
                            "cache group count=%zu topology count=%zu",
                            single_type_managers_.size(),
                            topology->groups().size());

    GroupedCacheLayerLayout::GroupLayouts groups;
    for (const auto& [tag, group_topology] : single_type_managers_) {
        std::vector<BlockBufferPtrInfo> layers(topology->layers().size());
        const auto                      layer_tensors = group_topology->allLayerCacheBase();
        const auto                      scale_tensors = group_topology->allLayerScaleCacheBase();
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
        groups.emplace(tag, CacheLayerLayout(std::move(layers)));
    }
    return GroupedCacheLayerLayout(topology, std::move(groups));
}

BlockAddrInfo
HybridPoolKVCacheAllocator::convertIndexToAddr(int layer_id, const std::string& tag, int block_id) const {
    validatePoolGroupForLayer(config_, layer_id, tag);
    return cacheGroupForTag(tag, "convertIndexToAddr")->convertIndexToAddr(layer_id, block_id);
}

std::vector<BlockInfo>
HybridPoolKVCacheAllocator::convertIndexToBuffer(int layer_id, const std::string& tag, int block_id) const {
    validatePoolGroupForLayer(config_, layer_id, tag);
    return cacheGroupForTag(tag, "convertIndexToBuffer")->convertIndexToBuffer(layer_id, block_id);
}

std::vector<BlockInfo> HybridPoolKVCacheAllocator::convertIndexToBuffer(
    int layer_id, const std::string& tag, int block_id, int partition_count, int partition_id) const {
    validatePoolGroupForLayer(config_, layer_id, tag);
    return cacheGroupForTag(tag, "convertIndexToBuffer partitioned")
        ->convertIndexToBuffer(layer_id, block_id, partition_count, partition_id);
}

void HybridPoolKVCacheAllocator::blockBatchCopy(const std::vector<GroupBlockIdPair>& copy_mapping) {
    if (copy_mapping.empty()) {
        return;
    }

    size_t copy_nums[BatchCopyParams::TYPE_SIZE] = {};
    for (const auto& mapping : copy_mapping) {
        const auto&  pool              = blockPoolForTag(mapping.tag, "blockBatchCopy reserve");
        const auto   copy_type         = BatchCopyParams::get_copy_type(pool->where(), pool->where());
        const auto&  group             = config_.topology().group(mapping.tag);
        const size_t buffers_per_layer = group.kv_scale_stride_bytes > 0 ? 2 : 1;
        copy_nums[copy_type] += config_.layerIdsForGroup(mapping.tag).size() * buffers_per_layer;
    }

    BatchCopyParams copy_params;
    for (size_t i = 0; i < BatchCopyParams::TYPE_SIZE; ++i) {
        copy_params.reserve(static_cast<BatchCopyParams::CopyType>(i), copy_nums[i]);
    }

    for (const auto& mapping : copy_mapping) {
        const auto&  pool                = blockPoolForTag(mapping.tag, "blockBatchCopy");
        const auto&  group               = config_.topology().group(mapping.tag);
        const size_t kv_block_size_bytes = group.kv_block_stride_bytes;
        const size_t scale_block_bytes   = group.kv_scale_stride_bytes;
        const auto   copy_type           = BatchCopyParams::get_copy_type(pool->where(), pool->where());

        for (int layer_id : config_.layerIdsForGroup(mapping.tag)) {
            auto src_addr_info = cacheGroupForTag(mapping.tag, "blockBatchCopy source")
                                     ->convertIndexToAddr(layer_id, mapping.src);
            auto dst_addr_info = cacheGroupForTag(mapping.tag, "blockBatchCopy destination")
                                     ->convertIndexToAddr(layer_id, mapping.dst);

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

// Capacity, metrics, eviction, and reservation.

std::vector<std::string> HybridPoolKVCacheAllocator::independentEvictionTags() const {
    std::vector<std::string> tags;
    for (const auto& [tag, group] : single_type_managers_) {
        if (group->evictPolicy() == CacheEvictPolicy::INDEPENDENT) {
            tags.push_back(tag);
        }
    }
    return tags;
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
            tags.AddTag("evict_policy",
                        evict_result.evicted_independent_tag.count(cache_key) ? "independent" : "chain");
            tags.AddTag("backing", "device");
            metrics_reporter_->report<RtpLLMCacheEvictionMetrics, RtpLLMCacheEvictionMetricsCollector>(&tags,
                                                                                                       &collector);
        }
    }

    auto batch_resource = std::make_shared<BatchKVCacheResource>();
    batch_resource->resetBatchSize(1);
    batch_resource->initGroups(config_.topologyPtr());
    batch_resource->setLastBlockAligned(true);

    for (const auto& group : config_.topology().groups()) {
        batch_resource->mutableBlockIds(0, group.tag).resize(evict_result.evicted_keys.size(), NULL_BLOCK_IDX);
    }

    CacheKeysType         evicted_keys;
    BlockDependenciesType evicted_dependencies;
    evicted_keys.reserve(evict_result.evicted_keys.size());
    evicted_dependencies.reserve(evict_result.evicted_keys.size());
    for (size_t evicted_idx = 0; evicted_idx < evict_result.evicted_keys.size(); ++evicted_idx) {
        const auto  cache_key       = evict_result.evicted_keys[evicted_idx];
        const auto& group_block_ids = evict_result.evicted_group_block_ids.at(cache_key);
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
        for (const auto& [tag, block_id] : group_block_ids) {
            if (!isNullBlockIdx(block_id)) {
                batch_resource->mutableBlockIds(0, tag).setAt(evicted_idx, block_id);
            }
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

void HybridPoolKVCacheAllocator::blockCacheFree(const BatchKVCacheResourcePtr& batch_kv_cache_resource) {
    if (!batch_kv_cache_resource) {
        return;
    }
    for (int batch_id = 0; batch_id < batch_kv_cache_resource->batchSize(); ++batch_id) {
        for (const auto& [tag, block_ids] : batch_kv_cache_resource->groupBlocks(batch_id)) {
            BlockIndicesType                 blocks_to_free;
            std::unordered_set<BlockIdxType> seen_blocks;
            for (auto block_idx : block_ids->blocks()) {
                if (isNullBlockIdx(block_idx) || !seen_blocks.insert(block_idx).second) {
                    continue;
                }
                blocks_to_free.push_back(block_idx);
            }
            if (!blocks_to_free.empty()) {
                blockPoolForTag(tag, "blockCacheFree")->blockCacheFree(blocks_to_free);
            }
        }
    }
}

size_t HybridPoolKVCacheAllocator::maxSequenceLengthForGroups(bool full_groups_only) const {
    if (group_block_pools_.empty()) {
        return 0;
    }

    auto calculate = [&](bool only_full_groups) {
        size_t min_tokens = std::numeric_limits<size_t>::max();
        bool   saw_group  = false;
        for (const auto& [tag, pool] : group_block_pools_) {
            if (only_full_groups && config_.typeForGroup(tag) != CacheGroupType::FULL) {
                continue;
            }
            if (!pool) {
                continue;
            }
            const size_t seq_size = logicalSeqSizePerBlockForCapacity(tag);
            min_tokens            = std::min(min_tokens, pool->totalBlocksNum() * seq_size);
            saw_group             = true;
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

size_t HybridPoolKVCacheAllocator::totalTokensNum() const {
    return minPoolTokens(/*use_available_blocks=*/false);
}

size_t HybridPoolKVCacheAllocator::availableTokensNum() const {
    return minPoolTokens(/*use_available_blocks=*/true);
}

size_t HybridPoolKVCacheAllocator::maxSequenceLength() const {
    return maxSequenceLengthForGroups(/*full_groups_only=*/true);
}

size_t HybridPoolKVCacheAllocator::minPoolTokens(bool use_available_blocks) const {
    if (group_block_pools_.empty()) {
        return 0;
    }
    size_t min_tokens = std::numeric_limits<size_t>::max();
    bool   has_pool   = false;
    for (const auto& [tag, pool] : group_block_pools_) {
        if (!pool) {
            continue;
        }
        const size_t seq_size = logicalSeqSizePerBlockForCapacity(tag);
        const size_t blocks   = use_available_blocks ? pool->availableBlocksNum() : pool->totalBlocksNum();
        min_tokens            = std::min(min_tokens, blocks * seq_size);
        has_pool              = true;
    }
    return has_pool ? min_tokens : 0;
}

std::vector<KVCachePoolMetricsSnapshot> HybridPoolKVCacheAllocator::poolMetricsSnapshots() const {
    std::vector<KVCachePoolMetricsSnapshot> snapshots;
    snapshots.reserve(group_block_pools_.size());
    const size_t reserve_blocks                    = reserveBlocksNum();
    const size_t total_reservable_available_blocks = totalReservableAvailableBlocks();
    for (const auto& [tag, pool] : group_block_pools_) {
        if (!pool) {
            continue;
        }
        KVCachePoolMetricsSnapshot snapshot;
        snapshot.pool_name            = pool->poolName();
        snapshot.total_blocks         = pool->totalBlocksNum();
        snapshot.available_blocks     = pool->availableBlocksNum();
        snapshot.free_blocks          = pool->freeBlocksNum();
        snapshot.request_ref_blocks   = pool->requestRefBlocksNum();
        snapshot.connector_ref_blocks = pool->connectorRefBlocksNum();
        snapshot.reserve_blocks       = reserveBlocksForPool(tag, reserve_blocks, total_reservable_available_blocks);
        snapshot.used_ratio           = (snapshot.total_blocks == 0) ?
                                            0.0f :
                                            static_cast<float>(100.0 * (snapshot.total_blocks - snapshot.available_blocks)
                                                     / static_cast<double>(snapshot.total_blocks));
        snapshots.push_back(snapshot);
    }
    return snapshots;
}

size_t HybridPoolKVCacheAllocator::totalReservableAvailableBlocks() const {
    size_t total = 0;
    for (const auto& [tag, pool] : group_block_pools_) {
        if (!pool || config_.usesExplicitIndependentBlocks(tag)) {
            continue;
        }
        total += pool->availableBlocksNum();
    }
    return total;
}

size_t HybridPoolKVCacheAllocator::reservableAvailableBlocksNum() const {
    return totalReservableAvailableBlocks();
}

size_t HybridPoolKVCacheAllocator::reserveBlocksForPool(std::string_view tag,
                                                                 size_t           reserve_blocks,
                                                                 size_t total_reservable_available_blocks) const {
    const auto pool = group_block_pools_.find(std::string(tag));
    if (pool == group_block_pools_.end() || !pool->second || config_.usesExplicitIndependentBlocks(tag)
        || total_reservable_available_blocks == 0) {
        return 0;
    }
    return reserve_blocks * pool->second->availableBlocksNum() / total_reservable_available_blocks;
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

    for (const auto& [tag, group] : single_type_managers_) {
        const int  group_common_seq       = cpEffectiveSeqLenForReserve(cp_mapper, config_, tag, raw_common_seq_len);
        const int  group_seq_len          = cpEffectiveSeqLenForReserve(cp_mapper, config_, tag, raw_seq_len);
        const int  group_reuse_blocks_len = reuse_enabled ? malloc_info.batch_kv_cache_resource->blocksNum(0, tag) : 0;
        const auto need =
            group->getNeedBlocks(group_common_seq, group_seq_len, reserve_step, group_reuse_blocks_len, reuse_enabled);
        const int need_blocks = need.common_blocks + batch_size * need.extra_blocks;
        if (need_blocks <= 0) {
            continue;
        }
        const auto&  pool             = blockPoolForTag(tag, __func__);
        const size_t available_blocks = pool->availableBlocksNum();
        const size_t total_blocks     = pool->totalBlocksNum();
        const size_t group_reserve_blocks =
            reserveBlocksForPool(tag, reserve_blocks, total_reservable_available_blocks);
        if (available_blocks < static_cast<size_t>(need_blocks) + group_reserve_blocks) {
            if (malloc_info.verbose) {
                RTP_LLM_LOG_INFO("HybridPool initMalloc rejected by reserve blocks: request_id=%ld pool_name=%s "
                                 "tag=%s need_blocks=%d total_blocks=%zu available_blocks=%zu "
                                 "reserve_blocks=%zu group_reserve_blocks=%zu",
                                 malloc_info.request_id,
                                 pool->poolName().c_str(),
                                 tag.c_str(),
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
