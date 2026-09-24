#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/core/OpData.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/cache/CoordinatorCacheManager.h"
#include "rtp_llm/cpp/cache/SingleTypeCacheManager.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadAsyncContext.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include <unordered_map>
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include <iomanip>
#include <numeric>
#include <sstream>
#include <string>
#include "rtp_llm/cpp/cache/DeviceBlockPoolConfigHelper.h"

namespace rtp_llm {
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

inline int cpEffectiveSeqLenForReserve(const std::shared_ptr<CPSlotMapper>& mapper,
                                       const CacheConfig&                   config,
                                       std::string_view                     tag,
                                       int                                  seq_len) {
    return (mapper && mapper->isSharded()) ? mapper->effectiveSeqLenForAlloc(config, tag, seq_len) : seq_len;
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

bool CoordinatorCacheManager::init() {
    RTP_LLM_CHECK_WITH_INFO(doInit(), "init failed");

    const int64_t reserve_ratio = reserve_block_ratio_;
    if (reserve_ratio > 0) {
        const size_t reservable_blocks = reservableFreeBlocksNum();
        const size_t reserve_blocks = static_cast<size_t>(reserve_ratio) * reservable_blocks / static_cast<size_t>(100);
        reserve_block_num_          = reserve_blocks;
        RTP_LLM_LOG_INFO(
            "CoordinatorCacheManager set reserve blocks: ratio=%ld%% reserve_blocks=%zu reservable_free_blocks=%zu",
            reserve_ratio,
            reserve_blocks,
            reservable_blocks);
    } else {
        reserve_block_num_ = 0;
    }

    return true;
}

MallocResult CoordinatorCacheManager::initMalloc(const MallocInfo& malloc_info) {
    auto finalize_init_failure = [this, &malloc_info](MallocResult result) {
        // Cache matching can satisfy part of a request from lower tiers, so
        // classify capacity only after materialization fails. Use the
        // failure-time snapshot before rollback; freeing first can make a
        // retryable shortage look like an internal error.
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
    if (malloc_info.batch_kv_cache_resource != nullptr) {
        const CacheKeysType& cache_keys        = malloc_info.batch_kv_cache_resource->cacheKeys(0);
        init_result.block_aligned_input_length = static_cast<int64_t>(cache_keys.size()) * config_.seq_size_per_block;
    }
    if (!init_result.success) {
        return finalize_init_failure(init_result);
    }

    std::shared_ptr<LoadAsyncContext> load_context =
        std::dynamic_pointer_cast<LoadAsyncContext>(init_result.async_context);
    if (!load_context || !load_context->needBackendMatch()) {
        std::shared_ptr<AsyncContext> pending_async_context = std::move(init_result.async_context);
        MallocResult                  incr_result           = incrMalloc(malloc_info);
        if (!incr_result.success) {
            if (load_context != nullptr) {
                load_context->abortPending();
            }
            pending_async_context.reset();
            load_context.reset();
            incr_result.match_cost_time_us         = init_result.match_cost_time_us;
            incr_result.match_end_time_us          = init_result.match_end_time_us;
            incr_result.block_aligned_input_length = init_result.block_aligned_input_length;
            incr_result.load_attempted             = init_result.load_attempted;
            return finalize_init_failure(std::move(incr_result));
        }
        if (pending_async_context != nullptr) {
            load_context = std::dynamic_pointer_cast<LoadAsyncContext>(pending_async_context);
            if (load_context == nullptr || !load_context->commit()) {
                load_context.reset();
                pending_async_context.reset();
                init_result.success   = false;
                init_result.reuse_len = 0;

                init_result.async_context = nullptr;
                return finalize_init_failure(std::move(init_result));
            }
            init_result.async_context = std::move(pending_async_context);
        }
    }

    return init_result;
}

size_t CoordinatorCacheManager::heldRequestBlocks(const MallocInfo& malloc_info, std::string_view tag) {
    const auto& resource = malloc_info.batch_kv_cache_resource;
    if (!resource) {
        return 0;
    }
    std::unordered_set<BlockIdxType> held_blocks;
    for (int batch = 0; batch < resource->batchSize(); ++batch) {
        for (const BlockIdxType block : resource->blocks(batch, tag)) {
            if (block > 0 && !isNullBlockIdx(block)) {
                held_blocks.insert(block);
            }
        }
    }
    return held_blocks.size();
}

CoordinatorCacheManager::InitBlockDemand
CoordinatorCacheManager::initBlockDemand(const MallocInfo& malloc_info, size_t planned_blocks, std::string_view tag) {
    const size_t held_blocks = heldRequestBlocks(malloc_info, tag);
    if (malloc_info.reuse_cache) {
        return {held_blocks, planned_blocks};
    }
    // No-reuse planners already describe the full request. Account only for
    // the portion not represented by the current partial allocation.
    return held_blocks > planned_blocks ? InitBlockDemand{held_blocks - planned_blocks, 0} :
                                          InitBlockDemand{0, planned_blocks - held_blocks};
}

MallocResult CoordinatorCacheManager::malloc(const MallocInfo& malloc_info) {
    // Keep capacity classification and the physical allocations it authorizes
    // in one transaction.  Decode-side P/D admission invokes this entry point
    // from concurrent RPC threads, while running streams can allocate from the
    // engine thread at the same time.
    std::unique_lock<std::mutex> lock(malloc_mutex_);

    if (!malloc_info.batch_kv_cache_resource) {
        RTP_LLM_LOG_ERROR("BatchKVCacheResource is null");
        return {false, 0};
    }

    if (!malloc_info.complete_token_ids) {
        RTP_LLM_LOG_ERROR("CompleteTokenIds is null");
        return {false, 0};
    }

    if (malloc_info.batch_kv_cache_resource->curBlocksNum() == 0) {
        auto result = initMalloc(malloc_info);
        lock.unlock();
        auto context = std::dynamic_pointer_cast<LoadAsyncContext>(result.async_context);
        if (result.success && context && context->needBackendMatch()) {
            context->startBackendMatch();
        }
        return result;
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

void CoordinatorCacheManager::attachBlockTreeCache(BlockTreeCachePtr block_tree_cache) {
    RTP_LLM_CHECK_WITH_INFO(block_tree_cache != nullptr, "cannot attach a null BlockTreeCache");
    RTP_LLM_CHECK_WITH_INFO(block_tree_cache_ == nullptr, "BlockTreeCache has already been attached");

    block_tree_cache_ = std::move(block_tree_cache);
    for (const auto& group : cacheGroups()) {
        const std::string tag = group->tag();
        group->setEvictCallback([cache = block_tree_cache_, tag](size_t need_blocks) {
            const int reclaimed = cache->evictForGroup(tag, need_blocks);
            return reclaimed > 0 ? static_cast<size_t>(reclaimed) : 0;
        });
    }
}

bool CoordinatorCacheManager::abortPendingLoad(const std::shared_ptr<AsyncContext>& context) {
    return block_tree_cache_ != nullptr && block_tree_cache_->abortPendingLoad(context);
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

void CoordinatorCacheManager::blockCopy(int src_block_index, int dest_block_index) {
    BlockIdPair copy_mapping{src_block_index, dest_block_index};
    blockBatchCopy(&copy_mapping, &copy_mapping + 1);
}

void CoordinatorCacheManager::blockBatchCopy(const std::vector<BlockIdPair>& copy_mapping) {
    blockBatchCopy(copy_mapping.data(), copy_mapping.data() + copy_mapping.size());
}

void CoordinatorCacheManager::blockBatchCopy(const torch::Tensor& copy_mapping) {
    RTP_LLM_CHECK_WITH_INFO(copy_mapping.device().is_cpu() && copy_mapping.scalar_type() == torch::kInt32
                                && copy_mapping.is_contiguous() && copy_mapping.dim() == 2,
                            "cache block copy mapping must be a contiguous CPU int32 matrix");
    RTP_LLM_CHECK_WITH_INFO(
        copy_mapping.size(1) == 2, "cache block copy mapping must have 2 columns, got %ld", copy_mapping.size(1));
    const auto* begin_ptr = reinterpret_cast<const BlockIdPair*>(copy_mapping.data_ptr());
    blockBatchCopy(begin_ptr, begin_ptr + copy_mapping.size(0));
}

size_t CoordinatorCacheManager::logicalSeqSizePerBlockForCapacity(const std::string& tag) const {
    if (cp_slot_mapper_ && cp_slot_mapper_->isSharded()) {
        return cp_slot_mapper_->logicalSeqSizePerBlock(config_, tag);
    }
    return config_.topology().group(tag).seqSizePerBlock();
}

int CoordinatorCacheManager::deviceCacheMetricTokensPerBlock() const {
    if (cp_slot_mapper_ && cp_slot_mapper_->isSharded()) {
        return cp_slot_mapper_->virtualBlockSize();
    }
    return seqSizePerBlock();
}

std::vector<KVCachePoolMetricsSnapshot> CoordinatorCacheManager::poolMetricsSnapshots() const {
    const std::vector<SingleTypeCacheManagerPtr> groups = cacheGroups();
    std::unordered_set<const IBlockPool*>        reported_pools;
    std::vector<KVCachePoolMetricsSnapshot>      snapshots;
    snapshots.reserve(groups.size());
    for (const SingleTypeCacheManagerPtr& group : groups) {
        if (group == nullptr || group->blockPool() == nullptr) {
            continue;
        }
        const DeviceBlockPoolPtr                                               pool = group->blockPool();
        const std::pair<std::unordered_set<const IBlockPool*>::iterator, bool> insert_result =
            reported_pools.insert(pool.get());
        if (!insert_result.second) {
            continue;
        }

        const size_t               pool_index = groupIdForTag(group->tag());
        KVCachePoolMetricsSnapshot snapshot;
        snapshot.pool_index                 = pool_index;
        snapshot.pool_name                  = pool->poolName();
        snapshot.block_size_bytes           = pool->blockSizeBytes();
        snapshot.total_blocks               = pool->totalBlocksNum();
        snapshot.free_blocks                = pool->freeBlocksNum();
        snapshot.used_blocks                = snapshot.total_blocks - snapshot.free_blocks;
        snapshot.active_blocks              = pool->activeBlocksNum();
        snapshot.available_blocks           = pool->availableBlocksNum();
        snapshot.reserve_blocks             = reserveBlocksForPoolMetrics(pool_index);
        snapshot.request_ref_blocks         = pool->referencedBlocksNum();
        snapshot.block_cache_ref_blocks     = pool->referencedBlocksNum(BlockTreeRefType::CACHE);
        snapshot.load_ref_blocks            = pool->referencedBlocksNum(BlockTreeRefType::LOAD);
        snapshot.eviction_target_ref_blocks = pool->referencedBlocksNum(BlockTreeRefType::EVICTION);
        snapshot.store_ref_blocks           = pool->referencedBlocksNum(BlockTreeRefType::STORE);
        snapshot.used_ratio =
            snapshot.total_blocks == 0 ?
                0.0f :
                static_cast<float>(100.0 * snapshot.used_blocks / static_cast<double>(snapshot.total_blocks));
        snapshots.push_back(std::move(snapshot));
    }
    return snapshots;
}

bool CoordinatorCacheManager::cpCompactSwaGroup(const std::string&                   tag,
                                                const std::shared_ptr<CPSlotMapper>& mapper) const {
    return mapper && mapper->isSharded() && mapper->compactLastRankGroup(config_, tag);
}

size_t CoordinatorCacheManager::loadTargetPosition(size_t                               path_index,
                                                   const std::string&                   tag,
                                                   const std::shared_ptr<CPSlotMapper>& mapper,
                                                   int                                  cp_scale) const {
    const CacheGroupType type = config_.topology().group(tag).policy.group_type;
    return type == CacheGroupType::LINEAR || (type == CacheGroupType::SWA && !cpCompactSwaGroup(tag, mapper)) ?
               (path_index + 1) * static_cast<size_t>(cp_scale) - 1 :
               path_index;
}

std::shared_ptr<LoadAsyncContext>
CoordinatorCacheManager::prepareKVCache(const CacheKeysType&                 cache_keys,
                                        BatchKVCacheResource&                kv_resource,
                                        const std::shared_ptr<CPSlotMapper>& cp_mapper,
                                        PreparedKVCache&                     prepared) {
    if (!block_tree_cache_ || cache_keys.empty()) {
        return nullptr;
    }
    const int                         cp_scale     = (cp_mapper && cp_mapper->isSharded()) ? cp_mapper->cpSize() : 1;
    BlockTreeMatchResult              match_result = block_tree_cache_->match(cache_keys);
    std::shared_ptr<LoadAsyncContext> load_context = std::move(match_result.async_context);
    prepared.matched_device_blocks                 = match_result.matched_device_blocks;
    prepared.total_logical_blocks =
        load_context ? load_context->localMatchedBlocks() : match_result.matched_device_blocks;
    const auto& group_sets = block_tree_cache_->groupSets();
    if (prepared.total_logical_blocks > 0) {
        for (const auto& group_set : group_sets) {
            for (const auto& tag : group_set->groupTags()) {
                const size_t reuse_size =
                    loadTargetPosition(prepared.total_logical_blocks - 1, tag, cp_mapper, cp_scale) + 1;
                kv_resource.mutableBlockIds(0, tag).assign(BlockIndicesType(reuse_size, NULL_BLOCK_IDX));
            }
        }
    }

    const auto set_group_set_blocks = [&](size_t group_set_id, size_t path_index, const BlockIndicesType& blocks) {
        const auto& group_tags = group_sets[group_set_id]->groupTags();
        for (size_t member_group_id = 0; member_group_id < group_tags.size(); ++member_group_id) {
            const size_t group_id = groupIdForTag(group_tags[member_group_id]);
            const size_t target_position =
                loadTargetPosition(path_index, group_tags[member_group_id], cp_mapper, cp_scale);
            kv_resource.mutableBlockIds(0, group_tags[member_group_id]).setAt(target_position, blocks[member_group_id]);
            prepared.referenced_blocks[group_id].push_back(blocks[member_group_id]);
        }
    };

    for (const auto& resource : match_result.matched_device_resources) {
        const size_t canonical_start = match_result.matched_device_blocks - resource.node_blocks.size();
        for (size_t i = 0; i < resource.node_blocks.size(); ++i) {
            set_group_set_blocks(resource.group_set_id, canonical_start + i, resource.node_blocks[i].second);
        }
    }

    if (load_context != nullptr) {
        for (size_t desc_index = 0; desc_index < load_context->loadDescs().size(); ++desc_index) {
            const auto& desc = load_context->loadDescs()[desc_index];
            if (desc.source_tier == Tier::DEVICE) {
                set_group_set_blocks(desc.group_set_id, desc.path_index, desc.source_blocks);
            } else if (load_context->joinedLoads()[desc_index]) {
                set_group_set_blocks(desc.group_set_id, desc.path_index, desc.target_blocks);
            }
        }
    }

    for (int group_id = 0; group_id < config_.groupNums(); ++group_id) {
        const auto& tag = config_.groupTags()[static_cast<size_t>(group_id)];
        prepared.original_sizes.push_back(kv_resource.blocksNum(0, tag));
    }
    return load_context;
}

MallocResult CoordinatorCacheManager::initMallocForCommonLen(const MallocInfo& malloc_info) {
    auto&       kv_resource       = malloc_info.batch_kv_cache_resource;
    const int   seq_len           = malloc_info.complete_token_ids->seqLength();
    const auto& cp_mapper         = cp_slot_mapper_;
    const int   reuse_unit_tokens = cp_mapper ? cp_mapper->reuseBlockTokens(config_) : seqSizePerBlock();

    const CacheKeysType& cache_keys         = kv_resource->cacheKeys(0);
    int64_t              match_cost_time_us = 0;
    int64_t              match_end_time_us  = 0;
    PreparedKVCache      prepared;
    bool                 load_attempted = false;
    prepared.referenced_blocks.resize(kv_cache_groups_.size());
    std::shared_ptr<LoadAsyncContext> load_context;

    if (malloc_info.enable_cache_lookup) {
        CacheKeysType match_keys = cpCanonicalCacheKeys(cp_mapper, cache_keys);
        match_keys.resize(std::min(match_keys.size(), maxReusableMatchKeys(seq_len, reuse_unit_tokens)));
        const int64_t begin_us = currentTimeUs();
        load_context           = prepareKVCache(match_keys, *kv_resource, cp_mapper, prepared);
        match_end_time_us      = currentTimeUs();
        match_cost_time_us     = match_end_time_us - begin_us;
        load_attempted         = load_context != nullptr;
        kv_resource->cacheResource(0).setDeviceReuseBlockNum(prepared.matched_device_blocks);
    }

    const auto rollback = [&]() -> MallocResult {
        load_attempted = load_attempted || load_context != nullptr;
        if (load_context != nullptr) {
            load_context->abortPending();
        }
        load_context.reset();
        rollbackInitMalloc(*kv_resource, prepared.referenced_blocks, prepared.original_sizes);
        MallocResult result{false, 0};
        result.match_cost_time_us = match_cost_time_us;
        result.match_end_time_us  = match_end_time_us;
        result.load_attempted     = load_attempted;
        if (prepared.materialize_status != MallocStatus::NONE) {
            result.status = prepared.materialize_status;
        }
        return result;
    };

    if (load_context && load_context->needBackendMatch()) {
        auto self              = shared_from_this();
        auto deferred_prepared = std::make_shared<PreparedKVCache>(std::move(prepared));
        load_context->setMatchCallback([self = std::move(self), malloc_info, deferred_prepared](
                                           LoadAsyncContext& context, size_t matched_blocks) mutable {
            auto       invocation_prepared = std::move(deferred_prepared);
            const bool success = self->finishDeferredMalloc(malloc_info, *invocation_prepared, context, matched_blocks);
            return LoadMatchResult{success, invocation_prepared->materialize_status};
        });
        MallocResult result{true,
                            static_cast<int>(deferred_prepared->matched_device_blocks) * reuse_unit_tokens,
                            match_cost_time_us,
                            load_context};
        result.match_end_time_us = match_end_time_us;
        result.load_attempted    = load_attempted;
        return result;
    }

    if (!materializeInitialBlocks(malloc_info, prepared, load_context.get(), prepared.total_logical_blocks)) {
        return rollback();
    }
    MallocResult result{
        true, static_cast<int>(prepared.matched_device_blocks) * reuse_unit_tokens, match_cost_time_us, load_context};
    result.match_end_time_us = match_end_time_us;
    result.load_attempted    = load_attempted;
    return result;
}

bool CoordinatorCacheManager::materializeInitialBlocks(const MallocInfo& malloc_info,
                                                       PreparedKVCache&  prepared,
                                                       LoadAsyncContext* context,
                                                       size_t            matched_blocks) {
    auto&       kv_resource = *malloc_info.batch_kv_cache_resource;
    const auto& cp_mapper   = cp_slot_mapper_;
    const int   cp_scale    = cp_mapper && cp_mapper->isSharded() ? cp_mapper->cpSize() : 1;
    prepared.required_positions.assign(kv_cache_groups_.size(), {});

    if (matched_blocks > 0) {
        for (const GroupSetPtr& group_set : block_tree_cache_->groupSets()) {
            for (const auto& tag : group_set->groupTags()) {
                kv_resource.mutableBlockIds(0, tag).resize(
                    loadTargetPosition(matched_blocks - 1, tag, cp_mapper, cp_scale) + 1);
            }
        }
    }
    auto add_target = [&](size_t path, size_t group_id, const std::string& tag) {
        prepared.required_positions[group_id].insert(loadTargetPosition(path, tag, cp_mapper, cp_scale));
    };
    if (context != nullptr) {
        for (size_t i = 0; i < context->loadDescs().size(); ++i) {
            const auto& desc = context->loadDescs()[i];
            if (desc.source_tier != Tier::DEVICE && !context->joinedLoads()[i]) {
                for (const auto& tag : block_tree_cache_->groupSets()[desc.group_set_id]->groupTags()) {
                    const size_t group_id = groupIdForTag(tag);
                    add_target(desc.path_index, group_id, tag);
                }
            }
        }
        const auto& backend_handles = context->backendHandles();
        for (size_t key_index = 0; key_index < backend_handles.size(); ++key_index) {
            for (const StorageBlockHandle& handle : backend_handles[key_index]) {
                add_target(key_index, groupIdForTag(handle.tag), handle.tag);
            }
        }
    }

    const int seq_len        = malloc_info.complete_token_ids->seqLength();
    const int common_seq_len = std::min(malloc_info.complete_token_ids->commonSeqLength(), seq_len);
    prepared.materialize_status =
        evaluatePreparedInitCapacity(malloc_info, reserveBlocksNum(), prepared, context != nullptr);
    if (prepared.materialize_status != MallocStatus::NONE) {
        return false;
    }
    for (int group_id = 0; group_id < config_.groupNums(); ++group_id) {
        const auto&         tag       = config_.groupTags()[static_cast<size_t>(group_id)];
        auto&               block_ids = kv_resource.mutableBlockIds(0, tag);
        std::vector<size_t> backfilled_positions;
        if (!kv_cache_groups_[static_cast<size_t>(group_id)]->malloc(
                block_ids,
                cpEffectiveSeqLenForGroup(cp_mapper, config_, tag, common_seq_len),
                malloc_info.reuse_cache,
                0,
                &backfilled_positions,
                prepared.required_positions[static_cast<size_t>(group_id)])) {
            return false;
        }
        const auto& blocks = block_ids.blocks();
        for (const size_t position : backfilled_positions) {
            prepared.referenced_blocks[static_cast<size_t>(group_id)].push_back(blocks[position]);
        }
    }

    auto target_blocks = [&](size_t path, size_t group_set_id) {
        BlockIndicesType blocks;
        for (const auto& tag : block_tree_cache_->groupSets()[group_set_id]->groupTags()) {
            blocks.push_back(kv_resource.blocks(0, tag)[loadTargetPosition(path, tag, cp_mapper, cp_scale)]);
        }
        return blocks;
    };
    if (context != nullptr) {
        for (size_t i = 0; i < context->loadDescs().size(); ++i) {
            const auto& desc = context->loadDescs()[i];
            if (!context->joinedLoads()[i]) {
                context->setTargetBlocks(i, target_blocks(desc.path_index, desc.group_set_id));
            }
        }
        const auto& backend_handles = context->backendHandles();
        for (size_t key_index = 0; key_index < backend_handles.size(); ++key_index) {
            for (size_t handle_index = 0; handle_index < backend_handles[key_index].size(); ++handle_index) {
                const auto& handle = backend_handles[key_index][handle_index];
                context->setBackendTargetBlock(
                    key_index,
                    handle_index,
                    kv_resource.blocks(0, handle.tag)[loadTargetPosition(key_index, handle.tag, cp_mapper, cp_scale)]);
            }
        }
    }
    for (int batch = 1; batch < kv_resource.batchSize(); ++batch) {
        for (int group_id = 0; group_id < config_.groupNums(); ++group_id) {
            const auto& tag = config_.groupTags()[static_cast<size_t>(group_id)];
            kv_cache_groups_[static_cast<size_t>(group_id)]->reference(kv_resource.mutableBlockIds(batch, tag),
                                                                       kv_resource.blocks(0, tag));
        }
    }
    return true;
}

bool CoordinatorCacheManager::finishDeferredMalloc(const MallocInfo& malloc_info,
                                                   PreparedKVCache&  prepared,
                                                   LoadAsyncContext& context,
                                                   size_t            matched_blocks) {
    std::lock_guard<std::mutex> lock(malloc_mutex_);
    bool                        success = materializeInitialBlocks(malloc_info, prepared, &context, matched_blocks);
    if (success) {
        const auto incr_result = incrMalloc(malloc_info);
        success                = incr_result.success;
        if (!success) {
            prepared.materialize_status = incr_result.status;
        }
    }
    success = success && context.commit();
    if (!success) {
        if (prepared.materialize_status == MallocStatus::NONE) {
            prepared.materialize_status = MallocStatus::INTERNAL_ERROR;
        }
        free(FreeInfo{malloc_info.batch_kv_cache_resource, malloc_info.complete_token_ids});
        return false;
    }
    return true;
}

MallocResult CoordinatorCacheManager::incrMalloc(const MallocInfo& malloc_info) {
    auto&       kv_resource  = malloc_info.batch_kv_cache_resource;
    const auto& cp_mapper    = cp_slot_mapper_;
    const int   batch_size   = kv_resource->batchSize();
    const int   raw_seq_len  = malloc_info.incrSeqLen();
    const int   reserve_step = malloc_info.complete_token_ids->getReserveStep();

    std::vector<std::vector<size_t>>              original_sizes(static_cast<size_t>(batch_size));
    std::vector<std::vector<std::vector<size_t>>> backfilled_positions(static_cast<size_t>(batch_size));
    for (int b = 0; b < batch_size; ++b) {
        original_sizes[static_cast<size_t>(b)].resize(kv_cache_groups_.size());
        backfilled_positions[static_cast<size_t>(b)].resize(kv_cache_groups_.size());
        for (int group_id = 0; group_id < config_.groupNums(); ++group_id) {
            const auto& tag = config_.groupTags()[static_cast<size_t>(group_id)];
            original_sizes[static_cast<size_t>(b)][static_cast<size_t>(group_id)] = kv_resource->blocksNum(b, tag);
        }
    }

    bool all_success        = true;
    int  failed_batch       = -1;
    int  failed_group       = -1;
    int  failed_need_blocks = -1;
    for (int b = 0; b < batch_size; ++b) {
        for (int group_id = 0; group_id < config_.groupNums(); ++group_id) {
            const auto& tag              = config_.groupTags()[static_cast<size_t>(group_id)];
            auto&       block_ids        = kv_resource->mutableBlockIds(b, tag);
            const int   group_seq_len    = cpEffectiveSeqLenForGroup(cp_mapper, config_, tag, raw_seq_len);
            auto&       filled_positions = backfilled_positions[static_cast<size_t>(b)][static_cast<size_t>(group_id)];
            if (!kv_cache_groups_[static_cast<size_t>(group_id)]->malloc(
                    block_ids, group_seq_len, malloc_info.reuse_cache, reserve_step, &filled_positions)) {
                all_success  = false;
                failed_batch = b;
                failed_group = group_id;
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
            for (int group_id = 0; group_id < config_.groupNums(); ++group_id) {
                const auto& tag = config_.groupTags()[static_cast<size_t>(group_id)];
                kv_cache_groups_[static_cast<size_t>(group_id)]->removeSkippedBlocks(
                    kv_resource->mutableBlockIds(b, tag), malloc_info.reuse_cache, reserve_step);
            }
        }
        return {true, 0};
    }

    // Emit the pool snapshot before rolling back: once the partially allocated
    // blocks go back to the pools, available_blocks no longer reflects the state
    // that caused the failure.
    logMallocFailure(malloc_info, "incremental_group_malloc", failed_batch, failed_group, true, failed_need_blocks);

    for (int b = 0; b <= failed_batch && b < batch_size; ++b) {
        for (int group_id = 0; group_id < config_.groupNums(); ++group_id) {
            const auto& tag              = config_.groupTags()[static_cast<size_t>(group_id)];
            auto&       block_ids        = kv_resource->mutableBlockIds(b, tag);
            const auto  original_size    = original_sizes[static_cast<size_t>(b)][static_cast<size_t>(group_id)];
            const auto& filled_positions = backfilled_positions[static_cast<size_t>(b)][static_cast<size_t>(group_id)];
            const auto& blocks           = block_ids.blocks();
            BlockIndicesType blocks_to_free;
            blocks_to_free.reserve(filled_positions.size() + blocks.size() - std::min(original_size, blocks.size()));
            for (size_t pos : filled_positions) {
                RTP_LLM_CHECK_WITH_INFO(pos < original_size && pos < blocks.size(),
                                        "invalid hybrid rollback backfill position=%zu original_size=%zu size=%zu",
                                        pos,
                                        original_size,
                                        blocks.size());
                blocks_to_free.push_back(blocks[pos]);
            }
            const size_t suffix_begin = std::min(original_size, blocks.size());
            blocks_to_free.insert(blocks_to_free.end(), blocks.begin() + suffix_begin, blocks.end());
            kv_cache_groups_[static_cast<size_t>(group_id)]->unreference(blocks_to_free);
            for (size_t pos : filled_positions) {
                block_ids.setAt(pos, NULL_BLOCK_IDX);
            }
            block_ids.resize(original_size);
        }
    }
    RTP_LLM_LOG_WARNING("Hybrid incrMalloc failed at batch=%d group=%d", failed_batch, failed_group);
    return {false, 0};
}

void CoordinatorCacheManager::free(const FreeInfo& free_info) {
    auto& kv_cache_resource = free_info.batch_kv_cache_resource;
    if (kv_cache_resource->curBlocksNum() == 0) {
        return;
    }

    const auto& group_tags = config_.groupTags();
    RTP_LLM_CHECK_WITH_INFO(kv_cache_groups_.size() == group_tags.size(),
                            "cache allocator group count mismatch: allocators=%zu topology=%zu",
                            kv_cache_groups_.size(),
                            group_tags.size());
    std::vector<std::vector<const BlockIndicesType*>> blocks_by_batch;
    blocks_by_batch.reserve(static_cast<size_t>(kv_cache_resource->batchSize()));
    for (int batch_id = 0; batch_id < kv_cache_resource->batchSize(); ++batch_id) {
        const auto& resource = kv_cache_resource->cacheResource(batch_id);
        resource.groupBlockIds().validate();
        RTP_LLM_CHECK_WITH_INFO(resource.groupNums() == static_cast<int>(group_tags.size()),
                                "cache resource group count mismatch: batch=%d resource=%d topology=%zu",
                                batch_id,
                                resource.groupNums(),
                                group_tags.size());
        auto& blocks_by_group = blocks_by_batch.emplace_back();
        blocks_by_group.reserve(group_tags.size());
        for (const auto& tag : group_tags) {
            blocks_by_group.push_back(&resource.blocks(tag));
        }
    }

    for (const auto& blocks_by_group : blocks_by_batch) {
        for (size_t group_id = 0; group_id < group_tags.size(); ++group_id) {
            kv_cache_groups_[group_id]->unreference(*blocks_by_group[group_id]);
        }
    }
    kv_cache_resource->clearBlocks();
}

void CoordinatorCacheManager::insertIntoCache(const InsertInfo& insert_info, size_t& resident_prefix_length) {
    resident_prefix_length  = 0;
    auto& kv_cache_resource = insert_info.batch_kv_cache_resource;
    RTP_LLM_CHECK(kv_cache_resource != nullptr);
    if (!block_tree_cache_) {
        return;
    }

    const auto& cp_mapper = cp_slot_mapper_;
    const bool  cp_active = cp_mapper && cp_mapper->isSharded();
    // Preserve the single-group publication contract: only batch zero becomes
    // reusable. Multi-group requests retain their per-batch publication semantics.
    const int batch_size =
        config_.groupNums() == 1 ? std::min(1, kv_cache_resource->batchSize()) : kv_cache_resource->batchSize();

    for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
        const auto& full_keys = kv_cache_resource->cacheKeys(batch_id);
        if (full_keys.empty()) {
            continue;
        }
        CacheKeysType insert_keys = cp_active ? cpCanonicalCacheKeys(cp_mapper, full_keys) : full_keys;
        if (insert_keys.empty()) {
            continue;
        }
        const auto&                                group_sets = block_tree_cache_->groupSets();
        std::vector<std::vector<GroupSetResource>> resources(insert_keys.size(),
                                                             std::vector<GroupSetResource>(group_sets.size()));
        bool                                       mapping_valid = true;
        for (size_t group_set_id = 0; group_set_id < group_sets.size(); ++group_set_id) {
            const auto& group_set = group_sets[group_set_id];
            if (!group_set || group_set->groupSetId() != group_set_id || group_set->groupTags().empty()
                || group_set->groupTags().size() != group_set->devicePools().size()) {
                mapping_valid = false;
                break;
            }
            for (auto& per_key_resources : resources) {
                per_key_resources[group_set_id].device_blocks.assign(group_set->devicePools().size(), NULL_BLOCK_IDX);
            }
            for (size_t member_group_id = 0; member_group_id < group_set->groupTags().size(); ++member_group_id) {
                const auto& tag      = group_set->groupTags()[member_group_id];
                const int   group_id = static_cast<int>(groupIdForTag(tag));
                if (!kv_cache_groups_[static_cast<size_t>(group_id)]->prefixReuseEnabled()) {
                    mapping_valid = false;
                    break;
                }
                const auto type           = config_.group(tag).policy.group_type;
                const bool sparse_logical = cp_active
                                            && (type == CacheGroupType::LINEAR
                                                || (type == CacheGroupType::SWA && !cpCompactSwaGroup(tag, cp_mapper)));
                const auto& blocks = kv_cache_resource->blocks(batch_id, tag);
                for (size_t i = 0; i < insert_keys.size(); ++i) {
                    const size_t position = sparse_logical ? (i + 1) * static_cast<size_t>(cp_mapper->cpSize()) - 1 : i;
                    if (position >= blocks.size() || isNullBlockIdx(blocks[position])) {
                        continue;
                    }
                    resources[i][group_set_id].device_blocks[member_group_id] = blocks[position];
                }
            }
            if (!mapping_valid) {
                break;
            }
        }
        if (!mapping_valid) {
            RTP_LLM_LOG_WARNING("Hybrid insert rejected inconsistent topology group/GroupSet mapping");
            continue;
        }

        size_t publish_prefix = 0;
        for (size_t i = 0; i < resources.size(); ++i) {
            bool key_valid    = true;
            bool key_has_data = false;
            for (size_t group_set_id = 0; group_set_id < group_sets.size(); ++group_set_id) {
                auto&       device_blocks = resources[i][group_set_id].device_blocks;
                const auto& group_set     = group_sets[group_set_id];
                const auto  valid_blocks  = static_cast<size_t>(
                    std::count_if(device_blocks.begin(), device_blocks.end(), [](BlockIdxType block) {
                        return !isNullBlockIdx(block);
                    }));
                if (valid_blocks == 0 && group_set->groupType() != CacheGroupType::FULL) {
                    device_blocks.clear();
                    continue;
                }
                if (valid_blocks != device_blocks.size()) {
                    key_valid = false;
                    break;
                }
                key_has_data = true;
            }
            if (!key_valid) {
                break;
            }
            if (key_has_data) {
                publish_prefix = i + 1;
            }
        }
        if (publish_prefix == 0) {
            continue;
        }
        insert_keys.resize(publish_prefix);
        resources.resize(publish_prefix);
        if (insert_info.is_resident) {
            const size_t batch_resident_prefix_length =
                block_tree_cache_->insert(insert_keys, resources, insert_info.target_tier, true);
            resident_prefix_length += batch_resident_prefix_length;
        } else {
            block_tree_cache_->insert(insert_keys, resources, insert_info.target_tier);
        }
    }
}

std::shared_ptr<KVCacheResource> CoordinatorCacheManager::incrKVCacheRef(const KVCacheResource& kvcache_resource,
                                                                         const CacheKeysType&   cache_keys,
                                                                         bool                   is_connector) {
    if (cache_keys.empty() || kvcache_resource.groupNums() <= 0) {
        return nullptr;
    }

    const auto& groups = config_.topology().groups();
    RTP_LLM_CHECK_WITH_INFO(kvcache_resource.groupNums() == config_.groupNums(),
                            "cache resource and coordinator group counts differ");
    std::vector<const BlockIndicesType*> source_blocks_by_group;
    source_blocks_by_group.reserve(groups.size());
    for (const auto& group : groups) {
        source_blocks_by_group.push_back(&kvcache_resource.blocks(group.tag));
    }
    std::unordered_map<CacheKeyType, size_t> key_to_pos;
    const auto&                              resource_keys = kvcache_resource.cacheKeys();
    for (size_t i = 0; i < resource_keys.size(); ++i) {
        key_to_pos.emplace(resource_keys[i], i);
    }

    auto selected_resource_ptr = new KVCacheResource(kvcache_resource);
    auto deleter               = [self = shared_from_this()](KVCacheResource* resource) {
        self->decrKVCacheRef(*resource);
        delete resource;
    };
    std::shared_ptr<KVCacheResource> selected_resource(selected_resource_ptr, deleter);
    selected_resource->initGroups(config_.topologyPtr());

    CacheKeysType                 selected_keys;
    BlockDependenciesType         selected_dependencies;
    std::vector<BlockIndicesType> selected_blocks(kv_cache_groups_.size());
    const auto&                   source_dependencies = kvcache_resource.blockDependencies();

    selected_dependencies.reserve(cache_keys.size());
    selected_keys.reserve(cache_keys.size());
    for (auto key : cache_keys) {
        auto it = key_to_pos.find(key);
        if (it == key_to_pos.end()) {
            continue;
        }
        const size_t              pos             = it->second;
        bool                      any_valid_block = false;
        std::vector<BlockIdxType> blocks_for_key(kv_cache_groups_.size(), NULL_BLOCK_IDX);
        for (int group_id = 0; group_id < config_.groupNums(); ++group_id) {
            const auto& src_blocks                        = *source_blocks_by_group[static_cast<size_t>(group_id)];
            const auto  block                             = pos < src_blocks.size() ? src_blocks[pos] : NULL_BLOCK_IDX;
            blocks_for_key[static_cast<size_t>(group_id)] = block;
            any_valid_block                               = any_valid_block || (!isNullBlockIdx(block) && block > 0);
        }
        const bool preserve_connector_tail = is_connector && !kvcache_resource.lastBlockAligned()
                                             && pos + 1 == resource_keys.size() && !selected_keys.empty();
        if (!any_valid_block && !preserve_connector_tail) {
            continue;
        }
        selected_keys.push_back(key);
        selected_dependencies.push_back(
            pos < source_dependencies.size() ?
                source_dependencies[pos] :
                BlockDependency{false, 0, static_cast<uint32_t>(selected_dependencies.size())});
        for (int group_id = 0; group_id < config_.groupNums(); ++group_id) {
            selected_blocks[static_cast<size_t>(group_id)].push_back(blocks_for_key[static_cast<size_t>(group_id)]);
        }
    }

    if (selected_keys.empty()) {
        return nullptr;
    }

    selected_resource->cacheKeys() = std::move(selected_keys);
    selected_resource->setBlockDependencies(std::move(selected_dependencies));
    for (int group_id = 0; group_id < config_.groupNums(); ++group_id) {
        BlockIndicesType valid;
        for (auto b : selected_blocks[static_cast<size_t>(group_id)]) {
            if (!isNullBlockIdx(b) && b > 0) {
                valid.push_back(b);
            }
        }
        kv_cache_groups_[static_cast<size_t>(group_id)]->reference(valid);
        selected_resource->mutableBlockIds(groups[static_cast<size_t>(group_id)].tag)
            .assign(std::move(selected_blocks[static_cast<size_t>(group_id)]));
    }
    return selected_resource;
}

void CoordinatorCacheManager::decrKVCacheRef(const KVCacheResource& kvcache_resource) {
    const auto& groups = config_.topology().groups();
    RTP_LLM_CHECK_WITH_INFO(kvcache_resource.groupNums() == config_.groupNums(),
                            "cache resource and coordinator group counts differ");
    std::vector<const BlockIndicesType*> blocks_by_group;
    blocks_by_group.reserve(groups.size());
    for (const auto& group : groups) {
        blocks_by_group.push_back(&kvcache_resource.blocks(group.tag));
    }
    for (size_t group_id = 0; group_id < groups.size(); ++group_id) {
        kv_cache_groups_[group_id]->unreference(*blocks_by_group[group_id]);
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
    const int group_nums     = config_.groupNums();

    std::vector<int> batch_fork_count(old_batch_size, 0);
    for (const int old_batch_idx : block_src_batch) {
        RTP_LLM_CHECK_WITH_INFO(old_batch_idx >= 0 && old_batch_idx < old_batch_size,
                                "try to reuse an old batch %d that out of range %d",
                                old_batch_idx,
                                old_batch_size);
        ++batch_fork_count[old_batch_idx];
    }

    // A sole FULL group can transfer request references and reserve forked
    // tails in one pool transaction. Multi-group updates retain their existing
    // all-group reservation path below.
    const auto* sole_group          = group_nums == 1 ? &config_.topology().groups().front() : nullptr;
    const bool  atomic_single_group = sole_group && sole_group->policy.group_type == CacheGroupType::FULL
                                     && sole_group->spec
                                     && (sole_group->spec->type == KVCacheSpecType::MultiHeadAttention
                                         || sole_group->spec->type == KVCacheSpecType::MultiHeadLatentAttention);
    if (atomic_single_group) {
        const auto&                                          tag  = config_.groupTags().front();
        const auto&                                          pool = group_block_pools_.front();
        std::vector<DeviceBlockPool::RequestReferenceUpdate> reference_updates;
        size_t                                               replacement_count = 0;
        for (int i = 0; i < old_batch_size; ++i) {
            const auto& blocks = batch_kv_cache_resource->blocks(i, tag);
            if (batch_fork_count[i] == 0 && !blocks.empty()) {
                reference_updates.push_back({blocks.back(), 1, 0});
            }
        }
        for (int i = 0; i < old_batch_size; ++i) {
            const auto& blocks = batch_kv_cache_resource->blocks(i, tag);
            const int   forks  = batch_fork_count[i];
            if (forks == 1 || blocks.empty()) {
                continue;
            }
            for (size_t j = 0; j + 1 < blocks.size(); ++j) {
                reference_updates.push_back({blocks[j], 1, forks});
            }
            if (forks > 1) {
                if (copy_last_block) {
                    replacement_count += static_cast<size_t>(forks - 1);
                } else {
                    reference_updates.push_back({blocks.back(), 1, forks});
                }
            }
        }

        BatchKVCacheResource staged_resource;
        staged_resource.resetBatchSize(new_batch_size);
        std::vector<TaggedBlockIdPair>            staged_mapping;
        std::vector<std::pair<BlockIds*, size_t>> replacement_slots;
        std::vector<std::pair<int, int>>          retained_slots;
        staged_mapping.reserve(replacement_count);
        replacement_slots.reserve(replacement_count);
        retained_slots.reserve(old_batch_size);
        for (int i = 0; i < new_batch_size; ++i) {
            const int   old_idx      = block_src_batch[i];
            auto&       forks        = batch_fork_count[old_idx];
            const auto& old_resource = batch_kv_cache_resource->cacheResource(old_idx);
            if (forks == 1) {
                retained_slots.emplace_back(i, old_idx);
            } else {
                auto& fork = staged_resource.cacheResource(i);
                fork.initGroups(config_.topologyPtr());
                fork.setCacheKeys(old_resource.cacheKeys());
                auto& block_ids = fork.mutableBlockIds(tag);
                block_ids.assign(old_resource.blocks(tag));
                if (copy_last_block && !block_ids.blocks().empty()) {
                    const auto& blocks = block_ids.blocks();
                    staged_mapping.push_back({tag, blocks.back(), NULL_BLOCK_IDX});
                    replacement_slots.emplace_back(&block_ids, blocks.size() - 1);
                    block_ids.setAt(blocks.size() - 1, NULL_BLOCK_IDX);
                }
            }
            --forks;
        }

        BlockIndicesType replacements;
        int              required_free_blocks = 0;
        if (!pool->tryReplaceRequestReferences(
                reference_updates, static_cast<int>(replacement_count), replacements, required_free_blocks)) {
            // Eviction takes the tree lock and must run outside the pool transaction.
            kv_cache_groups_.front()->ensureFreeBlocks(required_free_blocks);
            if (!pool->tryReplaceRequestReferences(
                    reference_updates, static_cast<int>(replacement_count), replacements, required_free_blocks)) {
                RTP_LLM_LOG_WARNING("atomic kv cache update failed, need %zu replacement blocks", replacement_count);
                return false;
            }
        }
        for (size_t i = 0; i < replacements.size(); ++i) {
            replacement_slots[i].first->setAt(replacement_slots[i].second, replacements[i]);
            staged_mapping[i].dst = replacements[i];
        }
        static_assert(std::is_nothrow_move_assignable_v<KVCacheResource>);
        for (const auto& [new_idx, old_idx] : retained_slots) {
            staged_resource.moveBatchResource(new_idx, std::move(batch_kv_cache_resource->cacheResource(old_idx)));
        }
        batch_kv_cache_resource->swap(staged_resource);
        block_update_mapping.swap(staged_mapping);
        return true;
    }

    std::vector<int> new_blocks_num(static_cast<size_t>(group_nums), 0);
    for (int old_batch_idx = 0; old_batch_idx < old_batch_size; ++old_batch_idx) {
        const int fork_count = batch_fork_count[old_batch_idx];
        if (fork_count > 1 && copy_last_block) {
            for (int group_id = 0; group_id < group_nums; ++group_id) {
                const auto& tag = config_.groupTags()[static_cast<size_t>(group_id)];
                if (!batch_kv_cache_resource->blocks(old_batch_idx, tag).empty()) {
                    new_blocks_num[static_cast<size_t>(group_id)] += fork_count - 1;
                }
            }
        }
    }

    // Transfer request ownership from dropped batches before allocating new
    // blocks. This keeps the operation transactional while allowing net-feasible
    // drop-and-fork updates to succeed when the pool is otherwise full.
    std::vector<BlockIndicesType>                      replacement_blocks(static_cast<size_t>(group_nums));
    std::vector<BlockIndicesType>                      allocated_replacements(static_cast<size_t>(group_nums));
    std::vector<std::unordered_map<BlockIdxType, int>> transferred_ref_counts(static_cast<size_t>(group_nums));
    for (int group_id = 0; group_id < group_nums; ++group_id) {
        const auto&                           tag = config_.groupTags()[static_cast<size_t>(group_id)];
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

        auto&     replacements = replacement_blocks[static_cast<size_t>(group_id)];
        auto&     transferred  = transferred_ref_counts[static_cast<size_t>(group_id)];
        const int need         = new_blocks_num[static_cast<size_t>(group_id)];
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
        for (int group_id = 0; group_id < group_nums; ++group_id) {
            auto& blocks = allocated_replacements[static_cast<size_t>(group_id)];
            kv_cache_groups_[static_cast<size_t>(group_id)]->unreference(blocks);
            blocks.clear();
        }
    };
    for (int group_id = 0; group_id < group_nums; ++group_id) {
        const int need_blocks = new_blocks_num[static_cast<size_t>(group_id)];
        auto&     reserved    = replacement_blocks[static_cast<size_t>(group_id)];
        reserved.reserve(static_cast<size_t>(need_blocks));
        for (int i = static_cast<int>(reserved.size()); i < need_blocks; ++i) {
            BlockIds   one_block;
            const bool ok = kv_cache_groups_[static_cast<size_t>(group_id)]->malloc(
                one_block, kv_cache_groups_[static_cast<size_t>(group_id)]->seqSizePerBlock());
            const auto& blocks = one_block.blocks();
            if (ok && blocks.size() == 1 && !isNullBlockIdx(blocks.front())) {
                reserved.push_back(blocks.front());
                allocated_replacements[static_cast<size_t>(group_id)].push_back(blocks.front());
                continue;
            }
            if (!blocks.empty()) {
                allocated_replacements[static_cast<size_t>(group_id)].insert(
                    allocated_replacements[static_cast<size_t>(group_id)].end(), blocks.begin(), blocks.end());
            }
            RTP_LLM_LOG_WARNING(
                "reserve replacement block failed for hybrid kv cache update, group=%d need=%d reserved=%zu",
                group_id,
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
        for (int group_id = 0; group_id < group_nums; ++group_id) {
            const auto&      tag = config_.groupTags()[static_cast<size_t>(group_id)];
            BlockIndicesType to_free;
            auto&            transferred = transferred_ref_counts[static_cast<size_t>(group_id)];
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
            kv_cache_groups_[static_cast<size_t>(group_id)]->unreference(to_free);
        }
    }

    std::vector<KVCacheResource> old_resources;
    batch_kv_cache_resource->resetAndReturnOldResources(new_batch_size, old_resources);
    batch_kv_cache_resource->initGroups(config_.topologyPtr());
    std::vector<size_t> next_replacement(static_cast<size_t>(group_nums), 0);

    for (int new_batch_idx = 0; new_batch_idx < new_batch_size; ++new_batch_idx) {
        const int old_batch_idx = block_src_batch[new_batch_idx];
        auto&     fork_count    = batch_fork_count[old_batch_idx];
        RTP_LLM_CHECK_WITH_INFO(fork_count > 0, "old batch %d has been forked too many times", old_batch_idx);

        if (fork_count == 1) {
            batch_kv_cache_resource->moveBatchResource(new_batch_idx, std::move(old_resources[old_batch_idx]));
        } else {
            batch_kv_cache_resource->setBatchCacheKeys(new_batch_idx, old_resources[old_batch_idx].cacheKeys());
            for (int group_id = 0; group_id < group_nums; ++group_id) {
                const auto& tag       = config_.groupTags()[static_cast<size_t>(group_id)];
                auto&       block_ids = batch_kv_cache_resource->mutableBlockIds(new_batch_idx, tag);
                kv_cache_groups_[static_cast<size_t>(group_id)]->reference(block_ids,
                                                                           old_resources[old_batch_idx].blocks(tag));

                if (copy_last_block && !block_ids.blocks().empty()) {
                    const int  old_block       = block_ids.popBack();
                    const bool old_block_valid = !isNullBlockIdx(old_block) && old_block > 0;
                    if (old_block_valid) {
                        kv_cache_groups_[static_cast<size_t>(group_id)]->unreference({old_block});
                    }

                    auto&      reserved     = replacement_blocks[static_cast<size_t>(group_id)];
                    const auto reserved_idx = next_replacement[static_cast<size_t>(group_id)]++;
                    RTP_LLM_CHECK_WITH_INFO(reserved_idx < reserved.size(),
                                            "missing reserved replacement block for hybrid kv cache update, group=%d",
                                            group_id);
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
    for (int group_id = 0; group_id < group_nums; ++group_id) {
        RTP_LLM_CHECK_WITH_INFO(
            next_replacement[static_cast<size_t>(group_id)] == replacement_blocks[static_cast<size_t>(group_id)].size(),
            "unused replacement blocks after hybrid kv cache update, group=%d used=%zu reserved=%zu",
            group_id,
            next_replacement[static_cast<size_t>(group_id)],
            replacement_blocks[static_cast<size_t>(group_id)].size());
    }
    return true;
}

int CoordinatorCacheManager::seqSizePerBlock() const {
    return static_cast<int>(config_.seq_size_per_block);
}

void CoordinatorCacheManager::rollbackBlockIdsToSize(int group_id, BlockIds& block_ids, size_t original_size) {
    if (block_ids.blocksNum() <= original_size) {
        return;
    }
    const auto&            blocks = block_ids.blocks();
    const BlockIndicesType blocks_to_free(blocks.begin() + original_size, blocks.end());
    block_ids.resize(original_size);
    kv_cache_groups_[static_cast<size_t>(group_id)]->unreference(blocks_to_free);
}

void CoordinatorCacheManager::rollbackInitMalloc(BatchKVCacheResource&                kv_resource,
                                                 const std::vector<BlockIndicesType>& referenced_blocks,
                                                 const std::vector<size_t>&           original_sizes) {
    for (int group_id = 0; group_id < config_.groupNums(); ++group_id) {
        const auto&  tag           = config_.groupTags()[static_cast<size_t>(group_id)];
        auto&        block_ids     = kv_resource.mutableBlockIds(0, tag);
        const size_t original_size = original_sizes.empty() ? 0 : original_sizes[static_cast<size_t>(group_id)];
        if (block_ids.blocksNum() > original_size) {
            rollbackBlockIdsToSize(group_id, block_ids, original_size);
        }
        if (static_cast<size_t>(group_id) < referenced_blocks.size()) {
            kv_cache_groups_[static_cast<size_t>(group_id)]->unreference(
                referenced_blocks[static_cast<size_t>(group_id)]);
        }
        block_ids.resize(0);
    }
    kv_resource.cacheResource(0).setDeviceReuseBlockNum(0);
}

MemoryType CoordinatorCacheManager::memoryTypeForGroup(int group_id) const {
    (void)group_id;
    return allocation_type_ == AllocationType::DEVICE ? MemoryType::MEMORY_GPU : MemoryType::MEMORY_CPU;
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
    for (int group_id = 0; group_id < static_cast<int>(kv_cache_groups_.size()); ++group_id) {
        const auto& tag              = config_.groupTags()[static_cast<size_t>(group_id)];
        const auto  group            = kv_cache_groups_[static_cast<size_t>(group_id)];
        const int   group_common_seq = cpEffectiveSeqLenForGroup(cp_mapper, config_, tag, raw_common_seq_len);
        const int   group_seq_len    = cpEffectiveSeqLenForGroup(cp_mapper, config_, tag, raw_seq_len);
        const auto  need             = kv_cache_groups_[static_cast<size_t>(group_id)]->getNeedBlocks(
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
    for (int group_id = 0; group_id < config_.groupNums(); ++group_id) {
        const auto& tag = config_.groupTags()[static_cast<size_t>(group_id)];
        need_blocks += kv_cache_groups_[static_cast<size_t>(group_id)]->estimatePeakNeedBlocks(
            seq_len, kv_cache_resource.blocks(tag), remaining_tokens, reserve_step, enable_reuse_cache);
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
    for (const auto& group : kv_cache_groups_) {
        peak_blocks += group->estimateInitialBatchPeakNeedBlocks(
            seq_len, common_seq_len, remaining_tokens, reserve_step, enable_reuse_cache, target_batch_size);
    }
    return peak_blocks;
}

void CoordinatorCacheManager::checkCPShardedMallocResult(const MallocInfo& malloc_info) const {
    if (!cp_slot_mapper_ || !cp_slot_mapper_->isSharded()) {
        return;
    }

    const auto& kv_resource  = malloc_info.batch_kv_cache_resource;
    const int   seq_len      = malloc_info.incrSeqLen();
    const int   reserve_step = malloc_info.complete_token_ids->getReserveStep();

    for (int batch_id = 0; batch_id < kv_resource->batchSize(); ++batch_id) {
        for (int group_id = 0; group_id < config_.groupNums(); ++group_id) {
            const auto& tag = config_.groupTags()[static_cast<size_t>(group_id)];
            if (!cpBlockRoundRobinGroup(cp_slot_mapper_, config_, tag)) {
                continue;
            }
            const int effective_seq_len = cpEffectiveSeqLenForGroup(cp_slot_mapper_, config_, tag, seq_len);
            const int expected_blocks =
                kv_cache_groups_[static_cast<size_t>(group_id)]->needBlocksNum(effective_seq_len, 0, reserve_step);
            const int actual_blocks = kv_resource->blocksNum(batch_id, tag);
            RTP_LLM_CHECK_WITH_INFO(actual_blocks == expected_blocks,
                                    "CP invariant violated: batch=%d group=%d blocks=%d != expected_local_blocks=%d "
                                    "(seq_len=%d, effective_seq_len=%d, reserve_step=%d, cp_size=%d, "
                                    "block_size=%d, cacheKeys=%zu)",
                                    batch_id,
                                    group_id,
                                    actual_blocks,
                                    expected_blocks,
                                    seq_len,
                                    effective_seq_len,
                                    reserve_step,
                                    cp_slot_mapper_->cpSize(),
                                    cp_slot_mapper_->blockSize(),
                                    kv_resource->cacheKeys(batch_id).size());
        }
    }
}

int CoordinatorCacheManager::singleBatchNeedBlocks(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                                   int                            seq_len,
                                                   int                            reserve_step) const {
    int need_blocks = 0;
    for (int group_id = 0; group_id < config_.groupNums(); ++group_id) {
        const auto& tag               = config_.groupTags()[static_cast<size_t>(group_id)];
        const int   effective_seq_len = cpEffectiveSeqLenForGroup(cp_slot_mapper_, config_, tag, seq_len);
        const int   cur_blocks        = batch_kv_cache_resource->blocksNum(0, tag);
        need_blocks +=
            kv_cache_groups_[static_cast<size_t>(group_id)]->needBlocksNum(effective_seq_len, cur_blocks, reserve_step);
    }
    return need_blocks;
}

bool CoordinatorCacheManager::doInit() {
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
    const auto& topology_groups = config_.topology().groups();
    for (const auto& group : topology_groups) {
        const int              group_id    = static_cast<int>(group_pool_configs.size());
        DeviceBlockPoolConfig  pool_config = DeviceBlockPoolConfigHelper::createConfigForGroup(config_, group);
        const CacheGroupPolicy policy      = group.policy;
        if (policy.memory_placement == CacheMemoryPlacement::DEVICE) {
            pool_config.use_pinned_cpu_backing = allocation_type_ == AllocationType::HOST;
            pool_config.use_device_malloc_backing =
                use_device_malloc_block_pool_ && allocation_type_ == AllocationType::DEVICE;
        } else if (policy.memory_placement == CacheMemoryPlacement::HOST_PINNED) {
            pool_config.use_pinned_cpu_backing    = true;
            pool_config.use_device_malloc_backing = false;
        } else {
            RTP_LLM_LOG_ERROR("unsupported cache memory placement=%d for group=%d",
                              static_cast<int>(policy.memory_placement),
                              group_id);
            return false;
        }
        const auto tag        = group.tag;
        const auto group_type = group.policy.group_type;
        appendPoolSummary(pool_summary, has_pool, group_id, tag, group_type, pool_config);
        pool_total_bytes += pool_config.total_size_bytes;
        pool_total_blocks += pool_config.physical_block_count;
        group_pool_configs.push_back(std::move(pool_config));
    }

    if (has_pool) {
        const auto summary = pool_summary.str();
        RTP_LLM_LOG_INFO("CoordinatorCacheManager pool summary: pools=[%s], total_size=%zu bytes total_size_mb=%.2f "
                         "total_blocks=%zu",
                         summary.c_str(),
                         pool_total_bytes,
                         static_cast<double>(pool_total_bytes) / kBytesPerMB,
                         pool_total_blocks);
    }

    int group_id = 0;
    for (const auto& cache_group : topology_groups) {
        const auto& pool_config = group_pool_configs[static_cast<size_t>(group_id)];
        const auto  group_type  = cache_group.policy.group_type;
        auto group_pool = std::make_shared<DeviceBlockPool>(std::make_shared<const DeviceBlockPoolConfig>(pool_config));
        RTP_LLM_CHECK_WITH_INFO(group_pool->init(),
                                "Failed to initialize block pool %s(group %d)",
                                pool_config.pool_name.c_str(),
                                group_id);

        SingleTypeCacheManagerPtr group;
        if (group_type == CacheGroupType::LINEAR) {
            group = std::make_shared<LinearCacheManager>(cache_group, group_pool, group_id, config_.linear_step);
            linear_group_ids_.push_back(group_id);
        } else if (group_type == CacheGroupType::SWA) {
            group = std::make_shared<SWACacheManager>(cache_group, group_pool, group_id, config_.linear_step);
            swa_group_ids_.push_back(group_id);
        } else {
            group = std::make_shared<FullCacheManager>(cache_group, group_pool, group_id);
            full_group_ids_.push_back(group_id);
        }

        RTP_LLM_CHECK_WITH_INFO(group->init(config_.layerIdsForGroup(cache_group.tag)),
                                "Failed to initialize SingleTypeCacheManager %s(group_id %d)",
                                pool_config.pool_name.c_str(),
                                group_id);
        RTP_LLM_CHECK_WITH_INFO(group->tag() == cache_group.tag && group->blockPool() == group_pool,
                                "cache manager/pool binding mismatch for tag %s",
                                cache_group.tag.c_str());
        group_block_pools_.push_back(group_pool);
        kv_cache_groups_.push_back(group);
        ++group_id;
    }

    RTP_LLM_LOG_INFO("CoordinatorCacheManager init success, group pools=%zu", group_block_pools_.size());
    return true;
}

size_t CoordinatorCacheManager::groupIdForTag(std::string_view tag) const {
    const auto& tags = config_.groupTags();
    const auto  it   = std::find(tags.begin(), tags.end(), tag);
    RTP_LLM_CHECK_WITH_INFO(it != tags.end(), "unknown cache group tag %.*s", static_cast<int>(tag.size()), tag.data());
    return static_cast<size_t>(std::distance(tags.begin(), it));
}

int CoordinatorCacheManager::defaultGroupIdForLayer(int layer_id) const {
    if (layer_id < 0 || static_cast<size_t>(layer_id) >= config_.layer_all_num()) {
        RTP_LLM_FAIL("invalid layer_id=%d", layer_id);
    }
    const auto& group    = config_.topology().soleGroupForLayer(layer_id);
    const int   group_id = static_cast<int>(groupIdForTag(group.tag));
    RTP_LLM_CHECK_WITH_INFO(group_id >= 0 && group_id < static_cast<int>(kv_cache_groups_.size()),
                            "invalid default group id %d for layer %d",
                            group_id,
                            layer_id);
    return group_id;
}

int CoordinatorCacheManager::validateGroupIdForLayer(int layer_id, int group_id) const {
    RTP_LLM_CHECK_WITH_INFO(group_id >= 0 && group_id < static_cast<int>(kv_cache_groups_.size()),
                            "invalid group id %d for layer %d",
                            group_id,
                            layer_id);
    RTP_LLM_CHECK_WITH_INFO(layer_id >= 0 && static_cast<size_t>(layer_id) < config_.layer_all_num(),
                            "invalid layer id %d for layer_all_num=%u",
                            layer_id,
                            config_.layer_all_num());
    (void)config_.groupForLayer(layer_id, config_.groupTags()[static_cast<size_t>(group_id)]);
    return group_id;
}

GroupedCacheLayerLayout CoordinatorCacheManager::allLayerCacheBase() const {
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
        groups.emplace(topology->groups()[group_id].tag, CacheLayerLayout(std::move(layers)));
    }
    return GroupedCacheLayerLayout(topology, std::move(groups));
}

BlockAddrInfo CoordinatorCacheManager::convertIndexToAddr(int layer_id, int block_id) const {
    const int group_id = defaultGroupIdForLayer(layer_id);
    return kv_cache_groups_[static_cast<size_t>(group_id)]->convertIndexToAddr(layer_id, block_id);
}

std::vector<BlockInfo> CoordinatorCacheManager::convertIndexToBuffer(int layer_id, int block_id) const {
    const int group_id = defaultGroupIdForLayer(layer_id);
    return kv_cache_groups_[static_cast<size_t>(group_id)]->convertIndexToBuffer(layer_id, block_id);
}

std::vector<BlockInfo>
CoordinatorCacheManager::convertIndexToBuffer(int layer_id, int block_id, int partition_count, int partition_id) const {
    const int group_id = defaultGroupIdForLayer(layer_id);
    return kv_cache_groups_[static_cast<size_t>(group_id)]->convertIndexToBuffer(
        layer_id, block_id, partition_count, partition_id);
}

BlockAddrInfo
CoordinatorCacheManager::convertIndexToAddr(int layer_id, const std::string& group_tag, int block_id) const {
    const auto group_id = static_cast<int>(groupIdForTag(group_tag));
    validateGroupIdForLayer(layer_id, group_id);
    return kv_cache_groups_[static_cast<size_t>(group_id)]->convertIndexToAddr(layer_id, block_id);
}

std::vector<BlockInfo>
CoordinatorCacheManager::convertIndexToBuffer(int layer_id, const std::string& group_tag, int block_id) const {
    const auto group_id = static_cast<int>(groupIdForTag(group_tag));
    validateGroupIdForLayer(layer_id, group_id);
    return kv_cache_groups_[static_cast<size_t>(group_id)]->convertIndexToBuffer(layer_id, block_id);
}

std::vector<BlockInfo> CoordinatorCacheManager::convertIndexToBuffer(
    int layer_id, const std::string& group_tag, int block_id, int partition_count, int partition_id) const {
    const auto group_id = static_cast<int>(groupIdForTag(group_tag));
    validateGroupIdForLayer(layer_id, group_id);
    return kv_cache_groups_[static_cast<size_t>(group_id)]->convertIndexToBuffer(
        layer_id, block_id, partition_count, partition_id);
}

void CoordinatorCacheManager::blockBatchCopy(const BlockIdPair* begin_ptr, const BlockIdPair* end_ptr) {
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
    blockBatchCopyByGroup(tagged_mappings);
}

void CoordinatorCacheManager::blockBatchCopyByGroup(const std::vector<TaggedBlockIdPair>& copy_mapping) {
    if (copy_mapping.empty()) {
        return;
    }

    size_t copy_nums[BatchCopyParams::TYPE_SIZE] = {};
    for (const auto& mapping : copy_mapping) {
        const auto group_id = static_cast<int>(groupIdForTag(mapping.tag));
        RTP_LLM_CHECK_WITH_INFO(
            static_cast<size_t>(group_id) < group_block_pools_.size(), "missing block pool for group %d", group_id);
        const auto copy_type =
            BatchCopyParams::get_copy_type(group_block_pools_[static_cast<size_t>(group_id)]->where(),
                                           group_block_pools_[static_cast<size_t>(group_id)]->where());
        for (int layer_id : config_.layerIdsForGroup(mapping.tag)) {
            const auto& physical_group = config_.physicalGroupForLayer(layer_id, mapping.tag);
            copy_nums[copy_type] += physical_group.kvScaleStrideBytes() > 0 ? 2 : 1;
        }
    }

    BatchCopyParams copy_params;
    for (size_t i = 0; i < BatchCopyParams::TYPE_SIZE; ++i) {
        copy_params.reserve(static_cast<BatchCopyParams::CopyType>(i), copy_nums[i]);
    }

    for (const auto& mapping : copy_mapping) {
        const auto group_id = static_cast<int>(groupIdForTag(mapping.tag));
        RTP_LLM_CHECK_WITH_INFO(
            static_cast<size_t>(group_id) < group_block_pools_.size(), "missing block pool for group %d", group_id);
        const auto copy_type =
            BatchCopyParams::get_copy_type(group_block_pools_[static_cast<size_t>(group_id)]->where(),
                                           group_block_pools_[static_cast<size_t>(group_id)]->where());

        for (int layer_id : config_.layerIdsForGroup(mapping.tag)) {
            const auto&  physical_group      = config_.physicalGroupForLayer(layer_id, mapping.tag);
            const size_t kv_block_size_bytes = physical_group.kvBlockStrideBytes();
            const size_t scale_block_bytes   = physical_group.kvScaleStrideBytes();
            auto         src_addr_info =
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
        if (pool) {
            total += pool->availableBlocksNum();
        }
    }
    return total;
}

size_t CoordinatorCacheManager::minTokenCapacity(bool use_available_blocks, bool full_groups_only) const {
    if (group_block_pools_.empty()) {
        return 0;
    }

    auto calculate = [&](bool only_full_groups) {
        size_t      min_tokens = std::numeric_limits<size_t>::max();
        bool        saw_group  = false;
        const auto& groups     = config_.topology().groups();
        for (size_t group_id = 0; group_id < group_block_pools_.size(); ++group_id) {
            if (only_full_groups && groups[group_id].policy.group_type != CacheGroupType::FULL) {
                continue;
            }
            if (!group_block_pools_[group_id]) {
                continue;
            }
            saw_group        = true;
            const auto block = use_available_blocks ? group_block_pools_[group_id]->availableBlocksNum() :
                                                      group_block_pools_[group_id]->totalBlocksNum();
            min_tokens       = std::min(min_tokens, block * logicalSeqSizePerBlockForCapacity(groups[group_id].tag));
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
    for (size_t group_id = 0; group_id < group_block_pools_.size(); ++group_id) {
        const auto& pool = group_block_pools_[group_id];
        if (!pool) {
            continue;
        }
        const size_t seq_size = config_.topology().groups()[group_id].seqSizePerBlock();
        total_tokens          = std::min(total_tokens, pool->totalBlocksNum() * seq_size);
        available_tokens      = std::min(available_tokens, pool->availableBlocksNum() * seq_size);
        has_pool              = true;
    }
    return has_pool ? KVCacheTokenCapacity{total_tokens, available_tokens} : KVCacheTokenCapacity{};
}

size_t CoordinatorCacheManager::reserveBlocksForPoolMetrics(size_t pool_index) const {
    return reserveBlocksForPool(pool_index);
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

size_t CoordinatorCacheManager::totalReservableFreeBlocks() const {
    size_t      total  = 0;
    const auto& groups = config_.topology().groups();
    for (size_t group_id = 0; group_id < group_block_pools_.size(); ++group_id) {
        if (!group_block_pools_[group_id] || group_id >= kv_cache_groups_.size() || !kv_cache_groups_[group_id]
            || !kv_cache_groups_[group_id]->isReservable() || groups[group_id].policy.explicit_block_num > 0) {
            continue;
        }
        total += group_block_pools_[group_id]->freeBlocksNum();
    }
    return total;
}

size_t CoordinatorCacheManager::reservableFreeBlocksNum() const {
    return totalReservableFreeBlocks();
}

size_t CoordinatorCacheManager::reserveBlocksForPool(size_t group_id) const {
    const auto& groups = config_.topology().groups();
    if (group_id >= group_block_pools_.size() || group_id >= kv_cache_groups_.size() || !group_block_pools_[group_id]
        || !kv_cache_groups_[group_id] || !kv_cache_groups_[group_id]->isReservable()
        || groups[group_id].policy.explicit_block_num > 0) {
        return 0;
    }

    size_t total_reservable_blocks = 0;
    for (size_t current_group_id = 0; current_group_id < group_block_pools_.size(); ++current_group_id) {
        if (!group_block_pools_[current_group_id] || current_group_id >= kv_cache_groups_.size()
            || !kv_cache_groups_[current_group_id] || !kv_cache_groups_[current_group_id]->isReservable()
            || groups[current_group_id].policy.explicit_block_num > 0) {
            continue;
        }
        total_reservable_blocks += group_block_pools_[current_group_id]->totalBlocksNum();
    }
    return total_reservable_blocks == 0 ?
               0 :
               reserveBlocksNum() * group_block_pools_[group_id]->totalBlocksNum() / total_reservable_blocks;
}

MallocStatus CoordinatorCacheManager::evaluateInitCapacity(const MallocInfo& malloc_info,
                                                           size_t            reserve_blocks,
                                                           InitCapacityMode  mode) const {
    return evaluateInitCapacityImpl(malloc_info, reserve_blocks, mode, nullptr);
}

MallocStatus
CoordinatorCacheManager::evaluateInitCapacityImpl(const MallocInfo&                     malloc_info,
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
    {
        const auto& groups = config_.topology().groups();
        for (size_t group_id = 0; group_id < group_block_pools_.size(); ++group_id) {
            if (!group_block_pools_[group_id] || group_id >= kv_cache_groups_.size() || !kv_cache_groups_[group_id]
                || !kv_cache_groups_[group_id]->isReservable() || groups[group_id].policy.explicit_block_num > 0) {
                continue;
            }
            total_reservable_blocks += group_block_pools_[group_id]->totalBlocksNum();
        }
    }

    MallocStatus            status = MallocStatus::NONE;
    const RequiredPositions no_required_positions;
    const auto&             topology_groups = config_.topology().groups();
    for (int group_id = 0; group_id < static_cast<int>(kv_cache_groups_.size()); ++group_id) {
        const size_t group_index            = static_cast<size_t>(group_id);
        const auto&  tag                    = topology_groups[group_index].tag;
        const int    group_common_seq       = cpEffectiveSeqLenForReserve(cp_mapper, config_, tag, raw_common_seq_len);
        const int    group_seq_len          = cpEffectiveSeqLenForReserve(cp_mapper, config_, tag, raw_seq_len);
        const int    group_reuse_blocks_len = malloc_info.batch_kv_cache_resource->blocksNum(0, tag);
        const auto&  group_required_positions =
            required_positions == nullptr ? no_required_positions : (*required_positions)[group_index];
        const auto   need           = kv_cache_groups_[group_index]->getNeedBlocks(group_common_seq,
                                                                       group_seq_len,
                                                                       reserve_step,
                                                                       group_reuse_blocks_len,
                                                                       reuse_enabled,
                                                                       group_required_positions);
        const int    need_blocks    = need.common_blocks + batch_size * need.extra_blocks;
        const size_t planned_blocks = static_cast<size_t>(std::max(need_blocks, 0));

        const auto&  pool         = group_block_pools_[group_index];
        const size_t total_blocks = pool->totalBlocksNum();
        const auto   demand       = initBlockDemand(malloc_info, planned_blocks, tag);
        const size_t group_reserve_blocks =
            (!kv_cache_groups_[group_index]->isReservable()
             || topology_groups[group_index].policy.explicit_block_num > 0 || total_reservable_blocks == 0) ?
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
                                 topology_groups[group_index].tag.c_str(),
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

bool CoordinatorCacheManager::hasAvailableBlocksForReserve(const MallocInfo& malloc_info, size_t reserve_blocks) const {
    return evaluateInitCapacity(malloc_info, reserve_blocks, InitCapacityMode::TOTAL_AND_AVAILABLE)
           == MallocStatus::NONE;
}

MallocStatus CoordinatorCacheManager::evaluatePreparedInitCapacity(const MallocInfo&      malloc_info,
                                                                   size_t                 reserve_blocks,
                                                                   const PreparedKVCache& prepared,
                                                                   bool                   has_load_context) const {
    if (reserve_blocks == 0 && !has_load_context) {
        return MallocStatus::NONE;
    }
    if (!has_load_context) {
        return evaluateInitCapacity(malloc_info, reserve_blocks, InitCapacityMode::TOTAL_AND_AVAILABLE);
    }
    return evaluateInitCapacityImpl(
        malloc_info, reserve_blocks, InitCapacityMode::TOTAL_AND_AVAILABLE, &prepared.required_positions);
}

void CoordinatorCacheManager::logMallocFailure(const MallocInfo& malloc_info,
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

    RTP_LLM_LOG_WARNING(
        "CoordinatorCacheManager malloc failure: error_code=602 request_id=%ld phase=%s failed_batch=%d "
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

    const auto& topology_groups = config_.topology().groups();
    for (int gid = 0; gid < static_cast<int>(kv_cache_groups_.size()); ++gid) {
        const size_t group_index   = static_cast<size_t>(gid);
        const auto&  group         = topology_groups[group_index];
        const auto   group_type    = group.policy.group_type;
        const int    group_seq_len = cpEffectiveSeqLenForReserve(cp_mapper, config_, group.tag, planning_raw_seq_len);

        int    need_blocks          = 0;
        int    need_slots           = 0;
        size_t current_slots        = 0;
        size_t current_valid_blocks = 0;
        for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
            const auto& blocks = resource->blocks(batch_id, group.tag);
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
            const int  group_common_len = cpEffectiveSeqLenForReserve(cp_mapper, config_, group.tag, raw_common_len);
            const int  reuse_blocks_len = malloc_info.reuse_cache ? resource->blocksNum(0, group.tag) : 0;
            const auto need             = kv_cache_groups_[group_index]->getNeedBlocks(
                group_common_len, group_seq_len, reserve_step, reuse_blocks_len, malloc_info.reuse_cache);
            need_blocks = need.common_blocks + batch_size * need.extra_blocks;
        }
        if (gid == failed_group && failed_need_blocks >= 0) {
            need_blocks = failed_need_blocks;
        }

        const auto&     pool               = group_block_pools_[group_index];
        const size_t    free_blocks        = pool->freeBlocksNum();
        const size_t    group_reserve      = reserve_admission ? reserveBlocksForPool(group_index) : 0;
        const long long required_available = need_blocks < 0 ? -1 : static_cast<long long>(need_blocks + group_reserve);
        const long long shortfall =
            required_available < 0 ? -1 : std::max(required_available - static_cast<long long>(free_blocks), 0LL);

        RTP_LLM_LOG_WARNING(
            "CoordinatorCacheManager malloc failure pool: error_code=602 request_id=%ld gid=%d pool_name=%s "
            "group_type=%s tag=%s failed=%d need_blocks=%d need_slots=%d "
            "group_reserve_blocks=%zu required_available_blocks=%lld shortfall_blocks=%lld "
            "current_slots=%zu "
            "current_valid_blocks=%zu total_blocks=%zu free_blocks=%zu "
            "request_ref_blocks=%zu block_cache_ref_blocks=%zu load_ref_blocks=%zu "
            "layer_count=%zu block_bytes=%zu seq_size_per_block=%zu",
            malloc_info.request_id,
            gid,
            pool->poolName().c_str(),
            cacheGroupTypeName(group_type),
            group.tag.c_str(),
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
            pool->referencedBlocksNum(),
            pool->referencedBlocksNum(BlockTreeRefType::CACHE),
            pool->referencedBlocksNum(BlockTreeRefType::LOAD),
            config_.layerIdsForGroup(group.tag).size(),
            config_.blockSizeBytesForGroup(group.tag),
            group.seqSizePerBlock());
    }
}

}  // namespace rtp_llm
