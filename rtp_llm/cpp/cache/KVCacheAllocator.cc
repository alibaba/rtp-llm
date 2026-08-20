#include <algorithm>
#include <cstdint>
#include <limits>
#include <unordered_set>
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/core/OpData.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache/KVCacheGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadAsyncContext.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"

namespace rtp_llm {

bool KVCacheAllocator::init() {
    RTP_LLM_CHECK_WITH_INFO(doInit(), "init failed");

    const int64_t reserve_ratio = reserve_block_ratio_;
    if (reserve_ratio > 0) {
        const size_t reservable_blocks = reservableFreeBlocksNum();
        const size_t reserve_blocks = static_cast<size_t>(reserve_ratio) * reservable_blocks / static_cast<size_t>(100);
        reserve_block_num_          = reserve_blocks;
        RTP_LLM_LOG_INFO(
            "KVCacheAllocator set reserve blocks: ratio=%ld%% reserve_blocks=%zu reservable_free_blocks=%zu",
            reserve_ratio,
            reserve_blocks,
            reservable_blocks);
    } else {
        reserve_block_num_ = 0;
    }

    return true;
}

MallocResult KVCacheAllocator::initMalloc(const MallocInfo& malloc_info) {
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
    if (load_context && load_context->needBackendMatch()) {
        load_context->startBackendMatch();
    } else {
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
                init_result.success        = false;
                init_result.reuse_len      = 0;
                init_result.host_reuse_len = 0;
                init_result.disk_reuse_len = 0;
                init_result.async_context  = nullptr;
                return finalize_init_failure(std::move(init_result));
            }
            init_result.async_context = std::move(pending_async_context);
        }
    }

    return init_result;
}

MallocStatus KVCacheAllocator::evaluateInitCapacity(const MallocInfo& malloc_info,
                                                    size_t            reserve_blocks,
                                                    InitCapacityMode  mode) const {
    const size_t planned_blocks = static_cast<size_t>(std::max(getNeedBlocks(malloc_info), 0));
    const auto   demand         = initBlockDemand(malloc_info, planned_blocks);
    auto         fits_total     = [demand, planned_blocks, reserve_blocks](size_t capacity) {
        return demand.retained_blocks <= capacity && planned_blocks <= capacity - demand.retained_blocks
               && reserve_blocks <= capacity - demand.retained_blocks - planned_blocks;
    };
    if (!fits_total(totalBlocksNum())) {
        return MallocStatus::PERMANENT_RESOURCE_EXHAUSTED;
    }
    auto fits_available = [demand, reserve_blocks](size_t capacity) {
        return demand.additional_blocks <= capacity && reserve_blocks <= capacity - demand.additional_blocks;
    };
    if (mode == InitCapacityMode::TOTAL_AND_AVAILABLE && !fits_available(availableBlocksNum())) {
        return MallocStatus::RETRYABLE_RESOURCE_EXHAUSTED;
    }
    return MallocStatus::NONE;
}

size_t KVCacheAllocator::heldRequestBlocks(const MallocInfo& malloc_info, int group_id) {
    const auto& resource = malloc_info.batch_kv_cache_resource;
    if (!resource) {
        return 0;
    }
    RTP_LLM_CHECK_WITH_INFO(group_id >= -1 && group_id < resource->groupNums(),
                            "held-block group out of range: group_id=%d groups=%d",
                            group_id,
                            resource->groupNums());
    const int group_begin = group_id < 0 ? 0 : group_id;
    const int group_end   = group_id < 0 ? resource->groupNums() : group_id + 1;
    std::unordered_set<BlockIdxType> held_blocks;
    for (int batch = 0; batch < resource->batchSize(); ++batch) {
        for (int group = group_begin; group < group_end; ++group) {
            for (const BlockIdxType block : resource->blocks(batch, group)) {
                if (block > 0 && !isNullBlockIdx(block)) {
                    held_blocks.insert(block);
                }
            }
        }
    }
    return held_blocks.size();
}

KVCacheAllocator::InitBlockDemand
KVCacheAllocator::initBlockDemand(const MallocInfo& malloc_info, size_t planned_blocks, int group_id) {
    const size_t held_blocks = heldRequestBlocks(malloc_info, group_id);
    if (malloc_info.reuse_cache) {
        return {held_blocks, planned_blocks};
    }
    // No-reuse planners already describe the full request. Account only for
    // the portion not represented by the current partial allocation.
    return held_blocks > planned_blocks ? InitBlockDemand{held_blocks - planned_blocks, 0} :
                                          InitBlockDemand{0, planned_blocks - held_blocks};
}

MallocResult KVCacheAllocator::malloc(const MallocInfo& malloc_info) {
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

int KVCacheAllocator::estimateBatchPeakNeedBlocks(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
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

void KVCacheAllocator::attachBlockTreeCache(BlockTreeCachePtr block_tree_cache) {
    RTP_LLM_CHECK_WITH_INFO(block_tree_cache != nullptr, "cannot attach a null BlockTreeCache");
    RTP_LLM_CHECK_WITH_INFO(block_tree_cache_ == nullptr, "BlockTreeCache has already been attached");

    block_tree_cache_ = std::move(block_tree_cache);
    for (const auto& group : cacheGroups()) {
        const size_t group_id = static_cast<size_t>(group->group_id());
        group->setEvictCallback([cache = block_tree_cache_, group_id](size_t need_blocks) {
            const int reclaimed = cache->evictForGroup(group_id, need_blocks);
            return reclaimed > 0 ? static_cast<size_t>(reclaimed) : 0;
        });
    }
}

bool KVCacheAllocator::abortPendingLoad(const std::shared_ptr<AsyncContext>& context) {
    return block_tree_cache_ != nullptr && block_tree_cache_->abortPendingLoad(context);
}

uint32_t KVCacheAllocator::convertToGlobalLayerId(size_t model_id, int local_layer_id) const {
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

BlockAddrInfo KVCacheAllocator::convertIndexToAddr(int layer_id, int group_id, int block_id) const {
    RTP_LLM_CHECK_WITH_INFO(group_id >= 0, "invalid cache topology group id=%d", group_id);
    return convertIndexToAddrByTag(layer_id, config_.topology().groupById(static_cast<size_t>(group_id)).tag, block_id);
}

std::vector<BlockInfo> KVCacheAllocator::convertIndexToBuffer(int layer_id, int group_id, int block_id) const {
    RTP_LLM_CHECK_WITH_INFO(group_id >= 0, "invalid cache topology group id=%d", group_id);
    return convertIndexToBufferByTag(
        layer_id, config_.topology().groupById(static_cast<size_t>(group_id)).tag, block_id);
}

std::vector<BlockInfo> KVCacheAllocator::convertIndexToBuffer(
    int layer_id, int group_id, int block_id, int partition_count, int partition_id) const {
    RTP_LLM_CHECK_WITH_INFO(group_id >= 0, "invalid cache topology group id=%d", group_id);
    return convertIndexToBufferByTag(layer_id,
                                     config_.topology().groupById(static_cast<size_t>(group_id)).tag,
                                     block_id,
                                     partition_count,
                                     partition_id);
}

BlockAddrInfo KVCacheAllocator::convertIndexToAddrByTag(int layer_id, const std::string& tag, int block_id) const {
    (void)config_.groupForLayer(layer_id, tag);
    return convertIndexToAddr(layer_id, block_id);
}

std::vector<BlockInfo>
KVCacheAllocator::convertIndexToBufferByTag(int layer_id, const std::string& tag, int block_id) const {
    (void)config_.groupForLayer(layer_id, tag);
    return convertIndexToBuffer(layer_id, block_id);
}

std::vector<BlockInfo> KVCacheAllocator::convertIndexToBufferByTag(
    int layer_id, const std::string& tag, int block_id, int partition_count, int partition_id) const {
    (void)config_.groupForLayer(layer_id, tag);
    return convertIndexToBuffer(layer_id, block_id, partition_count, partition_id);
}

void KVCacheAllocator::blockCopy(int src_block_index, int dest_block_index) {
    BlockIdPair copy_mapping{src_block_index, dest_block_index};
    blockBatchCopy(&copy_mapping, &copy_mapping + 1);
}

void KVCacheAllocator::blockBatchCopy(const std::vector<BlockIdPair>& copy_mapping) {
    blockBatchCopy(copy_mapping.data(), copy_mapping.data() + copy_mapping.size());
}

void KVCacheAllocator::blockBatchCopy(const torch::Tensor& copy_mapping) {
    RTP_LLM_CHECK_WITH_INFO(copy_mapping.device().is_cpu() && copy_mapping.scalar_type() == torch::kInt32
                                && copy_mapping.is_contiguous() && copy_mapping.dim() == 2,
                            "cache block copy mapping must be a contiguous CPU int32 matrix");
    if (copy_mapping.size(1) == 2) {
        const auto* begin_ptr = reinterpret_cast<const BlockIdPair*>(copy_mapping.data_ptr());
        blockBatchCopy(begin_ptr, begin_ptr + copy_mapping.size(0));
        return;
    }
    RTP_LLM_CHECK_WITH_INFO(copy_mapping.size(1) == 3,
                            "cache block copy mapping must have 2 legacy columns or 3 tagged columns, got %ld",
                            copy_mapping.size(1));
    const auto*                    mappings = reinterpret_cast<const GroupBlockIdPair*>(copy_mapping.data_ptr());
    std::vector<TaggedBlockIdPair> tagged_mappings;
    tagged_mappings.reserve(static_cast<size_t>(copy_mapping.size(0)));
    for (int64_t i = 0; i < copy_mapping.size(0); ++i) {
        RTP_LLM_CHECK_WITH_INFO(
            mappings[i].group_id >= 0, "cache block copy mapping has invalid group_id=%d", mappings[i].group_id);
        tagged_mappings.push_back({config_.topology().groupById(static_cast<size_t>(mappings[i].group_id)).tag,
                                   mappings[i].src,
                                   mappings[i].dst});
    }
    blockBatchCopyByTag(tagged_mappings);
}

void KVCacheAllocator::blockBatchCopy(const BlockIdPair* begin_ptr, const BlockIdPair* end_ptr) {
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

void KVCacheAllocator::blockBatchCopyByTag(const std::vector<TaggedBlockIdPair>& copy_mapping) {
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
            const auto src_addr = convertIndexToAddrByTag(layer_id, mapping.tag, mapping.src);
            const auto dst_addr = convertIndexToAddrByTag(layer_id, mapping.tag, mapping.dst);
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

size_t KVCacheAllocator::freeBlocksNum() const {
    return block_pool_ ? block_pool_->freeBlocksNum() : 0;
}

size_t KVCacheAllocator::availableBlocksNum() const {
    return freeBlocksNum() + activeTreeCachedBlocksNum();
}

size_t KVCacheAllocator::reservableFreeBlocksNum() const {
    return freeBlocksNum();
}

int64_t KVCacheAllocator::getMrCostTimeMs() const {
    return block_pool_ ? block_pool_->getMrCostTimeMs() : 0;
}

size_t KVCacheAllocator::activeTreeCachedBlocksNum() const {
    return block_pool_ ? block_pool_->activeTreeCachedBlocksNum() : 0;
}

size_t KVCacheAllocator::availableTokensNum() const {
    return block_pool_ ? (block_pool_->freeBlocksNum() * logicalSeqSizePerBlockForCapacity(/*gid=*/0)) : 0;
}

size_t KVCacheAllocator::totalTokensNum() const {
    return block_pool_ ? (block_pool_->totalBlocksNum() * logicalSeqSizePerBlockForCapacity(/*gid=*/0)) : 0;
}

size_t KVCacheAllocator::totalBlocksNum() const {
    return block_pool_ ? block_pool_->totalBlocksNum() : 0;
}

size_t KVCacheAllocator::maxAvailableTokensNum() const {
    return totalTokensNum();
}

bool KVCacheAllocator::cpShardThisGroupForCapacity(size_t gid) const {
    return cp_slot_mapper_ && cp_slot_mapper_->isSharded() && cp_slot_mapper_->blockRoundRobinGroup(config_, gid);
}

size_t KVCacheAllocator::logicalSeqSizePerBlockForCapacity(size_t gid) const {
    if (cp_slot_mapper_ && cp_slot_mapper_->isSharded()) {
        return cp_slot_mapper_->logicalSeqSizePerBlock(config_, gid);
    }
    return config_.seqSizePerBlockForGroup(gid);
}

int KVCacheAllocator::cpEffectiveSeqLenForAlloc(size_t gid, int seq_len) const {
    return (cp_slot_mapper_ && cp_slot_mapper_->isSharded()) ?
               cp_slot_mapper_->effectiveSeqLenForAlloc(config_, gid, seq_len) :
               seq_len;
}

int KVCacheAllocator::deviceCacheMetricTokensPerBlock() const {
    if (cp_slot_mapper_ && cp_slot_mapper_->isSharded()) {
        return cp_slot_mapper_->virtualBlockSize();
    }
    return seqSizePerBlock();
}

KVCacheTokenCapacity KVCacheAllocator::tokenCapacity(size_t default_seq_size_per_block) const {
    const size_t total_blocks     = totalBlocksNum();
    const size_t available_blocks = freeBlocksNum();
    return {total_blocks * default_seq_size_per_block, available_blocks * default_seq_size_per_block};
}

size_t KVCacheAllocator::reserveBlocksForPoolMetrics(size_t pool_index) const {
    (void)pool_index;
    return reserveBlocksNum();
}

std::vector<KVCachePoolMetricsSnapshot> KVCacheAllocator::poolMetricsSnapshots() const {
    const std::vector<KVCacheGroupPtr>      groups = cacheGroups();
    std::unordered_set<const IBlockPool*>   reported_pools;
    std::vector<KVCachePoolMetricsSnapshot> snapshots;
    snapshots.reserve(groups.size());
    for (const KVCacheGroupPtr& group : groups) {
        if (group == nullptr || group->blockPool() == nullptr) {
            continue;
        }
        const DeviceBlockPoolPtr                                               pool = group->blockPool();
        const std::pair<std::unordered_set<const IBlockPool*>::iterator, bool> insert_result =
            reported_pools.insert(pool.get());
        if (!insert_result.second) {
            continue;
        }

        const size_t               pool_index = static_cast<size_t>(group->group_id());
        KVCachePoolMetricsSnapshot snapshot;
        snapshot.pool_index                = pool_index;
        snapshot.pool_name                 = pool->poolName();
        snapshot.block_size_bytes          = pool->blockSizeBytes();
        snapshot.total_blocks              = pool->totalBlocksNum();
        snapshot.free_blocks               = pool->freeBlocksNum();
        snapshot.used_blocks               = snapshot.total_blocks - snapshot.free_blocks;
        snapshot.active_tree_cached_blocks = pool->activeTreeCachedBlocksNum();
        snapshot.reserve_blocks            = reserveBlocksForPoolMetrics(pool_index);
        snapshot.request_ref_blocks        = pool->referencedBlocksNum();
        snapshot.block_cache_ref_blocks    = pool->referencedBlocksNum(BlockTreeRefType::CACHE);
        snapshot.load_ref_blocks           = pool->referencedBlocksNum(BlockTreeRefType::LOAD);
        snapshot.eviction_ref_blocks       = pool->referencedBlocksNum(BlockTreeRefType::EVICTION);
        snapshot.store_ref_blocks          = pool->referencedBlocksNum(BlockTreeRefType::STORE);
        snapshot.used_ratio =
            snapshot.total_blocks == 0 ?
                0.0f :
                static_cast<float>(100.0 * snapshot.used_blocks / static_cast<double>(snapshot.total_blocks));
        snapshots.push_back(std::move(snapshot));
    }
    return snapshots;
}

void KVCacheAllocator::regUserMr(size_t model_id, std::shared_ptr<CacheStore> cache_store) {
    if (block_pool_) {
        block_pool_->regUserMr(model_id, std::move(cache_store));
    }
}

}  // namespace rtp_llm
