#include "rtp_llm/cpp/cache/HybridKVCacheAllocator.h"

#include <algorithm>
#include <limits>
#include <unordered_map>
#include <unordered_set>

#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadAsyncContext.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/core/OpData.h"

namespace rtp_llm {
namespace {

// CP shard helpers: when mapper is null/passthrough, all helpers no-op.
inline CacheKeysType cpCanonicalCacheKeys(const std::shared_ptr<CPSlotMapper>& mapper, const CacheKeysType& full) {
    return (mapper && mapper->isSharded()) ? mapper->canonicalCacheKeys(full) : full;
}

inline bool
cpBlockRoundRobinGroup(const std::shared_ptr<CPSlotMapper>& mapper, const CacheConfig& config, int group_id) {
    return mapper && mapper->isSharded() && group_id >= 0
           && mapper->blockRoundRobinGroup(config, static_cast<size_t>(group_id));
}

inline int cpEffectiveSeqLenForGroup(const std::shared_ptr<CPSlotMapper>& mapper,
                                     const CacheConfig&                   config,
                                     int                                  group_id,
                                     int                                  seq_len) {
    return cpBlockRoundRobinGroup(mapper, config, group_id) ?
               mapper->effectiveSeqLenForAlloc(config, static_cast<size_t>(group_id), seq_len) :
               seq_len;
}

}  // namespace

bool HybridKVCacheAllocator::cpCompactSwaGroup(size_t group_id, const std::shared_ptr<CPSlotMapper>& mapper) const {
    return mapper && mapper->isSharded() && group_id < kv_cache_groups_.size()
           && mapper->compactLastRankGroup(config_, group_id);
}

size_t HybridKVCacheAllocator::loadTargetPosition(size_t                               path_index,
                                                  size_t                               group_id,
                                                  const std::shared_ptr<CPSlotMapper>& mapper,
                                                  int                                  cp_scale) const {
    const CacheGroupType type = config_.typeForGroup(group_id);
    return type == CacheGroupType::LINEAR || (type == CacheGroupType::SWA && !cpCompactSwaGroup(group_id, mapper)) ?
               (path_index + 1) * static_cast<size_t>(cp_scale) - 1 :
               path_index;
}

HybridKVCacheAllocator::HybridKVCacheAllocator(const CacheConfig&                 config,
                                               AllocationType                     allocation_type,
                                               const kmonitor::MetricsReporterPtr metrics_reporter,
                                               int64_t                            reserve_block_ratio):
    KVCacheAllocator(config, allocation_type, metrics_reporter, reserve_block_ratio) {}

std::shared_ptr<LoadAsyncContext> HybridKVCacheAllocator::prepareKVCache(const CacheKeysType&  cache_keys,
                                                                         BatchKVCacheResource& kv_resource,
                                                                         const std::shared_ptr<CPSlotMapper>& cp_mapper,
                                                                         PreparedKVCache& prepared) {
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
            for (const size_t group_id : group_set->groupIds()) {
                const size_t reuse_size =
                    loadTargetPosition(prepared.total_logical_blocks - 1, group_id, cp_mapper, cp_scale) + 1;
                kv_resource.mutableBlockIds(0, static_cast<int>(group_id))
                    .assign(BlockIndicesType(reuse_size, NULL_BLOCK_IDX));
            }
        }
    }

    const auto set_group_set_blocks = [&](size_t group_set_id, size_t path_index, const BlockIndicesType& blocks) {
        const auto& group_ids = group_sets[group_set_id]->groupIds();
        for (size_t member_group_id = 0; member_group_id < group_ids.size(); ++member_group_id) {
            const size_t group_id        = group_ids[member_group_id];
            const size_t target_position = loadTargetPosition(path_index, group_id, cp_mapper, cp_scale);
            kv_resource.mutableBlockIds(0, static_cast<int>(group_id)).setAt(target_position, blocks[member_group_id]);
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

    for (int group_id = 0; group_id < kv_resource.groupNums(); ++group_id) {
        prepared.original_sizes.push_back(kv_resource.blocksNum(0, group_id));
    }
    return load_context;
}

MallocResult HybridKVCacheAllocator::initMallocForCommonLen(const MallocInfo& malloc_info) {
    auto&       kv_resource       = malloc_info.batch_kv_cache_resource;
    const int   seq_len           = malloc_info.complete_token_ids->seqLength();
    const auto& cp_mapper         = cp_slot_mapper_;
    const int   reuse_unit_tokens = cp_mapper ? cp_mapper->reuseBlockTokens(config_) : seqSizePerBlock();

    const CacheKeysType& cache_keys         = kv_resource->cacheKeys(0);
    int64_t              match_cost_time_us = 0;
    int64_t              match_end_time_us  = 0;
    PreparedKVCache      prepared;
    bool                 load_attempted = false;
    prepared.referenced_blocks.resize(static_cast<size_t>(kv_resource->groupNums()));
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

bool HybridKVCacheAllocator::materializeInitialBlocks(const MallocInfo& malloc_info,
                                                      PreparedKVCache&  prepared,
                                                      LoadAsyncContext* context,
                                                      size_t            matched_blocks) {
    auto&       kv_resource = *malloc_info.batch_kv_cache_resource;
    const auto& cp_mapper   = cp_slot_mapper_;
    const int   cp_scale    = cp_mapper && cp_mapper->isSharded() ? cp_mapper->cpSize() : 1;
    prepared.required_positions.assign(static_cast<size_t>(kv_resource.groupNums()), {});

    if (matched_blocks > 0) {
        for (const GroupSetPtr& group_set : block_tree_cache_->groupSets()) {
            for (size_t group_id : group_set->groupIds()) {
                kv_resource.mutableBlockIds(0, static_cast<int>(group_id))
                    .resize(loadTargetPosition(matched_blocks - 1, group_id, cp_mapper, cp_scale) + 1);
            }
        }
    }
    auto add_target = [&](size_t path, size_t group_id) {
        prepared.required_positions[group_id].insert(loadTargetPosition(path, group_id, cp_mapper, cp_scale));
    };
    if (context != nullptr) {
        for (size_t i = 0; i < context->loadDescs().size(); ++i) {
            const auto& desc = context->loadDescs()[i];
            if (desc.source_tier != Tier::DEVICE && !context->joinedLoads()[i]) {
                for (size_t group_id : block_tree_cache_->groupSets()[desc.group_set_id]->groupIds()) {
                    add_target(desc.path_index, group_id);
                }
            }
        }
        const auto& backend_handles = context->backendHandles();
        for (size_t key_index = 0; key_index < backend_handles.size(); ++key_index) {
            for (const StorageBlockHandle& handle : backend_handles[key_index]) {
                add_target(key_index, handle.group_id);
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
    for (int group_id = 0; group_id < kv_resource.groupNums(); ++group_id) {
        auto&               block_ids = kv_resource.mutableBlockIds(0, group_id);
        std::vector<size_t> backfilled_positions;
        if (!kv_cache_groups_[static_cast<size_t>(group_id)]->malloc(
                block_ids,
                cpEffectiveSeqLenForGroup(cp_mapper, config_, group_id, common_seq_len),
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
        for (size_t group_id : block_tree_cache_->groupSets()[group_set_id]->groupIds()) {
            blocks.push_back(kv_resource.blocks(
                0, static_cast<int>(group_id))[loadTargetPosition(path, group_id, cp_mapper, cp_scale)]);
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
                    kv_resource.blocks(
                        0,
                        static_cast<int>(
                            handle.group_id))[loadTargetPosition(key_index, handle.group_id, cp_mapper, cp_scale)]);
            }
        }
    }
    for (int batch = 1; batch < kv_resource.batchSize(); ++batch) {
        for (int group_id = 0; group_id < kv_resource.groupNums(); ++group_id) {
            kv_cache_groups_[static_cast<size_t>(group_id)]->reference(kv_resource.mutableBlockIds(batch, group_id),
                                                                       kv_resource.blocks(0, group_id));
        }
    }
    return true;
}

bool HybridKVCacheAllocator::finishDeferredMalloc(const MallocInfo& malloc_info,
                                                  PreparedKVCache&  prepared,
                                                  LoadAsyncContext& context,
                                                  size_t            matched_blocks) {
    bool success = materializeInitialBlocks(malloc_info, prepared, &context, matched_blocks);
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

MallocResult HybridKVCacheAllocator::incrMalloc(const MallocInfo& malloc_info) {
    auto&       kv_resource  = malloc_info.batch_kv_cache_resource;
    const auto& cp_mapper    = cp_slot_mapper_;
    const int   batch_size   = kv_resource->batchSize();
    const int   raw_seq_len  = malloc_info.incrSeqLen();
    const int   reserve_step = malloc_info.complete_token_ids->getReserveStep();

    std::vector<std::vector<size_t>>              original_sizes(static_cast<size_t>(batch_size));
    std::vector<std::vector<std::vector<size_t>>> backfilled_positions(static_cast<size_t>(batch_size));
    for (int b = 0; b < batch_size; ++b) {
        original_sizes[static_cast<size_t>(b)].resize(static_cast<size_t>(kv_resource->groupNums()));
        backfilled_positions[static_cast<size_t>(b)].resize(static_cast<size_t>(kv_resource->groupNums()));
        for (int group_id = 0; group_id < kv_resource->groupNums(); ++group_id) {
            original_sizes[static_cast<size_t>(b)][static_cast<size_t>(group_id)] = kv_resource->blocksNum(b, group_id);
        }
    }

    bool all_success        = true;
    int  failed_batch       = -1;
    int  failed_group       = -1;
    int  failed_need_blocks = -1;
    for (int b = 0; b < batch_size; ++b) {
        for (int group_id = 0; group_id < kv_resource->groupNums(); ++group_id) {
            auto&     block_ids        = kv_resource->mutableBlockIds(b, group_id);
            const int group_seq_len    = cpEffectiveSeqLenForGroup(cp_mapper, config_, group_id, raw_seq_len);
            auto&     filled_positions = backfilled_positions[static_cast<size_t>(b)][static_cast<size_t>(group_id)];
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
            for (int group_id = 0; group_id < kv_resource->groupNums(); ++group_id) {
                kv_cache_groups_[static_cast<size_t>(group_id)]->removeSkippedBlocks(
                    kv_resource->mutableBlockIds(b, group_id), malloc_info.reuse_cache, reserve_step);
            }
        }
        return {true, 0};
    }

    // Emit the pool snapshot before rolling back: once the partially allocated
    // blocks go back to the pools, available_blocks no longer reflects the state
    // that caused the failure.
    logMallocFailure(malloc_info, "incremental_group_malloc", failed_batch, failed_group, true, failed_need_blocks);

    for (int b = 0; b <= failed_batch && b < batch_size; ++b) {
        for (int group_id = 0; group_id < kv_resource->groupNums(); ++group_id) {
            auto&       block_ids        = kv_resource->mutableBlockIds(b, group_id);
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

void HybridKVCacheAllocator::free(const FreeInfo& free_info) {
    auto& kv_cache_resource = free_info.batch_kv_cache_resource;
    if (kv_cache_resource->curBlocksNum() == 0) {
        return;
    }
    for (int batch_id = 0; batch_id < kv_cache_resource->batchSize(); ++batch_id) {
        for (int group_id = 0; group_id < kv_cache_resource->groupNums(); ++group_id) {
            kv_cache_groups_[static_cast<size_t>(group_id)]->unreference(kv_cache_resource->blocks(batch_id, group_id));
        }
    }
    kv_cache_resource->clearBlocks();
}

void HybridKVCacheAllocator::insertIntoCache(const InsertInfo& insert_info) {
    auto& kv_cache_resource = insert_info.batch_kv_cache_resource;
    RTP_LLM_CHECK(kv_cache_resource != nullptr);
    if (!block_tree_cache_) {
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
            if (!group_set || group_set->groupSetId() != group_set_id || group_set->groupIds().empty()
                || group_set->groupIds().size() != group_set->devicePools().size()) {
                mapping_valid = false;
                break;
            }
            for (auto& per_key_resources : resources) {
                per_key_resources[group_set_id].device_blocks.assign(group_set->devicePools().size(), NULL_BLOCK_IDX);
            }
            for (size_t member_group_id = 0; member_group_id < group_set->groupIds().size(); ++member_group_id) {
                const int group_id = static_cast<int>(group_set->groupIds()[member_group_id]);
                if (!kv_cache_groups_[static_cast<size_t>(group_id)]->prefixReuseEnabled()) {
                    mapping_valid = false;
                    break;
                }
                const auto type = config_.typeForGroup(static_cast<size_t>(group_id));
                const bool sparse_logical =
                    cp_active
                    && (type == CacheGroupType::LINEAR
                        || (type == CacheGroupType::SWA && !cpCompactSwaGroup(group_id, cp_mapper)));
                const auto& blocks = kv_cache_resource->blocks(batch_id, group_id);
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
        block_tree_cache_->insert(insert_keys, resources, insert_info.target_tier, insert_info.write_remote);
    }
}

std::shared_ptr<KVCacheResource> HybridKVCacheAllocator::incrKVCacheRef(const KVCacheResource& kvcache_resource,
                                                                        const CacheKeysType&   cache_keys,
                                                                        bool                   is_connector) {
    if (cache_keys.empty() || kvcache_resource.groupNums() <= 0) {
        return nullptr;
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
    std::vector<BlockIndicesType> selected_blocks(static_cast<size_t>(kvcache_resource.groupNums()));
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
        std::vector<BlockIdxType> blocks_for_key(static_cast<size_t>(kvcache_resource.groupNums()), NULL_BLOCK_IDX);
        for (int group_id = 0; group_id < kvcache_resource.groupNums(); ++group_id) {
            const auto& src_blocks                        = kvcache_resource.blocks(group_id);
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
        for (int group_id = 0; group_id < kvcache_resource.groupNums(); ++group_id) {
            selected_blocks[static_cast<size_t>(group_id)].push_back(blocks_for_key[static_cast<size_t>(group_id)]);
        }
    }

    if (selected_keys.empty()) {
        return nullptr;
    }

    selected_resource->cacheKeys() = std::move(selected_keys);
    selected_resource->setBlockDependencies(std::move(selected_dependencies));
    for (int group_id = 0; group_id < kvcache_resource.groupNums(); ++group_id) {
        BlockIndicesType valid;
        for (auto b : selected_blocks[static_cast<size_t>(group_id)]) {
            if (!isNullBlockIdx(b) && b > 0) {
                valid.push_back(b);
            }
        }
        kv_cache_groups_[static_cast<size_t>(group_id)]->reference(valid);
        selected_resource->mutableBlockIds(group_id).assign(std::move(selected_blocks[static_cast<size_t>(group_id)]));
    }
    return selected_resource;
}

void HybridKVCacheAllocator::decrKVCacheRef(const KVCacheResource& kvcache_resource) {
    for (int group_id = 0; group_id < kvcache_resource.groupNums(); ++group_id) {
        kv_cache_groups_[static_cast<size_t>(group_id)]->unreference(kvcache_resource.blocks(group_id));
    }
}

bool HybridKVCacheAllocator::updateKVBlock(const BatchKVCacheResourcePtr&  batch_kv_cache_resource,
                                           const std::vector<int>&         block_src_batch,
                                           bool                            copy_last_block,
                                           std::vector<TaggedBlockIdPair>& block_update_mapping) {
    block_update_mapping.clear();
    if (block_src_batch.empty()) {
        return true;
    }
    const int old_batch_size = batch_kv_cache_resource->batchSize();
    const int new_batch_size = static_cast<int>(block_src_batch.size());
    const int group_nums     = batch_kv_cache_resource->groupNums();

    std::vector<int> batch_fork_count(old_batch_size, 0);
    for (const int old_batch_idx : block_src_batch) {
        RTP_LLM_CHECK_WITH_INFO(old_batch_idx >= 0 && old_batch_idx < old_batch_size,
                                "try to reuse an old batch %d that out of range %d",
                                old_batch_idx,
                                old_batch_size);
        ++batch_fork_count[old_batch_idx];
    }

    std::vector<int> new_blocks_num(static_cast<size_t>(group_nums), 0);
    for (int old_batch_idx = 0; old_batch_idx < old_batch_size; ++old_batch_idx) {
        const int fork_count = batch_fork_count[old_batch_idx];
        if (fork_count > 1 && copy_last_block) {
            for (int group_id = 0; group_id < group_nums; ++group_id) {
                if (!batch_kv_cache_resource->blocks(old_batch_idx, group_id).empty()) {
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
        std::unordered_set<BlockIdxType>      retained_blocks;
        std::unordered_map<BlockIdxType, int> dropped_block_counts;
        for (int old_batch_idx = 0; old_batch_idx < old_batch_size; ++old_batch_idx) {
            for (const auto block : batch_kv_cache_resource->blocks(old_batch_idx, group_id)) {
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
            const auto& dropped = batch_kv_cache_resource->blocks(old_batch_idx, group_id);
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
            BlockIndicesType to_free;
            auto&            transferred = transferred_ref_counts[static_cast<size_t>(group_id)];
            for (const auto block : batch_kv_cache_resource->blocks(old_batch_idx, group_id)) {
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
                auto& block_ids = batch_kv_cache_resource->mutableBlockIds(new_batch_idx, group_id);
                kv_cache_groups_[static_cast<size_t>(group_id)]->reference(
                    block_ids, old_resources[old_batch_idx].blocks(group_id));

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
                        block_update_mapping.push_back(
                            {config_.topology().groupById(static_cast<size_t>(group_id)).tag, old_block, new_block});
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

int HybridKVCacheAllocator::seqSizePerBlock() const {
    return static_cast<int>(config_.seq_size_per_block);
}

bool HybridKVCacheAllocator::hasAvailableBlocksForReserve(const MallocInfo& malloc_info, size_t reserve_blocks) const {
    const int need_blocks = getNeedBlocks(malloc_info);
    if (need_blocks <= 0) {
        return true;
    }
    const size_t available_blocks = freeBlocksNum();
    const bool   accepted         = available_blocks >= static_cast<size_t>(need_blocks) + reserve_blocks;
    if (!accepted && malloc_info.verbose) {
        RTP_LLM_LOG_INFO("Hybrid initMalloc rejected by reserve blocks: request_id=%ld "
                         "need_blocks=%d available_blocks=%zu reserve_blocks=%zu",
                         malloc_info.request_id,
                         need_blocks,
                         available_blocks,
                         reserve_blocks);
    }
    return accepted;
}

// Primary field-debug record for KV-exhaustion incidents. HybridPool overrides
// it with a per-pool breakdown; this base version reports the aggregate view.
void HybridKVCacheAllocator::logMallocFailure(const MallocInfo& malloc_info,
                                              const char*       phase,
                                              int               failed_batch,
                                              int               failed_group,
                                              bool              incremental,
                                              int               failed_need_blocks) const {
    if (!malloc_info.verbose) {
        return;
    }
    RTP_LLM_LOG_WARNING("Hybrid malloc failure: error_code=602 request_id=%ld phase=%s failed_batch=%d failed_group=%d "
                        "incremental=%d failed_need_blocks=%d need_blocks=%d free_blocks=%zu reserve_blocks=%zu",
                        malloc_info.request_id,
                        phase,
                        failed_batch,
                        failed_group,
                        incremental,
                        failed_need_blocks,
                        getNeedBlocks(malloc_info),
                        freeBlocksNum(),
                        reserveBlocksNum());
}

MallocStatus HybridKVCacheAllocator::evaluatePreparedInitCapacity(const MallocInfo&      malloc_info,
                                                                  size_t                 reserve_blocks,
                                                                  const PreparedKVCache& prepared,
                                                                  bool                   has_load_context) const {
    if (reserve_blocks == 0 && !has_load_context) {
        return MallocStatus::NONE;
    }
    if (!has_load_context) {
        if (hasAvailableBlocksForReserve(malloc_info, reserve_blocks)) {
            return MallocStatus::NONE;
        }
        return evaluateInitCapacity(malloc_info, reserve_blocks, InitCapacityMode::TOTAL_ONLY)
                       == MallocStatus::PERMANENT_RESOURCE_EXHAUSTED ?
                   MallocStatus::PERMANENT_RESOURCE_EXHAUSTED :
                   MallocStatus::RETRYABLE_RESOURCE_EXHAUSTED;
    }
    RTP_LLM_CHECK_WITH_INFO(malloc_info.batch_kv_cache_resource && malloc_info.complete_token_ids
                                && prepared.required_positions.size() == kv_cache_groups_.size(),
                            "prepared shared-pool capacity input mismatch");
    const int batch_size         = malloc_info.batch_kv_cache_resource->batchSize();
    const int total_seq_len      = malloc_info.complete_token_ids->totalSeqLength();
    const int raw_common_seq_len = std::min(malloc_info.complete_token_ids->commonSeqLength(), total_seq_len);
    const int raw_seq_len        = malloc_info.complete_token_ids->seqLength();
    const int reserve_step       = malloc_info.complete_token_ids->getReserveStep();
    size_t    planned_blocks     = 0;
    for (int group_id = 0; group_id < static_cast<int>(kv_cache_groups_.size()); ++group_id) {
        const int  group_common_seq = cpEffectiveSeqLenForGroup(cp_slot_mapper_, config_, group_id, raw_common_seq_len);
        const int  group_seq_len    = cpEffectiveSeqLenForGroup(cp_slot_mapper_, config_, group_id, raw_seq_len);
        const auto need             = kv_cache_groups_[static_cast<size_t>(group_id)]->getNeedBlocks(
            group_common_seq,
            group_seq_len,
            reserve_step,
            malloc_info.batch_kv_cache_resource->blocksNum(0, group_id),
            malloc_info.reuse_cache,
            prepared.required_positions[static_cast<size_t>(group_id)]);
        planned_blocks += static_cast<size_t>(std::max(need.common_blocks, 0));
        planned_blocks += static_cast<size_t>(batch_size) * static_cast<size_t>(std::max(need.extra_blocks, 0));
    }
    const auto   demand       = initBlockDemand(malloc_info, planned_blocks);
    const size_t total_blocks = totalBlocksNum();
    if (demand.retained_blocks > total_blocks || planned_blocks > total_blocks - demand.retained_blocks
        || reserve_blocks > total_blocks - demand.retained_blocks - planned_blocks) {
        return MallocStatus::PERMANENT_RESOURCE_EXHAUSTED;
    }
    const size_t required_free_blocks = demand.additional_blocks + reserve_blocks;
    size_t       free_blocks          = freeBlocksNum();
    if (free_blocks < required_free_blocks
        && required_free_blocks <= static_cast<size_t>(std::numeric_limits<int>::max())) {
        for (const auto& group : kv_cache_groups_) {
            (void)group->ensureFreeBlocks(static_cast<int>(required_free_blocks));
            free_blocks = freeBlocksNum();
            if (free_blocks >= required_free_blocks) {
                break;
            }
        }
    }
    return free_blocks >= required_free_blocks ? MallocStatus::NONE : MallocStatus::RETRYABLE_RESOURCE_EXHAUSTED;
}

void HybridKVCacheAllocator::rollbackBlockIdsToSize(int group_id, BlockIds& block_ids, size_t original_size) {
    if (block_ids.blocksNum() <= original_size) {
        return;
    }
    const auto&            blocks = block_ids.blocks();
    const BlockIndicesType blocks_to_free(blocks.begin() + original_size, blocks.end());
    block_ids.resize(original_size);
    kv_cache_groups_[static_cast<size_t>(group_id)]->unreference(blocks_to_free);
}

void HybridKVCacheAllocator::rollbackInitMalloc(BatchKVCacheResource&                kv_resource,
                                                const std::vector<BlockIndicesType>& referenced_blocks,
                                                const std::vector<size_t>&           original_sizes) {
    for (int group_id = 0; group_id < kv_resource.groupNums(); ++group_id) {
        auto&        block_ids     = kv_resource.mutableBlockIds(0, group_id);
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

MemoryType HybridKVCacheAllocator::memoryTypeForGroup(int group_id) const {
    (void)group_id;
    return allocation_type_ == AllocationType::DEVICE ? MemoryType::MEMORY_GPU : MemoryType::MEMORY_CPU;
}

void HybridKVCacheAllocator::copyBlockMappingForGroup(int                             group_id,
                                                      const std::vector<BlockIdPair>& block_update_mapping) const {
    if (block_update_mapping.empty()) {
        return;
    }

    const auto   memory_type         = memoryTypeForGroup(group_id);
    const auto   copy_type           = BatchCopyParams::get_copy_type(memory_type, memory_type);
    const auto&  spec                = config_.specForGroup(static_cast<size_t>(group_id));
    const size_t kv_block_size_bytes = spec->block_size_bytes();
    const size_t scale_block_bytes   = spec->scale_block_size_bytes();
    const size_t buffers_per_layer   = scale_block_bytes > 0 ? 2 : 1;

    BatchCopyParams copy_params;
    copy_params.reserve(copy_type,
                        config_.layerIdsForGroup(static_cast<size_t>(group_id)).size() * block_update_mapping.size()
                            * buffers_per_layer);

    for (const auto& [src_block_index, dest_block_index] : block_update_mapping) {
        for (int layer_id : config_.layerIdsForGroup(static_cast<size_t>(group_id))) {
            auto src_addr_info =
                kv_cache_groups_[static_cast<size_t>(group_id)]->convertIndexToAddr(layer_id, src_block_index);
            auto dst_addr_info =
                kv_cache_groups_[static_cast<size_t>(group_id)]->convertIndexToAddr(layer_id, dest_block_index);

            RTP_LLM_CHECK_WITH_INFO(src_addr_info.kv_addr && dst_addr_info.kv_addr,
                                    "failed to get block address for group %d layer %d src_block %d dst_block %d",
                                    group_id,
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

int HybridKVCacheAllocator::getNeedBlocks(const MallocInfo& malloc_info) const {
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
        const auto group            = kv_cache_groups_[static_cast<size_t>(group_id)];
        const int  group_common_seq = cpEffectiveSeqLenForGroup(cp_mapper, config_, group_id, raw_common_seq_len);
        const int  group_seq_len    = cpEffectiveSeqLenForGroup(cp_mapper, config_, group_id, raw_seq_len);
        const auto need             = kv_cache_groups_[static_cast<size_t>(group_id)]->getNeedBlocks(
            group_common_seq, group_seq_len, reserve_step, reuse_blocks_len, reuse_enabled);
        common_blocks_total += need.common_blocks;
        extra_blocks_total += need.extra_blocks;
    }
    return common_blocks_total + batch_size * extra_blocks_total;
}

int HybridKVCacheAllocator::estimatePeakNeedBlocks(const KVCacheResource& kv_cache_resource,
                                                   int                    seq_len,
                                                   int                    remaining_tokens,
                                                   int                    reserve_step,
                                                   bool                   enable_reuse_cache) const {
    int need_blocks = 0;
    for (int group_id = 0; group_id < kv_cache_resource.groupNums(); ++group_id) {
        need_blocks += kv_cache_groups_[static_cast<size_t>(group_id)]->estimatePeakNeedBlocks(
            seq_len, kv_cache_resource.blocks(group_id), remaining_tokens, reserve_step, enable_reuse_cache);
    }
    return need_blocks;
}

int HybridKVCacheAllocator::estimateInitialBatchPeakNeedBlocks(int  seq_len,
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

void HybridKVCacheAllocator::checkCPShardedMallocResult(const MallocInfo& malloc_info) const {
    if (!cp_slot_mapper_ || !cp_slot_mapper_->isSharded()) {
        return;
    }

    const auto& kv_resource  = malloc_info.batch_kv_cache_resource;
    const int   seq_len      = malloc_info.incrSeqLen();
    const int   reserve_step = malloc_info.complete_token_ids->getReserveStep();

    for (int batch_id = 0; batch_id < kv_resource->batchSize(); ++batch_id) {
        for (int group_id = 0; group_id < kv_resource->groupNums(); ++group_id) {
            if (!cpBlockRoundRobinGroup(cp_slot_mapper_, config_, group_id)) {
                continue;
            }
            const int effective_seq_len = cpEffectiveSeqLenForGroup(cp_slot_mapper_, config_, group_id, seq_len);
            const int expected_blocks =
                kv_cache_groups_[static_cast<size_t>(group_id)]->needBlocksNum(effective_seq_len, 0, reserve_step);
            const int actual_blocks = kv_resource->blocksNum(batch_id, group_id);
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

int HybridKVCacheAllocator::singleBatchNeedBlocks(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                                  int                            seq_len,
                                                  int                            reserve_step) const {
    int need_blocks = 0;
    for (int group_id = 0; group_id < batch_kv_cache_resource->groupNums(); ++group_id) {
        const int effective_seq_len = cpEffectiveSeqLenForGroup(cp_slot_mapper_, config_, group_id, seq_len);
        const int cur_blocks        = batch_kv_cache_resource->blocksNum(0, group_id);
        need_blocks +=
            kv_cache_groups_[static_cast<size_t>(group_id)]->needBlocksNum(effective_seq_len, cur_blocks, reserve_step);
    }
    return need_blocks;
}

}  // namespace rtp_llm
