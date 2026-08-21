#include "rtp_llm/cpp/cache/SingleTypeKVCacheAllocator.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <unordered_map>

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/cache/DeviceBlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/BlockReleaseBatch.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadAsyncContext.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"

namespace rtp_llm {

int SingleTypeKVCacheAllocator::getNeedBlocks(const MallocInfo& malloc_info) const {
    if (!malloc_info.batch_kv_cache_resource || !malloc_info.complete_token_ids) {
        return 0;
    }
    const bool reuse_enabled    = malloc_info.reuse_cache;
    const int  reuse_blocks_len = reuse_enabled ? malloc_info.batch_kv_cache_resource->curBlocksNum() : 0;
    const int  batch_size       = malloc_info.batch_kv_cache_resource->batchSize();
    const int  seq_len          = malloc_info.complete_token_ids->seqLength();
    const int  reserve_step     = malloc_info.complete_token_ids->getReserveStep();
    const int  common_seq_len   = std::min(malloc_info.complete_token_ids->commonSeqLength(), seq_len);

    const auto need =
        full_kv_cache_group_->getNeedBlocks(common_seq_len, seq_len, reserve_step, reuse_blocks_len, reuse_enabled);
    return (batch_size <= 0) ? 0 : (need.common_blocks + batch_size * need.extra_blocks);
}

SingleTypeKVCacheAllocator::SingleTypeKVCacheAllocator(const CacheConfig&                 config,
                                                       AllocationType                     allocation_type,
                                                       const kmonitor::MetricsReporterPtr metrics_reporter,
                                                       int64_t                            reserve_block_ratio):
    KVCacheAllocator(config, allocation_type, metrics_reporter, reserve_block_ratio) {}

bool SingleTypeKVCacheAllocator::doInit() {
    RTP_LLM_CHECK_WITH_INFO(config_.groupNums() == 1,
                            "SingleTypeKVCacheAllocator requires exactly one cache group, got %d",
                            config_.groupNums());
    const auto& cache_group = config_.topology().groupById(0);
    const auto& spec        = cache_group.spec;
    RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "cache spec[0] is null");
    const bool is_full_attention = config_.typeForGroup(0) == CacheGroupType::FULL
                                   && (spec->type == rtp_llm::KVCacheSpecType::MultiHeadAttention
                                       || spec->type == rtp_llm::KVCacheSpecType::MultiHeadLatentAttention);
    RTP_LLM_CHECK_WITH_INFO(is_full_attention, "SingleTypeKVCacheAllocator requires one FULL MHA/MLA cache group");

    auto pool_config = std::make_shared<DeviceBlockPoolConfig>(DeviceBlockPoolConfigHelper::createConfig(config_));
    pool_config->use_cuda_malloc_backing = use_cuda_malloc_block_pool_;
    block_pool_ = std::make_shared<DeviceBlockPool>(std::shared_ptr<const DeviceBlockPoolConfig>(pool_config));
    if (!block_pool_->init()) {
        RTP_LLM_LOG_ERROR("Failed to initialize block pool for SingleTypeKVCacheAllocator");
        return false;
    }

    full_kv_cache_group_ = std::make_shared<FullKVCacheGroup>(cache_group, block_pool_, 0);

    if (!full_kv_cache_group_->init()) {
        RTP_LLM_LOG_ERROR("Failed to initialize FullKVCacheGroup");
        return false;
    }

    RTP_LLM_LOG_INFO("SingleTypeKVCacheAllocator initialized successfully");
    return true;
}

MallocResult SingleTypeKVCacheAllocator::initMallocForCommonLen(const MallocInfo& malloc_info) {
    auto&       kv_resource = malloc_info.batch_kv_cache_resource;
    auto&       block_ids_0 = kv_resource->mutableBlockIds(0, 0);
    const auto& cache_keys  = kv_resource->cacheKeys(0);
    const int   seq_len     = malloc_info.complete_token_ids->seqLength();
    const int   reuse_unit_tokens =
        cp_slot_mapper_ ? cp_slot_mapper_->reuseBlockTokens(config_) : full_kv_cache_group_->seqSizePerBlock();

    int64_t                           match_cost_time_us    = 0;
    int64_t                           match_end_time_us     = 0;
    size_t                            matched_device_blocks = 0;
    size_t                            total_logical_blocks  = 0;
    std::shared_ptr<LoadAsyncContext> load_context;
    std::vector<MultiNodeResource>    matched_resources;
    bool                              matched_blocks_released = false;
    bool                              load_attempted          = false;
    MallocStatus                      materialize_status      = MallocStatus::NONE;
    const std::function<void()>       release_matched_blocks  = [&]() {
        if (!matched_blocks_released) {
            block_tree_cache_->releaseMatchedResources(matched_resources);
            matched_blocks_released = true;
        }
    };
    auto rollback = [&]() -> MallocResult {
        if (load_context != nullptr) {
            load_context->abort();
        }
        load_context.reset();
        release_matched_blocks();
        BlockIndicesType valid_blocks;
        for (const auto block : block_ids_0.blocks()) {
            if (!isNullBlockIdx(block)) {
                valid_blocks.push_back(block);
            }
        }
        if (!valid_blocks.empty()) {
            BlockReleaseBatch releases;
            releases.append(/*group_id=*/0, full_kv_cache_group_->release(valid_blocks, BlockRefType::REQUEST));
            submitBlockReleases(releases);
        }
        block_ids_0.resize(0);
        kv_resource->cacheResource(0).setDeviceReuseBlockNum(0);
        MallocResult result{false, 0};
        result.status             = materialize_status;
        result.match_cost_time_us = match_cost_time_us;
        result.match_end_time_us  = match_end_time_us;
        result.load_attempted     = load_attempted;
        return result;
    };

    if (malloc_info.enable_cache_lookup && full_kv_cache_group_->prefixReuseEnabled()) {
        CacheKeysType match_keys = cp_slot_mapper_ && cp_slot_mapper_->isSharded() ?
                                       cp_slot_mapper_->localCacheKeys(config_, 0, cache_keys) :
                                       cache_keys;
        match_keys.resize(std::min(match_keys.size(), maxReusableMatchKeys(seq_len, reuse_unit_tokens)));
        const int64_t        match_begin_time_us = currentTimeUs();
        BlockTreeMatchResult match_result        = block_tree_cache_->match(match_keys);
        match_end_time_us                        = currentTimeUs();
        match_cost_time_us                       = match_end_time_us - match_begin_time_us;
        const bool has_async_context             = match_result.async_context != nullptr;
        load_attempted                           = has_async_context;
        load_context = std::dynamic_pointer_cast<LoadAsyncContext>(match_result.async_context);
        match_result.async_context.reset();
        matched_resources = std::move(match_result.matched_device_resources);
        if (has_async_context && load_context == nullptr) {
            return rollback();
        }
        matched_device_blocks               = match_result.matched_device_blocks;
        total_logical_blocks                = load_context ? load_context->localMatchedBlocks() : matched_device_blocks;
        BlockIndicesType ready_group_blocks = block_tree_cache_->matchedBlocksForGroup(0, matched_resources);
        block_ids_0.assign(BlockIndicesType(total_logical_blocks, NULL_BLOCK_IDX));
        for (size_t i = 0; i < ready_group_blocks.size(); ++i) {
            block_ids_0.setAt(i, ready_group_blocks[i]);
        }

        if (load_context && !load_context->empty()) {
            for (size_t desc_index = 0; desc_index < load_context->loadDescs().size(); ++desc_index) {
                const BlockIndicesType& source_blocks = load_context->loadDescs()[desc_index].source_blocks;
                const size_t            path_index    = load_context->loadDescs()[desc_index].path_index;
                if (load_context->loadDescs()[desc_index].source_tier == Tier::DEVICE) {
                    const BlockIdxType current = block_ids_0.blocks()[path_index];
                    if (!isNullBlockIdx(current) && current != source_blocks.front()) {
                        return rollback();
                    }
                    block_ids_0.setAt(path_index, source_blocks.front());
                    continue;
                }
                if (load_context->joinedLoads()[desc_index]) {
                    const std::vector<BlockIdxType>& joined_targets =
                        load_context->loadDescs()[desc_index].target_blocks;
                    const BlockIdxType current = block_ids_0.blocks()[path_index];
                    if (!isNullBlockIdx(current) && current != joined_targets.front()) {
                        return rollback();
                    }
                    block_ids_0.setAt(path_index, joined_targets.front());
                }
            }
        }

        BlockIndicesType resident_blocks;
        for (const auto block : block_ids_0.blocks()) {
            if (!isNullBlockIdx(block)) {
                resident_blocks.push_back(block);
            }
        }
        if (!resident_blocks.empty()) {
            full_kv_cache_group_->KVCacheGroup::reference(resident_blocks);
        }
        release_matched_blocks();
        kv_resource->cacheResource(0).setDeviceReuseBlockNum(matched_device_blocks);
    } else {
        release_matched_blocks();
    }

    if (load_context && load_context->needBackendMatch()) {
        auto self = shared_from_this();
        load_context->setMatchCallback(
            [self = std::move(self), malloc_info](LoadAsyncContext& context, size_t matched_blocks) {
                return self->finishDeferredMalloc(malloc_info, context, matched_blocks);
            });
        MallocResult result{
            true, static_cast<int>(matched_device_blocks) * reuse_unit_tokens, match_cost_time_us, load_context};
        result.match_end_time_us = match_end_time_us;
        result.load_attempted    = load_attempted;
        return result;
    }

    if (!materializeInitialBlocks(malloc_info, load_context.get(), total_logical_blocks, materialize_status)) {
        return rollback();
    }
    const int    reuse_len = static_cast<int>(matched_device_blocks) * reuse_unit_tokens;
    MallocResult result{true, reuse_len, match_cost_time_us, load_context};
    result.match_end_time_us = match_end_time_us;
    result.load_attempted    = load_attempted;
    return result;
}

bool SingleTypeKVCacheAllocator::materializeInitialBlocks(const MallocInfo& malloc_info,
                                                          LoadAsyncContext* context,
                                                          size_t            matched_blocks,
                                                          MallocStatus&     materialize_status) {
    materialize_status = MallocStatus::NONE;
    auto& kv_resource = *malloc_info.batch_kv_cache_resource;
    auto& block_ids   = kv_resource.mutableBlockIds(0, 0);
    block_ids.resize(matched_blocks, NULL_BLOCK_IDX);

    RequiredPositions positions;
    if (context != nullptr) {
        for (size_t i = 0; i < context->loadDescs().size(); ++i) {
            const auto& desc = context->loadDescs()[i];
            if (desc.source_tier != Tier::DEVICE && !context->joinedLoads()[i]) {
                positions.insert(desc.path_index);
            }
        }
        const auto& backend_handles = context->backendHandles();
        for (size_t key_index = 0; key_index < backend_handles.size(); ++key_index) {
            if (!backend_handles[key_index].empty()) {
                positions.insert(key_index);
            }
        }
    }

    size_t missing_targets = 0;
    for (const size_t position : positions) {
        if (position >= block_ids.blocksNum()) {
            materialize_status = MallocStatus::INTERNAL_ERROR;
            return false;
        }
        missing_targets += isNullBlockIdx(block_ids.blocks()[position]) ? 1 : 0;
    }
    materialize_status = evaluatePreparedInitCapacity(malloc_info, missing_targets);
    if (materialize_status != MallocStatus::NONE) {
        return false;
    }

    int common_seq_len =
        std::min(malloc_info.complete_token_ids->commonSeqLength(), malloc_info.complete_token_ids->totalSeqLength());
    if (cp_slot_mapper_ && cp_slot_mapper_->isSharded()) {
        common_seq_len = cp_slot_mapper_->effectiveSeqLenForAlloc(config_, 0, common_seq_len);
    }
    if (!full_kv_cache_group_->malloc(block_ids, common_seq_len, false, 0, nullptr, positions)) {
        materialize_status = evaluatePreparedInitCapacity(malloc_info, missing_targets);
        if (materialize_status == MallocStatus::NONE) {
            materialize_status = MallocStatus::INTERNAL_ERROR;
        }
        return false;
    }

    if (context != nullptr) {
        for (size_t i = 0; i < context->loadDescs().size(); ++i) {
            const auto& desc = context->loadDescs()[i];
            if (desc.path_index >= block_ids.blocksNum() || isNullBlockIdx(block_ids.blocks()[desc.path_index])) {
                materialize_status = MallocStatus::INTERNAL_ERROR;
                return false;
            }
            const BlockIdxType target = block_ids.blocks()[desc.path_index];
            if (context->joinedLoads()[i]) {
                if (desc.target_blocks.size() != 1 || target != desc.target_blocks.front()) {
                    materialize_status = MallocStatus::INTERNAL_ERROR;
                    return false;
                }
            } else {
                context->setTargetBlocks(i, {target});
            }
        }
        const auto& backend_handles = context->backendHandles();
        for (size_t key_index = 0; key_index < backend_handles.size(); ++key_index) {
            for (size_t handle_index = 0; handle_index < backend_handles[key_index].size(); ++handle_index) {
                context->setBackendTargetBlock(key_index, handle_index, block_ids.blocks()[key_index]);
            }
        }
    }

    for (int batch = 1; batch < kv_resource.batchSize(); ++batch) {
        full_kv_cache_group_->reference(kv_resource.mutableBlockIds(batch, 0), block_ids.blocks());
    }
    return true;
}

LoadMatchResult SingleTypeKVCacheAllocator::finishDeferredMalloc(const MallocInfo& malloc_info,
                                                                 LoadAsyncContext& context,
                                                                 size_t            matched_blocks) {
    MallocStatus materialize_status = MallocStatus::NONE;
    bool         success = materializeInitialBlocks(malloc_info, &context, matched_blocks, materialize_status);
    if (success && !context.isRequestCanceled()) {
        const auto incr_result = incrMalloc(malloc_info);
        success                = incr_result.success;
        if (!success) {
            materialize_status = incr_result.status;
            if (materialize_status == MallocStatus::NONE) {
                materialize_status = evaluateInitCapacity(
                    malloc_info, reserveBlocksNum(), InitCapacityMode::TOTAL_AND_AVAILABLE);
            }
        }
    } else {
        success = false;
    }
    success = success && context.commit();
    if (!success) {
        if (materialize_status == MallocStatus::NONE) {
            materialize_status = MallocStatus::INTERNAL_ERROR;
        }
        free(FreeInfo{malloc_info.batch_kv_cache_resource, malloc_info.complete_token_ids});
        return {false, materialize_status};
    }
    return {true};
}

MallocStatus SingleTypeKVCacheAllocator::evaluatePreparedInitCapacity(const MallocInfo& malloc_info,
                                                                       size_t missing_targets) const {
    const size_t planned_blocks = static_cast<size_t>(std::max(getNeedBlocks(malloc_info), 0))
                                  + (malloc_info.reuse_cache ? missing_targets : 0);
    const auto   demand         = initBlockDemand(malloc_info, planned_blocks, /*group_id=*/0);
    const size_t total_blocks   = totalBlocksNum();
    const size_t reserve_blocks = reserveBlocksNum();
    if (demand.retained_blocks > total_blocks || planned_blocks > total_blocks - demand.retained_blocks
        || reserve_blocks > total_blocks - demand.retained_blocks - planned_blocks) {
        return MallocStatus::PERMANENT_RESOURCE_EXHAUSTED;
    }
    const size_t required_free_blocks = demand.additional_blocks + reserve_blocks;
    if (required_free_blocks <= static_cast<size_t>(std::numeric_limits<int>::max())) {
        (void)full_kv_cache_group_->ensureFreeBlocks(static_cast<int>(required_free_blocks));
    }
    return freeBlocksNum() >= required_free_blocks ? MallocStatus::NONE :
                                                     MallocStatus::RETRYABLE_RESOURCE_EXHAUSTED;
}

MallocResult SingleTypeKVCacheAllocator::incrMalloc(const MallocInfo& malloc_info) {
    auto& kv_resource    = malloc_info.batch_kv_cache_resource;
    int   batch_size     = kv_resource->batchSize();
    int   current_blocks = kv_resource->curBlocksNum();
    int   seq_len        = malloc_info.incrSeqLen();
    int   reserve_step   = malloc_info.complete_token_ids->getReserveStep();

    auto need_blocks = full_kv_cache_group_->needBlocksNum(seq_len, current_blocks, reserve_step);
    if (need_blocks == 0) {
        return {true, 0};
    }

    // Record original sizes for rollback in case any subsequent allocation fails
    std::vector<size_t> original_blocks_num;
    for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
        original_blocks_num.push_back(kv_resource->blocksNum(batch_id, 0));
    }

    bool all_success   = true;
    int  current_batch = 0;
    for (; current_batch < batch_size; ++current_batch) {
        auto& block_ids = kv_resource->mutableBlockIds(current_batch, 0);
        if (!full_kv_cache_group_->malloc(block_ids, seq_len, false, reserve_step)) {
            all_success = false;
            break;
        }
    }

    if (all_success) {
        return {true, 0};
    }

    // rollback kvcache blocks
    BlockIndicesType blocks_to_free;
    for (int batch_id = 0; batch_id <= current_batch; ++batch_id) {
        auto& block_ids    = kv_resource->mutableBlockIds(batch_id, 0);
        auto  original_num = original_blocks_num[batch_id];
        if (block_ids.blocksNum() > original_num) {
            const auto& blk = block_ids.blocks();
            blocks_to_free.insert(blocks_to_free.end(), blk.begin() + original_num, blk.end());
            block_ids.resize(original_num);
        }
    }
    if (!blocks_to_free.empty()) {
        (void)full_kv_cache_group_->release(blocks_to_free, BlockRefType::REQUEST);
    }
    return {false, 0};
}

void SingleTypeKVCacheAllocator::free(const FreeInfo& free_info) {
    auto& kv_cache_resource = free_info.batch_kv_cache_resource;

    if (kv_cache_resource->curBlocksNum() == 0) {
        return;
    }

    BlockReleaseBatch releases;
    auto              all_blocks = kv_cache_resource->getAllBatchBlocks(0);
    for (const auto& blocks : all_blocks) {
        releases.append(/*group_id=*/0, full_kv_cache_group_->release(blocks, BlockRefType::REQUEST));
    }
    kv_cache_resource->clearBlocks();
    submitBlockReleases(releases);
}

void SingleTypeKVCacheAllocator::insertIntoCache(const InsertInfo& insert_info) {
    if (!full_kv_cache_group_->prefixReuseEnabled()) {
        return;
    }
    if (!block_tree_cache_) {
        return;
    }

    auto&     kv_resource = insert_info.batch_kv_cache_resource;
    const int batch_size  = std::min(kv_resource->batchSize(), 1);
    for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
        const auto&   full_keys   = kv_resource->cacheKeys(batch_id);
        CacheKeysType insert_keys = cp_slot_mapper_ && cp_slot_mapper_->isSharded() ?
                                        cp_slot_mapper_->localCacheKeys(config_, 0, full_keys) :
                                        full_keys;
        const auto&   blocks      = kv_resource->blocks(batch_id, 0);
        const size_t  block_num   = std::min(insert_keys.size(), blocks.size());
        if (block_num == 0) {
            continue;
        }
        insert_keys.resize(block_num);
        const auto& group_sets = block_tree_cache_->groupSets();
        if (group_sets.size() != 1 || !group_sets[0] || group_sets[0]->groupIds() != std::vector<size_t>{0}
            || group_sets[0]->devicePools().size() != 1) {
            RTP_LLM_LOG_WARNING("SingleType insert rejected inconsistent GroupSet membership");
            continue;
        }
        std::vector<std::vector<GroupSetResource>> resources(block_num, std::vector<GroupSetResource>(1));
        for (auto& per_key_resources : resources) {
            per_key_resources[0].device_blocks.assign(1, NULL_BLOCK_IDX);
        }
        size_t publish_prefix = 0;
        for (size_t i = 0; i < block_num; ++i) {
            if (isNullBlockIdx(blocks[i])) {
                break;
            }
            resources[i][0].device_blocks[0] = blocks[i];
            publish_prefix                   = i + 1;
        }
        if (publish_prefix > 0) {
            insert_keys.resize(publish_prefix);
            resources.resize(publish_prefix);
            block_tree_cache_->insert(insert_keys, resources, insert_info.target_tier);
        }
    }
}

GroupedCacheLayerLayout SingleTypeKVCacheAllocator::allLayerCacheBase() const {
    const auto layer_tensors = full_kv_cache_group_->allLayerCacheBase();
    const auto scale_tensors = full_kv_cache_group_->allLayerScaleCacheBase();
    const auto topology      = config_.topologyPtr();

    GroupedCacheLayerLayout::GroupLayouts groups;
    for (const auto& group : topology->groups()) {
        std::vector<BlockBufferPtrInfo> layers(topology->layers().size());
        for (int layer_id : group.layer_ids) {
            const auto layer = static_cast<size_t>(layer_id);
            const auto kv_it = layer_tensors.find(layer_id);
            if (kv_it != layer_tensors.end() && kv_it->second.defined()) {
                layers[layer].kv_addr = kv_it->second;
            }
            const auto scale_it = scale_tensors.find(layer_id);
            if (scale_it != scale_tensors.end() && scale_it->second.defined()) {
                layers[layer].kv_scale_addr = scale_it->second;
            }
        }
        groups.emplace(group.tag, CacheLayerLayout(std::move(layers)));
    }
    return GroupedCacheLayerLayout(topology, std::move(groups));
}

BlockAddrInfo SingleTypeKVCacheAllocator::convertIndexToAddr(int layer_id, int block_id) const {
    return full_kv_cache_group_->convertIndexToAddr(layer_id, block_id);
}

std::vector<BlockInfo> SingleTypeKVCacheAllocator::convertIndexToBuffer(int layer_id, int block_id) const {
    return full_kv_cache_group_->convertIndexToBuffer(layer_id, block_id);
}

std::vector<BlockInfo> SingleTypeKVCacheAllocator::convertIndexToBuffer(int layer_id,
                                                                        int block_id,
                                                                        int partition_count,
                                                                        int partition_id) const {
    return full_kv_cache_group_->convertIndexToBuffer(layer_id, block_id, partition_count, partition_id);
}

std::shared_ptr<KVCacheResource> SingleTypeKVCacheAllocator::incrKVCacheRef(const KVCacheResource& kvcache_resource,
                                                                            const CacheKeysType&   cache_keys,
                                                                            bool                   is_connector) {
    if (cache_keys.empty()) {
        return nullptr;
    }

    RTP_LLM_CHECK_WITH_INFO(
        kvcache_resource.groupNums() == 1, "incrKVCacheRef expects groupNums==1, got %d", kvcache_resource.groupNums());

    std::unordered_map<CacheKeyType, size_t> key_to_pos;
    const auto&                              resource_keys = kvcache_resource.cacheKeys();
    key_to_pos.reserve(resource_keys.size());
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

    CacheKeysType    selected_cache_keys;
    BlockIndicesType selected_blocks;

    const auto& src_blocks = kvcache_resource.blocks(0);

    BlockIndicesType real_blocks;
    for (auto key : cache_keys) {
        auto it = key_to_pos.find(key);
        if (it == key_to_pos.end()) {
            continue;
        }
        const size_t pos = it->second;
        if (pos < src_blocks.size()) {
            const auto block = src_blocks[pos];
            if (block > 0 && !isNullBlockIdx(block)) {
                selected_cache_keys.push_back(key);
                selected_blocks.push_back(block);
                real_blocks.push_back(block);
            }
        } else if (is_connector && !kvcache_resource.lastBlockAligned()) {
            selected_cache_keys.push_back(key);
            selected_blocks.push_back(NULL_BLOCK_IDX);
        }
    }

    if (real_blocks.empty()) {
        return nullptr;
    }

    const BlockRefType ref_type = is_connector ? BlockRefType::STORAGE_BACKEND : BlockRefType::REQUEST;
    full_kv_cache_group_->KVCacheGroup::reference(real_blocks, ref_type);
    selected_resource->mutableBlockIds(0).assign(std::move(selected_blocks));
    selected_resource->cacheKeys() = std::move(selected_cache_keys);

    return selected_resource;
}

void SingleTypeKVCacheAllocator::decrKVCacheRef(const KVCacheResource& kvcache_resource, bool is_connector) {
    RTP_LLM_CHECK_WITH_INFO(
        kvcache_resource.groupNums() == 1, "decrKVCacheRef expects groupNums==1, got %d", kvcache_resource.groupNums());

    BlockIndicesType blocks_to_free;
    for (const auto block : kvcache_resource.blocks(0)) {
        if (block > 0 && !isNullBlockIdx(block)) {
            blocks_to_free.push_back(block);
        }
    }
    if (!blocks_to_free.empty()) {
        const BlockRefType ref_type = is_connector ? BlockRefType::STORAGE_BACKEND : BlockRefType::REQUEST;
        BlockReleaseBatch  releases;
        releases.append(/*group_id=*/0, full_kv_cache_group_->release(blocks_to_free, ref_type));
        submitBlockReleases(releases);
    }
}

// Update kv blocks for beam search or multi-return sequences.
// - batch_kv_cache_resource: in/out, batch blocks and cache_keys will be rearranged based on block_src_batch
// - block_src_batch: new batch i forks from old batch block_src_batch[i]
// - copy_last_block: whether to copy the last block for each forked batch (instead of sharing)
// - block_update_mapping: out, mapping from old block to new block for batch copy
bool SingleTypeKVCacheAllocator::updateKVBlock(const BatchKVCacheResourcePtr&  kv_cache_resource,
                                               const std::vector<int>&         block_src_batch,
                                               bool                            copy_last_block,
                                               std::vector<TaggedBlockIdPair>& block_update_mapping) {
    block_update_mapping.clear();
    if (block_src_batch.empty()) {
        return true;
    }

    const int        old_batch_size = kv_cache_resource->batchSize();
    const int        new_batch_size = static_cast<int>(block_src_batch.size());
    std::vector<int> batch_fork_count(old_batch_size, 0);
    for (const int old_batch_idx : block_src_batch) {
        RTP_LLM_CHECK_WITH_INFO(old_batch_idx < old_batch_size,
                                "try to reuse an old batch %d that out of range %d",
                                old_batch_idx,
                                old_batch_size);
        ++batch_fork_count[old_batch_idx];
    }

    BlockReleaseBatch releases;
    uint32_t          new_blocks_num = 0;
    for (int old_batch_idx = 0; old_batch_idx < old_batch_size; ++old_batch_idx) {
        const int fork_count = batch_fork_count[old_batch_idx];
        if (fork_count == 0) {
            releases.append(
                /*group_id=*/0,
                full_kv_cache_group_->release(kv_cache_resource->blocks(old_batch_idx, 0), BlockRefType::REQUEST));
        } else if (fork_count > 1 && copy_last_block) {
            new_blocks_num += static_cast<uint32_t>(fork_count - 1);
        }
    }

    // Free disused blocks first and publish their transitions before requesting
    // replacement capacity from the tree cache.
    submitBlockReleases(releases);

    // ensure there are enough free blocks for last-block copies
    if (new_blocks_num > 0) {
        if (!full_kv_cache_group_->ensureFreeBlocks(static_cast<int>(new_blocks_num))) {
            RTP_LLM_LOG_WARNING("ensure free blocks failed for kv cache update, need %u", new_blocks_num);
            return false;
        }
    }

    // rebuild batch_kv_cache_resource and generate mapping
    std::vector<KVCacheResource> old_resources;
    kv_cache_resource->resetAndReturnOldResources(new_batch_size, old_resources);

    // init for all batch
    kv_cache_resource->initGroups(config_.topologyPtr());

    for (int new_batch_idx = 0; new_batch_idx < new_batch_size; ++new_batch_idx) {
        const int old_batch_idx = block_src_batch[new_batch_idx];
        auto&     fork_count    = batch_fork_count[old_batch_idx];
        RTP_LLM_CHECK_WITH_INFO(fork_count > 0, "old batch %d has been forked too many times", old_batch_idx);

        if (fork_count == 1) {
            kv_cache_resource->moveBatchResource(new_batch_idx, std::move(old_resources[old_batch_idx]));
        } else {
            auto& block_ids = kv_cache_resource->mutableBlockIds(new_batch_idx, 0);
            kv_cache_resource->setBatchCacheKeys(new_batch_idx, old_resources[old_batch_idx].cacheKeys());
            full_kv_cache_group_->reference(block_ids, old_resources[old_batch_idx].blocks(0));

            if (copy_last_block && !block_ids.blocks().empty()) {
                const int old_block = block_ids.popBack();
                (void)full_kv_cache_group_->release({old_block}, BlockRefType::REQUEST);

                // allocate exactly one new block via kvCacheGroup
                int seq_len_target =
                    (static_cast<int>(block_ids.blocks().size()) + 1) * full_kv_cache_group_->seqSizePerBlock();
                bool ok = full_kv_cache_group_->malloc(block_ids, seq_len_target);
                RTP_LLM_CHECK_WITH_INFO(ok, "malloc one block via kvCacheGroup failed during kv cache update");
                const int new_block = block_ids.blocks().back();
                block_update_mapping.push_back(
                    TaggedBlockIdPair{config_.topology().soleGroupForLayer(0).tag, old_block, new_block});
            }
        }
        --fork_count;
    }
    return true;
}

int SingleTypeKVCacheAllocator::seqSizePerBlock() const {
    return full_kv_cache_group_->seqSizePerBlock();
}

int SingleTypeKVCacheAllocator::singleBatchNeedBlocks(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                                      int                            seq_len,
                                                      int                            reserve_step) const {
    const int current_blocks    = batch_kv_cache_resource ? batch_kv_cache_resource->blocksNum(0, 0) : 0;
    const int effective_seq_len = cpEffectiveSeqLenForAlloc(/*gid=*/0, seq_len);
    return full_kv_cache_group_->needBlocksNum(effective_seq_len, current_blocks, reserve_step);
}

int SingleTypeKVCacheAllocator::estimatePeakNeedBlocks(const KVCacheResource& kv_cache_resource,
                                                       int                    seq_len,
                                                       int                    remaining_tokens,
                                                       int                    reserve_step,
                                                       bool                   enable_reuse_cache) const {
    return full_kv_cache_group_->estimatePeakNeedBlocks(
        seq_len, kv_cache_resource.blocks(0), remaining_tokens, reserve_step, enable_reuse_cache);
}

int SingleTypeKVCacheAllocator::estimateInitialBatchPeakNeedBlocks(int  seq_len,
                                                                   int  common_seq_len,
                                                                   int  remaining_tokens,
                                                                   int  reserve_step,
                                                                   bool enable_reuse_cache,
                                                                   int  target_batch_size) const {
    return full_kv_cache_group_->estimateInitialBatchPeakNeedBlocks(
        seq_len, common_seq_len, remaining_tokens, reserve_step, enable_reuse_cache, target_batch_size);
}

}  // namespace rtp_llm
