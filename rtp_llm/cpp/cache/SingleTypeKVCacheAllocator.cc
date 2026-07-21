#include "rtp_llm/cpp/cache/SingleTypeKVCacheAllocator.h"

#include <algorithm>
#include <unordered_map>

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/cache/DeviceBlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
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

    full_kv_cache_group_ = std::make_shared<FullKVCacheGroup>(cache_group, block_pool_, 0, nullptr);

    if (!full_kv_cache_group_->init()) {
        RTP_LLM_LOG_ERROR("Failed to initialize FullKVCacheGroup");
        return false;
    }

    RTP_LLM_LOG_INFO("SingleTypeKVCacheAllocator initialized successfully");
    return true;
}

MallocResult SingleTypeKVCacheAllocator::initMallocForCommonLen(const MallocInfo& malloc_info) {
    auto& kv_resource = malloc_info.batch_kv_cache_resource;
    auto& block_ids_0 = kv_resource->mutableBlockIds(0, 0);
    int   common_seq_len =
        std::min(malloc_info.complete_token_ids->commonSeqLength(), malloc_info.complete_token_ids->totalSeqLength());
    if (cp_slot_mapper_ && cp_slot_mapper_->isSharded()) {
        common_seq_len = cp_slot_mapper_->effectiveSeqLenForAlloc(config_, 0, common_seq_len);
    }
    const auto&        cache_keys = kv_resource->cacheKeys(0);
    const std::string& tag        = config_.tagForGroup(0);
    RTP_LLM_CHECK_WITH_INFO(block_tree_cache_ != nullptr, "BlockTreeCache must be injected before allocation");

    int64_t                         match_cost_time_us = 0;
    size_t                          reuse_blocks       = 0;
    std::shared_ptr<LoadBackTicket> load_back_ticket;
    std::vector<GroupBlockSet>      matched_block_sets;
    bool                            matched_blocks_released = false;
    auto                            release_matched_blocks  = [&]() {
        if (!matched_blocks_released) {
            block_tree_cache_->releaseMatchedBlocks(matched_block_sets);
            matched_blocks_released = true;
        }
    };
    auto rollback = [&]() -> MallocResult {
        load_back_ticket.reset();
        release_matched_blocks();
        BlockIndicesType valid_blocks;
        for (const auto block : block_ids_0.blocks()) {
            if (!isNullBlockIdx(block)) {
                valid_blocks.push_back(block);
            }
        }
        if (!valid_blocks.empty()) {
            full_kv_cache_group_->free(valid_blocks);
        }
        block_ids_0.resize(0);
        kv_resource->cacheResource(0).setDeviceReuseBlockNum(0);
        return {false, 0};
    };

    if (malloc_info.enable_device_cache && full_kv_cache_group_->prefixReuseEnabled()) {
        CacheKeysType local_keys = cp_slot_mapper_ && cp_slot_mapper_->isSharded() ?
                                       cp_slot_mapper_->localCacheKeys(config_, 0, cache_keys) :
                                       cache_keys;
        CacheKeysType match_keys(local_keys.begin(), local_keys.empty() ? local_keys.end() : local_keys.end() - 1);
        const auto    match_begin_time_us = currentTimeUs();
        auto          match_result        = block_tree_cache_->match(match_keys);
        match_cost_time_us                = currentTimeUs() - match_begin_time_us;
        load_back_ticket                  = match_result.load_back_ticket;
        matched_block_sets                = std::move(match_result.matched_block_sets);
        const size_t ready_blocks         = match_result.matched_blocks;
        reuse_blocks =
            load_back_ticket && !load_back_ticket->empty() ? load_back_ticket->logicalMatchedBlocks() : ready_blocks;
        const auto       group_it = match_result.group_block_indices.find(tag);
        BlockIndicesType ready_group_blocks =
            group_it == match_result.group_block_indices.end() ? BlockIndicesType{} : group_it->second;
        if (ready_group_blocks.size() != ready_blocks || ready_blocks > reuse_blocks) {
            return rollback();
        }
        block_ids_0.assign(BlockIndicesType(reuse_blocks, NULL_BLOCK_IDX));
        for (size_t i = 0; i < ready_group_blocks.size(); ++i) {
            if (isNullBlockIdx(ready_group_blocks[i])) {
                return rollback();
            }
            block_ids_0.setAt(i, ready_group_blocks[i]);
        }

        if (load_back_ticket && !load_back_ticket->empty()) {
            for (size_t item_index = 0; item_index < load_back_ticket->itemCount(); ++item_index) {
                const int   group_id          = load_back_ticket->groupId(item_index);
                const auto& device_group_tags = load_back_ticket->deviceGroupTags(item_index);
                const auto& source_blocks     = load_back_ticket->sourceBlocks(item_index);
                const auto  path_index        = load_back_ticket->pathIndex(item_index);
                if (!block_tree_cache_->validateDeviceGroupTagsForComponentGroup(group_id, device_group_tags)
                    || device_group_tags.size() != 1 || device_group_tags.front() != tag
                    || path_index >= reuse_blocks || source_blocks.size() != 1
                    || isNullBlockIdx(source_blocks.front())) {
                    return rollback();
                }
                if (load_back_ticket->sourceTier(item_index) == Tier::DEVICE) {
                    const auto current = block_ids_0.blocks()[path_index];
                    if (!isNullBlockIdx(current) && current != source_blocks.front()) {
                        return rollback();
                    }
                    block_ids_0.setAt(path_index, source_blocks.front());
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
        kv_resource->cacheResource(0).setDeviceReuseBlockNum(reuse_blocks);
    } else {
        release_matched_blocks();
    }

    std::vector<size_t> materialize_positions;
    if (load_back_ticket && !load_back_ticket->empty()) {
        for (size_t item_index = 0; item_index < load_back_ticket->itemCount(); ++item_index) {
            if (load_back_ticket->sourceTier(item_index) == Tier::DEVICE) {
                continue;
            }
            const size_t path_index = load_back_ticket->pathIndex(item_index);
            if (std::find(materialize_positions.begin(), materialize_positions.end(), path_index)
                == materialize_positions.end()) {
                materialize_positions.push_back(path_index);
            }
        }
        for (size_t position = 0; position < reuse_blocks; ++position) {
            if (isNullBlockIdx(block_ids_0.blocks()[position])
                && std::find(materialize_positions.begin(), materialize_positions.end(), position)
                       == materialize_positions.end()) {
                return rollback();
            }
        }
    }

    size_t missing_targets = 0;
    for (const size_t position : materialize_positions) {
        if (position >= block_ids_0.blocksNum()) {
            return rollback();
        }
        missing_targets += isNullBlockIdx(block_ids_0.blocks()[position]) ? 1 : 0;
    }
    const size_t reserve_blocks = reserveBlocksNum();
    if (reserve_blocks > 0) {
        const int    need_blocks = getNeedBlocks(malloc_info);
        const size_t required    = static_cast<size_t>(std::max(need_blocks, 0)) + missing_targets + reserve_blocks;
        if (freeBlocksNum() < required) {
            return rollback();
        }
    }
    if (!full_kv_cache_group_->materializePositions(block_ids_0, materialize_positions)) {
        return rollback();
    }
    if (!full_kv_cache_group_->malloc(block_ids_0, common_seq_len)) {
        return rollback();
    }

    if (load_back_ticket && !load_back_ticket->empty()) {
        for (size_t item_index = 0; item_index < load_back_ticket->itemCount(); ++item_index) {
            const size_t path_index = load_back_ticket->pathIndex(item_index);
            if (path_index >= block_ids_0.blocksNum() || isNullBlockIdx(block_ids_0.blocks()[path_index])) {
                return rollback();
            }
            const auto target = block_ids_0.blocks()[path_index];
            if (load_back_ticket->sourceTier(item_index) == Tier::DEVICE
                && target != load_back_ticket->sourceBlocks(item_index).front()) {
                return rollback();
            }
            if (!load_back_ticket->bindTargetDeviceBlocks(item_index, {target})) {
                return rollback();
            }
        }
    }

    for (int batch_id = 1; batch_id < kv_resource->batchSize(); ++batch_id) {
        full_kv_cache_group_->reference(kv_resource->mutableBlockIds(batch_id, 0), block_ids_0.blocks());
    }
    const int reuse_len =
            static_cast<int>(reuse_blocks
                             * (cp_slot_mapper_ && cp_slot_mapper_->isSharded() ?
                                    cp_slot_mapper_->logicalSeqSizePerBlock(config_, 0) :
                                static_cast<size_t>(full_kv_cache_group_->seqSizePerBlock())));
    MallocResult result{true, reuse_len, match_cost_time_us, nullptr, load_back_ticket};
    if (load_back_ticket != nullptr && reuse_blocks > 0) {
        const int reuse_unit_tokens = reuse_len / static_cast<int>(reuse_blocks);
        result.memory_reuse_len =
            static_cast<int>(load_back_ticket->logicalMatchedBlocks(Tier::HOST)) * reuse_unit_tokens;
        result.disk_reuse_len =
            static_cast<int>(load_back_ticket->logicalMatchedBlocks(Tier::DISK)) * reuse_unit_tokens;
    }
    return result;
}

MallocResult SingleTypeKVCacheAllocator::incrMalloc(const MallocInfo& malloc_info) {
    auto& kv_resource    = malloc_info.batch_kv_cache_resource;
    int   batch_size     = kv_resource->batchSize();
    int   current_blocks = kv_resource->curBlocksNum();
    int   seq_len        = malloc_info.complete_token_ids->seqLength();
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
        full_kv_cache_group_->free(blocks_to_free);
    }
    return {false, 0};
}

void SingleTypeKVCacheAllocator::free(const FreeInfo& free_info) {
    auto& kv_cache_resource = free_info.batch_kv_cache_resource;

    if (kv_cache_resource->curBlocksNum() == 0) {
        return;
    }

    auto all_blocks = kv_cache_resource->getAllBatchBlocks(0);
    for (const auto& blocks : all_blocks) {
        full_kv_cache_group_->free(blocks);
    }
    kv_cache_resource->clearBlocks();
    if (block_tree_cache_) {
        block_tree_cache_->onBlocksReleased();
    }
}

void SingleTypeKVCacheAllocator::insertIntoCache(const InsertInfo& insert_info) {
    if (!full_kv_cache_group_->prefixReuseEnabled() || !block_tree_cache_) {
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
        const auto& component_groups = block_tree_cache_->componentGroups();
        if (component_groups.size() != 1 || !component_groups[0] || component_groups[0]->tags().size() != 1
            || component_groups[0]->tags().front() != config_.tagForGroup(0)
            || component_groups[0]->devicePoolCount() != 1) {
            RTP_LLM_LOG_WARNING("SingleType insert rejected inconsistent stable tag/component mapping");
            continue;
        }
        std::vector<std::vector<GroupSlot>> slots(block_num, std::vector<GroupSlot>(1));
        for (auto& per_key_slots : slots) {
            per_key_slots[0].device_blocks.assign(1, NULL_BLOCK_IDX);
        }
        bool has_valid = false;
        for (size_t i = 0; i < block_num; ++i) {
            if (isNullBlockIdx(blocks[i])) {
                continue;
            }
            slots[i][0].device_blocks[0] = blocks[i];
            has_valid                    = true;
        }
        if (has_valid) {
            block_tree_cache_->insert(nullptr, insert_keys, slots);
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

    const BlockRefType ref_type = is_connector ? BlockRefType::CONNECTOR : BlockRefType::REQUEST;
    block_pool_->incRef(real_blocks, ref_type);
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
        const BlockRefType ref_type = is_connector ? BlockRefType::CONNECTOR : BlockRefType::REQUEST;
        block_pool_->decRef(blocks_to_free, ref_type);
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

    std::vector<int> disused_kv_blocks;
    uint32_t         new_blocks_num = 0;
    for (int old_batch_idx = 0; old_batch_idx < old_batch_size; ++old_batch_idx) {
        const int fork_count = batch_fork_count[old_batch_idx];
        if (fork_count == 0) {
            const auto& blocks = kv_cache_resource->blocks(old_batch_idx, 0);
            disused_kv_blocks.insert(disused_kv_blocks.end(), blocks.begin(), blocks.end());
        } else if (fork_count > 1 && copy_last_block) {
            new_blocks_num += static_cast<uint32_t>(fork_count - 1);
        }
    }

    // free disused first to reclaim capacity
    if (!disused_kv_blocks.empty()) {
        full_kv_cache_group_->free(disused_kv_blocks);
    }

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
                full_kv_cache_group_->free({old_block});

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
