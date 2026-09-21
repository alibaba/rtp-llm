#include "rtp_llm/cpp/cache/SingleTypeKVCacheAllocator.h"

#include <algorithm>
#include <unordered_map>

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
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

    BlockPoolConfig pool_config;

    pool_config = BlockPoolConfigHelper::createConfig(config_);
    block_pool_ = std::make_shared<BlockPool>(pool_config, allocation_type_);
    if (!block_pool_->init()) {
        RTP_LLM_LOG_ERROR("Failed to initialize block pool for SingleTypeKVCacheAllocator");
        return false;
    }

    SharedBlockCache* shared_cache_raw = shared_block_cache_ ? shared_block_cache_.get() : nullptr;
    if (shared_block_cache_) {
        shared_block_cache_->init(1, std::vector<BlockPoolPtr>{block_pool_});
    }

    full_kv_cache_group_ = std::make_shared<FullKVCacheGroup>(cache_group, block_pool_, 0, shared_cache_raw, nullptr);

    if (!full_kv_cache_group_->init()) {
        RTP_LLM_LOG_ERROR("Failed to initialize FullKVCacheGroup");
        return false;
    }

    RTP_LLM_LOG_INFO("SingleTypeKVCacheAllocator initialized successfully");
    return true;
}

MallocResult SingleTypeKVCacheAllocator::initMallocForCommonLen(const MallocInfo& malloc_info) {
    auto& kv_resource = malloc_info.batch_kv_cache_resource;
    int   reuse_len   = 0;
    int   common_seq_len =
        std::min(malloc_info.complete_token_ids->commonSeqLength(), malloc_info.complete_token_ids->totalSeqLength());

    const auto& cache_keys         = kv_resource->cacheKeys(0);
    auto&       block_ids_0        = kv_resource->mutableBlockIds(0, 0);
    int64_t     match_cost_time_us = 0;

    const size_t reserve_blocks   = reserveBlocksNum();
    const int    estimated_blocks = (reserve_blocks > 0) ? getNeedBlocks(malloc_info) : 0;
    int          reuse_blocks     = 0;

    // drop the last cache key of the partial block to avoid reuse it for two reasons:
    // 1. if the last block is partial, it actually cannot be reused, because only full blocks will be inserted into the
    // cache.
    // 2. if the last block is full and matched, the reuse length will be equal to the seq_len, which causes core dump
    // in computing ops.
    if (malloc_info.enable_device_cache && full_kv_cache_group_->prefixReuseEnabled()) {
        CacheKeysType match_keys(cache_keys.begin(), cache_keys.empty() ? cache_keys.end() : cache_keys.end() - 1);
        auto          match_begin_time_us = currentTimeUs();
        auto          match_result        = full_kv_cache_group_->match(match_keys);
        match_cost_time_us                = currentTimeUs() - match_begin_time_us;
        reuse_len                         = static_cast<int>(match_result.reuse_length);
        reuse_blocks                      = static_cast<int>(match_result.reuse_blocks);
        kv_resource->cacheResource(0).setDeviceReuseBlockNum(reuse_blocks);
        full_kv_cache_group_->reference(block_ids_0, match_result.block_indices);
    }

    // Check if available blocks are enough for the request.
    if (reserve_blocks > 0 && estimated_blocks > 0) {
        const size_t available_blocks = availableBlocksNum();
        const int    actual_blocks    = std::max(estimated_blocks - reuse_blocks, 0);
        if (actual_blocks > 0 && available_blocks < static_cast<size_t>(actual_blocks) + reserve_blocks) {
            if (malloc_info.verbose) {
                RTP_LLM_LOG_INFO("SingleTypeKVCacheAllocator initMalloc rejected by reserve blocks: request_id=%ld "
                                 "need_blocks=%d reuse_blocks=%d adjusted_need_blocks=%d available_blocks=%zu "
                                 "reserve_blocks=%zu",
                                 malloc_info.request_id,
                                 estimated_blocks,
                                 reuse_blocks,
                                 actual_blocks,
                                 available_blocks,
                                 reserve_blocks);
            }
            return {false, 0};
        }
    }

    if (!full_kv_cache_group_->malloc(block_ids_0, common_seq_len)) {
        return {false, 0};
    }

    // other batches reference batch 0's blocks
    for (int batch_id = 1; batch_id < kv_resource->batchSize(); ++batch_id) {
        full_kv_cache_group_->reference(kv_resource->mutableBlockIds(batch_id, 0), block_ids_0.blocks());
    }

    return {true, reuse_len, match_cost_time_us};
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
        full_kv_cache_group_->free(blocks_to_free);
    }
    return {false, 0};
}

void SingleTypeKVCacheAllocator::free(const FreeInfo& free_info) {
    auto& kv_cache_resource = free_info.batch_kv_cache_resource;

    if (kv_cache_resource->curBlocksNum() == 0) {
        return;
    }

    std::vector<const BlockIndicesType*> blocks;
    blocks.reserve(kv_cache_resource->batchSize());
    for (int i = 0; i < kv_cache_resource->batchSize(); ++i) {
        blocks.push_back(&kv_cache_resource->blocks(i, 0));
    }
    {
        RTP_LLM_PROFILE_SCOPE("kv_free.batch_release");
        full_kv_cache_group_->freeBatch(blocks);
    }
    {
        RTP_LLM_PROFILE_SCOPE("kv_free.clear_blocks");
        kv_cache_resource->clearBlocks();
    }
}

void SingleTypeKVCacheAllocator::insertIntoCache(const InsertInfo& insert_info) {
    if (!full_kv_cache_group_->prefixReuseEnabled()) {
        return;
    }

    auto& kv_resource = insert_info.batch_kv_cache_resource;
    int   batch_size  = kv_resource->batchSize();

    // TODO(chanyin): set batch_size to 1 for now
    batch_size = 1;

    for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
        const auto& cache_keys = kv_resource->cacheKeys(batch_id);
        const auto& blocks     = kv_resource->blocks(batch_id, 0);

        size_t block_num = std::min(size_t(cache_keys.size()), size_t(blocks.size()));
        if (block_num == 0) {
            continue;
        }

        CacheKeysType    put_cache_keys(cache_keys.begin(), cache_keys.begin() + block_num);
        BlockIndicesType put_block_ids(blocks.begin(), blocks.begin() + block_num);

        full_kv_cache_group_->insertIntoCache(put_cache_keys, put_block_ids, insert_info.is_resident);
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

    if (is_connector) {
        block_pool_->connectorReference(real_blocks);
    } else {
        block_pool_->requestReference(real_blocks);
    }
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
        if (is_connector) {
            block_pool_->connectorFree(blocks_to_free);
        } else {
            block_pool_->requestFree(blocks_to_free);
        }
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
    std::vector<BlockPool::RequestRefDelta> deltas;
    int new_blocks_num = 0;
    {
        RTP_LLM_PROFILE_SCOPE("kv_cpu.plan_parents");
        for (const int parent : block_src_batch) {
            RTP_LLM_CHECK(parent >= 0 && parent < old_batch_size);
            ++batch_fork_count[parent];
        }
        std::unordered_map<BlockIdxType, int> counts;
        for (int parent = 0; parent < old_batch_size; ++parent) {
            const auto& blocks = kv_cache_resource->blocks(parent, 0);
            const int forks = batch_fork_count[parent];
            if (forks == 0) {
                for (const auto block : blocks) {
                    --counts[block];
                }
            } else if (forks > 1) {
                const bool replace_tail = copy_last_block && !blocks.empty();
                const size_t shared_count = blocks.size() - (replace_tail ? 1 : 0);
                for (size_t i = 0; i < shared_count; ++i) {
                    counts[blocks[i]] += forks - 1;
                }
                new_blocks_num += replace_tail ? forks - 1 : 0;
            }
        }
        deltas.reserve(counts.size());
        for (const auto& item : counts) {
            if (item.second != 0) {
                deltas.push_back({item.first, item.second});
            }
        }
        std::sort(deltas.begin(), deltas.end(), [](const auto& a, const auto& b) { return a.block_id < b.block_id; });
    }

    // Prepare all host allocations before committing pool ownership. Survivor
    // slots remain empty until commit; their old resources will be moved once.
    std::vector<KVCacheResource> new_resources(new_batch_size);
    std::vector<std::pair<int, int>> survivors;
    std::vector<int> tail_destinations;
    std::vector<TaggedBlockIdPair> new_mapping;
    survivors.reserve(old_batch_size);
    tail_destinations.reserve(new_blocks_num);
    new_mapping.reserve(new_blocks_num);
    {
        RTP_LLM_PROFILE_SCOPE("kv_cpu.build_beam_tables");
        for (int child = 0; child < new_batch_size; ++child) {
            const int parent = block_src_batch[child];
            auto& remaining = batch_fork_count[parent];
            if (--remaining == 0) {
                survivors.emplace_back(child, parent);
                continue;
            }
            auto& resource = new_resources[child];
            resource.initGroups(config_.topologyPtr(), /*materialize_layer_views=*/false);
            resource.cacheKeys() = kv_cache_resource->cacheKeys(parent);
            const auto& old_blocks = kv_cache_resource->blocks(parent, 0);
            auto& blocks = resource.mutableBlockIds(0);
            blocks.assign(old_blocks);
            if (copy_last_block && !old_blocks.empty()) {
                blocks.setAt(old_blocks.size() - 1, NULL_BLOCK_IDX);
                tail_destinations.push_back(child);
                new_mapping.push_back({config_.topology().soleGroupForLayer(0).tag, old_blocks.back(), NULL_BLOCK_IDX});
            }
        }
    }

    BlockIndicesType new_blocks;
    int required_free_blocks = 0;
    {
        RTP_LLM_PROFILE_SCOPE("kv_cpu.commit_pool");
        if (!full_kv_cache_group_->reassignRequestBlocks(deltas, new_blocks_num, new_blocks, required_free_blocks)) {
            // The failed transaction left all old refs intact. Evict cached
            // pages only when necessary, then atomically recheck capacity.
            if (!full_kv_cache_group_->ensureFreeBlocks(required_free_blocks)
                || !full_kv_cache_group_->reassignRequestBlocks(deltas, new_blocks_num, new_blocks, required_free_blocks)) {
                return false;
            }
        }
    }
    {
        RTP_LLM_PROFILE_SCOPE("kv_cpu.publish_tables");
        for (size_t i = 0; i < new_blocks.size(); ++i) {
            auto& blocks = new_resources[tail_destinations[i]].mutableBlockIds(0);
            blocks.setAt(blocks.blocksNum() - 1, new_blocks[i]);
            new_mapping[i].dst = new_blocks[i];
        }
        for (const auto& survivor : survivors) {
            new_resources[survivor.first] = std::move(kv_cache_resource->cacheResource(survivor.second));
        }
        kv_cache_resource->swapResources(new_resources);
        block_update_mapping.swap(new_mapping);
    }
    {
        RTP_LLM_PROFILE_SCOPE("kv_cpu.retire_old_resources");
        new_resources.clear();
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
