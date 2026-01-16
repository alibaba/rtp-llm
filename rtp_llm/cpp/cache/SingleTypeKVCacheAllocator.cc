#include "rtp_llm/cpp/cache/SingleTypeKVCacheAllocator.h"

#include <algorithm>
#include <unordered_map>

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include <sstream>

namespace rtp_llm {

namespace {

std::string formatBlockIdxVec(const BlockIndicesType& v) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) {
            oss << ",";
        }
        oss << v[i];
    }
    oss << "]";
    return oss.str();
}

}  // namespace

int SingleTypeKVCacheAllocator::getNeedBlocks(const MallocInfo& malloc_info) const {
    if (!malloc_info.batch_kv_cache_resource || !malloc_info.complete_token_ids) {
        return 0;
    }

    const int batch_size     = malloc_info.batch_kv_cache_resource->batchSize();
    const int total_seq_len  = malloc_info.complete_token_ids->totalSeqLength();
    const int common_seq_len = std::min(malloc_info.complete_token_ids->commonSeqLength(), total_seq_len);

    const int total_blocks_per_batch = full_kv_cache_group_->needBlocksNum(total_seq_len);
    const int common_blocks          = full_kv_cache_group_->needBlocksNum(common_seq_len);
    const int extra_blocks_per_batch = std::max(total_blocks_per_batch - common_blocks, 0);

    return (batch_size <= 0) ? 0 : (common_blocks + batch_size * extra_blocks_per_batch);
}

SingleTypeKVCacheAllocator::SingleTypeKVCacheAllocator(const CacheConfig&                 config,
                                                       rtp_llm::DeviceBase*               device,
                                                       AllocationType                     allocation_type,
                                                       const kmonitor::MetricsReporterPtr metrics_reporter):
    KVCacheAllocator(config, device, allocation_type, metrics_reporter) {}

bool SingleTypeKVCacheAllocator::init() {
    auto& spec = config_.cache_specs[0];

    BlockPoolConfig pool_config;

    pool_config = BlockPoolConfigHelper::createLayerFirstConfig(config_);
    block_pool_ = std::make_shared<BlockPool>(pool_config, device_, allocation_type_);
    if (!block_pool_->init()) {
        RTP_LLM_LOG_ERROR("Failed to initialize block pool for SingleTypeKVCacheAllocator");
        return false;
    }

    std::vector<int> layer_ids(config_.global_layer_ids[0]);
    if (config_.cache_specs.empty()) {
        RTP_LLM_LOG_ERROR("no cache_specs found in CacheConfig");
        return false;
    }

    full_kv_cache_group_ = std::make_shared<FullKVCacheGroup>(layer_ids, spec, block_pool_, 0);

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
    auto&       blocks_0           = kv_resource->mutableBlocks(0);
    int64_t     match_cost_time_us = 0;

    const size_t reserve_blocks   = reserveBlockNum();
    const int    estimated_blocks = (reserve_blocks > 0) ? getNeedBlocks(malloc_info) : 0;
    int          reuse_blocks     = 0;

    // drop the last cache key of the partial block to avoid reuse it for two reasons:
    // 1. if the last block is partial, it actually cannot be reused, because only full blocks will be inserted into the
    // cache.
    // 2. if the last block is full and matched, the reuse length will be equal to the seq_len, which causes core dump
    // in computing ops.
    if (kv_resource->enable_device_cache) {
        RTP_LLM_LOG_INFO(
            "KVCache upstream(SingleTypeKVCacheAllocator::initMallocForCommonLen) request_id=%ld stage=match "
            "seq_len=%d common_seq_len=%d curBlocks=%d keys0=%zu blocks0=%zu kv_ptr=%p",
            malloc_info.request_id,
            malloc_info.complete_token_ids->totalSeqLength(),
            common_seq_len,
            kv_resource->curBlocksNum(),
            cache_keys.size(),
            blocks_0.size(),
            kv_resource.get());
        CacheKeysType match_keys(cache_keys.begin(), cache_keys.empty() ? cache_keys.end() : cache_keys.end() - 1);
        auto          match_begin_time_us = currentTimeUs();
        auto          match_result        = full_kv_cache_group_->match(match_keys);
        match_cost_time_us                = currentTimeUs() - match_begin_time_us;
        reuse_len                         = static_cast<int>(match_result.reuse_length);
        reuse_blocks                      = static_cast<int>(match_result.reuse_blocks);
        RTP_LLM_LOG_INFO(
            "KVCache upstream(SingleTypeKVCacheAllocator::initMallocForCommonLen) request_id=%ld stage=reference_reuse "
            "reuse_len=%d reuse_blocks=%d matched_blocks=%s",
            malloc_info.request_id,
            reuse_len,
            reuse_blocks,
            formatBlockIdxVec(match_result.block_indices).c_str());
        full_kv_cache_group_->reference(blocks_0, match_result.block_indices);
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

    if (!full_kv_cache_group_->malloc(blocks_0, common_seq_len)) {
        return {false, 0};
    }
    RTP_LLM_LOG_INFO(
        "KVCache upstream(SingleTypeKVCacheAllocator::initMallocForCommonLen) request_id=%ld stage=malloc_common "
        "blocks0=%s",
        malloc_info.request_id,
        formatBlockIdxVec(blocks_0).c_str());

    // other batches reference batch 0's blocks
    for (int batch_id = 1; batch_id < kv_resource->batchSize(); ++batch_id) {
        RTP_LLM_LOG_INFO(
            "KVCache upstream(SingleTypeKVCacheAllocator::initMallocForCommonLen) request_id=%ld stage=reference_other_batch "
            "batch_id=%d blocks0_size=%zu",
            malloc_info.request_id,
            batch_id,
            blocks_0.size());
        full_kv_cache_group_->reference(kv_resource->mutableBlocks(batch_id), blocks_0);
    }

    return {true, reuse_len, match_cost_time_us};
}

MallocResult SingleTypeKVCacheAllocator::incrMalloc(const MallocInfo& malloc_info) {
    auto& kv_resource    = malloc_info.batch_kv_cache_resource;
    int   batch_size     = kv_resource->batchSize();
    int   current_blocks = kv_resource->curBlocksNum();
    int   seq_len        = malloc_info.complete_token_ids->totalSeqLength();

    auto need_blocks = full_kv_cache_group_->needBlocksNum(seq_len, current_blocks);
    if (need_blocks == 0) {
        return {true, 0};
    }
    RTP_LLM_LOG_INFO("KVCache upstream(SingleTypeKVCacheAllocator::incrMalloc) request_id=%ld stage=incr_malloc "
                     "seq_len=%d curBlocks=%d need_blocks=%d batch_size=%d kv_ptr=%p",
                     malloc_info.request_id,
                     seq_len,
                     current_blocks,
                     need_blocks,
                     batch_size,
                     kv_resource.get());

    // Record original sizes for rollback in case any subsequent allocation fails
    std::vector<size_t> original_blocks_num;
    for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
        original_blocks_num.push_back(kv_resource->blocksNum(batch_id));
    }

    bool all_success   = true;
    int  current_batch = 0;
    for (; current_batch < batch_size; ++current_batch) {
        auto& blocks = kv_resource->mutableBlocks(current_batch);
        if (!full_kv_cache_group_->malloc(blocks, seq_len)) {
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
        auto& blocks       = kv_resource->mutableBlocks(batch_id);
        auto  original_num = original_blocks_num[batch_id];
        if (blocks.size() > original_num) {
            blocks_to_free.insert(blocks_to_free.end(), blocks.begin() + original_num, blocks.end());
            blocks.resize(original_num);
        }
    }
    if (!blocks_to_free.empty()) {
        RTP_LLM_LOG_INFO("KVCache upstream(SingleTypeKVCacheAllocator::incrMalloc) request_id=%ld stage=rollback_free "
                         "blocks_to_free=%s",
                         malloc_info.request_id,
                         formatBlockIdxVec(blocks_to_free).c_str());
        full_kv_cache_group_->free(blocks_to_free);
    }
    return {false, 0};
}

void SingleTypeKVCacheAllocator::free(const FreeInfo& free_info) {
    auto& kv_cache_resource = free_info.batch_kv_cache_resource;

    if (kv_cache_resource->curBlocksNum() == 0) {
        return;
    }

    RTP_LLM_LOG_INFO(
        "KVCache upstream(SingleTypeKVCacheAllocator::free) request_id=%ld kv_ptr=%p batch_size=%d curBlocks=%d blocks0=%s",
        free_info.request_id,
        kv_cache_resource.get(),
        kv_cache_resource->batchSize(),
        kv_cache_resource->curBlocksNum(),
        formatBlockIdxVec(kv_cache_resource->blocks(0)).c_str());
    auto all_blocks = kv_cache_resource->getAllBatchBlocks();
    for (const auto& blocks : all_blocks) {
        full_kv_cache_group_->free(blocks);
    }
    kv_cache_resource->clearBlocks();
}

void SingleTypeKVCacheAllocator::insertIntoCache(const InsertInfo& insert_info) {
    auto& kv_resource = insert_info.batch_kv_cache_resource;
    int   batch_size  = kv_resource->batchSize();

    // TODO(chanyin): set batch_size to 1 for now
    batch_size = 1;

    for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
        const auto& cache_keys = kv_resource->cacheKeys(batch_id);
        const auto& blocks     = kv_resource->blocks(batch_id);

        size_t block_num = std::min(size_t(cache_keys.size()), size_t(blocks.size()));
        if (block_num == 0) {
            continue;
        }

        CacheKeysType    put_cache_keys(cache_keys.begin(), cache_keys.begin() + block_num);
        BlockIndicesType put_block_ids(blocks.begin(), blocks.begin() + block_num);

        RTP_LLM_LOG_INFO(
            "KVCache upstream(SingleTypeKVCacheAllocator::insertIntoCache) kv_ptr=%p batch_id=%d is_resident=%d "
            "put_blocks=%s put_keys=%zu",
            kv_resource.get(),
            batch_id,
            insert_info.is_resident,
            formatBlockIdxVec(put_block_ids).c_str(),
            put_cache_keys.size());
        full_kv_cache_group_->insertIntoCache(put_cache_keys, put_block_ids, insert_info.is_resident);
    }
}

CacheLayerLayout SingleTypeKVCacheAllocator::allLayerCacheBase() const {
    CacheLayerLayout layout;
    auto             layer_tensors = full_kv_cache_group_->allLayerCacheBase();
    auto             scale_tensors = full_kv_cache_group_->allLayerScaleCacheBase();

    layout.layers_to_buffer_ptrs.clear();
    layout.layers_to_buffer_ptrs.resize(config_.layer_num);
    layout.layers_to_scale_buffer_ptrs.clear();
    layout.layers_to_scale_buffer_ptrs.resize(config_.layer_num);

    for (int layer_id = 0; layer_id < config_.layer_num; ++layer_id) {
        if (layer_tensors[layer_id].defined() && layer_tensors[layer_id].numel() > 0) {
            layout.layers_to_buffer_ptrs[layer_id] = torchTensor2Buffer(layer_tensors[layer_id]);
        } else {
            layout.layers_to_buffer_ptrs[layer_id] = nullptr;
        }
        if (scale_tensors[layer_id].defined() && scale_tensors[layer_id].numel() > 0) {
            layout.layers_to_scale_buffer_ptrs[layer_id] = torchTensor2Buffer(scale_tensors[layer_id]);
        } else {
            layout.layers_to_scale_buffer_ptrs[layer_id] = nullptr;
        }
    }

    return layout;
}

BlockAddrInfo SingleTypeKVCacheAllocator::convertIndexToAddr(int layer_id, int block_id) const {
    return full_kv_cache_group_->convertIndexToAddr(layer_id, block_id);
}

BlockBufferPtrInfo SingleTypeKVCacheAllocator::convertIndexToBuffer(int layer_id, int block_id) const {
    return full_kv_cache_group_->convertIndexToBuffer(layer_id, block_id);
}

std::vector<BufferPtr> SingleTypeKVCacheAllocator::convertIndexToBuffer(int layer_id,
                                                                        int block_id,
                                                                        int partition_count,
                                                                        int partition_id) const {
    return full_kv_cache_group_->convertIndexToBuffer(layer_id, block_id, partition_count, partition_id);
}

std::shared_ptr<KVCacheResource> SingleTypeKVCacheAllocator::incrKVCacheRef(const KVCacheResource& kvcache_resource,
                                                                            const CacheKeysType&   cache_keys) {
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

    auto selected_resource = std::make_shared<KVCacheResource>();
    selected_resource->initGroups(1, config_.layer_num);

    CacheKeysType&   selected_cache_keys = selected_resource->cacheKeys();
    BlockIndicesType selected_blocks;

    const auto& src_blocks = kvcache_resource.blocks(0);

    for (auto key : cache_keys) {
        auto it = key_to_pos.find(key);
        if (it == key_to_pos.end()) {
            continue;
        }
        const size_t pos = it->second;
        if (pos >= src_blocks.size()) {
            continue;
        }
        const auto block = src_blocks[pos];
        if (block > 0 && !isNullBlockIdx(block)) {
            selected_cache_keys.push_back(key);
            selected_blocks.push_back(block);
        }
    }

    if (selected_blocks.empty()) {
        return nullptr;
    }

    RTP_LLM_LOG_INFO("KVCache upstream(SingleTypeKVCacheAllocator::incrKVCacheRef) stage=blockCacheReference blocks=%s",
                     formatBlockIdxVec(selected_blocks).c_str());
    block_pool_->blockCacheReference(selected_blocks);
    selected_resource->blocks(0) = std::move(selected_blocks);

    return selected_resource;
}

void SingleTypeKVCacheAllocator::decrKVCacheRef(const KVCacheResource& kvcache_resource) {
    RTP_LLM_CHECK_WITH_INFO(
        kvcache_resource.groupNums() == 1, "decrKVCacheRef expects groupNums==1, got %d", kvcache_resource.groupNums());

    const auto& blocks_to_free = kvcache_resource.blocks(0);
    if (!blocks_to_free.empty()) {
        RTP_LLM_LOG_INFO("KVCache upstream(SingleTypeKVCacheAllocator::decrKVCacheRef) stage=blockCacheFree blocks=%s",
                         formatBlockIdxVec(blocks_to_free).c_str());
        block_pool_->blockCacheFree(blocks_to_free);
    }
}

// Update kv blocks for beam search or multi-return sequences.
// - batch_kv_cache_resource: in/out, batch blocks and cache_keys will be rearranged based on block_src_batch
// - block_src_batch: new batch i forks from old batch block_src_batch[i]
// - copy_last_block: whether to copy the last block for each forked batch (instead of sharing)
// - block_update_mapping: out, mapping from old block to new block for batch copy
bool SingleTypeKVCacheAllocator::updateKVBlock(const BatchKVCacheResourcePtr& kv_cache_resource,
                                               const std::vector<int>&        block_src_batch,
                                               bool                           copy_last_block,
                                               std::vector<BlockIdPair>&      block_update_mapping) {
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
            const auto& blocks = kv_cache_resource->blocks(old_batch_idx);
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
    kv_cache_resource->initGroups(1, config_.layer_num);

    for (int new_batch_idx = 0; new_batch_idx < new_batch_size; ++new_batch_idx) {
        const int old_batch_idx = block_src_batch[new_batch_idx];
        auto&     fork_count    = batch_fork_count[old_batch_idx];
        RTP_LLM_CHECK_WITH_INFO(fork_count > 0, "old batch %d has been forked too many times", old_batch_idx);

        if (fork_count == 1) {
            kv_cache_resource->moveBatchResource(new_batch_idx, std::move(old_resources[old_batch_idx]));
        } else {
            auto& blocks = kv_cache_resource->mutableBlocks(new_batch_idx);
            kv_cache_resource->setBatchCacheKeys(new_batch_idx, old_resources[old_batch_idx].cacheKeys());
            full_kv_cache_group_->reference(blocks, old_resources[old_batch_idx].blocks());

            if (copy_last_block && !blocks.empty()) {
                const int old_block = blocks.back();
                full_kv_cache_group_->free({old_block});
                blocks.pop_back();

                // allocate exactly one new block via kvCacheGroup
                int  seq_len_target = (static_cast<int>(blocks.size()) + 1) * full_kv_cache_group_->seqSizePerBlock();
                bool ok             = full_kv_cache_group_->malloc(blocks, seq_len_target);
                RTP_LLM_CHECK_WITH_INFO(ok, "malloc one block via kvCacheGroup failed during kv cache update");
                const int new_block = blocks.back();

                block_update_mapping.push_back(BlockIdPair{old_block, new_block});
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
                                                      int                            seq_len) const {
    const int tokens_per_block = seqSizePerBlock();
    const int cur_blocks_num   = batch_kv_cache_resource ? batch_kv_cache_resource->curBlocksNum() : 0;
    return std::max((seq_len + tokens_per_block - 1) / tokens_per_block - cur_blocks_num, 0);
}

}  // namespace rtp_llm
