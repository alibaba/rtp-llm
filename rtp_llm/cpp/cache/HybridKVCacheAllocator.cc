#include "rtp_llm/cpp/cache/HybridKVCacheAllocator.h"

#include <algorithm>
#include <cstdlib>
#include <unordered_map>
#include <unordered_set>

#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {
namespace {

bool dsv4DebugReuseEnabled() {
    return std::getenv("DSV4_DEBUG_REUSE") != nullptr;
}

int64_t debugCacheKeyAt(const CacheKeysType& cache_keys, size_t pos) {
    if (cache_keys.empty() || pos >= cache_keys.size()) {
        return 0;
    }
    return cache_keys[pos];
}

size_t validBlockCount(const BlockIndicesType& blocks) {
    size_t count = 0;
    for (const auto block : blocks) {
        if (!isNullBlockIdx(block)) {
            ++count;
        }
    }
    return count;
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

HybridKVCacheAllocator::HybridKVCacheAllocator(const CacheConfig&                 config,
                                               AllocationType                     allocation_type,
                                               const kmonitor::MetricsReporterPtr metrics_reporter,
                                               int64_t                            reserve_block_ratio):
    KVCacheAllocator(config, allocation_type, metrics_reporter, reserve_block_ratio) {}

int HybridKVCacheAllocator::reuseCache(const CacheKeysType& cache_keys, BatchKVCacheResource& kv_resource) {
    int                           min_full_reuse_blocks = static_cast<int>(cache_keys.size());
    std::vector<BlockIndicesType> full_matched_blocks(kv_cache_groups_.size());
    const bool                    debug_reuse = dsv4DebugReuseEnabled();

    if (debug_reuse) {
        RTP_LLM_LOG_INFO("DSV4_DEBUG_REUSE reuseCache begin: keys=%zu first_key=%ld last_key=%ld full_groups=%zu "
                         "linear_groups=%zu swa_groups=%zu",
                         cache_keys.size(),
                         debugCacheKeyAt(cache_keys, 0),
                         debugCacheKeyAt(cache_keys, cache_keys.empty() ? 0 : cache_keys.size() - 1),
                         full_group_ids_.size(),
                         linear_group_ids_.size(),
                         swa_group_ids_.size());
    }

    for (int gid : full_group_ids_) {
        auto match_result     = kv_cache_groups_[static_cast<size_t>(gid)]->match(cache_keys);
        min_full_reuse_blocks = std::min(min_full_reuse_blocks, static_cast<int>(match_result.reuse_blocks));
        full_matched_blocks[static_cast<size_t>(gid)] = std::move(match_result.block_indices);
        if (debug_reuse) {
            RTP_LLM_LOG_INFO("DSV4_DEBUG_REUSE reuseCache full gid=%d reuse_blocks=%zu matched_blocks=%zu",
                             gid,
                             match_result.reuse_blocks,
                             full_matched_blocks[static_cast<size_t>(gid)].size());
        }
    }

    int                       pos = min_full_reuse_blocks - 1;
    std::vector<BlockIdxType>    linear_tail_blocks(linear_group_ids_.size(), NULL_BLOCK_IDX);
    std::vector<BlockIndicesType> swa_tail_blocks(swa_group_ids_.size());
    const bool                has_tail_groups = !linear_group_ids_.empty() || !swa_group_ids_.empty();
    for (; pos >= 0 && has_tail_groups; --pos) {
        bool                     all_tail_groups_matched = true;
        std::vector<BlockIdxType> candidate_linear_tail_blocks(linear_group_ids_.size(), NULL_BLOCK_IDX);
        std::vector<BlockIndicesType> candidate_swa_tail_blocks(swa_group_ids_.size());
        for (size_t i = 0; i < linear_group_ids_.size(); ++i) {
            const int gid      = linear_group_ids_[i];
            auto* linear_group = dynamic_cast<LinearKVCacheGroup*>(kv_cache_groups_[static_cast<size_t>(gid)].get());
            RTP_LLM_CHECK_WITH_INFO(linear_group != nullptr, "group %d is not LinearKVCacheGroup", gid);
            auto result = linear_group->matchSingleKey(cache_keys[static_cast<size_t>(pos)]);
            if (result.block_indices.empty()) {
                if (debug_reuse) {
                    RTP_LLM_LOG_INFO("DSV4_DEBUG_REUSE reuseCache linear tail miss gid=%d pos=%d key=%ld",
                                     gid,
                                     pos,
                                     cache_keys[static_cast<size_t>(pos)]);
                }
                all_tail_groups_matched = false;
                break;
            }
            candidate_linear_tail_blocks[i] = result.block_indices[0];
        }
        if (!all_tail_groups_matched) {
            continue;
        }
        for (size_t i = 0; i < swa_group_ids_.size(); ++i) {
            const int gid       = swa_group_ids_[i];
            auto*     swa_group = dynamic_cast<SWAKVCacheGroup*>(kv_cache_groups_[static_cast<size_t>(gid)].get());
            RTP_LLM_CHECK_WITH_INFO(swa_group != nullptr, "group %d is not SWAKVCacheGroup", gid);
            auto result = swa_group->matchSingleKey(cache_keys[static_cast<size_t>(pos)]);
            if (result.block_indices.empty()) {
                if (debug_reuse) {
                    RTP_LLM_LOG_INFO("DSV4_DEBUG_REUSE reuseCache swa tail miss gid=%d pos=%d key=%ld",
                                     gid,
                                     pos,
                                     cache_keys[static_cast<size_t>(pos)]);
                }
                all_tail_groups_matched = false;
                break;
            }
            candidate_swa_tail_blocks[i].push_back(result.block_indices[0]);
            if (debug_reuse && all_tail_groups_matched) {
                RTP_LLM_LOG_INFO("DSV4_DEBUG_REUSE reuseCache swa tail hit gid=%d pos=%d tail_blocks=%zu",
                                 gid,
                                 pos,
                                 candidate_swa_tail_blocks[i].size());
            }
            if (!all_tail_groups_matched) {
                break;
            }
        }
        if (all_tail_groups_matched) {
            linear_tail_blocks = std::move(candidate_linear_tail_blocks);
            swa_tail_blocks    = std::move(candidate_swa_tail_blocks);
            break;
        }
    }

    const int reuse_blocks_len = has_tail_groups ? std::max(pos + 1, 0) : std::max(min_full_reuse_blocks, 0);
    if (reuse_blocks_len <= 0) {
        if (debug_reuse) {
            RTP_LLM_LOG_INFO("DSV4_DEBUG_REUSE reuseCache no hit: min_full_reuse_blocks=%d", min_full_reuse_blocks);
        }
        return 0;
    }
    if (debug_reuse) {
        RTP_LLM_LOG_INFO("DSV4_DEBUG_REUSE reuseCache hit: reuse_blocks=%d reuse_len=%d",
                         reuse_blocks_len,
                         reuse_blocks_len * seqSizePerBlock());
    }

    for (int gid : full_group_ids_) {
        BlockIndicesType full_blocks = full_matched_blocks[static_cast<size_t>(gid)];
        if (static_cast<int>(full_blocks.size()) > reuse_blocks_len) {
            full_blocks.resize(static_cast<size_t>(reuse_blocks_len));
        }
        kv_resource.mutableBlockIds(0, gid).assign(std::move(full_blocks));
    }

    for (size_t i = 0; i < linear_group_ids_.size(); ++i) {
        const int gid = linear_group_ids_[i];
        kv_resource.mutableBlockIds(0, gid).assign(
            BlockIndicesType(static_cast<size_t>(reuse_blocks_len), NULL_BLOCK_IDX));
        kv_resource.mutableBlockIds(0, gid).setAt(static_cast<size_t>(reuse_blocks_len - 1), linear_tail_blocks[i]);
    }
    for (size_t i = 0; i < swa_group_ids_.size(); ++i) {
        const int gid = swa_group_ids_[i];
        kv_resource.mutableBlockIds(0, gid).assign(
            BlockIndicesType(static_cast<size_t>(reuse_blocks_len), NULL_BLOCK_IDX));
        const size_t tail_begin =
            static_cast<size_t>(std::max(reuse_blocks_len - static_cast<int>(swa_tail_blocks[i].size()), 0));
        for (size_t j = 0; j < swa_tail_blocks[i].size(); ++j) {
            if (debug_reuse) {
                RTP_LLM_LOG_INFO("DSV4_DEBUG_REUSE reuseCache restore swa gid=%d dst_pos=%zu block=%d",
                                 gid,
                                 tail_begin + j,
                                 swa_tail_blocks[i][j]);
            }
            kv_resource.mutableBlockIds(0, gid).setAt(tail_begin + j, swa_tail_blocks[i][j]);
        }
    }
    return reuse_blocks_len;
}

MallocResult HybridKVCacheAllocator::initMallocForCommonLen(const MallocInfo& malloc_info) {
    auto&     kv_resource = malloc_info.batch_kv_cache_resource;
    const int batch_size  = kv_resource->batchSize();

    const int seq_len        = malloc_info.complete_token_ids->seqLength();
    const int common_seq_len = std::min(malloc_info.complete_token_ids->commonSeqLength(), seq_len);

    const auto&                   cache_keys         = kv_resource->cacheKeys(0);
    int64_t                       match_cost_time_us = 0;
    const size_t                  reserve_blocks     = reserveBlockNum();
    int                           reuse_blocks       = 0;
    std::vector<BlockIndicesType> referenced_blocks(static_cast<size_t>(kv_resource->groupNums()));

    if (malloc_info.enable_device_cache) {
        CacheKeysType match_keys(cache_keys.begin(), cache_keys.empty() ? cache_keys.end() : cache_keys.end() - 1);
        auto          begin_us = currentTimeUs();
        reuse_blocks           = reuseCache(match_keys, *kv_resource);
        match_cost_time_us     = currentTimeUs() - begin_us;

        for (int gid = 0; gid < kv_resource->groupNums(); ++gid) {
            const auto&      blocks = kv_resource->blocks(0, gid);
            BlockIndicesType valid;
            valid.reserve(blocks.size());
            for (auto b : blocks) {
                if (!isNullBlockIdx(b)) {
                    valid.push_back(b);
                }
            }
            if (!valid.empty()) {
                referenceBlocksInGroup(gid, valid);
                referenced_blocks[static_cast<size_t>(gid)] = std::move(valid);
            }
        }
        kv_resource->cacheResource(0).setDeviceReuseBlockNum(reuse_blocks);
    }

    if (reserve_blocks > 0 && !hasAvailableBlocksForReserve(malloc_info, reserve_blocks)) {
        rollbackInitMalloc(*kv_resource, referenced_blocks, {});
        return {false, 0};
    }

    std::vector<size_t> original_sizes(static_cast<size_t>(kv_resource->groupNums()));
    for (int gid = 0; gid < kv_resource->groupNums(); ++gid) {
        original_sizes[static_cast<size_t>(gid)] = kv_resource->blocksNum(0, gid);
    }
    for (int gid = 0; gid < kv_resource->groupNums(); ++gid) {
        auto& block_ids_0 = kv_resource->mutableBlockIds(0, gid);
        if (!kv_cache_groups_[static_cast<size_t>(gid)]->malloc(
                block_ids_0, common_seq_len, malloc_info.reuse_cache, 0)) {
            rollbackInitMalloc(*kv_resource, referenced_blocks, original_sizes);
            return {false, 0};
        }
    }

    for (int b = 1; b < batch_size; ++b) {
        for (int gid = 0; gid < kv_resource->groupNums(); ++gid) {
            kv_cache_groups_[static_cast<size_t>(gid)]->reference(kv_resource->mutableBlockIds(b, gid),
                                                                  kv_resource->blocks(0, gid));
        }
    }
    return {true, reuse_blocks * seqSizePerBlock(), match_cost_time_us};
}

MallocResult HybridKVCacheAllocator::incrMalloc(const MallocInfo& malloc_info) {
    auto&     kv_resource  = malloc_info.batch_kv_cache_resource;
    const int batch_size   = kv_resource->batchSize();
    const int seq_len      = malloc_info.complete_token_ids->seqLength();
    const int reserve_step = malloc_info.complete_token_ids->getReserveStep();

    std::vector<std::vector<size_t>> original_sizes(static_cast<size_t>(batch_size));
    for (int b = 0; b < batch_size; ++b) {
        original_sizes[static_cast<size_t>(b)].resize(static_cast<size_t>(kv_resource->groupNums()));
        for (int gid = 0; gid < kv_resource->groupNums(); ++gid) {
            original_sizes[static_cast<size_t>(b)][static_cast<size_t>(gid)] = kv_resource->blocksNum(b, gid);
        }
    }

    for (int b = 0; b < batch_size; ++b) {
        for (int gid = 0; gid < kv_resource->groupNums(); ++gid) {
            kv_cache_groups_[static_cast<size_t>(gid)]->removeSkippedBlocks(
                kv_resource->mutableBlockIds(b, gid), malloc_info.reuse_cache, reserve_step);
        }
    }

    bool all_success  = true;
    int  failed_batch = -1;
    int  failed_group = -1;
    for (int b = 0; b < batch_size; ++b) {
        for (int gid = 0; gid < kv_resource->groupNums(); ++gid) {
            auto& block_ids = kv_resource->mutableBlockIds(b, gid);
            if (!kv_cache_groups_[static_cast<size_t>(gid)]->malloc(
                    block_ids, seq_len, malloc_info.reuse_cache, reserve_step)) {
                all_success  = false;
                failed_batch = b;
                failed_group = gid;
                break;
            }
        }
        if (!all_success) {
            break;
        }
    }

    if (all_success) {
        return {true, 0};
    }

    rollbackIncrMalloc(*kv_resource, original_sizes, failed_batch);
    RTP_LLM_LOG_WARNING("Hybrid incrMalloc failed at batch=%d group=%d", failed_batch, failed_group);
    return {false, 0};
}

void HybridKVCacheAllocator::free(const FreeInfo& free_info) {
    auto& kv_cache_resource = free_info.batch_kv_cache_resource;
    if (kv_cache_resource->curBlocksNum() == 0) {
        return;
    }
    for (int batch_id = 0; batch_id < kv_cache_resource->batchSize(); ++batch_id) {
        for (int gid = 0; gid < kv_cache_resource->groupNums(); ++gid) {
            kv_cache_groups_[static_cast<size_t>(gid)]->free(kv_cache_resource->blocks(batch_id, gid));
        }
    }
    kv_cache_resource->clearBlocks();
}

void HybridKVCacheAllocator::insertIntoCache(const InsertInfo& insert_info) {
    auto& kv_cache_resource = insert_info.batch_kv_cache_resource;
    RTP_LLM_CHECK(kv_cache_resource != nullptr);

    const int batch_size = kv_cache_resource->batchSize();
    const bool debug_reuse = dsv4DebugReuseEnabled();
    for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
        const auto& cache_keys = kv_cache_resource->cacheKeys(batch_id);
        auto        token_ids  = insert_info.complete_token_ids->completeTokenIdsVec(batch_id);
        if (token_ids.size() <= 1 || cache_keys.empty()) {
            if (debug_reuse) {
                RTP_LLM_LOG_INFO("DSV4_DEBUG_REUSE insert skip batch=%d token_ids=%zu keys=%zu",
                                 batch_id,
                                 token_ids.size(),
                                 cache_keys.size());
            }
            continue;
        }
        const size_t token_len = token_ids.size() - 1;
        if (debug_reuse) {
            RTP_LLM_LOG_INFO("DSV4_DEBUG_REUSE insert begin: batch=%d token_len=%zu keys=%zu first_key=%ld "
                             "last_key=%ld groups=%d is_resident=%d",
                             batch_id,
                             token_len,
                             cache_keys.size(),
                             debugCacheKeyAt(cache_keys, 0),
                             debugCacheKeyAt(cache_keys, cache_keys.empty() ? 0 : cache_keys.size() - 1),
                             kv_cache_resource->groupNums(),
                             insert_info.is_resident);
        }
        for (int gid = 0; gid < kv_cache_resource->groupNums(); ++gid) {
            const int    group_seq_size  = kv_cache_groups_[static_cast<size_t>(gid)]->seqSizePerBlock();
            const size_t full_blocks_num = token_len / static_cast<size_t>(group_seq_size);
            const size_t n               = std::min(cache_keys.size(), full_blocks_num);
            if (n == 0) {
                if (debug_reuse) {
                    RTP_LLM_LOG_INFO("DSV4_DEBUG_REUSE insert skip group: batch=%d gid=%d group_seq=%d "
                                     "full_blocks=%zu keys=%zu blocks=%zu",
                                     batch_id,
                                     gid,
                                     group_seq_size,
                                     full_blocks_num,
                                     cache_keys.size(),
                                     kv_cache_resource->blocks(batch_id, gid).size());
                }
                continue;
            }
            CacheKeysType    put_cache_keys(cache_keys.begin(), cache_keys.begin() + n);
            const auto&      blocks = kv_cache_resource->blocks(batch_id, gid);
            BlockIndicesType put_blocks;
            put_blocks.reserve(n);
            for (size_t i = 0; i < n && i < blocks.size(); ++i) {
                put_blocks.push_back(blocks[i]);
            }
            if (debug_reuse) {
                RTP_LLM_LOG_INFO("DSV4_DEBUG_REUSE insert group: batch=%d gid=%d group_seq=%d n=%zu blocks=%zu "
                                 "valid_blocks=%zu first_key=%ld last_key=%ld tail_aligned=%d",
                                 batch_id,
                                 gid,
                                 group_seq_size,
                                 n,
                                 blocks.size(),
                                 validBlockCount(put_blocks),
                                 debugCacheKeyAt(put_cache_keys, 0),
                                 debugCacheKeyAt(put_cache_keys, put_cache_keys.empty() ? 0 : put_cache_keys.size() - 1),
                                 0);
            }
            kv_cache_groups_[static_cast<size_t>(gid)]->insertIntoCache(
                put_cache_keys, put_blocks, insert_info.is_resident);
        }
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
    auto deleter               = [self = shared_from_this(), is_connector](KVCacheResource* resource) {
        self->decrKVCacheRef(*resource, is_connector);
        delete resource;
    };
    std::shared_ptr<KVCacheResource> selected_resource(selected_resource_ptr, deleter);
    selected_resource->initGroups(kvcache_resource.groupNums(),
                                  static_cast<int>(config_.layer_all_num),
                                  config_.layer_to_group_id,
                                  config_.kernelBlocksPerKvBlock(),
                                  config_.group_types,
                                  config_.layer_region_to_group_id);

    CacheKeysType&                selected_keys = selected_resource->cacheKeys();
    std::vector<BlockIndicesType> selected_blocks(static_cast<size_t>(kvcache_resource.groupNums()));

    for (auto key : cache_keys) {
        auto it = key_to_pos.find(key);
        if (it == key_to_pos.end()) {
            continue;
        }
        const size_t pos = it->second;
        for (int gid = 0; gid < kvcache_resource.groupNums(); ++gid) {
            const auto& src_blocks = kvcache_resource.blocks(gid);
            if (pos >= src_blocks.size()) {
                continue;
            }
            selected_blocks[static_cast<size_t>(gid)].push_back(src_blocks[pos]);
        }
    }

    selected_keys.assign(cache_keys.begin(), cache_keys.end());
    for (int gid = 0; gid < kvcache_resource.groupNums(); ++gid) {
        BlockIndicesType valid;
        for (auto b : selected_blocks[static_cast<size_t>(gid)]) {
            if (!isNullBlockIdx(b) && b > 0) {
                valid.push_back(b);
            }
        }
        if (!valid.empty()) {
            referenceBlocksInGroup(gid, valid, is_connector);
        }
        selected_resource->mutableBlockIds(gid).assign(std::move(selected_blocks[static_cast<size_t>(gid)]));
    }
    return selected_resource;
}

void HybridKVCacheAllocator::decrKVCacheRef(const KVCacheResource& kvcache_resource, bool is_connector) {
    for (int gid = 0; gid < kvcache_resource.groupNums(); ++gid) {
        BlockIndicesType valid;
        for (auto b : kvcache_resource.blocks(gid)) {
            if (!isNullBlockIdx(b) && b > 0) {
                valid.push_back(b);
            }
        }
        if (!valid.empty()) {
            freeBlocksInGroup(gid, valid, is_connector);
        }
    }
}

bool HybridKVCacheAllocator::updateKVBlock(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                           const std::vector<int>&        block_src_batch,
                                           bool                           copy_last_block,
                                           std::vector<BlockIdPair>&      block_update_mapping) {
    (void)batch_kv_cache_resource;
    (void)block_src_batch;
    (void)copy_last_block;
    (void)block_update_mapping;
    RTP_LLM_FAIL("HybridKVCacheAllocator::updateKVBlock is not supported");
}

int HybridKVCacheAllocator::seqSizePerBlock() const {
    return static_cast<int>(config_.seq_size_per_block);
}

bool HybridKVCacheAllocator::hasAvailableBlocksForReserve(const MallocInfo& malloc_info, size_t reserve_blocks) const {
    const int need_blocks = getNeedBlocks(malloc_info);
    if (need_blocks <= 0) {
        return true;
    }
    const size_t available_blocks = availableBlocksNum();
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

void HybridKVCacheAllocator::rollbackBlockIdsToSize(int gid, BlockIds& block_ids, size_t original_size) {
    if (block_ids.blocksNum() <= original_size) {
        return;
    }
    const auto blocks_to_free = validBlocksAfter(block_ids.blocks(), original_size);
    block_ids.resize(original_size);
    if (!blocks_to_free.empty()) {
        freeBlocksInGroup(gid, blocks_to_free);
    }
}

void HybridKVCacheAllocator::rollbackInitMalloc(BatchKVCacheResource&                kv_resource,
                                                const std::vector<BlockIndicesType>& referenced_blocks,
                                                const std::vector<size_t>&           original_sizes) {
    for (int gid = 0; gid < kv_resource.groupNums(); ++gid) {
        auto& block_ids = kv_resource.mutableBlockIds(0, gid);
        if (!original_sizes.empty() && static_cast<size_t>(gid) < original_sizes.size()
            && block_ids.blocksNum() > original_sizes[static_cast<size_t>(gid)]) {
            rollbackBlockIdsToSize(gid, block_ids, original_sizes[static_cast<size_t>(gid)]);
        }
        if (static_cast<size_t>(gid) < referenced_blocks.size()
            && !referenced_blocks[static_cast<size_t>(gid)].empty()) {
            freeBlocksInGroup(gid, referenced_blocks[static_cast<size_t>(gid)]);
        }
        block_ids.resize(0);
    }
    kv_resource.cacheResource(0).setDeviceReuseBlockNum(0);
}

void HybridKVCacheAllocator::rollbackIncrMalloc(BatchKVCacheResource&                   kv_resource,
                                                const std::vector<std::vector<size_t>>& original_sizes,
                                                int                                     failed_batch) {
    const int last_touched_batch = std::min(failed_batch, kv_resource.batchSize() - 1);
    for (int b = 0; b <= last_touched_batch; ++b) {
        for (int gid = 0; gid < kv_resource.groupNums(); ++gid) {
            auto&        block_ids    = kv_resource.mutableBlockIds(b, gid);
            const size_t original_num = original_sizes[static_cast<size_t>(b)][static_cast<size_t>(gid)];
            rollbackBlockIdsToSize(gid, block_ids, original_num);
        }
    }
}

int HybridKVCacheAllocator::getNeedBlocks(const MallocInfo& malloc_info) const {
    if (!malloc_info.batch_kv_cache_resource || !malloc_info.complete_token_ids) {
        return 0;
    }
    const int  batch_size       = malloc_info.batch_kv_cache_resource->batchSize();
    const int  total_seq_len    = malloc_info.complete_token_ids->totalSeqLength();
    const int  common_seq_len   = std::min(malloc_info.complete_token_ids->commonSeqLength(), total_seq_len);
    const int  seq_len          = malloc_info.complete_token_ids->seqLength();
    const int  reserve_step     = malloc_info.complete_token_ids->getReserveStep();
    const bool reuse_enabled    = malloc_info.reuse_cache;
    const int  reuse_blocks_len = reuse_enabled ? malloc_info.batch_kv_cache_resource->curBlocksNum() : 0;

    int common_blocks_total = 0;
    int extra_blocks_total  = 0;
    for (int gid = 0; gid < static_cast<int>(kv_cache_groups_.size()); ++gid) {
        const auto need = kv_cache_groups_[static_cast<size_t>(gid)]->getNeedBlocks(
            common_seq_len, seq_len, reserve_step, reuse_blocks_len, reuse_enabled);
        common_blocks_total += need.common_blocks;
        extra_blocks_total += need.extra_blocks;
    }
    return common_blocks_total + batch_size * extra_blocks_total;
}

int HybridKVCacheAllocator::singleBatchNeedBlocks(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                                  int                            seq_len,
                                                  int                            reserve_step) const {
    int need_blocks = 0;
    for (int gid = 0; gid < batch_kv_cache_resource->groupNums(); ++gid) {
        const int cur_blocks = batch_kv_cache_resource->blocksNum(0, gid);
        need_blocks += kv_cache_groups_[static_cast<size_t>(gid)]->needBlocksNum(seq_len, cur_blocks, reserve_step);
    }
    return need_blocks;
}

}  // namespace rtp_llm
