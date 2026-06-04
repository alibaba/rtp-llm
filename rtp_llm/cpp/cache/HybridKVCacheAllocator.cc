#include "rtp_llm/cpp/cache/HybridKVCacheAllocator.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/ZeroSwaCacheHelper.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {
namespace {

// CP shard helpers: when mapper is null/passthrough, all helpers no-op.
inline int cpEffectiveSeqLen(const std::shared_ptr<CPSlotMapper>& mapper, int seq_len) {
    return (mapper && mapper->isSharded()) ? mapper->effectiveSeqLenForAlloc(seq_len) : seq_len;
}

inline CacheKeysType cpEffectiveCacheKeys(const std::shared_ptr<CPSlotMapper>& mapper, const CacheKeysType& full) {
    if (!mapper || !mapper->isSharded()) {
        return full;
    }
    CacheKeysType local;
    const int     cp_size = mapper->cpSize();
    const int     start   = cp_size - 1;
    for (int i = start; i < static_cast<int>(full.size()); i += cp_size) {
        local.push_back(full[i]);
    }
    return local;
}

inline int cpVirtualBlockSize(const std::shared_ptr<CPSlotMapper>& mapper, int block_size) {
    return (mapper && mapper->isSharded()) ? mapper->virtualBlockSize() : block_size;
}

// Per-group gate: only RR-shard groups that participate in cache_keys reuse
// (paged FULL groups). Non-paged groups (SWA / STATE pools with
// fixed_blocks_per_req) keep raw seq_len semantics — sharing the cp shrink
// would shrink their per-request block count incorrectly.
inline bool cpShardThisGroup(const std::shared_ptr<CPSlotMapper>& mapper, CacheGroupType group_type) {
    return mapper && mapper->isSharded() && group_type == CacheGroupType::FULL;
}

inline bool containsGroupId(const std::vector<int>& group_ids, int gid) {
    return std::find(group_ids.begin(), group_ids.end(), gid) != group_ids.end();
}

inline int
cpEffectiveSeqLenForGroup(const std::shared_ptr<CPSlotMapper>& mapper, CacheGroupType group_type, int seq_len) {
    return cpShardThisGroup(mapper, group_type) ? mapper->effectiveSeqLenForAlloc(seq_len) : seq_len;
}

inline int
cpVirtualBlockSizeForGroup(const std::shared_ptr<CPSlotMapper>& mapper, CacheGroupType group_type, int block_size) {
    return cpShardThisGroup(mapper, group_type) ? mapper->virtualBlockSize() : block_size;
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

bool HybridKVCacheAllocator::skipReuseCacheGroup(int gid) const {
    return static_cast<size_t>(gid) < config_.group_region_names.size()
           && skipReuseCacheRegion(config_.group_region_names[static_cast<size_t>(gid)]);
}

HybridKVCacheAllocator::HybridKVCacheAllocator(const CacheConfig&                 config,
                                               AllocationType                     allocation_type,
                                               const kmonitor::MetricsReporterPtr metrics_reporter,
                                               int64_t                            reserve_block_ratio):
    KVCacheAllocator(config, allocation_type, metrics_reporter, reserve_block_ratio) {}

int HybridKVCacheAllocator::reuseCache(const CacheKeysType&                 cache_keys,
                                       BatchKVCacheResource&                kv_resource,
                                       const std::shared_ptr<CPSlotMapper>& cp_mapper,
                                       int                                  seq_len) {
    // Under cp shard, FULL groups index block_ids by cp-virtual-block units
    // (one entry covers cp_size physical blocks). LINEAR/SWA groups index by
    // raw block_size logical blocks. So when populating tail blocks for
    // LINEAR/SWA we need to scale the array length and matched-block position
    // back to the logical-block coordinate system.
    const int                     cp_scale = (cp_mapper && cp_mapper->isSharded()) ? cp_mapper->cpSize() : 1;
    int                           min_full_reuse_blocks = static_cast<int>(cache_keys.size());
    std::vector<BlockIndicesType> full_matched_blocks(kv_cache_groups_.size());

    for (int gid : full_group_ids_) {
        auto match_result     = kv_cache_groups_[static_cast<size_t>(gid)]->match(cache_keys);
        min_full_reuse_blocks = std::min(min_full_reuse_blocks, static_cast<int>(match_result.reuse_blocks));
        full_matched_blocks[static_cast<size_t>(gid)] = std::move(match_result.block_indices);
    }

    const int reuse_unit_tokens = cpVirtualBlockSize(cp_mapper, seqSizePerBlock());
    const int capped_full_reuse_blocks =
        capReuseBlocksForZeroSwaCaching(config_, min_full_reuse_blocks, reuse_unit_tokens, seq_len);
    const int zero_swa_restore_blocks_to_drop = std::max(min_full_reuse_blocks - capped_full_reuse_blocks, 0);
    if (config_.dsv4_zero_swa_caching && min_full_reuse_blocks != capped_full_reuse_blocks) {
        RTP_LLM_LOG_DEBUG("zero SWA caching caps device reuse blocks: full_matched=%d capped=%d "
                          "restore_tokens=%llu reuse_unit_tokens=%d seq_len=%d drop_blocks=%d",
                          min_full_reuse_blocks,
                          capped_full_reuse_blocks,
                          static_cast<unsigned long long>(zeroSwaRestoreWindowTokens(config_)),
                          reuse_unit_tokens,
                          seq_len,
                          zero_swa_restore_blocks_to_drop);
    }

    int                           pos = capped_full_reuse_blocks - 1;
    std::vector<BlockIdxType>     linear_tail_blocks(linear_group_ids_.size(), NULL_BLOCK_IDX);
    std::vector<BlockIndicesType> swa_tail_blocks(swa_group_ids_.size());
    bool                          has_tail_groups = !linear_group_ids_.empty();
    for (int gid : swa_group_ids_) {
        if (!skipSwaKvForZeroSwaCaching(config_, gid)) {
            has_tail_groups = true;
            break;
        }
    }
    for (; pos >= 0 && has_tail_groups; --pos) {
        bool                          all_tail_groups_matched = true;
        std::vector<BlockIdxType>     candidate_linear_tail_blocks(linear_group_ids_.size(), NULL_BLOCK_IDX);
        std::vector<BlockIndicesType> candidate_swa_tail_blocks(swa_group_ids_.size());
        for (size_t i = 0; i < linear_group_ids_.size(); ++i) {
            const int gid      = linear_group_ids_[i];
            auto* linear_group = dynamic_cast<LinearKVCacheGroup*>(kv_cache_groups_[static_cast<size_t>(gid)].get());
            RTP_LLM_CHECK_WITH_INFO(linear_group != nullptr, "group %d is not LinearKVCacheGroup", gid);
            auto result = linear_group->matchSingleKey(cache_keys[static_cast<size_t>(pos)]);
            if (result.block_indices.empty()) {
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
            if (skipSwaKvForZeroSwaCaching(config_, gid)) {
                continue;
            }
            auto*     swa_group = dynamic_cast<SWAKVCacheGroup*>(kv_cache_groups_[static_cast<size_t>(gid)].get());
            RTP_LLM_CHECK_WITH_INFO(swa_group != nullptr, "group %d is not SWAKVCacheGroup", gid);
            if (skipReuseCacheGroup(gid)) {
                continue;
            }
            auto result = swa_group->matchSingleKey(cache_keys[static_cast<size_t>(pos)]);
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

    const int reuse_blocks_len = has_tail_groups ? std::max(pos + 1, 0) : std::max(capped_full_reuse_blocks, 0);
    if (reuse_blocks_len <= 0) {
        // Under zero-SWA the STATE tail groups (INDEXER/CSA/HCA_STATE) are still
        // required to anchor the tail. If they miss at every candidate position,
        // reuse collapses to 0 even though the FULL groups matched a long prefix --
        // an all-or-nothing cliff that silently defeats the cache-capacity benefit.
        // Surface it so the regression is observable instead of looking like a miss.
        if (config_.dsv4_zero_swa_caching && capped_full_reuse_blocks > 0) {
            RTP_LLM_LOG_WARNING("zero SWA caching: reuse collapsed to 0 although FULL groups matched %d blocks "
                                "(capped=%d); STATE tail groups failed to anchor the tail",
                                min_full_reuse_blocks,
                                capped_full_reuse_blocks);
        }
        kv_resource.cacheResource(0).setZeroSwaFullReuseTokenNum(0);
        return 0;
    }

    // Per-region reuse for zero-SWA TRIM: paged FULL groups (CSA_KV/HCA_KV/INDEXER_KV) keep
    // the full tail-anchored match so recomputed restore-window tokens READ cached compressed/
    // indexer KV instead of recomputing it. The returned reuse_blocks_len (-> reuse_length_ ->
    // prefix_lengths) stays capped so the restore window is still materialized for SWA recompute
    // + write-skipped compressed writes (Python). STATE groups (gid 3/4/5) and SWA_KV (gid 6)
    // are NOT in full_group_ids_ and stay capped (they are ring pools recomputed-as-scratch over
    // the restore window; the compressor raw path supplies the in-flight positions). extend_full
    // gates on dsv4_zero_swa_trim (NOT dsv4_zero_swa_caching) so the C++ FULL extension and the
    // Python compressor write-skip engage as ONE unit. Default path (trim off) -> restore_blocks
    // is 0 -> full_cover == reuse_blocks_len -> byte-identical to the capped reuse below.
    const bool extend_full    = config_.dsv4_zero_swa_trim;
    const int  restore_blocks = extend_full ? zero_swa_restore_blocks_to_drop : 0;
    const int full_cover = std::min(reuse_blocks_len + std::max(restore_blocks, 0), min_full_reuse_blocks);
    if (extend_full && full_cover != reuse_blocks_len) {
        RTP_LLM_LOG_DEBUG("zero SWA trim FULL coverage: reuse_blocks_len=%d full_cover=%d restore_blocks=%d",
                          reuse_blocks_len,
                          full_cover,
                          restore_blocks);
    }
    kv_resource.cacheResource(0).setZeroSwaFullReuseTokenNum(static_cast<size_t>(full_cover) * reuse_unit_tokens);
    for (int gid : full_group_ids_) {
        BlockIndicesType full_blocks = full_matched_blocks[static_cast<size_t>(gid)];
        if (static_cast<int>(full_blocks.size()) > full_cover) {
            full_blocks.resize(static_cast<size_t>(full_cover));
        }
        kv_resource.mutableBlockIds(0, gid).assign(std::move(full_blocks));
    }

    // LINEAR/SWA arrays are sized in logical-block units (cp_size× larger
    // than the FULL groups' cp-virtual-block units). The matched tail block
    // corresponds to the LAST logical block in the canonical (last-rank)
    // namespace, so its index is `(reuse_blocks_len * cp_size) - 1` in
    // logical units, NOT `reuse_blocks_len - 1`.
    const int logical_reuse_len = reuse_blocks_len * cp_scale;
    for (size_t i = 0; i < linear_group_ids_.size(); ++i) {
        const int gid = linear_group_ids_[i];
        kv_resource.mutableBlockIds(0, gid).assign(
            BlockIndicesType(static_cast<size_t>(logical_reuse_len), NULL_BLOCK_IDX));
        kv_resource.mutableBlockIds(0, gid).setAt(static_cast<size_t>(logical_reuse_len - 1), linear_tail_blocks[i]);
    }
    for (size_t i = 0; i < swa_group_ids_.size(); ++i) {
        const int gid = swa_group_ids_[i];
        kv_resource.mutableBlockIds(0, gid).assign(
            BlockIndicesType(static_cast<size_t>(logical_reuse_len), NULL_BLOCK_IDX));
        if (skipReuseCacheGroup(gid) || skipSwaKvForZeroSwaCaching(config_, gid)) {
            continue;
        }
        const size_t tail_begin =
            static_cast<size_t>(std::max(logical_reuse_len - static_cast<int>(swa_tail_blocks[i].size()), 0));
        for (size_t j = 0; j < swa_tail_blocks[i].size(); ++j) {
            kv_resource.mutableBlockIds(0, gid).setAt(tail_begin + j, swa_tail_blocks[i][j]);
        }
    }
    return reuse_blocks_len;
}

MallocResult HybridKVCacheAllocator::initMallocForCommonLen(const MallocInfo& malloc_info) {
    auto&     kv_resource = malloc_info.batch_kv_cache_resource;
    const int batch_size  = kv_resource->batchSize();
    RTP_LLM_CHECK_WITH_INFO(batch_size == 1, "currently batch size should be 1 in hybrid attention but %d", batch_size);

    const int   seq_len        = malloc_info.complete_token_ids->seqLength();
    const int   common_seq_len = std::min(malloc_info.complete_token_ids->commonSeqLength(), seq_len);
    const auto& cp_mapper      = malloc_info.cp_slot_mapper;
    // reuse_unit_tokens is computed against the canonical (paged FULL) group's
    // block_size: cache_keys reuse only happens for paged groups so virtual block
    // size = canonical block_size * cp_size; non-paged groups don't enter reuse.
    const int reuse_unit_tokens = cpVirtualBlockSizeForGroup(cp_mapper, CacheGroupType::FULL, seqSizePerBlock());

    const auto&                   cache_keys         = kv_resource->cacheKeys(0);
    int64_t                       match_cost_time_us = 0;
    const size_t                  reserve_blocks     = reserveBlockNum();
    int                           reuse_blocks       = 0;
    std::vector<BlockIndicesType> referenced_blocks(static_cast<size_t>(kv_resource->groupNums()));

    if (malloc_info.enable_device_cache) {
        // CP-sharded: subsample to last-rank canonical key namespace before matching.
        CacheKeysType cp_keys = cpEffectiveCacheKeys(cp_mapper, cache_keys);
        // Off mode drops the last key to skip the partial trailing block. Under
        // CP sharding cpEffectiveCacheKeys already excludes the partial block
        // (last-rank stride lands inside completed full blocks only), so the
        // extra drop would discard a valid full-block key — costing the SWA
        // tail-loop its only matchable key (full_keys[cp_size-1 + (n-1)*cp_size]
        // is exactly what the non-sharded SWA group caches).
        const bool    cp_active = cp_mapper && cp_mapper->isSharded();
        CacheKeysType match_keys(cp_keys.begin(),
                                 cp_active ? cp_keys.end() : (cp_keys.empty() ? cp_keys.end() : cp_keys.end() - 1));
        auto          begin_us = currentTimeUs();
        reuse_blocks           = reuseCache(match_keys, *kv_resource, cp_mapper, common_seq_len);
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
        auto&      block_ids_0   = kv_resource->mutableBlockIds(0, gid);
        const auto group_type    = static_cast<size_t>(gid) < config_.group_types.size() ?
                                       config_.group_types[static_cast<size_t>(gid)] :
                                       CacheGroupType::FULL;
        const int  group_seq_len = cpEffectiveSeqLenForGroup(cp_mapper, group_type, common_seq_len);
        if (!kv_cache_groups_[static_cast<size_t>(gid)]->malloc(
                block_ids_0, group_seq_len, malloc_info.reuse_cache, 0)) {
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
    return {true, reuse_blocks * reuse_unit_tokens, match_cost_time_us};
}

MallocResult HybridKVCacheAllocator::incrMalloc(const MallocInfo& malloc_info) {
    auto&       kv_resource  = malloc_info.batch_kv_cache_resource;
    const auto& cp_mapper    = malloc_info.cp_slot_mapper;
    const int   batch_size   = kv_resource->batchSize();
    const int   raw_seq_len  = malloc_info.incrSeqLen();
    const int   reserve_step = malloc_info.complete_token_ids->getReserveStep();

    std::vector<std::vector<BlockIndicesType>> original_blocks(static_cast<size_t>(batch_size));
    for (int b = 0; b < batch_size; ++b) {
        original_blocks[static_cast<size_t>(b)].resize(static_cast<size_t>(kv_resource->groupNums()));
        for (int gid = 0; gid < kv_resource->groupNums(); ++gid) {
            original_blocks[static_cast<size_t>(b)][static_cast<size_t>(gid)] = kv_resource->blocks(b, gid);
        }
    }

    bool all_success  = true;
    int  failed_batch = -1;
    int  failed_group = -1;
    for (int b = 0; b < batch_size; ++b) {
        for (int gid = 0; gid < kv_resource->groupNums(); ++gid) {
            auto&      block_ids     = kv_resource->mutableBlockIds(b, gid);
            const auto group_type    = static_cast<size_t>(gid) < config_.group_types.size() ?
                                           config_.group_types[static_cast<size_t>(gid)] :
                                           CacheGroupType::FULL;
            const int  group_seq_len = cpEffectiveSeqLenForGroup(cp_mapper, group_type, raw_seq_len);
            if (!kv_cache_groups_[static_cast<size_t>(gid)]->malloc(
                    block_ids, group_seq_len, malloc_info.reuse_cache, reserve_step)) {
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
        if (!malloc_info.enable_remove_skipped_blocks) {
            return {true, 0};
        }
        for (int b = 0; b < batch_size; ++b) {
            for (int gid = 0; gid < kv_resource->groupNums(); ++gid) {
                kv_cache_groups_[static_cast<size_t>(gid)]->removeSkippedBlocks(
                    kv_resource->mutableBlockIds(b, gid), malloc_info.reuse_cache, reserve_step);
            }
        }
        return {true, 0};
    }

    for (int b = 0; b <= failed_batch && b < batch_size; ++b) {
        for (int gid = 0; gid < kv_resource->groupNums(); ++gid) {
            auto&       block_ids = kv_resource->mutableBlockIds(b, gid);
            const auto& original  = original_blocks[static_cast<size_t>(b)][static_cast<size_t>(gid)];

            std::unordered_set<BlockIdxType> original_valid_blocks;
            original_valid_blocks.reserve(original.size());
            for (auto block : original) {
                if (!isNullBlockIdx(block)) {
                    original_valid_blocks.insert(block);
                }
            }

            BlockIndicesType blocks_to_free;
            for (auto block : block_ids.blocks()) {
                if (!isNullBlockIdx(block) && original_valid_blocks.find(block) == original_valid_blocks.end()) {
                    blocks_to_free.push_back(block);
                }
            }
            if (!blocks_to_free.empty()) {
                freeBlocksInGroup(gid, blocks_to_free);
            }
            block_ids.assign(original);
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
        for (int gid = 0; gid < kv_cache_resource->groupNums(); ++gid) {
            kv_cache_groups_[static_cast<size_t>(gid)]->free(kv_cache_resource->blocks(batch_id, gid));
        }
    }
    kv_cache_resource->clearBlocks();
}

void HybridKVCacheAllocator::insertIntoCache(const InsertInfo& insert_info) {
    auto& kv_cache_resource = insert_info.batch_kv_cache_resource;
    RTP_LLM_CHECK(kv_cache_resource != nullptr);
    if (!shared_block_cache_) {
        return;
    }

    const auto& cp_mapper  = insert_info.cp_slot_mapper;
    const bool  cp_active  = cp_mapper && cp_mapper->isSharded();
    const int   group_nums = kv_cache_resource->groupNums();
    const int   batch_size = kv_cache_resource->batchSize();

    for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
        kv_cache_resource->cacheResource(batch_id).ensureLinearBlockDependencies();
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
                const size_t              i = pos - 1;
                std::vector<BlockIdxType> group_slots(static_cast<size_t>(group_nums), NULL_BLOCK_IDX);
                bool                      has_valid = false;
                for (int gid = 0; gid < group_nums; ++gid) {
                    if (skipReuseCacheGroup(gid) || skipSwaKvForZeroSwaCaching(config_, gid)) {
                        continue;
                    }
                    const auto& blocks = kv_cache_resource->blocks(batch_id, gid);
                    if (i >= blocks.size()) {
                        continue;
                    }
                    if (!isNullBlockIdx(blocks[i])) {
                        group_slots[static_cast<size_t>(gid)] = blocks[i];
                        has_valid                             = true;
                    }
                }
                if (has_valid) {
                    const auto dependency =
                        i < full_dependencies.size() ? full_dependencies[i] :
                                                       BlockDependency{false, 0, static_cast<uint32_t>(i)};
                    shared_block_cache_->put(full_keys[i],
                                             group_slots,
                                             insert_info.is_resident,
                                             SharedBlockCache::kGpuLogicalNamespace,
                                             dependency);
                }
            }
            continue;
        }

        // Per-group key namespace, per-(key, group) put. SharedBlockCache::put
        // merges multiple puts on the same key into a single item with each group's slot
        // populated independently (NULL_BLOCK_IDX entries are skipped by the merge path).
        //
        // CP per-group key namespace: paged FULL groups use cp-subsampled (last-rank) keys
        // to align 1:1 with rank-local blocks; non-paged groups (SWA / LINEAR) keep the
        // full key sequence so their tail blocks (real entries at positions >= length-2)
        // get inserted alongside the keys that the reuseCache tail-loop later queries.
        CacheKeysType cp_keys   = cpEffectiveCacheKeys(cp_mapper, full_keys);
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
        auto          token_ids = insert_info.complete_token_ids->completeTokenIdsVec(batch_id);
        if (token_ids.size() <= 1) {
            continue;
        }
        const size_t token_len = token_ids.size() - 1;

        for (int gid = 0; gid < group_nums; ++gid) {
            if (skipReuseCacheGroup(gid) || skipSwaKvForZeroSwaCaching(config_, gid)) {
                continue;
            }
            const int  raw_group_seq = kv_cache_groups_[static_cast<size_t>(gid)]->seqSizePerBlock();
            const auto group_type    = static_cast<size_t>(gid) < config_.group_types.size() ?
                                           config_.group_types[static_cast<size_t>(gid)] :
                                           (containsGroupId(linear_group_ids_, gid) ?
                                                CacheGroupType::LINEAR :
                                                (containsGroupId(swa_group_ids_, gid) ? CacheGroupType::SWA :
                                                                                        CacheGroupType::FULL));
            if (static_cast<size_t>(gid) >= config_.group_types.size() && group_type != CacheGroupType::FULL) {
                continue;
            }
            const bool           gp_sharded    = cpShardThisGroup(cp_mapper, group_type);
            const CacheKeysType& src_keys      = cp_active && gp_sharded ? cp_keys : full_keys;
            const auto&          dependencies  = cp_active && gp_sharded ? cp_dependencies : full_dependencies;
            const auto           namespace_id  = gp_sharded ? SharedBlockCache::kGpuCpCanonicalNamespace :
                                                               SharedBlockCache::kGpuLogicalNamespace;
            if (src_keys.empty()) {
                continue;
            }
            const int    group_seq_size  = cpVirtualBlockSizeForGroup(cp_mapper, group_type, raw_group_seq);
            const size_t full_blocks_num = token_len / static_cast<size_t>(group_seq_size);
            const size_t n               = std::min(src_keys.size(), full_blocks_num);
            const auto&  blocks          = kv_cache_resource->blocks(batch_id, gid);
            const size_t loop_end        = std::min(n, blocks.size());

            // Reverse iterate so prefix-base keys land at MRU end (matches non-CP path).
            for (size_t pos = loop_end; pos > 0; --pos) {
                const size_t i = pos - 1;
                if (isNullBlockIdx(blocks[i])) {
                    continue;
                }
                std::vector<BlockIdxType> group_slots(static_cast<size_t>(group_nums), NULL_BLOCK_IDX);
                std::vector<bool>         matchable_slots(static_cast<size_t>(group_nums), true);
                group_slots[static_cast<size_t>(gid)] = blocks[i];
                if (static_cast<size_t>(gid) >= config_.group_types.size() && group_type != CacheGroupType::FULL) {
                    matchable_slots[static_cast<size_t>(gid)] = false;
                }
                const auto dependency =
                    i < dependencies.size() ? dependencies[i] : BlockDependency{false, 0, static_cast<uint32_t>(i)};
                shared_block_cache_->put(
                    src_keys[i], group_slots, insert_info.is_resident, namespace_id, dependency, matchable_slots);
            }
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
        const size_t pos = it->second;
        bool any_valid_block = false;
        std::vector<BlockIdxType> blocks_for_key(static_cast<size_t>(kvcache_resource.groupNums()), NULL_BLOCK_IDX);
        for (int gid = 0; gid < kvcache_resource.groupNums(); ++gid) {
            const auto& src_blocks = kvcache_resource.blocks(gid);
            const auto  block      = pos < src_blocks.size() ? src_blocks[pos] : NULL_BLOCK_IDX;
            blocks_for_key[static_cast<size_t>(gid)] = block;
            any_valid_block = any_valid_block || (!isNullBlockIdx(block) && block > 0);
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
        for (int gid = 0; gid < kvcache_resource.groupNums(); ++gid) {
            selected_blocks[static_cast<size_t>(gid)].push_back(blocks_for_key[static_cast<size_t>(gid)]);
        }
    }

    if (selected_keys.empty()) {
        return nullptr;
    }

    selected_resource->setCacheKeys(std::move(selected_keys));
    selected_resource->setCacheKeysAreCpCanonical(kvcache_resource.cacheKeysAreCpCanonical());
    selected_resource->setBlockDependencies(std::move(selected_dependencies));
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
    const auto& cp_mapper          = malloc_info.cp_slot_mapper;
    const int   batch_size         = malloc_info.batch_kv_cache_resource->batchSize();
    const int   total_seq_len      = malloc_info.complete_token_ids->totalSeqLength();
    const int   raw_common_seq_len = std::min(malloc_info.complete_token_ids->commonSeqLength(), total_seq_len);
    const int   raw_seq_len        = malloc_info.complete_token_ids->seqLength();
    const int   reserve_step       = malloc_info.complete_token_ids->getReserveStep();
    const bool  reuse_enabled      = malloc_info.reuse_cache;
    int         reuse_blocks_len   = reuse_enabled ? malloc_info.batch_kv_cache_resource->curBlocksNum() : 0;
    if (reuse_enabled && config_.dsv4_zero_swa_trim) {
        // After reuseCache may extend FULL coverage into the restore window, curBlocksNum()
        // reflects the FULL groups (full_cover). The SWA/STATE ring pools only reuse the
        // returned prefix boundary, so size their reserve need against reuseBlockNum().
        const int capped_boundary =
            static_cast<int>(malloc_info.batch_kv_cache_resource->cacheResource(0).reuseBlockNum());
        if (capped_boundary > 0) {
            reuse_blocks_len = std::min(reuse_blocks_len, capped_boundary);
        }
    }

    int common_blocks_total = 0;
    int extra_blocks_total  = 0;
    for (int gid = 0; gid < static_cast<int>(kv_cache_groups_.size()); ++gid) {
        const auto group_type       = static_cast<size_t>(gid) < config_.group_types.size() ?
                                          config_.group_types[static_cast<size_t>(gid)] :
                                          CacheGroupType::FULL;
        const int  group_common_seq = cpEffectiveSeqLenForGroup(cp_mapper, group_type, raw_common_seq_len);
        const int  group_seq_len    = cpEffectiveSeqLenForGroup(cp_mapper, group_type, raw_seq_len);
        const auto need             = kv_cache_groups_[static_cast<size_t>(gid)]->getNeedBlocks(
            group_common_seq, group_seq_len, reserve_step, reuse_blocks_len, reuse_enabled);
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
