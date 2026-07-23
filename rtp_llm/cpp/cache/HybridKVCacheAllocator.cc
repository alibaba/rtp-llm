#include "rtp_llm/cpp/cache/HybridKVCacheAllocator.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
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
cpBlockRoundRobinGroup(const std::shared_ptr<CPSlotMapper>& mapper, const CacheConfig& config, int group_index) {
    return mapper && mapper->isSharded() && group_index >= 0
           && mapper->blockRoundRobinGroup(config.topology().groups().at(static_cast<size_t>(group_index)));
}

inline int cpEffectiveSeqLenForGroup(const std::shared_ptr<CPSlotMapper>& mapper,
                                     const CacheConfig&                   config,
                                     int                                  group_index,
                                     int                                  seq_len) {
    return cpBlockRoundRobinGroup(mapper, config, group_index) ? mapper->effectiveSeqLenForAlloc(seq_len) : seq_len;
}

inline int cpLogicalSeqSizeForGroup(const std::shared_ptr<CPSlotMapper>& mapper,
                                    const CacheConfig&                   config,
                                    int                                  group_index,
                                    int                                  fallback) {
    return (mapper && mapper->isSharded() && group_index >= 0) ?
               static_cast<int>(
                   mapper->logicalSeqSizePerBlock(config.topology().groups().at(static_cast<size_t>(group_index)))) :
               fallback;
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

bool HybridKVCacheAllocator::skipReuseCacheGroup(int group_index) const {
    return group_index >= 0 && static_cast<size_t>(group_index) < kv_cache_groups_.size()
           && !kv_cache_groups_[static_cast<size_t>(group_index)]->prefixReuseEnabled();
}

std::vector<std::string> HybridKVCacheAllocator::independentEvictionGroupTags() const {
    std::vector<std::string> tags;
    for (size_t group_index = 0; group_index < kv_cache_groups_.size(); ++group_index) {
        if (kv_cache_groups_[group_index]->evictPolicy() == CacheEvictPolicy::INDEPENDENT) {
            tags.push_back(kv_cache_groups_[group_index]->tag());
        }
    }
    return tags;
}

bool HybridKVCacheAllocator::cpCompactSwaGroup(int group_index, const std::shared_ptr<CPSlotMapper>& mapper) const {
    return mapper && mapper->isSharded() && group_index >= 0
           && static_cast<size_t>(group_index) < kv_cache_groups_.size()
           && mapper->compactLastRankGroup(config_.topology().groups().at(static_cast<size_t>(group_index)));
}

HybridKVCacheAllocator::HybridKVCacheAllocator(const CacheConfig&                 config,
                                               AllocationType                     allocation_type,
                                               const kmonitor::MetricsReporterPtr metrics_reporter,
                                               int64_t                            reserve_block_ratio):
    KVCacheAllocator(config, allocation_type, metrics_reporter, reserve_block_ratio) {}

int HybridKVCacheAllocator::reuseCache(const CacheKeysType&                 cache_keys,
                                       BatchKVCacheResource&                kv_resource,
                                       const std::shared_ptr<CPSlotMapper>& cp_mapper) {
    // Under cp shard, FULL groups index block_ids by cp-virtual-block units
    // (one entry covers cp_size physical blocks). LINEAR/SWA groups index by
    // raw block_size logical blocks. So when populating tail blocks for
    // LINEAR/SWA we need to scale the array length and matched-block position
    // back to the logical-block coordinate system.
    const int                     cp_scale = (cp_mapper && cp_mapper->isSharded()) ? cp_mapper->cpSize() : 1;
    int                           min_full_reuse_blocks = static_cast<int>(cache_keys.size());
    std::vector<BlockIndicesType> full_matched_blocks(kv_cache_groups_.size());

    for (int group_index : full_group_indices_) {
        auto match_result     = kv_cache_groups_[static_cast<size_t>(group_index)]->match(cache_keys);
        min_full_reuse_blocks = std::min(min_full_reuse_blocks, static_cast<int>(match_result.reuse_blocks));
        full_matched_blocks[static_cast<size_t>(group_index)] = std::move(match_result.block_indices);
    }

    int                           pos = min_full_reuse_blocks - 1;
    std::vector<BlockIdxType>     linear_tail_blocks(linear_group_indices_.size(), NULL_BLOCK_IDX);
    std::vector<BlockIndicesType> swa_tail_blocks(swa_group_indices_.size());
    const bool                    has_tail_groups = !linear_group_indices_.empty() || !swa_group_indices_.empty();
    for (; pos >= 0 && has_tail_groups; --pos) {
        bool                          all_tail_groups_matched = true;
        std::vector<BlockIdxType>     candidate_linear_tail_blocks(linear_group_indices_.size(), NULL_BLOCK_IDX);
        std::vector<BlockIndicesType> candidate_swa_tail_blocks(swa_group_indices_.size());
        for (size_t i = 0; i < linear_group_indices_.size(); ++i) {
            const int group_index = linear_group_indices_[i];
            auto      result      = kv_cache_groups_[static_cast<size_t>(group_index)]->matchSingleKey(
                cache_keys[static_cast<size_t>(pos)]);
            if (result.block_indices.empty()) {
                all_tail_groups_matched = false;
                break;
            }
            candidate_linear_tail_blocks[i] = result.block_indices[0];
        }
        if (!all_tail_groups_matched) {
            continue;
        }
        for (size_t i = 0; i < swa_group_indices_.size(); ++i) {
            const int group_index = swa_group_indices_[i];
            if (skipReuseCacheGroup(group_index)) {
                continue;
            }
            auto result = kv_cache_groups_[static_cast<size_t>(group_index)]->matchSingleKey(
                cache_keys[static_cast<size_t>(pos)]);
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

    const int reuse_blocks_len = has_tail_groups ? std::max(pos + 1, 0) : std::max(min_full_reuse_blocks, 0);
    if (reuse_blocks_len <= 0) {
        return 0;
    }

    for (int group_index : full_group_indices_) {
        BlockIndicesType full_blocks = full_matched_blocks[static_cast<size_t>(group_index)];
        if (static_cast<int>(full_blocks.size()) > reuse_blocks_len) {
            full_blocks.resize(static_cast<size_t>(reuse_blocks_len));
        }
        kv_resource.mutableBlockIdsByIndex(0, static_cast<size_t>(group_index)).assign(std::move(full_blocks));
    }

    // LINEAR/SWA arrays are sized in logical-block units (cp_size× larger
    // than the FULL groups' cp-virtual-block units). The matched tail block
    // corresponds to the LAST logical block in the canonical (last-rank)
    // namespace, so its index is `(reuse_blocks_len * cp_size) - 1` in
    // logical units, NOT `reuse_blocks_len - 1`.
    const int logical_reuse_len = reuse_blocks_len * cp_scale;
    for (size_t i = 0; i < linear_group_indices_.size(); ++i) {
        const int group_index = linear_group_indices_[i];
        kv_resource.mutableBlockIdsByIndex(0, static_cast<size_t>(group_index))
            .assign(BlockIndicesType(static_cast<size_t>(logical_reuse_len), NULL_BLOCK_IDX));
        kv_resource.mutableBlockIdsByIndex(0, static_cast<size_t>(group_index))
            .setAt(static_cast<size_t>(logical_reuse_len - 1), linear_tail_blocks[i]);
    }
    for (size_t i = 0; i < swa_group_indices_.size(); ++i) {
        const int group_index     = swa_group_indices_[i];
        const int group_reuse_len = cpCompactSwaGroup(group_index, cp_mapper) ? reuse_blocks_len : logical_reuse_len;
        kv_resource.mutableBlockIdsByIndex(0, static_cast<size_t>(group_index))
            .assign(BlockIndicesType(static_cast<size_t>(group_reuse_len), NULL_BLOCK_IDX));
        if (skipReuseCacheGroup(group_index)) {
            continue;
        }
        const size_t tail_begin =
            static_cast<size_t>(std::max(group_reuse_len - static_cast<int>(swa_tail_blocks[i].size()), 0));
        for (size_t j = 0; j < swa_tail_blocks[i].size(); ++j) {
            kv_resource.mutableBlockIdsByIndex(0, static_cast<size_t>(group_index))
                .setAt(tail_begin + j, swa_tail_blocks[i][j]);
        }
    }
    return reuse_blocks_len;
}

MallocResult HybridKVCacheAllocator::initMallocForCommonLen(const MallocInfo& malloc_info) {
    auto&     kv_resource = malloc_info.batch_kv_cache_resource;
    const int batch_size  = kv_resource->batchSize();

    const int   seq_len        = malloc_info.complete_token_ids->seqLength();
    const int   common_seq_len = std::min(malloc_info.complete_token_ids->commonSeqLength(), seq_len);
    const auto& cp_mapper      = cp_slot_mapper_;
    // reuse_unit_tokens is computed against the canonical (paged FULL) group's
    // block_size: cache_keys reuse only happens for paged groups so virtual block
    // size = canonical block_size * cp_size; non-paged groups don't enter reuse.
    const int             reuse_group_index = full_group_indices_.empty() ? -1 : full_group_indices_.front();
    const KVCacheGroupPtr reuse_group =
        reuse_group_index < 0 ? KVCacheGroupPtr{} : kv_cache_groups_[static_cast<size_t>(reuse_group_index)];
    const int reuse_unit_tokens =
        (reuse_group ? cpLogicalSeqSizeForGroup(cp_mapper, config_, reuse_group_index, seqSizePerBlock()) :
                       seqSizePerBlock());

    const auto&                   cache_keys         = kv_resource->cacheKeys(0);
    int64_t                       match_cost_time_us = 0;
    const size_t                  reserve_blocks     = reserveBlocksNum();
    int                           reuse_blocks       = 0;
    std::vector<BlockIndicesType> referenced_blocks(static_cast<size_t>(kv_resource->groupNums()));

    if (malloc_info.enable_device_cache) {
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

        for (int group_index = 0; group_index < kv_resource->groupNums(); ++group_index) {
            const auto&      blocks = kv_resource->blocksByIndex(0, static_cast<size_t>(group_index));
            BlockIndicesType valid;
            valid.reserve(blocks.size());
            for (auto b : blocks) {
                if (!isNullBlockIdx(b)) {
                    valid.push_back(b);
                }
            }
            if (!valid.empty()) {
                referenceBlocksInGroup(group_index, valid);
                referenced_blocks[static_cast<size_t>(group_index)] = std::move(valid);
            }
        }
        kv_resource->cacheResource(0).setDeviceReuseBlockNum(reuse_blocks);
    }

    if (reserve_blocks > 0 && !hasAvailableBlocksForReserve(malloc_info, reserve_blocks)) {
        rollbackInitMalloc(*kv_resource, referenced_blocks, {});
        return {false, 0};
    }

    std::vector<size_t> original_sizes(static_cast<size_t>(kv_resource->groupNums()));
    for (int group_index = 0; group_index < kv_resource->groupNums(); ++group_index) {
        original_sizes[static_cast<size_t>(group_index)] =
            kv_resource->blocksNumByIndex(0, static_cast<size_t>(group_index));
    }
    for (int group_index = 0; group_index < kv_resource->groupNums(); ++group_index) {
        auto&     block_ids_0   = kv_resource->mutableBlockIdsByIndex(0, static_cast<size_t>(group_index));
        const int group_seq_len = cpEffectiveSeqLenForGroup(cp_mapper, config_, group_index, common_seq_len);
        if (!kv_cache_groups_[static_cast<size_t>(group_index)]->malloc(
                block_ids_0, group_seq_len, malloc_info.reuse_cache, 0)) {
            rollbackInitMalloc(*kv_resource, referenced_blocks, original_sizes);
            return {false, 0};
        }
    }

    for (int b = 1; b < batch_size; ++b) {
        for (int group_index = 0; group_index < kv_resource->groupNums(); ++group_index) {
            kv_cache_groups_[static_cast<size_t>(group_index)]->reference(
                kv_resource->mutableBlockIdsByIndex(b, static_cast<size_t>(group_index)),
                kv_resource->blocksByIndex(0, static_cast<size_t>(group_index)));
        }
    }
    return {true, reuse_blocks * reuse_unit_tokens, match_cost_time_us};
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
        for (int group_index = 0; group_index < kv_resource->groupNums(); ++group_index) {
            original_sizes[static_cast<size_t>(b)][static_cast<size_t>(group_index)] =
                kv_resource->blocksNumByIndex(b, static_cast<size_t>(group_index));
        }
    }

    bool all_success  = true;
    int  failed_batch = -1;
    int  failed_group = -1;
    for (int b = 0; b < batch_size; ++b) {
        for (int group_index = 0; group_index < kv_resource->groupNums(); ++group_index) {
            auto&     block_ids        = kv_resource->mutableBlockIdsByIndex(b, static_cast<size_t>(group_index));
            const int group_seq_len    = cpEffectiveSeqLenForGroup(cp_mapper, config_, group_index, raw_seq_len);
            auto&     filled_positions = backfilled_positions[static_cast<size_t>(b)][static_cast<size_t>(group_index)];
            if (!kv_cache_groups_[static_cast<size_t>(group_index)]->malloc(
                    block_ids, group_seq_len, malloc_info.reuse_cache, reserve_step, &filled_positions)) {
                all_success  = false;
                failed_batch = b;
                failed_group = group_index;
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
            for (int group_index = 0; group_index < kv_resource->groupNums(); ++group_index) {
                kv_cache_groups_[static_cast<size_t>(group_index)]->removeSkippedBlocks(
                    kv_resource->mutableBlockIdsByIndex(b, static_cast<size_t>(group_index)),
                    malloc_info.reuse_cache,
                    reserve_step);
            }
        }
        return {true, 0};
    }

    for (int b = 0; b <= failed_batch && b < batch_size; ++b) {
        for (int group_index = 0; group_index < kv_resource->groupNums(); ++group_index) {
            auto&       block_ids     = kv_resource->mutableBlockIdsByIndex(b, static_cast<size_t>(group_index));
            const auto  original_size = original_sizes[static_cast<size_t>(b)][static_cast<size_t>(group_index)];
            const auto& filled_positions =
                backfilled_positions[static_cast<size_t>(b)][static_cast<size_t>(group_index)];
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
                freeBlocksInGroup(group_index, blocks_to_free);
            }
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
        for (int group_index = 0; group_index < kv_cache_resource->groupNums(); ++group_index) {
            kv_cache_groups_[static_cast<size_t>(group_index)]->free(
                kv_cache_resource->blocksByIndex(batch_id, static_cast<size_t>(group_index)));
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

    const auto& cp_mapper  = cp_slot_mapper_;
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
                const size_t                      i = pos - 1;
                SharedBlockCache::IndexedBlockIds group_block_ids(static_cast<size_t>(group_nums), NULL_BLOCK_IDX);
                SharedBlockCache::IndexedMatches  matchable_groups(static_cast<size_t>(group_nums), 0);
                bool                              has_group = false;
                for (int group_index = 0; group_index < group_nums; ++group_index) {
                    if (skipReuseCacheGroup(group_index)) {
                        continue;
                    }
                    const auto& blocks = kv_cache_resource->blocksByIndex(batch_id, static_cast<size_t>(group_index));
                    if (i >= blocks.size()) {
                        continue;
                    }
                    if (!isNullBlockIdx(blocks[i])) {
                        group_block_ids[static_cast<size_t>(group_index)]  = blocks[i];
                        matchable_groups[static_cast<size_t>(group_index)] = 1;
                        has_group                                          = true;
                    }
                }
                if (has_group) {
                    const auto dependency = i < full_dependencies.size() ?
                                                full_dependencies[i] :
                                                BlockDependency{false, 0, static_cast<uint32_t>(i)};
                    shared_block_cache_->putIndexed(full_keys[i],
                                                    group_block_ids,
                                                    insert_info.is_resident,
                                                    SharedBlockCache::kGpuLogicalNamespace,
                                                    dependency,
                                                    matchable_groups);
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

        for (int group_index = 0; group_index < group_nums; ++group_index) {
            if (skipReuseCacheGroup(group_index)) {
                continue;
            }
            const int            raw_group_seq = kv_cache_groups_[static_cast<size_t>(group_index)]->seqSizePerBlock();
            const bool           gp_sharded    = cpBlockRoundRobinGroup(cp_mapper, config_, group_index);
            const bool           compact_swa   = cpCompactSwaGroup(group_index, cp_mapper);
            const bool           use_cp_keys   = cp_active && (gp_sharded || compact_swa);
            const CacheKeysType& src_keys      = use_cp_keys ? cp_keys : full_keys;
            const auto&          dependencies  = use_cp_keys ? cp_dependencies : full_dependencies;
            const auto           namespace_id =
                use_cp_keys ? SharedBlockCache::kGpuCpCanonicalNamespace : SharedBlockCache::kGpuLogicalNamespace;
            if (src_keys.empty()) {
                continue;
            }
            const int    group_seq_size  = cpLogicalSeqSizeForGroup(cp_mapper, config_, group_index, raw_group_seq);
            const size_t full_blocks_num = token_len / static_cast<size_t>(group_seq_size);
            const size_t n               = std::min(src_keys.size(), full_blocks_num);
            const auto&  blocks          = kv_cache_resource->blocksByIndex(batch_id, static_cast<size_t>(group_index));
            const size_t loop_end        = std::min(n, blocks.size());
            SharedBlockCache::IndexedBlockIds group_block_ids(static_cast<size_t>(group_nums), NULL_BLOCK_IDX);
            SharedBlockCache::IndexedMatches  matchable_groups(static_cast<size_t>(group_nums), 0);
            matchable_groups[static_cast<size_t>(group_index)] = 1;

            // Reverse iterate so prefix-base keys land at MRU end (matches non-CP path).
            for (size_t pos = loop_end; pos > 0; --pos) {
                const size_t i = pos - 1;
                if (isNullBlockIdx(blocks[i])) {
                    continue;
                }
                group_block_ids[static_cast<size_t>(group_index)] = blocks[i];
                const auto dependency =
                    i < dependencies.size() ? dependencies[i] : BlockDependency{false, 0, static_cast<uint32_t>(i)};
                shared_block_cache_->putIndexed(
                    src_keys[i], group_block_ids, insert_info.is_resident, namespace_id, dependency, matchable_groups);
                group_block_ids[static_cast<size_t>(group_index)] = NULL_BLOCK_IDX;
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
        for (int group_index = 0; group_index < kvcache_resource.groupNums(); ++group_index) {
            const auto& src_blocks = kvcache_resource.groupBlocks().at(static_cast<size_t>(group_index))->blocks();
            const auto  block      = pos < src_blocks.size() ? src_blocks[pos] : NULL_BLOCK_IDX;
            blocks_for_key[static_cast<size_t>(group_index)] = block;
            any_valid_block                                  = any_valid_block || (!isNullBlockIdx(block) && block > 0);
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
        for (int group_index = 0; group_index < kvcache_resource.groupNums(); ++group_index) {
            selected_blocks[static_cast<size_t>(group_index)].push_back(
                blocks_for_key[static_cast<size_t>(group_index)]);
        }
    }

    if (selected_keys.empty()) {
        return nullptr;
    }

    selected_resource->cacheKeys() = std::move(selected_keys);
    selected_resource->setBlockDependencies(std::move(selected_dependencies));
    for (int group_index = 0; group_index < kvcache_resource.groupNums(); ++group_index) {
        BlockIndicesType valid;
        for (auto b : selected_blocks[static_cast<size_t>(group_index)]) {
            if (!isNullBlockIdx(b) && b > 0) {
                valid.push_back(b);
            }
        }
        if (!valid.empty()) {
            referenceBlocksInGroup(group_index, valid, is_connector);
        }
        (*selected_resource->groupBlocks().at(static_cast<size_t>(group_index)))
            .assign(std::move(selected_blocks[static_cast<size_t>(group_index)]));
    }
    return selected_resource;
}

void HybridKVCacheAllocator::decrKVCacheRef(const KVCacheResource& kvcache_resource, bool is_connector) {
    for (int group_index = 0; group_index < kvcache_resource.groupNums(); ++group_index) {
        BlockIndicesType valid;
        for (auto b : kvcache_resource.groupBlocks().at(static_cast<size_t>(group_index))->blocks()) {
            if (!isNullBlockIdx(b) && b > 0) {
                valid.push_back(b);
            }
        }
        if (!valid.empty()) {
            freeBlocksInGroup(group_index, valid, is_connector);
        }
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
            for (int group_index = 0; group_index < group_nums; ++group_index) {
                if (!batch_kv_cache_resource->blocksByIndex(old_batch_idx, static_cast<size_t>(group_index)).empty()) {
                    new_blocks_num[static_cast<size_t>(group_index)] += fork_count - 1;
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
    for (int group_index = 0; group_index < group_nums; ++group_index) {
        std::unordered_set<BlockIdxType>      retained_blocks;
        std::unordered_map<BlockIdxType, int> dropped_block_counts;
        for (int old_batch_idx = 0; old_batch_idx < old_batch_size; ++old_batch_idx) {
            for (const auto block :
                 batch_kv_cache_resource->blocksByIndex(old_batch_idx, static_cast<size_t>(group_index))) {
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

        auto&     replacements = replacement_blocks[static_cast<size_t>(group_index)];
        auto&     transferred  = transferred_ref_counts[static_cast<size_t>(group_index)];
        const int need         = new_blocks_num[static_cast<size_t>(group_index)];
        for (int old_batch_idx = 0; old_batch_idx < old_batch_size && static_cast<int>(replacements.size()) < need;
             ++old_batch_idx) {
            if (batch_fork_count[old_batch_idx] != 0) {
                continue;
            }
            const auto& dropped =
                batch_kv_cache_resource->blocksByIndex(old_batch_idx, static_cast<size_t>(group_index));
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
        for (int group_index = 0; group_index < group_nums; ++group_index) {
            auto& blocks = allocated_replacements[static_cast<size_t>(group_index)];
            if (!blocks.empty()) {
                kv_cache_groups_[static_cast<size_t>(group_index)]->free(blocks);
                blocks.clear();
            }
        }
    };
    for (int group_index = 0; group_index < group_nums; ++group_index) {
        const int need_blocks = new_blocks_num[static_cast<size_t>(group_index)];
        auto&     reserved    = replacement_blocks[static_cast<size_t>(group_index)];
        reserved.reserve(static_cast<size_t>(need_blocks));
        for (int i = static_cast<int>(reserved.size()); i < need_blocks; ++i) {
            BlockIds   one_block;
            const bool ok = kv_cache_groups_[static_cast<size_t>(group_index)]->malloc(
                one_block, kv_cache_groups_[static_cast<size_t>(group_index)]->seqSizePerBlock());
            const auto& blocks = one_block.blocks();
            if (ok && blocks.size() == 1 && !isNullBlockIdx(blocks.front())) {
                reserved.push_back(blocks.front());
                allocated_replacements[static_cast<size_t>(group_index)].push_back(blocks.front());
                continue;
            }
            if (!blocks.empty()) {
                allocated_replacements[static_cast<size_t>(group_index)].insert(
                    allocated_replacements[static_cast<size_t>(group_index)].end(), blocks.begin(), blocks.end());
            }
            RTP_LLM_LOG_WARNING(
                "reserve replacement block failed for hybrid kv cache update, group=%d need=%d reserved=%zu",
                group_index,
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
        for (int group_index = 0; group_index < group_nums; ++group_index) {
            BlockIndicesType to_free;
            auto&            transferred = transferred_ref_counts[static_cast<size_t>(group_index)];
            for (const auto block :
                 batch_kv_cache_resource->blocksByIndex(old_batch_idx, static_cast<size_t>(group_index))) {
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
                kv_cache_groups_[static_cast<size_t>(group_index)]->free(to_free);
            }
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
            for (int group_index = 0; group_index < group_nums; ++group_index) {
                auto& block_ids =
                    batch_kv_cache_resource->mutableBlockIdsByIndex(new_batch_idx, static_cast<size_t>(group_index));
                kv_cache_groups_[static_cast<size_t>(group_index)]->reference(
                    block_ids,
                    old_resources[old_batch_idx].groupBlocks().at(static_cast<size_t>(group_index))->blocks());

                if (copy_last_block && !block_ids.blocks().empty()) {
                    const int  old_block       = block_ids.popBack();
                    const bool old_block_valid = !isNullBlockIdx(old_block) && old_block > 0;
                    if (old_block_valid) {
                        kv_cache_groups_[static_cast<size_t>(group_index)]->free({old_block});
                    }

                    auto&      reserved     = replacement_blocks[static_cast<size_t>(group_index)];
                    const auto reserved_idx = next_replacement[static_cast<size_t>(group_index)]++;
                    RTP_LLM_CHECK_WITH_INFO(reserved_idx < reserved.size(),
                                            "missing reserved replacement block for hybrid kv cache update, group=%d",
                                            group_index);
                    const int new_block = reserved[reserved_idx];
                    block_ids.add({new_block});
                    if (old_block_valid && !isNullBlockIdx(new_block) && new_block > 0) {
                        block_update_mapping.push_back({groupTag(group_index), old_block, new_block});
                    }
                }
            }
        }
        --fork_count;
    }
    for (int group_index = 0; group_index < group_nums; ++group_index) {
        RTP_LLM_CHECK_WITH_INFO(
            next_replacement[static_cast<size_t>(group_index)]
                == replacement_blocks[static_cast<size_t>(group_index)].size(),
            "unused replacement blocks after hybrid kv cache update, group=%d used=%zu reserved=%zu",
            group_index,
            next_replacement[static_cast<size_t>(group_index)],
            replacement_blocks[static_cast<size_t>(group_index)].size());
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

void HybridKVCacheAllocator::rollbackBlockIdsToSize(int group_index, BlockIds& block_ids, size_t original_size) {
    if (block_ids.blocksNum() <= original_size) {
        return;
    }
    const auto blocks_to_free = validBlocksAfter(block_ids.blocks(), original_size);
    block_ids.resize(original_size);
    if (!blocks_to_free.empty()) {
        freeBlocksInGroup(group_index, blocks_to_free);
    }
}

void HybridKVCacheAllocator::rollbackInitMalloc(BatchKVCacheResource&                kv_resource,
                                                const std::vector<BlockIndicesType>& referenced_blocks,
                                                const std::vector<size_t>&           original_sizes) {
    for (int group_index = 0; group_index < kv_resource.groupNums(); ++group_index) {
        auto& block_ids = kv_resource.mutableBlockIdsByIndex(0, static_cast<size_t>(group_index));
        if (!original_sizes.empty() && static_cast<size_t>(group_index) < original_sizes.size()
            && block_ids.blocksNum() > original_sizes[static_cast<size_t>(group_index)]) {
            rollbackBlockIdsToSize(group_index, block_ids, original_sizes[static_cast<size_t>(group_index)]);
        }
        if (static_cast<size_t>(group_index) < referenced_blocks.size()
            && !referenced_blocks[static_cast<size_t>(group_index)].empty()) {
            freeBlocksInGroup(group_index, referenced_blocks[static_cast<size_t>(group_index)]);
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
        for (int group_index = 0; group_index < kv_resource.groupNums(); ++group_index) {
            auto&        block_ids    = kv_resource.mutableBlockIdsByIndex(b, static_cast<size_t>(group_index));
            const size_t original_num = original_sizes[static_cast<size_t>(b)][static_cast<size_t>(group_index)];
            rollbackBlockIdsToSize(group_index, block_ids, original_num);
        }
    }
}

MemoryType HybridKVCacheAllocator::memoryTypeForGroup(int group_index) const {
    (void)group_index;
    return allocation_type_ == AllocationType::DEVICE ? MemoryType::MEMORY_GPU : MemoryType::MEMORY_CPU;
}

void HybridKVCacheAllocator::copyBlockMappingForGroup(int                             group_index,
                                                      const std::vector<BlockIdPair>& block_update_mapping) const {
    if (block_update_mapping.empty()) {
        return;
    }

    const auto&  group               = config_.topology().groups().at(static_cast<size_t>(group_index));
    const auto   memory_type         = memoryTypeForGroup(group_index);
    const auto   copy_type           = BatchCopyParams::get_copy_type(memory_type, memory_type);
    const auto&  spec                = group.spec;
    const size_t kv_block_size_bytes = spec->block_size_bytes();
    const size_t scale_block_bytes   = spec->scale_block_size_bytes();
    const size_t buffers_per_layer   = scale_block_bytes > 0 ? 2 : 1;

    BatchCopyParams copy_params;
    copy_params.reserve(copy_type, group.layer_ids.size() * block_update_mapping.size() * buffers_per_layer);

    for (const auto& [src_block_index, dest_block_index] : block_update_mapping) {
        for (int layer_id : group.layer_ids) {
            auto src_addr_info =
                kv_cache_groups_[static_cast<size_t>(group_index)]->convertIndexToAddr(layer_id, src_block_index);
            auto dst_addr_info =
                kv_cache_groups_[static_cast<size_t>(group_index)]->convertIndexToAddr(layer_id, dest_block_index);

            RTP_LLM_CHECK_WITH_INFO(src_addr_info.kv_addr && dst_addr_info.kv_addr,
                                    "failed to get block address for group %d layer %d src_block %d dst_block %d",
                                    group_index,
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
    for (int group_index = 0; group_index < static_cast<int>(kv_cache_groups_.size()); ++group_index) {
        const auto group            = kv_cache_groups_[static_cast<size_t>(group_index)];
        const int  group_common_seq = cpEffectiveSeqLenForGroup(cp_mapper, config_, group_index, raw_common_seq_len);
        const int  group_seq_len    = cpEffectiveSeqLenForGroup(cp_mapper, config_, group_index, raw_seq_len);
        const auto need             = kv_cache_groups_[static_cast<size_t>(group_index)]->getNeedBlocks(
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
    for (int group_index = 0; group_index < kv_cache_resource.groupNums(); ++group_index) {
        need_blocks += kv_cache_groups_[static_cast<size_t>(group_index)]->estimatePeakNeedBlocks(
            seq_len,
            kv_cache_resource.groupBlocks().at(static_cast<size_t>(group_index))->blocks(),
            remaining_tokens,
            reserve_step,
            enable_reuse_cache);
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
        for (int group_index = 0; group_index < kv_resource->groupNums(); ++group_index) {
            if (!cpBlockRoundRobinGroup(cp_slot_mapper_, config_, group_index)) {
                continue;
            }
            const int effective_seq_len = cpEffectiveSeqLenForGroup(cp_slot_mapper_, config_, group_index, seq_len);
            const int expected_blocks =
                kv_cache_groups_[static_cast<size_t>(group_index)]->needBlocksNum(effective_seq_len, 0, reserve_step);
            const int actual_blocks = kv_resource->blocksNumByIndex(batch_id, static_cast<size_t>(group_index));
            RTP_LLM_CHECK_WITH_INFO(actual_blocks == expected_blocks,
                                    "CP invariant violated: batch=%d group=%d blocks=%d != expected_local_blocks=%d "
                                    "(seq_len=%d, effective_seq_len=%d, reserve_step=%d, cp_size=%d, "
                                    "block_size=%d, cacheKeys=%zu)",
                                    batch_id,
                                    group_index,
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
    for (int group_index = 0; group_index < batch_kv_cache_resource->groupNums(); ++group_index) {
        const int effective_seq_len = cpEffectiveSeqLenForGroup(cp_slot_mapper_, config_, group_index, seq_len);
        const int cur_blocks        = batch_kv_cache_resource->blocksNumByIndex(0, static_cast<size_t>(group_index));
        need_blocks += kv_cache_groups_[static_cast<size_t>(group_index)]->needBlocksNum(
            effective_seq_len, cur_blocks, reserve_step);
    }
    return need_blocks;
}

}  // namespace rtp_llm
