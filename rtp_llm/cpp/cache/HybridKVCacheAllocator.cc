#include "rtp_llm/cpp/cache/HybridKVCacheAllocator.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
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

inline bool cpBlockRoundRobinGroup(const std::shared_ptr<CPSlotMapper>& mapper, const CacheConfig& config, int gid) {
    return mapper && mapper->isSharded() && gid >= 0 && mapper->blockRoundRobinGroup(config, static_cast<size_t>(gid));
}

inline int cpEffectiveSeqLenForGroup(const std::shared_ptr<CPSlotMapper>& mapper,
                                     const CacheConfig&                   config,
                                     int                                  gid,
                                     int                                  seq_len) {
    return cpBlockRoundRobinGroup(mapper, config, gid) ?
               mapper->effectiveSeqLenForAlloc(config, static_cast<size_t>(gid), seq_len) :
               seq_len;
}

inline int cpLogicalSeqSizeForGroup(const std::shared_ptr<CPSlotMapper>& mapper,
                                    const CacheConfig&                   config,
                                    int                                  gid,
                                    int                                  fallback) {
    return (mapper && mapper->isSharded() && gid >= 0) ?
               static_cast<int>(mapper->logicalSeqSizePerBlock(config, static_cast<size_t>(gid))) :
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

int groupIdForStableTag(const CacheConfig& config, const std::string& tag) {
    const auto tags = config.groupTagsSnapshot();
    const auto it   = std::find(tags.begin(), tags.end(), tag);
    return it == tags.end() ? -1 : static_cast<int>(std::distance(tags.begin(), it));
}

}  // namespace

bool HybridKVCacheAllocator::skipReuseCacheGroup(int gid) const {
    return gid >= 0 && static_cast<size_t>(gid) < kv_cache_groups_.size()
           && !kv_cache_groups_[static_cast<size_t>(gid)]->prefixReuseEnabled();
}

std::vector<int> HybridKVCacheAllocator::independentEvictionGroupIds() const {
    std::vector<int> group_ids;
    for (size_t gid = 0; gid < kv_cache_groups_.size(); ++gid) {
        if (kv_cache_groups_[gid]->evictPolicy() == CacheEvictPolicy::INDEPENDENT) {
            group_ids.push_back(static_cast<int>(gid));
        }
    }
    return group_ids;
}

bool HybridKVCacheAllocator::cpCompactSwaGroup(int gid, const std::shared_ptr<CPSlotMapper>& mapper) const {
    return mapper && mapper->isSharded() && gid >= 0 && static_cast<size_t>(gid) < kv_cache_groups_.size()
           && mapper->compactLastRankGroup(config_, static_cast<size_t>(gid));
}

HybridKVCacheAllocator::HybridKVCacheAllocator(const CacheConfig&                 config,
                                               AllocationType                     allocation_type,
                                               const kmonitor::MetricsReporterPtr metrics_reporter,
                                               int64_t                            reserve_block_ratio):
    KVCacheAllocator(config, allocation_type, metrics_reporter, reserve_block_ratio) {}

bool HybridKVCacheAllocator::preflightLoadBackMappings(const std::shared_ptr<LoadBackTicket>& ticket) const {
    if (ticket == nullptr || ticket->empty()) {
        return true;
    }
    for (size_t item_index = 0; item_index < ticket->itemCount(); ++item_index) {
        const int   group_id          = ticket->groupId(item_index);
        const auto& device_group_tags = ticket->deviceGroupTags(item_index);
        if (!block_tree_cache_
            || !block_tree_cache_->validateDeviceGroupTagsForComponentGroup(group_id, device_group_tags)
            || device_group_tags.empty()) {
            return false;
        }
        std::unordered_set<std::string> unique_tags;
        for (const auto& tag : device_group_tags) {
            const int gid = groupIdForStableTag(config_, tag);
            if (tag.empty() || !unique_tags.emplace(tag).second || gid < 0 || skipReuseCacheGroup(gid)) {
                return false;
            }
        }
    }
    return true;
}

int HybridKVCacheAllocator::reuseCache(const CacheKeysType&                 cache_keys,
                                       BatchKVCacheResource&                kv_resource,
                                       const std::shared_ptr<CPSlotMapper>& cp_mapper,
                                       std::shared_ptr<LoadBackTicket>&     ticket,
                                       std::vector<BlockIndicesType>&       referenced_blocks) {
    ticket.reset();
    if (!block_tree_cache_ || cache_keys.empty()) {
        return 0;
    }
    const int cp_scale     = (cp_mapper && cp_mapper->isSharded()) ? cp_mapper->cpSize() : 1;
    auto      match_result = block_tree_cache_->match(cache_keys);
    ticket                 = match_result.load_back_ticket;
    auto fail_match        = [&]() {
        ticket.reset();
        auto matched_block_sets = std::move(match_result.matched_block_sets);
        block_tree_cache_->releaseMatchedBlocks(matched_block_sets);
        return -1;
    };
    if (!preflightLoadBackMappings(ticket)) {
        return fail_match();
    }
    const int ready_reuse_blocks = static_cast<int>(match_result.matched_blocks);
    const int logical_reuse_blocks =
        ticket != nullptr && !ticket->empty() ? static_cast<int>(ticket->logicalMatchedBlocks()) : ready_reuse_blocks;
    auto groupBlocks = [&](int gid) -> const BlockIndicesType& {
        static const BlockIndicesType empty;
        const auto it = match_result.group_block_indices.find(config_.tagForGroup(static_cast<size_t>(gid)));
        return it == match_result.group_block_indices.end() ? empty : it->second;
    };

    for (int gid : full_group_ids_) {
        if (skipReuseCacheGroup(gid)) {
            continue;
        }
        BlockIndicesType blocks = groupBlocks(gid);
        blocks.resize(static_cast<size_t>(logical_reuse_blocks), NULL_BLOCK_IDX);
        kv_resource.mutableBlockIds(0, gid).assign(std::move(blocks));
    }
    const int logical_reuse_len = logical_reuse_blocks * cp_scale;
    const int ready_reuse_len   = ready_reuse_blocks * cp_scale;
    for (int gid : linear_group_ids_) {
        if (skipReuseCacheGroup(gid)) {
            continue;
        }
        kv_resource.mutableBlockIds(0, gid).assign(
            BlockIndicesType(static_cast<size_t>(logical_reuse_len), NULL_BLOCK_IDX));
        const auto& blocks = groupBlocks(gid);
        if (!blocks.empty() && ready_reuse_len > 0) {
            kv_resource.mutableBlockIds(0, gid).setAt(static_cast<size_t>(ready_reuse_len - 1), blocks.back());
        }
    }
    for (int gid : swa_group_ids_) {
        if (skipReuseCacheGroup(gid)) {
            continue;
        }
        const int group_reuse_len = cpCompactSwaGroup(gid, cp_mapper) ? logical_reuse_blocks : logical_reuse_len;
        kv_resource.mutableBlockIds(0, gid).assign(
            BlockIndicesType(static_cast<size_t>(group_reuse_len), NULL_BLOCK_IDX));
        const auto& blocks = groupBlocks(gid);
        if (blocks.size() > static_cast<size_t>(ready_reuse_blocks)) {
            return fail_match();
        }
        const size_t canonical_start = static_cast<size_t>(ready_reuse_blocks) - blocks.size();
        for (size_t i = 0; i < blocks.size(); ++i) {
            const size_t canonical_position = canonical_start + i;
            const size_t target_position    = cpCompactSwaGroup(gid, cp_mapper) ?
                                                  canonical_position :
                                                  (canonical_position + 1) * static_cast<size_t>(cp_scale) - 1;
            kv_resource.mutableBlockIds(0, gid).setAt(target_position, blocks[i]);
        }
    }

    if (ticket != nullptr) {
        for (size_t item_index = 0; item_index < ticket->itemCount(); ++item_index) {
            if (ticket->sourceTier(item_index) != Tier::DEVICE) {
                continue;
            }
            const auto& source_blocks     = ticket->sourceBlocks(item_index);
            const auto& device_group_tags = ticket->deviceGroupTags(item_index);
            if (source_blocks.size() != device_group_tags.size() || source_blocks.empty()) {
                return fail_match();
            }
            for (size_t local = 0; local < device_group_tags.size(); ++local) {
                const int gid = groupIdForStableTag(config_, device_group_tags[local]);
                if (gid < 0 || gid >= kv_resource.groupNums() || skipReuseCacheGroup(gid)
                    || isNullBlockIdx(source_blocks[local])) {
                    return fail_match();
                }
                const auto   type = config_.typeForGroup(static_cast<size_t>(gid));
                const size_t target_position =
                    type == CacheGroupType::LINEAR
                            || (type == CacheGroupType::SWA && !cpCompactSwaGroup(gid, cp_mapper)) ?
                        (ticket->pathIndex(item_index) + 1) * static_cast<size_t>(cp_scale) - 1 :
                        ticket->pathIndex(item_index);
                auto& target = kv_resource.mutableBlockIds(0, gid);
                if (target_position >= target.blocksNum()
                    || (!isNullBlockIdx(target.blocks()[target_position])
                        && target.blocks()[target_position] != source_blocks[local])) {
                    return fail_match();
                }
                target.setAt(target_position, source_blocks[local]);
            }
        }
    }

    for (int gid = 0; gid < kv_resource.groupNums(); ++gid) {
        BlockIndicesType valid;
        for (const auto block : kv_resource.blocks(0, gid)) {
            if (!isNullBlockIdx(block)) {
                valid.push_back(block);
            }
        }
        if (!valid.empty()) {
            referenceBlocksInGroup(gid, valid);
            referenced_blocks[static_cast<size_t>(gid)] = std::move(valid);
        }
    }
    block_tree_cache_->releaseMatchedBlocks(match_result.matched_block_sets);
    return logical_reuse_blocks;
}

MallocResult HybridKVCacheAllocator::initMallocForCommonLen(const MallocInfo& malloc_info) {
    auto&       kv_resource    = malloc_info.batch_kv_cache_resource;
    const int   batch_size     = kv_resource->batchSize();
    const int   seq_len        = malloc_info.complete_token_ids->seqLength();
    const int   common_seq_len = std::min(malloc_info.complete_token_ids->commonSeqLength(), seq_len);
    const auto& cp_mapper      = cp_slot_mapper_;
    const int   cp_scale       = (cp_mapper && cp_mapper->isSharded()) ? cp_mapper->cpSize() : 1;
    RTP_LLM_CHECK_WITH_INFO(block_tree_cache_ != nullptr, "BlockTreeCache must be injected before allocation");
    const KVCacheGroupPtr reuse_group =
        full_group_ids_.empty() ? KVCacheGroupPtr{} : kv_cache_groups_[static_cast<size_t>(full_group_ids_.front())];
    const int reuse_unit_tokens =
        reuse_group ? cpLogicalSeqSizeForGroup(cp_mapper, config_, reuse_group->group_id(), seqSizePerBlock()) :
                      seqSizePerBlock();

    const auto&                     cache_keys         = kv_resource->cacheKeys(0);
    int64_t                         match_cost_time_us = 0;
    const size_t                    reserve_blocks     = reserveBlocksNum();
    int                             reuse_blocks       = 0;
    std::vector<BlockIndicesType>   referenced_blocks(static_cast<size_t>(kv_resource->groupNums()));
    std::shared_ptr<LoadBackTicket> load_back_ticket;
    auto                            rollback = [&](const std::vector<size_t>& original_sizes) -> MallocResult {
        load_back_ticket.reset();
        rollbackInitMalloc(*kv_resource, referenced_blocks, original_sizes);
        return {false, 0};
    };

    if (malloc_info.enable_device_cache) {
        CacheKeysType cp_keys   = cpCanonicalCacheKeys(cp_mapper, cache_keys);
        const bool    cp_active = cp_mapper && cp_mapper->isSharded();
        CacheKeysType match_keys(cp_keys.begin(),
                                 cp_active ? cp_keys.end() : (cp_keys.empty() ? cp_keys.end() : cp_keys.end() - 1));
        const auto    begin_us = currentTimeUs();
        reuse_blocks           = reuseCache(match_keys, *kv_resource, cp_mapper, load_back_ticket, referenced_blocks);
        match_cost_time_us     = currentTimeUs() - begin_us;
        if (reuse_blocks < 0) {
            load_back_ticket.reset();
            kv_resource->clearBlocks();
            return {false, 0};
        }
        kv_resource->cacheResource(0).setDeviceReuseBlockNum(static_cast<size_t>(reuse_blocks));
    }

    std::vector<std::vector<size_t>> load_back_positions(static_cast<size_t>(kv_resource->groupNums()));
    if (load_back_ticket != nullptr && !load_back_ticket->empty()) {
        for (size_t item_index = 0; item_index < load_back_ticket->itemCount(); ++item_index) {
            if (load_back_ticket->sourceTier(item_index) == Tier::DEVICE) {
                continue;
            }
            for (const auto& tag : load_back_ticket->deviceGroupTags(item_index)) {
                const int gid = groupIdForStableTag(config_, tag);
                if (gid < 0 || gid >= kv_resource->groupNums() || skipReuseCacheGroup(gid)) {
                    return rollback({});
                }
                const auto   type = config_.typeForGroup(static_cast<size_t>(gid));
                const size_t target_position =
                    type == CacheGroupType::LINEAR
                            || (type == CacheGroupType::SWA && !cpCompactSwaGroup(gid, cp_mapper)) ?
                        (load_back_ticket->pathIndex(item_index) + 1) * static_cast<size_t>(cp_scale) - 1 :
                        load_back_ticket->pathIndex(item_index);
                auto& positions = load_back_positions[static_cast<size_t>(gid)];
                if (std::find(positions.begin(), positions.end(), target_position) == positions.end()) {
                    positions.push_back(target_position);
                }
            }
        }

        size_t pending_targets = 0;
        for (int gid = 0; gid < kv_resource->groupNums(); ++gid) {
            const auto& blocks = kv_resource->blocks(0, gid);
            for (const size_t position : load_back_positions[static_cast<size_t>(gid)]) {
                if (position >= blocks.size()) {
                    return rollback({});
                }
                pending_targets += isNullBlockIdx(blocks[position]) ? 1 : 0;
            }
        }
        if (reserve_blocks > 0) {
            const int    need_blocks = getNeedBlocks(malloc_info);
            const size_t required    = static_cast<size_t>(std::max(need_blocks, 0)) + pending_targets + reserve_blocks;
            if (freeBlocksNum() < required) {
                return rollback({});
            }
        }

        for (int gid = 0; gid < kv_resource->groupNums(); ++gid) {
            const auto& positions = load_back_positions[static_cast<size_t>(gid)];
            if (positions.empty()) {
                continue;
            }
            const auto before = kv_resource->blocks(0, gid);
            auto&      group  = kv_cache_groups_[static_cast<size_t>(gid)];
            if (!group->materializePositions(kv_resource->mutableBlockIds(0, gid), positions)) {
                return rollback({});
            }
            const auto& after = kv_resource->blocks(0, gid);
            for (const size_t position : positions) {
                if (isNullBlockIdx(before[position]) && !isNullBlockIdx(after[position])) {
                    referenced_blocks[static_cast<size_t>(gid)].push_back(after[position]);
                }
            }
        }
    } else if (reserve_blocks > 0 && !hasAvailableBlocksForReserve(malloc_info, reserve_blocks)) {
        return rollback({});
    }

    std::vector<size_t> original_sizes(static_cast<size_t>(kv_resource->groupNums()));
    for (int gid = 0; gid < kv_resource->groupNums(); ++gid) {
        original_sizes[static_cast<size_t>(gid)] = kv_resource->blocksNum(0, gid);
    }
    for (int gid = 0; gid < kv_resource->groupNums(); ++gid) {
        auto&     block_ids     = kv_resource->mutableBlockIds(0, gid);
        const int group_seq_len = cpEffectiveSeqLenForGroup(cp_mapper, config_, gid, common_seq_len);
        if (!kv_cache_groups_[static_cast<size_t>(gid)]->malloc(block_ids, group_seq_len, malloc_info.reuse_cache, 0)) {
            return rollback(original_sizes);
        }
    }
    if (load_back_ticket != nullptr && !load_back_ticket->empty()) {
        for (size_t item_index = 0; item_index < load_back_ticket->itemCount(); ++item_index) {
            const auto& device_group_tags = load_back_ticket->deviceGroupTags(item_index);
            const auto& source_blocks     = load_back_ticket->sourceBlocks(item_index);
            BlockIndicesType target_device_blocks;
            bool             valid = !device_group_tags.empty();
            for (size_t local = 0; local < device_group_tags.size(); ++local) {
                const int gid = groupIdForStableTag(config_, device_group_tags[local]);
                if (gid < 0 || gid >= kv_resource->groupNums()) {
                    valid = false;
                    break;
                }
                const auto   type     = config_.typeForGroup(static_cast<size_t>(gid));
                const size_t position = type == CacheGroupType::LINEAR
                                                || (type == CacheGroupType::SWA && !cpCompactSwaGroup(gid, cp_mapper)) ?
                                            (load_back_ticket->pathIndex(item_index) + 1) * static_cast<size_t>(cp_scale) - 1 :
                                            load_back_ticket->pathIndex(item_index);
                const auto&  blocks   = kv_resource->blocks(0, gid);
                if (position >= blocks.size() || isNullBlockIdx(blocks[position])
                    || (load_back_ticket->sourceTier(item_index) == Tier::DEVICE
                        && (local >= source_blocks.size() || blocks[position] != source_blocks[local]))) {
                    valid = false;
                    break;
                }
                target_device_blocks.push_back(blocks[position]);
            }
            if (!valid || !load_back_ticket->bindTargetDeviceBlocks(item_index, std::move(target_device_blocks))) {
                return rollback(original_sizes);
            }
        }
    }

    for (int batch = 1; batch < batch_size; ++batch) {
        for (int gid = 0; gid < kv_resource->groupNums(); ++gid) {
            kv_cache_groups_[static_cast<size_t>(gid)]->reference(kv_resource->mutableBlockIds(batch, gid),
                                                                  kv_resource->blocks(0, gid));
        }
    }
    return {true, reuse_blocks * reuse_unit_tokens, match_cost_time_us, nullptr, load_back_ticket};
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
        for (int gid = 0; gid < kv_resource->groupNums(); ++gid) {
            original_sizes[static_cast<size_t>(b)][static_cast<size_t>(gid)] = kv_resource->blocksNum(b, gid);
        }
    }

    bool all_success  = true;
    int  failed_batch = -1;
    int  failed_group = -1;
    for (int b = 0; b < batch_size; ++b) {
        for (int gid = 0; gid < kv_resource->groupNums(); ++gid) {
            auto&     block_ids        = kv_resource->mutableBlockIds(b, gid);
            const int group_seq_len    = cpEffectiveSeqLenForGroup(cp_mapper, config_, gid, raw_seq_len);
            auto&     filled_positions = backfilled_positions[static_cast<size_t>(b)][static_cast<size_t>(gid)];
            if (!kv_cache_groups_[static_cast<size_t>(gid)]->malloc(
                    block_ids, group_seq_len, malloc_info.reuse_cache, reserve_step, &filled_positions)) {
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
            auto&            block_ids        = kv_resource->mutableBlockIds(b, gid);
            const auto       original_size    = original_sizes[static_cast<size_t>(b)][static_cast<size_t>(gid)];
            const auto&      filled_positions = backfilled_positions[static_cast<size_t>(b)][static_cast<size_t>(gid)];
            const auto&      blocks           = block_ids.blocks();
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
                freeBlocksInGroup(gid, blocks_to_free);
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
        for (int gid = 0; gid < kv_cache_resource->groupNums(); ++gid) {
            kv_cache_groups_[static_cast<size_t>(gid)]->free(kv_cache_resource->blocks(batch_id, gid));
        }
    }
    kv_cache_resource->clearBlocks();
    if (block_tree_cache_) {
        block_tree_cache_->onBlocksReleased();
    }
}

void HybridKVCacheAllocator::insertIntoCache(const InsertInfo& insert_info) {
    auto& kv_cache_resource = insert_info.batch_kv_cache_resource;
    RTP_LLM_CHECK(kv_cache_resource != nullptr);
    if (!block_tree_cache_) {
        return;
    }

    const auto& cp_mapper  = cp_slot_mapper_;
    const bool  cp_active  = cp_mapper && cp_mapper->isSharded();
    const int   group_nums = kv_cache_resource->groupNums();
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
        const auto&                         component_groups = block_tree_cache_->componentGroups();
        std::vector<std::vector<GroupSlot>> slots(insert_keys.size(), std::vector<GroupSlot>(component_groups.size()));
        bool                                mapping_valid = true;
        for (size_t component_index = 0; component_index < component_groups.size(); ++component_index) {
            const auto& component_group = component_groups[component_index];
            if (!component_group || component_group->component_group_id != static_cast<int>(component_index)
                || component_group->tags().empty()
                || component_group->tags().size() != component_group->devicePoolCount()) {
                mapping_valid = false;
                break;
            }
            for (auto& per_key_slots : slots) {
                per_key_slots[component_index].device_blocks.assign(component_group->devicePoolCount(), NULL_BLOCK_IDX);
            }
            for (size_t local_pool = 0; local_pool < component_group->tags().size(); ++local_pool) {
                const int gid = groupIdForStableTag(config_, component_group->tags()[local_pool]);
                if (gid < 0 || gid >= group_nums || skipReuseCacheGroup(gid)) {
                    mapping_valid = false;
                    break;
                }
                const auto type           = config_.typeForGroup(static_cast<size_t>(gid));
                const bool sparse_logical = cp_active
                                            && (type == CacheGroupType::LINEAR
                                                || (type == CacheGroupType::SWA && !cpCompactSwaGroup(gid, cp_mapper)));
                const auto& blocks = kv_cache_resource->blocks(batch_id, gid);
                for (size_t i = 0; i < insert_keys.size(); ++i) {
                    const size_t position = sparse_logical ? (i + 1) * static_cast<size_t>(cp_mapper->cpSize()) - 1 : i;
                    if (position >= blocks.size() || isNullBlockIdx(blocks[position])) {
                        continue;
                    }
                    slots[i][component_index].device_blocks[local_pool] = blocks[position];
                }
            }
            if (!mapping_valid) {
                break;
            }
        }
        if (!mapping_valid) {
            RTP_LLM_LOG_WARNING("Hybrid insert rejected inconsistent stable tag/component mapping");
            continue;
        }

        size_t publish_prefix = 0;
        for (size_t i = 0; i < slots.size(); ++i) {
            bool key_valid    = true;
            bool key_has_data = false;
            for (size_t component_index = 0; component_index < component_groups.size(); ++component_index) {
                auto&       device_blocks = slots[i][component_index].device_blocks;
                const auto& component     = component_groups[component_index];
                const auto  valid_blocks  = static_cast<size_t>(
                    std::count_if(device_blocks.begin(), device_blocks.end(), [](BlockIdxType block) {
                        return !isNullBlockIdx(block);
                    }));
                if (valid_blocks == 0 && component->group_type != CacheGroupType::FULL) {
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
        slots.resize(publish_prefix);
        block_tree_cache_->insertSparse(nullptr, insert_keys, slots);
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
        for (int gid = 0; gid < kvcache_resource.groupNums(); ++gid) {
            const auto& src_blocks                   = kvcache_resource.blocks(gid);
            const auto  block                        = pos < src_blocks.size() ? src_blocks[pos] : NULL_BLOCK_IDX;
            blocks_for_key[static_cast<size_t>(gid)] = block;
            any_valid_block                          = any_valid_block || (!isNullBlockIdx(block) && block > 0);
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

    selected_resource->cacheKeys() = std::move(selected_keys);
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
            for (int gid = 0; gid < group_nums; ++gid) {
                if (!batch_kv_cache_resource->blocks(old_batch_idx, gid).empty()) {
                    new_blocks_num[static_cast<size_t>(gid)] += fork_count - 1;
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
    for (int gid = 0; gid < group_nums; ++gid) {
        std::unordered_set<BlockIdxType>      retained_blocks;
        std::unordered_map<BlockIdxType, int> dropped_block_counts;
        for (int old_batch_idx = 0; old_batch_idx < old_batch_size; ++old_batch_idx) {
            for (const auto block : batch_kv_cache_resource->blocks(old_batch_idx, gid)) {
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

        auto&     replacements = replacement_blocks[static_cast<size_t>(gid)];
        auto&     transferred  = transferred_ref_counts[static_cast<size_t>(gid)];
        const int need         = new_blocks_num[static_cast<size_t>(gid)];
        for (int old_batch_idx = 0; old_batch_idx < old_batch_size && static_cast<int>(replacements.size()) < need;
             ++old_batch_idx) {
            if (batch_fork_count[old_batch_idx] != 0) {
                continue;
            }
            const auto& dropped = batch_kv_cache_resource->blocks(old_batch_idx, gid);
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
        for (int gid = 0; gid < group_nums; ++gid) {
            auto& blocks = allocated_replacements[static_cast<size_t>(gid)];
            if (!blocks.empty()) {
                kv_cache_groups_[static_cast<size_t>(gid)]->free(blocks);
                blocks.clear();
            }
        }
    };
    for (int gid = 0; gid < group_nums; ++gid) {
        const int need_blocks = new_blocks_num[static_cast<size_t>(gid)];
        auto&     reserved    = replacement_blocks[static_cast<size_t>(gid)];
        reserved.reserve(static_cast<size_t>(need_blocks));
        for (int i = static_cast<int>(reserved.size()); i < need_blocks; ++i) {
            BlockIds   one_block;
            const bool ok = kv_cache_groups_[static_cast<size_t>(gid)]->malloc(
                one_block, kv_cache_groups_[static_cast<size_t>(gid)]->seqSizePerBlock());
            const auto& blocks = one_block.blocks();
            if (ok && blocks.size() == 1 && !isNullBlockIdx(blocks.front())) {
                reserved.push_back(blocks.front());
                allocated_replacements[static_cast<size_t>(gid)].push_back(blocks.front());
                continue;
            }
            if (!blocks.empty()) {
                allocated_replacements[static_cast<size_t>(gid)].insert(
                    allocated_replacements[static_cast<size_t>(gid)].end(), blocks.begin(), blocks.end());
            }
            RTP_LLM_LOG_WARNING(
                "reserve replacement block failed for hybrid kv cache update, group=%d need=%d reserved=%zu",
                gid,
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
        for (int gid = 0; gid < group_nums; ++gid) {
            BlockIndicesType to_free;
            auto&            transferred = transferred_ref_counts[static_cast<size_t>(gid)];
            for (const auto block : batch_kv_cache_resource->blocks(old_batch_idx, gid)) {
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
                kv_cache_groups_[static_cast<size_t>(gid)]->free(to_free);
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
            for (int gid = 0; gid < group_nums; ++gid) {
                auto& block_ids = batch_kv_cache_resource->mutableBlockIds(new_batch_idx, gid);
                kv_cache_groups_[static_cast<size_t>(gid)]->reference(block_ids,
                                                                      old_resources[old_batch_idx].blocks(gid));

                if (copy_last_block && !block_ids.blocks().empty()) {
                    const int  old_block       = block_ids.popBack();
                    const bool old_block_valid = !isNullBlockIdx(old_block) && old_block > 0;
                    if (old_block_valid) {
                        kv_cache_groups_[static_cast<size_t>(gid)]->free({old_block});
                    }

                    auto&      reserved     = replacement_blocks[static_cast<size_t>(gid)];
                    const auto reserved_idx = next_replacement[static_cast<size_t>(gid)]++;
                    RTP_LLM_CHECK_WITH_INFO(reserved_idx < reserved.size(),
                                            "missing reserved replacement block for hybrid kv cache update, group=%d",
                                            gid);
                    const int new_block = reserved[reserved_idx];
                    block_ids.add({new_block});
                    if (old_block_valid && !isNullBlockIdx(new_block) && new_block > 0) {
                        block_update_mapping.push_back(
                            {config_.topology().groupById(static_cast<size_t>(gid)).tag, old_block, new_block});
                    }
                }
            }
        }
        --fork_count;
    }
    for (int gid = 0; gid < group_nums; ++gid) {
        RTP_LLM_CHECK_WITH_INFO(
            next_replacement[static_cast<size_t>(gid)] == replacement_blocks[static_cast<size_t>(gid)].size(),
            "unused replacement blocks after hybrid kv cache update, group=%d used=%zu reserved=%zu",
            gid,
            next_replacement[static_cast<size_t>(gid)],
            replacement_blocks[static_cast<size_t>(gid)].size());
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

MemoryType HybridKVCacheAllocator::memoryTypeForGroup(int gid) const {
    (void)gid;
    return allocation_type_ == AllocationType::DEVICE ? MemoryType::MEMORY_GPU : MemoryType::MEMORY_CPU;
}

void HybridKVCacheAllocator::copyBlockMappingForGroup(int                             gid,
                                                      const std::vector<BlockIdPair>& block_update_mapping) const {
    if (block_update_mapping.empty()) {
        return;
    }

    const auto   memory_type         = memoryTypeForGroup(gid);
    const auto   copy_type           = BatchCopyParams::get_copy_type(memory_type, memory_type);
    const auto&  spec                = config_.specForGroup(static_cast<size_t>(gid));
    const size_t kv_block_size_bytes = spec->block_size_bytes();
    const size_t scale_block_bytes   = spec->scale_block_size_bytes();
    const size_t buffers_per_layer   = scale_block_bytes > 0 ? 2 : 1;

    BatchCopyParams copy_params;
    copy_params.reserve(copy_type,
                        config_.layerIdsForGroup(static_cast<size_t>(gid)).size() * block_update_mapping.size()
                            * buffers_per_layer);

    for (const auto& [src_block_index, dest_block_index] : block_update_mapping) {
        for (int layer_id : config_.layerIdsForGroup(static_cast<size_t>(gid))) {
            auto src_addr_info =
                kv_cache_groups_[static_cast<size_t>(gid)]->convertIndexToAddr(layer_id, src_block_index);
            auto dst_addr_info =
                kv_cache_groups_[static_cast<size_t>(gid)]->convertIndexToAddr(layer_id, dest_block_index);

            RTP_LLM_CHECK_WITH_INFO(src_addr_info.kv_addr && dst_addr_info.kv_addr,
                                    "failed to get block address for group %d layer %d src_block %d dst_block %d",
                                    gid,
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
    for (int gid = 0; gid < static_cast<int>(kv_cache_groups_.size()); ++gid) {
        const auto group            = kv_cache_groups_[static_cast<size_t>(gid)];
        const int  group_common_seq = cpEffectiveSeqLenForGroup(cp_mapper, config_, gid, raw_common_seq_len);
        const int  group_seq_len    = cpEffectiveSeqLenForGroup(cp_mapper, config_, gid, raw_seq_len);
        const auto need             = kv_cache_groups_[static_cast<size_t>(gid)]->getNeedBlocks(
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
    for (int gid = 0; gid < kv_cache_resource.groupNums(); ++gid) {
        need_blocks += kv_cache_groups_[static_cast<size_t>(gid)]->estimatePeakNeedBlocks(
            seq_len, kv_cache_resource.blocks(gid), remaining_tokens, reserve_step, enable_reuse_cache);
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
        for (int gid = 0; gid < kv_resource->groupNums(); ++gid) {
            if (!cpBlockRoundRobinGroup(cp_slot_mapper_, config_, gid)) {
                continue;
            }
            const int effective_seq_len = cpEffectiveSeqLenForGroup(cp_slot_mapper_, config_, gid, seq_len);
            const int expected_blocks =
                kv_cache_groups_[static_cast<size_t>(gid)]->needBlocksNum(effective_seq_len, 0, reserve_step);
            const int actual_blocks = kv_resource->blocksNum(batch_id, gid);
            RTP_LLM_CHECK_WITH_INFO(actual_blocks == expected_blocks,
                                    "CP invariant violated: batch=%d group=%d blocks=%d != expected_local_blocks=%d "
                                    "(seq_len=%d, effective_seq_len=%d, reserve_step=%d, cp_size=%d, "
                                    "block_size=%d, cacheKeys=%zu)",
                                    batch_id,
                                    gid,
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
    for (int gid = 0; gid < batch_kv_cache_resource->groupNums(); ++gid) {
        const int effective_seq_len = cpEffectiveSeqLenForGroup(cp_slot_mapper_, config_, gid, seq_len);
        const int cur_blocks        = batch_kv_cache_resource->blocksNum(0, gid);
        need_blocks +=
            kv_cache_groups_[static_cast<size_t>(gid)]->needBlocksNum(effective_seq_len, cur_blocks, reserve_step);
    }
    return need_blocks;
}

}  // namespace rtp_llm
