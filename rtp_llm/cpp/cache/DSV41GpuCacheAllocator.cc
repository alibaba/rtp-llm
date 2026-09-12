#include "rtp_llm/cpp/cache/HybridPoolKVCacheAllocator.h"

#include <algorithm>
#include <numeric>

#include "rtp_llm/cpp/cache/DSV41KVCacheSpec.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm {

size_t HybridPoolKVCacheAllocator::dsv41ReuseUnit() const {
    auto spec = std::dynamic_pointer_cast<DSV41KVCacheSpec>(config_.cache_specs.at(5));
    if (!spec)
        throw std::logic_error("V4.1 reuse requires its physical SWA layout");
    return config_.seq_size_per_block * spec->cp_size;
}

size_t HybridPoolKVCacheAllocator::dsv41DataUnit() const {
    const auto spec = std::dynamic_pointer_cast<DSV41KVCacheSpec>(config_.cache_specs.at(5));
    if (!spec)
        throw std::logic_error("V4.1 data matching requires its physical layout");
    return config_.seq_size_per_block * (spec->prefill_byte_slice ? spec->cp_size : 1);
}

DSV41CacheIdentity HybridPoolKVCacheAllocator::dsv41Identity(const DSV41CacheIdentity& identity) const {
    identity.validate();
    if (config_.dsv41_cache_layout_version != 1 || (config_.layer_all_num != 40 && config_.layer_all_num != 43)
        || config_.cache_specs.size() != 6 || config_.global_layer_ids.size() != 6 || !config_.use_typed_cache_regions
        || !config_.use_opaque_kv_cache_store || !config_.use_independent_block_pools)
        throw std::logic_error("V4.1 reuse requires its declared physical owner layout");
    const std::array<KVCacheRegionName, 6> regions{KVCacheRegionName::DSV41_GLOBAL_KV,
                                                   KVCacheRegionName::DSV41_GLOBAL_KV,
                                                   KVCacheRegionName::DSV41_INDEX_KV,
                                                   KVCacheRegionName::DSV41_INDEX_KV,
                                                   KVCacheRegionName::DSV41_PAIR_STATE,
                                                   KVCacheRegionName::SWA_KV};
    for (size_t group = 0; group < regions.size(); ++group) {
        auto             spec   = std::dynamic_pointer_cast<DSV41KVCacheSpec>(config_.cache_specs[group]);
        std::vector<int> owners = group == 1 || group == 3 ? std::vector<int>{20} : std::vector<int>{2, 8, 14};
        if (group == 5) {
            owners.resize(config_.layer_all_num);
            std::iota(owners.begin(), owners.end(), 0);
        }
        if (!spec || spec->region != regions[group] || config_.global_layer_ids[group] != owners)
            throw std::logic_error("V4.1 reuse has inconsistent physical owners");
    }
    const auto swa = std::dynamic_pointer_cast<DSV41KVCacheSpec>(config_.cache_specs[5]);
    DSV41CacheIdentity expected{identity.model_revision,
                                config_.dsv41LayoutFingerprint(),
                                identity.replay_mode,
                                1,
                                128,
                                swa->entries_per_block};
    if (!(identity == expected))
        throw std::invalid_argument("V4.1 request identity differs from its physical cache layout");
    return expected;
}

void HybridPoolKVCacheAllocator::insertIntoCache(const InsertInfo& info) {
    if (config_.dsv41_cache_layout_version == 0) {
        HybridKVCacheAllocator::insertIntoCache(info);
        return;
    }
    if (!info.batch_kv_cache_resource || !info.complete_token_ids || !shared_block_cache_)
        throw std::invalid_argument("V4.1 publication requires its materialized request");
    for (int b = 0; b < info.batch_kv_cache_resource->batchSize(); ++b) {
        auto& resource = info.batch_kv_cache_resource->cacheResource(b);
        if (!resource.dsv41CacheState() || resource.groupNums() != 6)
            throw std::logic_error("V4.1 publication requires request identity and physical groups");
        const auto view = resource.dsv41CacheState()->view();
        dsv41Identity(view.identity);
        const size_t unit = dsv41DataUnit();
        const bool   cp   = unit != config_.seq_size_per_block;
        if (cp && (!info.cp_slot_mapper || !info.cp_slot_mapper->isSharded() || info.cp_slot_mapper->cpSize() != 8))
            throw std::invalid_argument("V4.1 publication requires the CP8 canonical mapper");
        CacheKeysType keys = resource.cacheKeys();
        if (cp && !resource.cacheKeysAreCpCanonical()) {
            keys.clear();
            for (size_t index = 7; index < resource.cacheKeys().size(); index += 8)
                keys.push_back(resource.cacheKeys()[index]);
        }
        const size_t count =
            std::min(keys.size(),
                     static_cast<size_t>(std::min<int64_t>(std::max(info.complete_token_ids->seqLength() - 1, 0),
                                                           view.encoder_materialized_end))
                         / unit);
        runtimeSyncAndCheck();
        for (size_t index = 0; index < count; ++index) {
            std::vector<BlockIdxType> slots(6, NULL_BLOCK_IDX);
            bool                      complete = true;
            for (size_t group = 0; group < 4; ++group) {
                const auto& ids = resource.blocks(group);
                if (index >= ids.size() || ids[index] <= 0) {
                    complete = false;
                    break;
                }
                slots[group] = ids[index];
            }
            if (!complete)
                break;
            std::shared_ptr<const DSV41CheckpointMetadata> metadata;
            if (view.completed && view.encoder_materialized_end == view.completed->materialized_end
                && view.decoder_checkpoint_end == view.completed->materialized_end
                && view.completed->materialized_end == static_cast<int64_t>((index + 1) * unit)) {
                view.completed->validate(dsv41ReuseUnit());
                metadata                 = std::make_shared<DSV41CheckpointMetadata>(*view.completed);
                const size_t fixed_index = view.completed->materialized_end / dsv41ReuseUnit() - 1;
                for (size_t group = 4; group < 6; ++group) {
                    const auto& ids = resource.blocks(group);
                    if (fixed_index >= ids.size() || ids[fixed_index] <= 0) {
                        metadata.reset();
                        break;
                    }
                    slots[group] = ids[fixed_index];
                }
                if (!metadata)
                    slots[4] = slots[5] = NULL_BLOCK_IDX;
            }
            BlockDependency dependency{index != 0, index ? keys[index - 1] : 0, static_cast<uint32_t>(index)};
            shared_block_cache_->put(keys[index],
                                     slots,
                                     info.is_resident,
                                     cp ? SharedBlockCache::kGpuCpCanonicalNamespace :
                                          SharedBlockCache::kGpuLogicalNamespace,
                                     dependency,
                                     {},
                                     std::move(metadata));
        }
    }
}

int HybridPoolKVCacheAllocator::reuseCache(const CacheKeysType&                 keys,
                                           BatchKVCacheResource&                batch,
                                           const std::shared_ptr<CPSlotMapper>& mapper) {
    if (config_.dsv41_cache_layout_version == 0)
        return HybridKVCacheAllocator::reuseCache(keys, batch, mapper);
    auto& resource = batch.cacheResource(0);
    if (!resource.dsv41CacheState() || !shared_block_cache_)
        throw std::logic_error("V4.1 block matching requires request identity");
    const auto view = resource.dsv41CacheState()->view();
    dsv41Identity(view.identity);
    if (dsv41DataUnit() != config_.seq_size_per_block && (!mapper || !mapper->isSharded() || mapper->cpSize() != 8))
        throw std::invalid_argument("V4.1 block matching requires the CP8 canonical mapper");
    std::array<BlockIndicesType, 6>                blocks;
    std::shared_ptr<const DSV41CheckpointMetadata> tail;
    size_t                                         tail_blocks = 0;
    for (size_t index = 0; index < keys.size(); ++index) {
        auto match = shared_block_cache_->matchAndReference(keys[index], {0, 1, 2, 3});
        if (!match.found)
            break;
        for (size_t group = 0; group < 4; ++group)
            blocks[group].push_back(match.group_blocks[group]);
        auto metadata   = match.recovery_metadata;
        bool valid_tail = metadata && metadata->identity == view.identity
                          && metadata->materialized_end == static_cast<int64_t>((index + 1) * dsv41DataUnit())
                          && match.group_blocks.size() == 6 && match.group_blocks[4] > 0 && match.group_blocks[5] > 0;
        if (valid_tail) {
            try {
                metadata->validate(dsv41ReuseUnit());
            } catch (const std::invalid_argument&) {
                valid_tail = false;
            }
        }
        for (size_t group = 4; group < 6; ++group) {
            if (valid_tail) {
                if (!blocks[group].empty())
                    group_block_pools_[group]->requestFree(blocks[group].back());
                blocks[group].assign(metadata->materialized_end / dsv41ReuseUnit(), NULL_BLOCK_IDX);
                blocks[group].back() = match.group_blocks[group];
            } else if (group < match.group_blocks.size() && match.group_blocks[group] > 0) {
                group_block_pools_[group]->requestFree(match.group_blocks[group]);
            }
        }
        if (valid_tail) {
            tail        = std::move(metadata);
            tail_blocks = index + 1;
        }
    }
    const size_t hits = blocks[0].size();
    for (size_t group = 0; group < blocks.size(); ++group)
        resource.mutableBlockIds(group).assign(std::move(blocks[group]));
    resource.clearDsv41RecoveryMetadata();
    if (tail)
        resource.setDsv41RecoveryMetadata(tail_blocks - 1, std::move(tail));
    return hits;
}

bool HybridPoolKVCacheAllocator::cloneDsv41WritableBacking(KVCacheResource& resource, size_t state_ready_blocks) {
    struct Replacement {
        size_t       group;
        size_t       index;
        BlockIdxType old;
        BlockIdxType fresh;
    };
    std::vector<Replacement> replacements;
    try {
        BatchCopyParams copies;
        for (size_t group = 0; group < 6; ++group) {
            const auto& ids = resource.blocks(group);
            for (size_t index = 0; index < ids.size(); ++index) {
                const size_t data_index = (index + 1) * dsv41ReuseUnit() / dsv41DataUnit() - 1;
                const bool   needs_copy = group < 4 ?
                                              index >= state_ready_blocks && index < resource.deviceReuseBlockNum() :
                                              static_cast<bool>(resource.dsv41RecoveryMetadata(data_index));
                if (!needs_copy || ids[index] <= 0)
                    continue;
                auto& pool = group_block_pools_[group];
                // The matched source must remain cached if private allocation fails.
                // Its request reference means evicting it cannot provide copy space.
                auto allocated = pool->malloc(1);
                if (allocated.size() != 1)
                    throw std::runtime_error("V4.1 reuse cannot allocate private writable backing");
                replacements.push_back({group, index, ids[index], allocated.front()});
                for (int owner : config_.global_layer_ids[group]) {
                    const auto src = kv_cache_groups_[group]->convertIndexToAddr(owner, ids[index]);
                    const auto dst = kv_cache_groups_[group]->convertIndexToAddr(owner, allocated.front());
                    copies.add(dst.kv_addr,
                               src.kv_addr,
                               config_.cache_specs[group]->block_size_bytes(),
                               BatchCopyParams::get_copy_type(pool->where(), pool->where()));
                }
            }
        }
        execBatchCopy(copies);
        runtimeSyncAndCheck();
        for (auto& item : replacements) {
            resource.mutableBlockIds(item.group).setAt(item.index, item.fresh);
            item.fresh = NULL_BLOCK_IDX;
            group_block_pools_[item.group]->requestFree(item.old);
        }
        return true;
    } catch (const std::exception& error) {
        for (const auto& item : replacements)
            if (item.fresh > 0)
                group_block_pools_[item.group]->requestFree(item.fresh);
        RTP_LLM_LOG_WARNING("V4.1 private reuse backing copy failed: %s", error.what());
        return false;
    }
}

MallocResult HybridPoolKVCacheAllocator::initMallocForCommonLen(const MallocInfo& info) {
    auto result = HybridKVCacheAllocator::initMallocForCommonLen(info);
    if (!result.success || config_.dsv41_cache_layout_version == 0)
        return result;
    auto&  resource = info.batch_kv_cache_resource->cacheResource(0);
    size_t ready    = 0;
    for (size_t index = 0; index < resource.deviceReuseBlockNum(); ++index) {
        auto metadata = resource.dsv41RecoveryMetadata(index);
        // Bounded recovery also needs its L20 replay-source payload.
        if (metadata && metadata->identity.replay_mode == DSV41ReplayMode::FULL)
            ready = index + 1;
    }
    if (!cloneDsv41WritableBacking(resource, ready)) {
        HybridKVCacheAllocator::free(FreeInfo{info.batch_kv_cache_resource, info.complete_token_ids});
        resource.setDeviceReuseBlockNum(0);
        return {false, 0};
    }
    result.reuse_len = ready * dsv41DataUnit();
    return result;
}

MallocResult HybridPoolKVCacheAllocator::incrMalloc(const MallocInfo& info) {
    auto adjusted = info;
    std::shared_ptr<const DSV41CheckpointMetadata> restore;
    if (config_.dsv41_cache_layout_version != 0 && info.batch_kv_cache_resource) {
        auto& resource = info.batch_kv_cache_resource->cacheResource(0);
        if (resource.dsv41CacheState() && resource.dsv41CacheState()->view().decoder_checkpoint_end == 0) {
            for (size_t index = 0; index < resource.deviceReuseBlockNum(); ++index)
                if (auto metadata = resource.dsv41RecoveryMetadata(index))
                    restore = std::move(metadata);
            if (restore)
                adjusted.enable_remove_skipped_blocks = false;
        }
    }
    auto result = HybridKVCacheAllocator::incrMalloc(adjusted);
    if (result.success && restore)
        info.batch_kv_cache_resource->cacheResource(0).dsv41CacheState()->restore(*restore, dsv41ReuseUnit());
    return result;
}

}  // namespace rtp_llm
