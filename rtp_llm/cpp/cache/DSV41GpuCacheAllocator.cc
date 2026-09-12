#include "rtp_llm/cpp/cache/HybridPoolKVCacheAllocator.h"

#include <algorithm>
#include <numeric>
#include <sstream>

#include "rtp_llm/cpp/cache/DSV41KVCacheSpec.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm {

size_t HybridPoolKVCacheAllocator::dsv41ReuseUnit() const {
    const auto spec = std::dynamic_pointer_cast<DSV41KVCacheSpec>(config_.cache_specs.at(5));
    if (!spec)
        throw std::logic_error("V4.1 GPU cache requires its declared SWA layout");
    return config_.seq_size_per_block * spec->cp_size;
}

DSV41CacheIdentity HybridPoolKVCacheAllocator::dsv41Identity(const DSV41CacheIdentity& identity) const {
    identity.validate();
    if (config_.dsv41_cache_layout_version != 1 || config_.layer_all_num != 43 || config_.cache_specs.size() != 6
        || config_.global_layer_ids.size() != 6 || !config_.use_typed_cache_regions
        || !config_.use_opaque_kv_cache_store || !config_.use_independent_block_pools)
        throw std::logic_error("V4.1 GPU checkpoints require the complete target/draft owner layout");
    std::ostringstream fingerprint;
    fingerprint << "dsv41-memory-v1:target40:draft3:";
    for (size_t group = 0; group < config_.cache_specs.size(); ++group) {
        const auto spec = std::dynamic_pointer_cast<DSV41KVCacheSpec>(config_.cache_specs[group]);
        if (!spec)
            throw std::logic_error("V4.1 GPU checkpoint contains an untyped physical pool");
        const std::array<KVCacheRegionName, 6> regions{KVCacheRegionName::DSV41_GLOBAL_KV,
                                                       KVCacheRegionName::DSV41_GLOBAL_KV,
                                                       KVCacheRegionName::DSV41_INDEX_KV,
                                                       KVCacheRegionName::DSV41_INDEX_KV,
                                                       KVCacheRegionName::DSV41_PAIR_STATE,
                                                       KVCacheRegionName::SWA_KV};
        std::vector<int> owners = group == 1 || group == 3 ? std::vector<int>{20} : std::vector<int>{2, 8, 14};
        if (group == 5) {
            owners.resize(43);
            std::iota(owners.begin(), owners.end(), 0);
        }
        if (spec->region != regions[group] || config_.global_layer_ids[group] != owners)
            throw std::logic_error("V4.1 GPU checkpoint has inconsistent physical owners");
        fingerprint << group << ':' << spec->debugString() << ":owners=";
        for (int owner : config_.global_layer_ids[group])
            fingerprint << owner << ',';
        fingerprint << ';';
    }
    const auto swa = std::dynamic_pointer_cast<DSV41KVCacheSpec>(config_.cache_specs[5]);
    if (swa->cp_size > 1 && !swa->prefill_byte_slice)
        throw std::logic_error("V4.1 decode GPU reuse requires explicit eight-shard global page assembly");
    DSV41CacheIdentity expected{
        identity.model_revision, fingerprint.str(), identity.replay_mode, 1, 128, swa->entries_per_block};
    if (!(identity == expected))
        throw std::invalid_argument("V4.1 GPU checkpoint identity does not match the allocated physical layout");
    return expected;
}

void HybridPoolKVCacheAllocator::insertIntoCache(const InsertInfo& info) {
    if (config_.dsv41_cache_layout_version == 0) {
        HybridKVCacheAllocator::insertIntoCache(info);
        return;
    }
    if (!info.batch_kv_cache_resource || !shared_block_cache_)
        throw std::invalid_argument("V4.1 GPU publication requires an explicit request and checkpoint cache");
    for (int batch = 0; batch < info.batch_kv_cache_resource->batchSize(); ++batch) {
        const auto& resource = info.batch_kv_cache_resource->cacheResource(batch);
        if (resource.groupNums() != 6)
            throw std::invalid_argument("V4.1 GPU checkpoint requires all six request pool mappings");
        const auto& state = resource.dsv41CacheState();
        if (!state)
            throw std::logic_error("V4.1 GPU publication requires typed producer completion");
        const auto view = state->view();
        if (!view.completed)
            throw std::invalid_argument("V4.1 GPU publication has no complete checkpoint");
        DSV41GpuCheckpointData data;
        data.metadata   = *view.completed;
        data.reuse_unit = dsv41ReuseUnit();
        dsv41Identity(data.metadata.identity);
        const auto& mapper  = info.cp_slot_mapper;
        const bool  sharded = data.reuse_unit != config_.seq_size_per_block;
        if (sharded && (!mapper || !mapper->isSharded() || mapper->cpSize() != 8))
            throw std::invalid_argument("V4.1 GPU publication requires explicit CP8 canonical key mapping");
        const auto&   raw = resource.cacheKeys();
        CacheKeysType keys;
        if (sharded && !resource.cacheKeysAreCpCanonical()) {
            for (size_t index = 7; index < raw.size(); index += 8)
                keys.push_back(raw[index]);
        } else {
            keys = raw;
        }
        data.metadata.validate(data.reuse_unit);
        const size_t count = data.metadata.materialized_end / data.reuse_unit;
        if (keys.size() < count)
            throw std::invalid_argument("V4.1 GPU checkpoint is missing its prefix keys");
        data.keys.assign(keys.begin(), keys.begin() + count);
        for (size_t group = 0; group < data.blocks.size(); ++group) {
            const auto& ids = resource.blocks(group);
            if (ids.size() < count)
                throw std::invalid_argument("V4.1 GPU checkpoint is missing its materialized page map");
            if (group < 4)
                data.blocks[group].assign(ids.begin(), ids.begin() + count);
            else
                data.blocks[group] = {ids[count - 1]};
        }
        data.validateProducer(view);
        runtimeSyncAndCheck();
        const bool published = state->publishGpuCheckpoint([&](const DSV41CacheState::View& current) {
            data.validateProducer(current);
            return shared_block_cache_->putDsv41Checkpoint(data, info.is_resident);
        });
        if (!published)
            RTP_LLM_LOG_WARNING("V4.1 GPU checkpoint publication was cancelled or conflicted; no entry published");
    }
}

int HybridPoolKVCacheAllocator::reuseCache(const CacheKeysType&                 keys,
                                           BatchKVCacheResource&                batch,
                                           const std::shared_ptr<CPSlotMapper>& mapper) {
    if (config_.dsv41_cache_layout_version == 0)
        return HybridKVCacheAllocator::reuseCache(keys, batch, mapper);
    auto& resource = batch.cacheResource(0);
    if (!resource.dsv41CacheState() || !shared_block_cache_)
        throw std::logic_error("V4.1 GPU matching requires typed request identity");
    const auto view = resource.dsv41CacheState()->view();
    if (view.finished || view.cancelled || view.encoder_materialized_end != 0 || view.decoder_checkpoint_end != 0)
        throw std::invalid_argument("V4.1 GPU matching requires a fresh active request");
    dsv41Identity(view.identity);
    if (dsv41ReuseUnit() != config_.seq_size_per_block && (!mapper || !mapper->isSharded() || mapper->cpSize() != 8))
        throw std::invalid_argument("V4.1 GPU matching requires explicit CP8 canonical key mapping");
    auto lease = shared_block_cache_->matchDsv41Checkpoint(view.identity, keys, dsv41ReuseUnit(), keys.size());
    if (!lease)
        return 0;
    const auto&  data  = lease->data;
    const size_t count = data.keys.size();
    for (size_t group = 0; group < data.blocks.size(); ++group) {
        if (group < 4)
            resource.mutableBlockIds(group).assign(data.blocks[group]);
        else {
            BlockIndicesType ids(count, NULL_BLOCK_IDX);
            ids.back() = data.blocks[group].front();
            resource.mutableBlockIds(group).assign(std::move(ids));
        }
    }
    resource.setDsv41GpuLease(std::move(lease), true);
    return count;
}

bool HybridPoolKVCacheAllocator::cloneDsv41FixedBacking(KVCacheResource& resource) {
    const auto& lease = resource.dsv41GpuLease();
    if (!lease)
        return true;
    const size_t           ordinal = lease->data.keys.size() - 1;
    std::array<int32_t, 2> fresh{NULL_BLOCK_IDX, NULL_BLOCK_IDX};
    try {
        BatchCopyParams copies;
        for (size_t offset = 0; offset < fresh.size(); ++offset) {
            const size_t group = 4 + offset;
            auto&        pool  = group_block_pools_[group];
            if (pool->freeBlocksNum() == 0)
                shared_block_cache_->evictAndFreeForGroup(group, 1);
            auto ids = pool->malloc(1);
            if (ids.size() != 1)
                throw std::runtime_error("V4.1 GPU restore cannot allocate private fixed backing");
            fresh[offset]     = ids.front();
            const auto source = lease->data.blocks[group].front();
            for (int owner : config_.global_layer_ids[group]) {
                const auto src = kv_cache_groups_[group]->convertIndexToAddr(owner, source);
                const auto dst = kv_cache_groups_[group]->convertIndexToAddr(owner, fresh[offset]);
                copies.add(dst.kv_addr,
                           src.kv_addr,
                           config_.cache_specs[group]->block_size_bytes(),
                           BatchCopyParams::get_copy_type(pool->where(), pool->where()));
            }
        }
        execBatchCopy(copies);
        runtimeSyncAndCheck();
        for (size_t offset = 0; offset < fresh.size(); ++offset) {
            const size_t group = 4 + offset;
            const auto   old   = resource.blocks(group)[ordinal];
            resource.mutableBlockIds(group).setAt(ordinal, fresh[offset]);
            fresh[offset] = NULL_BLOCK_IDX;
            group_block_pools_[group]->requestFree(old);
        }
        return true;
    } catch (const std::exception& error) {
        for (size_t offset = 0; offset < fresh.size(); ++offset) {
            if (fresh[offset] > 0)
                group_block_pools_[4 + offset]->requestFree(fresh[offset]);
        }
        RTP_LLM_LOG_WARNING("V4.1 GPU checkpoint restore failed: %s", error.what());
        return false;
    }
}

MallocResult HybridPoolKVCacheAllocator::initMallocForCommonLen(const MallocInfo& info) {
    auto result = HybridKVCacheAllocator::initMallocForCommonLen(info);
    if (!result.success || config_.dsv41_cache_layout_version == 0)
        return result;
    if (!cloneDsv41FixedBacking(info.batch_kv_cache_resource->cacheResource(0))) {
        FreeInfo free_info{info.batch_kv_cache_resource, info.complete_token_ids};
        HybridKVCacheAllocator::free(free_info);
        info.batch_kv_cache_resource->cacheResource(0).setDeviceReuseBlockNum(0);
        return {false, 0};
    }
    return result;
}

MallocResult HybridPoolKVCacheAllocator::incrMalloc(const MallocInfo& info) {
    auto adjusted = info;
    if (config_.dsv41_cache_layout_version != 0 && info.batch_kv_cache_resource
        && info.batch_kv_cache_resource->cacheResource(0).dsv41GpuRestorePending()) {
        // The first forward still reads N's restored ring even when its suffix
        // allocation spans several blocks. Cleanup resumes after that boundary.
        adjusted.enable_remove_skipped_blocks = false;
    }
    auto result = HybridKVCacheAllocator::incrMalloc(adjusted);
    if (!result.success || config_.dsv41_cache_layout_version == 0)
        return result;
    auto& resource = info.batch_kv_cache_resource->cacheResource(0);
    if (resource.dsv41GpuRestorePending()) {
        try {
            resource.dsv41CacheState()->restore(resource.dsv41GpuLease()->data.metadata, dsv41ReuseUnit());
            resource.completeDsv41GpuRestore();
        } catch (const std::exception& error) {
            RTP_LLM_LOG_WARNING("V4.1 GPU checkpoint progress could not be committed: %s", error.what());
            return {false, 0};
        }
    }
    return result;
}

BatchKVCacheResourcePtr HybridPoolKVCacheAllocator::leaseDsv41ForMemoryTransfer() {
    if (!shared_block_cache_)
        return nullptr;
    auto lease = shared_block_cache_->leaseDsv41CheckpointForTransfer();
    if (!lease)
        return nullptr;
    const auto& data = lease->data;
    dsv41Identity(data.metadata.identity);
    auto batch = std::make_shared<BatchKVCacheResource>();
    batch->resetBatchSize(1);
    batch->initGroups(6,
                      43,
                      config_.layer_to_group_id,
                      config_.kernelBlocksPerKvBlock(),
                      config_.group_types,
                      config_.layer_region_to_group_id);
    auto& resource = batch->cacheResource(0);
    resource.setCacheKeys(data.keys);
    resource.rebuildLinearBlockDependencies();
    resource.setCacheKeysAreCpCanonical(true);
    resource.setLastBlockAligned(true);
    for (size_t group = 0; group < data.blocks.size(); ++group) {
        if (group < 4)
            resource.mutableBlockIds(group).assign(data.blocks[group]);
        else {
            BlockIndicesType ids(data.keys.size(), NULL_BLOCK_IDX);
            ids.back() = data.blocks[group].front();
            resource.mutableBlockIds(group).assign(std::move(ids));
        }
    }
    auto state = std::make_shared<DSV41CacheState>(data.metadata.identity);
    state->advanceEncoder(data.metadata.materialized_end);
    state->completeDecoder(data.metadata, data.reuse_unit);
    state->finish(data.metadata.materialized_end);
    resource.setDsv41CacheState(std::move(state));
    resource.setDsv41GpuLease(lease);
    resource.setDsv41GpuTransferCommit(
        [cache = shared_block_cache_, lease] { cache->commitDsv41CheckpointTransfer(lease); });
    return batch;
}

}  // namespace rtp_llm
