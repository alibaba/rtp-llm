#include "rtp_llm/cpp/cache/connector/memory/KVCacheMemoryConnector.h"

#include <algorithm>
#include <cstring>
#include <set>

#include "rtp_llm/cpp/cache/DSV41KVCacheSpec.h"
#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/models_py/bindings/NoBlockCopy.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm {
namespace {

class DSV41ProtectedSnapshot final: public DSV41CheckpointSnapshot {
public:
    DSV41ProtectedSnapshot(const DSV41CheckpointMetadata&          metadata,
                           std::shared_ptr<KVCacheMemoryConnector> owner,
                           std::shared_ptr<void>                   copy_plan):
        DSV41CheckpointSnapshot(metadata), owner_(std::move(owner)), copy_plan_(std::move(copy_plan)) {}

private:
    // CopyPlan releases pins through its owner, so destroy the plan first.
    std::shared_ptr<KVCacheMemoryConnector> owner_;
    std::shared_ptr<void>                   copy_plan_;
};

}  // namespace

bool KVCacheMemoryConnector::isDsv41TypedCacheLayout(const std::vector<LayerRegionSlot>& slots) const {
    if (cache_config_.dsv41_cache_layout_version != 1
        || (cache_config_.layer_all_num != 40 && cache_config_.layer_all_num != 43)
        || cache_config_.cache_specs.size() != 6 || slots.size() != cache_config_.layer_all_num + 11
        || !cache_config_.use_typed_cache_regions || !cache_config_.use_opaque_kv_cache_store
        || !cache_config_.use_independent_block_pools)
        return false;
    std::set<int> global, index, pair, swa;
    for (const auto& slot : slots) {
        if (slot.group_id < 0 || static_cast<size_t>(slot.group_id) >= cache_config_.cache_specs.size())
            return false;
        auto spec = std::dynamic_pointer_cast<DSV41KVCacheSpec>(cache_config_.cache_specs[slot.group_id]);
        if (!spec || spec->region != slot.region_name || spec->block_size_bytes() != slot.stride_bytes
            || cache_config_.physicalOwner(slot.layer_id, slot.region_name) != slot.layer_id)
            return false;
        switch (slot.region_name) {
            case KVCacheRegionName::DSV41_GLOBAL_KV:
                global.insert(slot.layer_id);
                break;
            case KVCacheRegionName::DSV41_INDEX_KV:
                index.insert(slot.layer_id);
                break;
            case KVCacheRegionName::DSV41_PAIR_STATE:
                pair.insert(slot.layer_id);
                break;
            case KVCacheRegionName::SWA_KV:
                swa.insert(slot.layer_id);
                break;
            default:
                return false;
        }
    }
    return global == std::set<int>({2, 8, 14, 20}) && index == global && pair == std::set<int>({2, 8, 14})
           && swa.size() == cache_config_.layer_all_num && *swa.begin() == 0
           && *swa.rbegin() == static_cast<int>(cache_config_.layer_all_num) - 1;
}

std::string KVCacheMemoryConnector::dsv41LayoutFingerprint() const {
    const auto slots = layerRegionSlots();
    if (!isDsv41TypedCacheLayout(slots))
        throw std::logic_error("invalid V4.1 memory layout");
    return cache_config_.dsv41LayoutFingerprint();
}

size_t KVCacheMemoryConnector::dsv41ReuseUnit() const {
    auto spec = std::dynamic_pointer_cast<DSV41KVCacheSpec>(cache_config_.cache_specs.at(5));
    if (!spec)
        throw std::logic_error("V4.1 cache has no typed SWA spec");
    return cache_config_.seq_size_per_block * spec->cp_size;
}

size_t KVCacheMemoryConnector::dsv41DataUnit() const {
    const auto spec = std::dynamic_pointer_cast<DSV41KVCacheSpec>(cache_config_.cache_specs.at(5));
    if (!spec)
        throw std::logic_error("V4.1 cache has no typed SWA spec");
    return cache_config_.seq_size_per_block * (spec->prefill_byte_slice ? spec->cp_size : 1);
}

DSV41CacheIdentity KVCacheMemoryConnector::dsv41CacheIdentity(const std::string& model_revision,
                                                              DSV41ReplayMode    replay_mode) const {
    auto               spec = std::dynamic_pointer_cast<DSV41KVCacheSpec>(cache_config_.cache_specs.at(5));
    DSV41CacheIdentity identity{model_revision, dsv41LayoutFingerprint(), replay_mode, 1, 128, spec->entries_per_block};
    identity.validate();
    return identity;
}

bool KVCacheMemoryConnector::dsv41ResourceCompatible(const KVCacheResource& resource) const {
    const auto spec = std::dynamic_pointer_cast<DSV41KVCacheSpec>(cache_config_.cache_specs.at(5));
    if (!spec || (spec->cp_size > 1 && spec->prefill_byte_slice && !resource.cacheKeysAreCpCanonical()))
        return false;
    if (resource.dsv41CacheState()) {
        const auto view = resource.dsv41CacheState()->view();
        return view.identity == dsv41CacheIdentity(view.identity.model_revision, view.identity.replay_mode);
    }
    return resource.blockIdsAreKeyAligned();
}

bool KVCacheMemoryConnector::validDsv41Recovery(const KVCacheResource&                                resource,
                                                size_t                                                block_index,
                                                const std::shared_ptr<const DSV41CheckpointMetadata>& metadata) const {
    if (!metadata || (!resource.dsv41CacheState() && !resource.blockIdsAreKeyAligned())
        || block_index >= resource.blockDependencies().size())
        return false;
    try {
        metadata->validate(dsv41ReuseUnit());
        const auto& dependency = resource.blockDependencies()[block_index];
        return metadata->identity
                   == dsv41CacheIdentity(metadata->identity.model_revision, metadata->identity.replay_mode)
               && (!resource.dsv41CacheState() || metadata->identity == resource.dsv41CacheState()->view().identity)
               && metadata->materialized_end
                      == static_cast<int64_t>((uint64_t{dependency.ordinal} + 1) * dsv41DataUnit());
    } catch (const std::exception&) {
        return false;
    }
}

std::optional<size_t> KVCacheMemoryConnector::dsv41SlotIndex(const KVCacheResource& resource,
                                                             size_t                 key_index,
                                                             const LayerRegionSlot& slot) const {
    if (resource.blockIdsAreKeyAligned() || kindForSlot(slot) == CacheBlockKind::COMPRESSED_KV)
        return key_index;
    if (key_index >= resource.blockDependencies().size())
        return std::nullopt;
    const uint64_t end = (uint64_t{resource.blockDependencies()[key_index].ordinal} + 1) * dsv41DataUnit();
    if (end == 0 || end % dsv41ReuseUnit() != 0)
        return std::nullopt;
    return end / dsv41ReuseUnit() - 1;
}

bool KVCacheMemoryConnector::bindDsv41ReadPlan(CopyPlan&                           plan,
                                               const KVCacheResource&              resource,
                                               const std::vector<LayerRegionSlot>& slots) {
    for (auto it = plan.copy_infos.begin(); it != plan.copy_infos.end();) {
        auto&      info  = *it;
        const auto found = std::find(resource.cacheKeys().begin(), resource.cacheKeys().end(), info.cache_key);
        if (found == resource.cacheKeys().end())
            return false;
        const size_t index  = found - resource.cacheKeys().begin();
        bool         usable = info.kind != CacheBlockKind::STATE_SWA_KV
                      || (resource.dsv41CacheState() && validDsv41Recovery(resource, index, info.recovery_metadata));
        info.gpu_blocks.assign(slots.size(), NULL_BLOCK_IDX);
        for (size_t s = 0; s < slots.size(); ++s) {
            if (kindForSlot(slots[s]) != info.kind)
                continue;
            const auto& blocks     = resource.blocks(slots[s].layer_id, slots[s].region_name);
            const auto  slot_index = dsv41SlotIndex(resource, index, slots[s]);
            if (!slot_index || *slot_index >= blocks.size() || blocks[*slot_index] <= 0) {
                usable = false;
                break;
            }
            info.gpu_blocks[s] = blocks[*slot_index];
        }
        if (!usable) {
            if (info.kind != CacheBlockKind::STATE_SWA_KV)
                return false;
            auto release = createCopyPlan({info}, CopyDirection::H2D);
            it           = plan.copy_infos.erase(it);
        } else {
            ++it;
        }
    }
    return !plan.copy_infos.empty();
}

bool KVCacheMemoryConnector::stageDsv41Checkpoint(const std::shared_ptr<KVCacheResource>& resource,
                                                  const std::function<void()>&            wait_for_producer,
                                                  const std::shared_ptr<Meta>&            meta) {
    if (!resource || !resource->dsv41CacheState() || !wait_for_producer || !meta || stop_.load())
        return false;
    auto owner = weak_from_this().lock();
    if (!owner)
        return false;
    const auto state = resource->dsv41CacheState();
    const auto view  = state->view();
    if (!view.completed || view.finished || view.cancelled || !dsv41ResourceCompatible(*resource)
        || view.encoder_materialized_end != view.decoder_checkpoint_end
        || view.completed->materialized_end != view.decoder_checkpoint_end)
        return false;
    resource->ensureLinearBlockDependencies();
    size_t count = 0;
    for (size_t i = 0; i < resource->cacheKeys().size(); ++i) {
        if (static_cast<int64_t>((uint64_t{resource->blockDependencies()[i].ordinal} + 1) * dsv41DataUnit())
            == view.completed->materialized_end) {
            count = i + 1;
            break;
        }
    }
    if (count == 0)
        return false;
    auto source = std::make_shared<KVCacheResource>(*resource);
    source->setCacheKeys(CacheKeysType(resource->cacheKeys().begin(), resource->cacheKeys().begin() + count));
    source->setBlockDependencies(
        BlockDependenciesType(resource->blockDependencies().begin(), resource->blockDependencies().begin() + count));
    source->setCacheKeysAreCpCanonical(resource->cacheKeysAreCpCanonical());
    source->setLastBlockAligned(true);
    source->clearDsv41RecoveryMetadata();
    source->setDsv41RecoveryMetadata(count - 1, std::make_shared<DSV41CheckpointMetadata>(*view.completed));
    try {
        wait_for_producer();
        const auto current = state->view();
        if (!current.completed || !(*current.completed == *view.completed)
            || current.encoder_materialized_end != view.completed->materialized_end)
            return false;
        auto write = asyncWrite(source, meta);
        if (!write)
            return false;
        write->waitDone();
        if (!write->success())
            return false;
        const auto slots = layerRegionSlots();
        auto       plan  = buildPrefixCopyPlanForRead(source->cacheKeys(),
                                               source->blockDependencies(),
                                               resourceLayerRegionBlocks(*source, slots),
                                               slots,
                                               0,
                                               count,
                                               source.get());
        if (!plan || plan->copy_infos.empty() || !plan->copy_infos.back().recovery_metadata
            || !(*plan->copy_infos.back().recovery_metadata == *view.completed))
            return false;
        state->protect(std::make_shared<DSV41ProtectedSnapshot>(*view.completed, std::move(owner), plan));
        return true;
    } catch (const std::exception& error) {
        RTP_LLM_LOG_WARNING("V4.1 optional checkpoint protection failed: %s", error.what());
        return false;
    }
}

bool KVCacheMemoryConnector::copyDsv41MemoryItems(const MemoryOperationRequestPB&     request,
                                                  CopyDirection                       direction,
                                                  const std::vector<LayerRegionSlot>& slots) {
    if (!isDsv41TypedCacheLayout(slots) || request.copy_items_size() == 0 || !allocator_)
        return false;
    if (request.copy_direction() != MemoryOperationRequestPB::D2H
        && request.copy_direction() != MemoryOperationRequestPB::H2D)
        return false;
    MultiCopyParams                       params;
    std::vector<std::pair<void*, size_t>> zero_pairs;
    for (const auto& item : request.copy_items()) {
        if (item.backing_type() != MemoryOperationRequestPB::MEMORY
            || item.src_backing_type() != MemoryOperationRequestPB::MEMORY
            || item.disk_slot_presence_case() == MemoryOperationRequestPB::CopyItem::kDiskSlot
            || item.src_disk_slot_presence_case() == MemoryOperationRequestPB::CopyItem::kSrcDiskSlot
            || item.src_mem_block_presence_case() == MemoryOperationRequestPB::CopyItem::kSrcMemBlock
            || item.mem_block() <= 0 || item.gpu_blocks_size() != static_cast<int>(slots.size())
            || item.slot_valid_mask_size() != static_cast<int>(slots.size()))
            return false;
        CacheBlockKind kind;
        if (item.cache_block_kind() == MemoryOperationRequestPB::COMPRESSED_KV)
            kind = CacheBlockKind::COMPRESSED_KV;
        else if (item.cache_block_kind() == MemoryOperationRequestPB::STATE_SWA_KV)
            kind = CacheBlockKind::STATE_SWA_KV;
        else
            return false;
        const auto pool = memoryPoolFor(kind);
        if (!pool || static_cast<size_t>(item.mem_block()) > pool->totalBlocksNum())
            return false;
        const auto host = pool->convertIndexToBuffer(0, item.mem_block());
        if (host.size() != 1 || host[0].is_cuda || host[0].size_bytes != prefixKindBlockSize(kind, slots))
            return false;
        size_t offset = 0;
        for (size_t s = 0; s < slots.size(); ++s) {
            const auto& slot     = slots[s];
            const bool  required = kindForSlot(slot) == kind;
            if (item.slot_valid_mask(s) != (required ? 1 : 0) || (required && item.gpu_blocks(s) <= 0)
                || (!required && item.gpu_blocks(s) != NULL_BLOCK_IDX))
                return false;
            if (!required)
                continue;
            if (static_cast<size_t>(slot.group_id) >= cache_config_.group_block_nums.size()
                || static_cast<uint32_t>(item.gpu_blocks(s)) >= cache_config_.group_block_nums[slot.group_id])
                return false;
            const auto gpu    = allocator_->convertIndexToBuffer(slot.layer_id, slot.region_name, item.gpu_blocks(s));
            size_t     copied = 0;
            for (const auto& segment : gpu) {
                if (!segment.is_cuda || !segment.addr || copied + segment.size_bytes > slot.stride_bytes)
                    return false;
                if (direction == CopyDirection::D2H && slot.region_name == KVCacheRegionName::DSV41_PAIR_STATE) {
                    zero_pairs.emplace_back(static_cast<char*>(host[0].addr) + offset + copied, segment.size_bytes);
                } else if (!appendCopyBytesToBuffers(
                               host[0], segment, offset + copied, direction, params.multi_dst, params.multi_src))
                    return false;
                copied += segment.size_bytes;
            }
            if (copied != slot.stride_bytes)
                return false;
            offset += slot.stride_bytes;
        }
        if (offset != host[0].size_bytes)
            return false;
    }
    try {
        // Aligned boundaries have no pending ratio2 pair, including speculative slack.
        for (const auto& [pointer, bytes] : zero_pairs)
            std::memset(pointer, 0, bytes);
        execNoBlockCopy(params);
        return true;
    } catch (const std::exception& error) {
        RTP_LLM_LOG_WARNING("V4.1 memory byte copy failed: %s", error.what());
        return false;
    }
}

}  // namespace rtp_llm
