#include "rtp_llm/cpp/cache/connector/memory/KVCacheMemoryConnector.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <set>
#include <sstream>

#include "rtp_llm/cpp/cache/DSV41KVCacheSpec.h"
#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache/connector/memory/MemoryAsyncContext.h"
#include "rtp_llm/models_py/bindings/NoBlockCopy.h"

namespace rtp_llm {
namespace {

class DSV41CompletedContext final: public AsyncContext {
public:
    explicit DSV41CompletedContext(bool success): success_(success) {}
    void waitDone() override {}
    bool done() const override {
        return true;
    }
    bool success() const override {
        return success_;
    }

private:
    bool success_;
};

struct DSV41ReadLease {
    std::shared_ptr<PrefixTreeMemoryBlockCache> tree;
    DSV41CacheIdentity                          identity;
    PrefixTreeMemoryBlockCache::DSV41Match      match;
    std::mutex                                  read_mutex;
    bool                                        consumed{false};
    ~DSV41ReadLease() {
        tree->releaseDsv41InFlight(identity, match);
    }
};

bool copyFinished(const std::shared_ptr<BroadcastResult<FunctionRequestPB, FunctionResponsePB>>& result) {
    if (!result)
        return false;
    result->waitDone();
    if (!result->success())
        return false;
    for (const auto& response : result->responses()) {
        if (!response.has_mem_response() || !response.mem_response().success())
            return false;
    }
    return true;
}

}  // namespace

struct KVCacheMemoryConnector::DSV41StagedSnapshot final: public DSV41CheckpointSnapshot {
    DSV41StagedSnapshot(const DSV41CheckpointMetadata&                      metadata,
                        std::shared_ptr<CopyPlan>                           plan,
                        std::shared_ptr<PrefixTreeMemoryBlockCache>         tree,
                        std::vector<PrefixTreeMemoryBlockCache::DSV41Entry> entries):
        DSV41CheckpointSnapshot(metadata), plan(std::move(plan)), tree(std::move(tree)), entries(std::move(entries)) {}
    std::shared_ptr<CopyPlan>                           plan;
    std::shared_ptr<PrefixTreeMemoryBlockCache>         tree;
    std::vector<PrefixTreeMemoryBlockCache::DSV41Entry> entries;
};

bool KVCacheMemoryConnector::isDsv41TypedCacheLayout(const std::vector<LayerRegionSlot>& slots) const {
    if (cache_config_.dsv41_cache_layout_version != 1 || cache_config_.layer_all_num != 43
        || cache_config_.cache_specs.size() != 6 || slots.size() != 54 || !cache_config_.use_typed_cache_regions
        || !cache_config_.use_opaque_kv_cache_store || !cache_config_.use_independent_block_pools)
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
           && swa.size() == 43 && *swa.begin() == 0 && *swa.rbegin() == 42;
}

std::string KVCacheMemoryConnector::dsv41LayoutFingerprint() const {
    const auto slots = layerRegionSlots();
    if (!isDsv41TypedCacheLayout(slots))
        throw std::logic_error("invalid V4.1 memory layout");
    std::ostringstream output;
    output << "dsv41-memory-v1:target40:draft3:";
    for (size_t group = 0; group < cache_config_.cache_specs.size(); ++group) {
        output << group << ':' << cache_config_.cache_specs[group]->debugString() << ":owners=";
        for (const auto& slot : slots) {
            if (static_cast<size_t>(slot.group_id) == group)
                output << slot.layer_id << ',';
        }
        output << ';';
    }
    return output.str();
}

size_t KVCacheMemoryConnector::dsv41ReuseUnit() const {
    auto spec = std::dynamic_pointer_cast<DSV41KVCacheSpec>(cache_config_.cache_specs.at(5));
    if (!spec)
        throw std::logic_error("V4.1 checkpoint has no typed SWA spec");
    return cache_config_.seq_size_per_block * spec->cp_size;
}

DSV41CacheIdentity KVCacheMemoryConnector::dsv41CacheIdentity(const std::string& model_revision,
                                                              DSV41ReplayMode    replay_mode) const {
    auto               spec = std::dynamic_pointer_cast<DSV41KVCacheSpec>(cache_config_.cache_specs.at(5));
    DSV41CacheIdentity identity{model_revision, dsv41LayoutFingerprint(), replay_mode, 1, 128, spec->entries_per_block};
    identity.validate();
    return identity;
}

std::shared_ptr<KVCacheMemoryConnector::CopyPlan>
KVCacheMemoryConnector::createDsv41CopyPlan(std::vector<CopyInfoPerKey> infos, CopyDirection direction) {
    auto compressed = compressed_pool_;
    auto state      = state_swa_pool_;
    auto deleter    = [compressed, state](CopyPlan* plan) {
        for (const auto& info : plan->copy_infos) {
            if (info.mem_block > 0) {
                auto pool = info.kind == CacheBlockKind::COMPRESSED_KV ? compressed : state;
                pool->requestFree(info.mem_block);
            }
        }
        delete plan;
    };
    auto result        = std::shared_ptr<CopyPlan>(new CopyPlan(), std::move(deleter));
    result->copy_infos = std::move(infos);
    result->direction  = direction;
    return result;
}

bool KVCacheMemoryConnector::allocateDsv41Backings(std::vector<CopyInfoPerKey>& infos) {
    size_t compressed = 0, state = 0;
    for (const auto& info : infos) {
        if (info.kind == CacheBlockKind::COMPRESSED_KV)
            ++compressed;
        else
            ++state;
    }
    if (compressed > compressed_pool_->totalBlocksNum() || state > state_swa_pool_->totalBlocksNum())
        return false;
    while (compressed_pool_->freeBlocksNum() < compressed || state_swa_pool_->freeBlocksNum() < state) {
        const auto evicted = prefix_block_cache_->popOldestDsv41JointEvictable();
        if (evicted.empty())
            return false;
        for (const auto& item : evicted) {
            memoryPoolFor(item.kind)->blockCacheFree(item.block_index);
            reportEvictionLifetime(item.kind, item.backing_type, item.created_time_us);
        }
    }
    std::vector<size_t> allocated;
    for (size_t i = 0; i < infos.size(); ++i) {
        auto blocks = memoryPoolFor(infos[i].kind)->malloc(1);
        if (blocks.size() != 1 || blocks.front() <= 0) {
            for (auto index : allocated) {
                memoryPoolFor(infos[index].kind)->requestFree(infos[index].mem_block);
                infos[index].mem_block = NULL_BLOCK_IDX;
            }
            return false;
        }
        infos[i].mem_block = blocks.front();
        allocated.push_back(i);
    }
    return true;
}

bool KVCacheMemoryConnector::stageDsv41Checkpoint(const std::shared_ptr<KVCacheResource>& resource,
                                                  const std::function<void()>&            wait_for_producer) {
    if (!resource || !resource->dsv41CacheState() || !wait_for_producer || stop_.load())
        return false;
    auto state = resource->dsv41CacheState();
    auto view  = state->view();
    if (!view.completed || view.finished || view.cancelled
        || view.encoder_materialized_end != view.decoder_checkpoint_end
        || view.completed->materialized_end != view.decoder_checkpoint_end
        || !(view.identity == dsv41CacheIdentity(view.identity.model_revision, view.identity.replay_mode)))
        return false;
    const auto unit     = dsv41ReuseUnit();
    const auto swa_spec = std::dynamic_pointer_cast<DSV41KVCacheSpec>(cache_config_.cache_specs[5]);
    if (swa_spec->cp_size > 1 && !swa_spec->prefill_byte_slice) {
        throw std::logic_error("V4.1 decode memory requires explicit eight-shard global copy planning");
    }
    view.completed->validate(unit);
    const size_t count = view.completed->materialized_end / unit;
    if (count == 0 || resource->cacheKeys().size() < count)
        return false;
    // CP8 uses the existing canonical last-rank-key resource. D-side gathering
    // of all eight CP shards is provided by its runtime coordinator, not guessed.
    if (unit != cache_config_.seq_size_per_block && !resource->cacheKeysAreCpCanonical())
        return false;
    resource->ensureLinearBlockDependencies();
    const auto& dependencies = resource->blockDependencies();
    if (dependencies.size() < count)
        return false;
    const auto                  slots = layerRegionSlots();
    std::vector<CopyInfoPerKey> infos;
    for (size_t key_index = 0; key_index < count; ++key_index) {
        const auto& dep = dependencies[key_index];
        if (dep.ordinal != key_index || dep.has_parent != (key_index > 0)
            || (key_index > 0 && dep.parent_key != resource->cacheKeys()[key_index - 1]))
            return false;
        for (auto kind : {CacheBlockKind::COMPRESSED_KV, CacheBlockKind::STATE_SWA_KV}) {
            if (kind == CacheBlockKind::STATE_SWA_KV && key_index + 1 != count)
                continue;
            CopyInfoPerKey info;
            info.cache_key  = resource->cacheKeys()[key_index];
            info.kind       = kind;
            info.block_size = prefixKindBlockSize(kind, slots);
            info.gpu_blocks.resize(slots.size(), NULL_BLOCK_IDX);
            info.slot_valid_mask.resize(slots.size(), 0);
            for (size_t slot_index = 0; slot_index < slots.size(); ++slot_index) {
                const auto& slot = slots[slot_index];
                if (kindForSlot(slot) != kind)
                    continue;
                const auto& blocks = resource->blocks(slot.layer_id, slot.region_name);
                if (key_index >= blocks.size() || blocks[key_index] <= 0)
                    return false;
                info.gpu_blocks[slot_index]      = blocks[key_index];
                info.slot_valid_mask[slot_index] = 1;
            }
            infos.push_back(std::move(info));
        }
    }
    try {
        wait_for_producer();
        std::shared_ptr<CopyPlan> plan;
        {
            std::lock_guard<std::mutex> lock(dsv41_transaction_mutex_);
            if (!allocateDsv41Backings(infos))
                return false;
            plan = createDsv41CopyPlan(std::move(infos), CopyDirection::D2H);
        }
        if (!copyFinished(sendCopyPlan(plan)))
            return false;
        std::vector<PrefixTreeMemoryBlockCache::DSV41Entry> entries;
        for (size_t index = 0; index < count; ++index) {
            const auto&                            info = plan->copy_infos[index];
            PrefixTreeMemoryBlockCache::DSV41Entry entry;
            entry.global.cache_key       = info.cache_key;
            entry.global.kind            = info.kind;
            entry.global.block_index     = info.mem_block;
            entry.global.block_size      = info.block_size;
            entry.global.slot_valid_mask = info.slot_valid_mask;
            entry.dependency             = dependencies[index];
            entries.push_back(std::move(entry));
        }
        const auto& swa           = plan->copy_infos.back();
        auto        item          = entries.back().global;
        item.kind                 = CacheBlockKind::STATE_SWA_KV;
        item.block_index          = swa.mem_block;
        item.block_size           = swa.block_size;
        item.slot_valid_mask      = swa.slot_valid_mask;
        entries.back().swa        = item;
        entries.back().checkpoint = view.completed;
        state->protect(
            std::make_shared<DSV41StagedSnapshot>(*view.completed, plan, prefix_block_cache_, std::move(entries)));
        return true;
    } catch (const std::exception& error) {
        RTP_LLM_LOG_WARNING("V4.1 checkpoint staging failed: %s", error.what());
        return false;
    }
}

std::shared_ptr<AsyncContext> KVCacheMemoryConnector::dsv41Write(const std::shared_ptr<KVCacheResource>& resource) {
    if (!resource->dsv41CacheState())
        throw std::logic_error("V4.1 memory write requires typed checkpoint state");
    const bool success = resource->dsv41CacheState()->publishSnapshots([this](const DSV41CacheState::View& view) {
        if (view.snapshots.empty())
            return view.protected_prefix_end == 0;
        std::lock_guard<std::mutex> lock(dsv41_transaction_mutex_);
        auto                        same_item_bytes = [this](const PrefixTreeMemoryBlockCache::CacheItem& lhs,
                                      const PrefixTreeMemoryBlockCache::CacheItem& rhs) {
            if (lhs.kind != rhs.kind || lhs.block_size != rhs.block_size || lhs.slot_valid_mask != rhs.slot_valid_mask)
                return false;
            const auto left  = memoryPoolFor(lhs.kind)->convertIndexToBuffer(0, lhs.block_index);
            const auto right = memoryPoolFor(rhs.kind)->convertIndexToBuffer(0, rhs.block_index);
            return left.size() == 1 && right.size() == 1 && left[0].size_bytes == right[0].size_bytes
                   && std::memcmp(left[0].addr, right[0].addr, left[0].size_bytes) == 0;
        };
        std::map<size_t, PrefixTreeMemoryBlockCache::DSV41Entry> merged;
        for (const auto& snapshot : view.snapshots) {
            auto staged = std::dynamic_pointer_cast<DSV41StagedSnapshot>(snapshot);
            if (!staged || staged->tree != prefix_block_cache_ || !(snapshot->metadata().identity == view.identity))
                return false;
            for (size_t i = 0; i < staged->entries.size(); ++i) {
                auto [it, inserted] = merged.emplace(i, staged->entries[i]);
                if (!inserted) {
                    if (it->second.global.cache_key != staged->entries[i].global.cache_key
                        || !same_item_bytes(it->second.global, staged->entries[i].global)
                        || (it->second.swa && staged->entries[i].swa
                            && (!same_item_bytes(*it->second.swa, *staged->entries[i].swa)
                                || !(*it->second.checkpoint == *staged->entries[i].checkpoint))))
                        return false;
                    if (staged->entries[i].swa && !it->second.swa) {
                        it->second.swa        = staged->entries[i].swa;
                        it->second.checkpoint = staged->entries[i].checkpoint;
                    }
                }
            }
        }
        std::vector<PrefixTreeMemoryBlockCache::DSV41Entry> entries;
        for (auto& [_, entry] : merged)
            entries.push_back(std::move(entry));
        auto committed =
            prefix_block_cache_->putDsv41Committed(view.identity, entries, [&](const auto& lhs, const auto& rhs) {
                return same_item_bytes(lhs.global, rhs.global)
                       && (!lhs.swa || !rhs.swa || same_item_bytes(*lhs.swa, *rhs.swa));
            });
        if (!committed.success)
            return false;
        for (const auto& item : committed.retained)
            memoryPoolFor(item.kind)->blockCacheReference(item.block_index);
        return true;
    });
    return std::make_shared<DSV41CompletedContext>(success);
}

std::shared_ptr<AsyncMatchContext>
KVCacheMemoryConnector::dsv41Match(const std::shared_ptr<KVCacheResource>& resource) {
    if (!resource->dsv41CacheState())
        throw std::logic_error("V4.1 memory match requires mode and layout identity");
    auto view = resource->dsv41CacheState()->view();
    if (view.cancelled || view.finished
        || !(view.identity == dsv41CacheIdentity(view.identity.model_revision, view.identity.replay_mode)))
        return nullptr;
    const auto swa_spec = std::dynamic_pointer_cast<DSV41KVCacheSpec>(cache_config_.cache_specs[5]);
    if (swa_spec->cp_size > 1 && !swa_spec->prefill_byte_slice) {
        throw std::logic_error("V4.1 decode memory requires explicit eight-shard global copy planning");
    }
    if (dsv41ReuseUnit() != cache_config_.seq_size_per_block && !resource->cacheKeysAreCpCanonical())
        return nullptr;
    resource->ensureLinearBlockDependencies();
    const size_t                limit = resource->cacheKeys().empty() ? 0 : resource->cacheKeys().size() - 1;
    std::lock_guard<std::mutex> lock(dsv41_transaction_mutex_);
    auto                        lease = std::make_shared<DSV41ReadLease>();
    lease->tree                       = prefix_block_cache_;
    lease->identity                   = view.identity;
    lease->match                      = prefix_block_cache_->matchDsv41AndMarkInFlight(
        view.identity, resource->cacheKeys(), resource->blockDependencies(), limit);
    const size_t count = lease->match.matched_blocks;
    const size_t start = resource->reuseBlockNum();
    if (count <= start)
        return nullptr;
    const auto metadata = *lease->match.chain.back().checkpoint;
    metadata.validate(dsv41ReuseUnit());
    if (metadata.materialized_end != static_cast<int64_t>(count * dsv41ReuseUnit()))
        return nullptr;
    // The lease protects the complete chain while CP selects a common boundary
    // and the allocator assigns its destination pages. No stale GPU IDs persist.
    auto plan              = createDsv41CopyPlan({}, CopyDirection::H2D);
    plan->dsv41_checkpoint = metadata;
    plan->dsv41_keys.assign(resource->cacheKeys().begin(), resource->cacheKeys().begin() + count);
    plan->dsv41_lease = lease;
    return std::make_shared<MemoryAsyncMatchContext>(
        count, static_cast<int>(start), static_cast<int>(count - start), plan);
}

std::vector<int64_t>
KVCacheMemoryConnector::dsv41MatchedCheckpointEnds(const std::shared_ptr<AsyncMatchContext>& match_context) const {
    auto match = std::dynamic_pointer_cast<MemoryAsyncMatchContext>(match_context);
    if (!match || !match->readCopyPlan())
        return {};
    auto plan = std::static_pointer_cast<CopyPlan>(match->readCopyPlan());
    if (!plan->dsv41_checkpoint || !plan->dsv41_lease)
        return {};
    auto                 lease = std::static_pointer_cast<DSV41ReadLease>(plan->dsv41_lease);
    std::vector<int64_t> ends;
    for (size_t i = static_cast<size_t>(match->startReadBlockIndex()); i < lease->match.chain.size(); ++i) {
        const auto& entry = lease->match.chain[i];
        if (entry.checkpoint)
            ends.push_back(entry.checkpoint->materialized_end);
    }
    return ends;
}

std::shared_ptr<AsyncContext> KVCacheMemoryConnector::dsv41Read(const std::shared_ptr<KVCacheResource>&   resource,
                                                                const std::shared_ptr<AsyncMatchContext>& match_context,
                                                                int                                       start_index,
                                                                int                                       read_num) {
    auto match  = std::dynamic_pointer_cast<MemoryAsyncMatchContext>(match_context);
    auto failed = std::make_shared<DSV41CompletedContext>(false);
    if (!match || !match->readCopyPlan() || !resource->dsv41CacheState() || read_num <= 0
        || start_index < match->startReadBlockIndex() || start_index < 0
        || static_cast<size_t>(start_index) != resource->reuseBlockNum())
        return failed;
    auto         matched_plan = std::static_pointer_cast<CopyPlan>(match->readCopyPlan());
    const size_t end_index    = static_cast<size_t>(start_index) + static_cast<size_t>(read_num);
    if (!matched_plan->dsv41_checkpoint || !matched_plan->dsv41_lease || end_index > matched_plan->dsv41_keys.size()
        || end_index > resource->cacheKeys().size()
        || !std::equal(matched_plan->dsv41_keys.begin(),
                       matched_plan->dsv41_keys.begin() + end_index,
                       resource->cacheKeys().begin()))
        return failed;
    auto                        lease = std::static_pointer_cast<DSV41ReadLease>(matched_plan->dsv41_lease);
    std::lock_guard<std::mutex> read_lock(lease->read_mutex);
    if (lease->tree != prefix_block_cache_ || end_index > lease->match.chain.size() || lease->consumed
        || !lease->match.chain[end_index - 1].checkpoint)
        return failed;
    const auto metadata = *lease->match.chain[end_index - 1].checkpoint;
    try {
        metadata.validate(dsv41ReuseUnit());
        const auto view = resource->dsv41CacheState()->view();
        if (view.finished || view.cancelled || !(view.identity == metadata.identity)
            || view.encoder_materialized_end != view.decoder_checkpoint_end
            || view.encoder_materialized_end != static_cast<int64_t>(start_index * dsv41ReuseUnit())
            || metadata.materialized_end != static_cast<int64_t>(end_index * dsv41ReuseUnit()))
            return failed;
        const auto                  slots = layerRegionSlots();
        std::vector<CopyInfoPerKey> infos;
        for (size_t index = static_cast<size_t>(start_index); index < end_index; ++index) {
            const auto&                                        entry = lease->match.chain[index];
            std::vector<PrefixTreeMemoryBlockCache::CacheItem> items{entry.global};
            if (index + 1 == end_index)
                items.push_back(*entry.swa);
            for (const auto& item : items) {
                CopyInfoPerKey info;
                info.cache_key       = item.cache_key;
                info.kind            = item.kind;
                info.mem_block       = item.block_index;
                info.block_size      = item.block_size;
                info.generation      = item.generation;
                info.slot_valid_mask = item.slot_valid_mask;
                info.gpu_blocks.resize(slots.size(), NULL_BLOCK_IDX);
                for (size_t s = 0; s < slots.size(); ++s) {
                    if (kindForSlot(slots[s]) != info.kind)
                        continue;
                    const auto& blocks = resource->blocks(slots[s].layer_id, slots[s].region_name);
                    if (index >= blocks.size() || blocks[index] <= 0)
                        return failed;
                    info.gpu_blocks[s] = blocks[index];
                }
                infos.push_back(std::move(info));
            }
        }
        for (const auto& info : infos)
            memoryPoolFor(info.kind)->requestReference(info.mem_block);
        auto plan         = createDsv41CopyPlan(std::move(infos), CopyDirection::H2D);
        plan->dsv41_lease = lease;
        if (!copyFinished(sendCopyPlan(plan)))
            return failed;
        resource->dsv41CacheState()->restore(metadata, dsv41ReuseUnit());
        resource->setMemoryReuseBlockNum(resource->memoryReuseBlockNum() + read_num);
        lease->consumed = true;
        match->clearReadCopyPlan();
        return std::make_shared<DSV41CompletedContext>(true);
    } catch (const std::exception& error) {
        RTP_LLM_LOG_WARNING("V4.1 checkpoint restore failed: %s", error.what());
        return failed;
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
        // Prefix boundaries are even. A restored request starts with no pending
        // ratio2 pair; clear all seven snapshots, including speculative slack.
        for (const auto& [pointer, bytes] : zero_pairs)
            std::memset(pointer, 0, bytes);
        execNoBlockCopy(params);  // Dedicated copy stream is synchronized before return.
        return true;
    } catch (const std::exception& error) {
        RTP_LLM_LOG_WARNING("V4.1 memory byte copy failed: %s", error.what());
        return false;
    }
}

}  // namespace rtp_llm
