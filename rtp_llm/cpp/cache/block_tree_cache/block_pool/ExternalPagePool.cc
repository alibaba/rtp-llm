#include "ExternalPagePool.h"

#include <cassert>
#include <stdexcept>
#include <tuple>
#include <unordered_set>

namespace rtp_llm {
namespace {
std::shared_ptr<BlockPoolConfigBase> externalConfig() {
    auto config                  = std::make_shared<BlockPoolConfigBase>();
    config->pool_name            = "external_device";
    config->pool_type            = BlockPoolType::DEVICE;
    config->physical_block_count = 1;  // the native pool reserves slot zero
    config->growable             = true;
    return config;
}

template<class Apply>
void applyRequestRefs(const BlockIdList& blocks, Apply apply) {
    if (blocks.size() < 2) {
        apply(blocks);
        return;
    }
    // An allocator hold can back several logical rows. The native bulk pool
    // operation checks uniqueness, so apply repeated claims once per row.
    std::unordered_set<BlockIdxType> seen;
    seen.reserve(blocks.size());
    BlockIdList unique, repeated;
    for (const auto block : blocks) {
        (seen.insert(block).second ? unique : repeated).push_back(block);
    }
    apply(unique);
    for (const auto block : repeated) {
        apply(BlockIdList{block});
    }
}
}  // namespace

ExternalPagePool::ExternalPagePool(): IBlockPool(externalConfig()) {
    markInitialized();
}

std::optional<BlockIdxType> ExternalPagePool::prepare(const btc_page& page, bool require_unused_slot) {
    if (!page.hold || page.pool == BTC_HOST_POOL || page.pool == BTC_DISK_POOL) {
        throw std::invalid_argument("Expected a device pool and a nonzero external hold");
    }
    std::unique_lock lock(mutex_);
    const Slot       slot{page.pool, page.slot};
    const auto       existing_slot = slots_.find(slot);
    if (existing_slot != slots_.end() && (require_unused_slot || existing_slot->second.generation != page.generation)) {
        return std::nullopt;
    }
    if (const auto known = by_hold_.find(page.hold); known != by_hold_.end()) {
        const auto& item = *known->second;
        if (item.invalidated || isNullBlockIdx(item.block) || item.page.pool != page.pool || item.page.slot != page.slot
            || item.page.generation != page.generation) {
            return std::nullopt;
        }
        incTreeRefLocked(item.block, BlockTreeRefType::LOAD);
        return item.block;
    }
    // prepare/finishPrepare are serialized by the host adapter. Native workers
    // can drop refs throughout the call, including while the pool grows.
    if (!availableFreeBlocksNoLock()) {
        lock.unlock();
        appendCapacity(1);
        lock.lock();
    }
    auto record = live_.insert(live_.end(), {page, NULL_BLOCK_IDX, false, false});
    try {
        by_hold_.emplace(page.hold, record);
        auto occupants = slots_.try_emplace(slot, Occupants{page.generation, 0}).first;
        ++occupants->second.holds;
        try {
            const auto allocated = mallocNoLock(1);
            assert(allocated && "external pool has a reserved free index");
            record->block = allocated->front();
            by_block_.emplace(record->block, record);
        } catch (...) {
            if (!isNullBlockIdx(record->block)) {
                const bool was_available = isAvailableNoLock(record->block);
                freeAllocatedBlockNoLock(record->block);
                updateAvailableBlocksNumNoLock(record->block, was_available);
            }
            if (!--occupants->second.holds) {
                slots_.erase(occupants);
            }
            throw;
        }
    } catch (...) {
        by_hold_.erase(page.hold);
        live_.erase(record);
        throw;
    }
    incTreeRefLocked(record->block, BlockTreeRefType::LOAD);
    return record->block;
}

void ExternalPagePool::finishPrepare(BlockIdxType block, bool adopted) {
    if (adopted) {
        std::lock_guard lock(mutex_);
        by_block_.at(block)->adopted = true;
    }
    decTreeRef(block, BlockTreeRefType::LOAD);
}

btc_page ExternalPagePool::describe(BlockIdxType block) const {
    std::lock_guard lock(mutex_);
    return by_block_.at(block)->page;
}

std::vector<BlockIdxType> ExternalPagePool::resolve(const btc_page& page) const {
    std::lock_guard lock(mutex_);
    if (page.hold) {
        const auto it = by_hold_.find(page.hold);
        if (it != by_hold_.end()) {
            const auto& item = *it->second;
            if (!isNullBlockIdx(item.block) && item.page.pool == page.pool && item.page.slot == page.slot
                && item.page.generation == page.generation) {
                return {item.block};
            }
        }
        return {};
    }
    std::vector<BlockIdxType> result;
    for (const auto& item : live_) {
        if (item.page.pool == page.pool && item.page.slot == page.slot && item.page.generation == page.generation) {
            result.push_back(item.block);
        }
    }
    return result;
}

ExternalPagePool::PhysicalResolution
ExternalPagePool::resolvePhysical(const btc_page* pages, size_t count, uint64_t pool) const {
    const auto identity = [](const btc_page& page) { return std::make_tuple(page.pool, page.slot, page.generation); };
    using Identity = std::tuple<uint64_t, uint64_t, uint64_t>;
    std::map<Identity, BlockIdList> requested, companions;
    for (size_t i = 0; i < count; ++i) {
        requested.try_emplace(identity(pages[i]));
    }
    std::lock_guard lock(mutex_);
    for (const auto& record : live_) {
        const auto found = requested.find(identity(record.page));
        if (found != requested.end()) {
            found->second.push_back(record.block);
        } else if (record.page.pool == pool) {
            companions[identity(record.page)].push_back(record.block);
        }
    }
    PhysicalResolution result;
    for (const auto& [identity, aliases] : requested) {
        const auto& [provider, slot, generation] = identity;
        if (provider == pool && aliases.empty()) {
            const auto occupied = slots_.find({provider, slot});
            result.returning += occupied != slots_.end() && occupied->second.generation == generation;
        }
    }
    result.provided.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        result.provided.push_back(requested.at(identity(pages[i])));
    }
    result.companions.reserve(companions.size());
    for (auto& item : companions) {
        result.companions.push_back(std::move(item.second));
    }
    return result;
}

void ExternalPagePool::invalidate(const BlockIdList& blocks) {
    std::lock_guard lock(mutex_);
    for (auto block : blocks) {
        const auto found = by_block_.find(block);
        if (found != by_block_.end()) {
            found->second->invalidated = true;
        }
    }
}

bool ExternalPagePool::canPublish(BlockIdxType block) const {
    std::lock_guard lock(mutex_);
    const auto      found = by_block_.find(block);
    return found != by_block_.end() && !found->second->invalidated;
}

void ExternalPagePool::forgetNoLock(Records& records, Records::iterator item) {
    const auto it = slots_.find({item->page.pool, item->page.slot});
    assert(it != slots_.end() && it->second.holds && "external slot lost its hold");
    if (!--it->second.holds) {
        slots_.erase(it);
    }
    by_hold_.erase(item->page.hold);
    records.erase(item);
}

bool ExternalPagePool::onLastTreeRefNoLock(BlockIdxType block) {
    const auto it = by_block_.find(block);
    assert(it != by_block_.end() && "native device reference missing external hold");
    const auto item = it->second;
    by_block_.erase(it);
    item->block = NULL_BLOCK_IDX;
    if (item->adopted) {
        released_.splice(released_.end(), live_, item);
    } else {
        forgetNoLock(live_, item);
    }
    return true;
}

std::vector<btc_page> ExternalPagePool::takeReleased() {
    std::lock_guard       lock(mutex_);
    std::vector<btc_page> result;
    result.reserve(released_.size());
    for (const auto& item : released_) {
        result.push_back(item.page);
    }
    awaiting_ack_.splice(awaiting_ack_.end(), released_);
    return result;
}

void ExternalPagePool::acknowledgeRelease(const btc_page& page) {
    std::lock_guard lock(mutex_);
    auto            item = by_hold_.at(page.hold);
    assert(isNullBlockIdx(item->block) && "external release acknowledged while native owner is live");
    forgetNoLock(awaiting_ack_, item);
}

bool ExternalPagePool::hasOutstandingHolds() const {
    std::lock_guard lock(mutex_);
    return !by_hold_.empty();
}

void ExternalPagePool::referenceRequestBlocks(const BlockIdList& blocks) {
    applyRequestRefs(blocks, [&](const auto& batch) { incTreeRef(batch, BlockTreeRefType::LOAD); });
}

void ExternalPagePool::unreferenceRequestBlocks(const BlockIdList& blocks) {
    applyRequestRefs(blocks, [&](const auto& batch) { decTreeRef(batch, BlockTreeRefType::LOAD); });
}
}  // namespace rtp_llm
