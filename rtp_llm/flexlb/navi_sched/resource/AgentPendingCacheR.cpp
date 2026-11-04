#include "rtp_llm/flexlb/navi_sched/resource/AgentPendingCacheR.h"

#include <algorithm>

#include "navi/engine/NaviRegisterMacro.h"

namespace navi_sched {

const std::string AgentPendingCacheR::RESOURCE_ID =
    "navi_sched.agent_pending_cache_r";

void AgentPendingCacheR::def(navi::ResourceDefBuilder &builder) const
{
    builder.name(RESOURCE_ID, navi::RS_BIZ);
}

navi::ErrorCode AgentPendingCacheR::init(navi::ResourceInitContext &)
{
    clear();
    return navi::EC_NONE;
}

bool AgentPendingCacheR::reserve(
    const NaviSchedPendingCacheReservation &reservation,
    AgentPendingCacheToMasterData &update, bool &changed)
{
    changed = false;
    if (!PendingCacheUtil::validReservation(reservation)) {
        return false;
    }
    autil::ScopedWriteLock lock(_lock);
    if (!_state.upsert(reservation, false, changed)) {
        return false;
    }
    if (!changed) {
        return true;
    }
    ++_revision;
    update.addedReservations.emplace_back(reservation);
    fillStateLocked(update);
    changed = true;
    return true;
}

bool AgentPendingCacheR::replace(
    const NaviSchedPendingCacheReservation &reservation,
    AgentPendingCacheToMasterData &update, bool &changed)
{
    changed = false;
    if (!PendingCacheUtil::validReservation(reservation)) {
        return false;
    }
    autil::ScopedWriteLock lock(_lock);
    if (!_state.upsert(reservation, true, changed)) {
        return false;
    }
    if (!changed) {
        return true;
    }
    ++_revision;
    update.addedReservations.emplace_back(reservation);
    fillStateLocked(update);
    return true;
}

bool AgentPendingCacheR::retain(
    const NaviSchedPendingCacheReservation &reservation,
    AgentPendingCacheToMasterData &update, bool &changed)
{
    changed = false;
    if (!PendingCacheUtil::validReservation(reservation)) {
        return false;
    }
    autil::ScopedWriteLock lock(_lock);
    NaviSchedPendingCacheReservation remaining;
    if (!_state.getReservation(reservation.batchId, remaining)) {
        return true;
    }
    if (remaining.scheduleBatchId != reservation.scheduleBatchId) {
        return false;
    }
    const auto intersect = [](auto &hashes, const auto &allowed) {
        size_t input = 0;
        size_t limit = 0;
        size_t output = 0;
        while (input < hashes.size() && limit < allowed.size()) {
            if (hashes[input] == allowed[limit]) {
                hashes[output++] = hashes[input++];
                ++limit;
            } else if (hashes[input] < allowed[limit]) {
                ++input;
            } else {
                ++limit;
            }
        }
        hashes.resize(output);
    };
    intersect(remaining.fullReadyBlockHashes, reservation.fullReadyBlockHashes);
    intersect(remaining.tailReadyBlockHashes, reservation.tailReadyBlockHashes);
    const bool retiring = remaining.fullReadyBlockHashes.empty();
    if (retiring ? !_state.erase(reservation.batchId, changed)
                 : !_state.upsert(remaining, true, changed)) {
        return false;
    }
    if (!changed) {
        return true;
    }
    ++_revision;
    if (retiring) {
        update.removedBatchIds.emplace_back(reservation.batchId);
    } else {
        update.addedReservations.emplace_back(std::move(remaining));
    }
    fillStateLocked(update);
    return true;
}

bool AgentPendingCacheR::retire(int64_t batchId,
                                AgentPendingCacheToMasterData &update,
                                bool &changed)
{
    changed = false;
    if (batchId <= 0) {
        return false;
    }
    autil::ScopedWriteLock lock(_lock);
    if (!_state.erase(batchId, changed)) {
        return false;
    }
    if (!changed) {
        return true;
    }
    ++_revision;
    update.removedBatchIds.emplace_back(batchId);
    fillStateLocked(update);
    changed = true;
    return true;
}

bool AgentPendingCacheR::getReservation(
    int64_t batchId, NaviSchedPendingCacheReservation &reservation) const
{
    autil::ScopedReadLock lock(_lock);
    return _state.getReservation(batchId, reservation);
}

int64_t AgentPendingCacheR::getMatchedBlockCount(
    const NaviSchedBlockHashes &blockHashes,
    const std::unordered_set<uint64_t> &cachedBlockHashes,
    const std::unordered_set<uint64_t> &fullReadyBlockHashes,
    const std::unordered_set<uint64_t> &tailReadyBlockHashes,
    bool hybridTailRequired, size_t matchableBlockCount) const
{
    autil::ScopedReadLock lock(_lock);
    const auto &physicalPrefixHashes =
        hybridTailRequired ? fullReadyBlockHashes : cachedBlockHashes;
    const auto limit = std::min(blockHashes.size(), matchableBlockCount);
    int64_t matchedBlockCount = 0;
    for (size_t blockIndex = 0; blockIndex < limit; ++blockIndex) {
        const auto blockHash = blockHashes[blockIndex];
        if (physicalPrefixHashes.count(blockHash) == 0 &&
            !_state.containsFullReady(blockHash)) {
            break;
        }
        ++matchedBlockCount;
    }
    if (hybridTailRequired) {
        while (matchedBlockCount > 0) {
            const auto blockHash =
                blockHashes[static_cast<size_t>(matchedBlockCount - 1)];
            if (tailReadyBlockHashes.count(blockHash) > 0 ||
                _state.containsTailReady(blockHash)) {
                break;
            }
            --matchedBlockCount;
        }
    }
    return matchedBlockCount;
}

AgentPendingCacheToMasterData AgentPendingCacheR::makeSnapshot() const
{
    AgentPendingCacheToMasterData update;
    update.fullSnapshot = true;
    autil::ScopedReadLock lock(_lock);
    update.addedReservations = _state.makeSnapshot();
    fillStateLocked(update);
    return update;
}

AgentPendingCacheR::State AgentPendingCacheR::getState() const
{
    autil::ScopedReadLock lock(_lock);
    const auto stats = _state.getStats();
    return {_revision, stats.batchCount, stats.uniqueBlockCount,
            stats.uniqueTailReadyBlockCount, stats.checksum};
}

void AgentPendingCacheR::clear()
{
    autil::ScopedWriteLock lock(_lock);
    _state.clear();
    _revision = 0;
}

void AgentPendingCacheR::fillStateLocked(
    AgentPendingCacheToMasterData &update) const
{
    const auto stats = _state.getStats();
    update.revision = _revision;
    update.batchCount = stats.batchCount;
    update.uniqueBlockCount = stats.uniqueBlockCount;
    update.uniqueTailReadyBlockCount = stats.uniqueTailReadyBlockCount;
    update.checksum = stats.checksum;
}

navi::SummaryErrorCode AgentPendingCacheR::getSummary(
    const std::string &, navi::SummaryBuilder &builder) const
{
    autil::ScopedReadLock lock(_lock);
    const auto stats = _state.getStats();
    auto table = builder.newTable("NaviSchedAgentPendingCacheOverview");
    TABLE_ADD_COLUMN_VALUE(table, "revision", _revision);
    TABLE_ADD_COLUMN_VALUE(table, "batch_count", stats.batchCount);
    TABLE_ADD_COLUMN_VALUE(table, "unique_block_count", stats.uniqueBlockCount);
    TABLE_ADD_COLUMN_VALUE(table, "unique_tail_ready_block_count",
                           stats.uniqueTailReadyBlockCount);
    TABLE_ADD_COLUMN_VALUE(table, "checksum", stats.checksum);
    return navi::SEC_OK;
}

REGISTER_RESOURCE(AgentPendingCacheR);

} // namespace navi_sched
