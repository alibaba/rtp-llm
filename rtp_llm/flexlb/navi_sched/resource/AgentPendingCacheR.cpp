#include "aios/navi_sched/resource/AgentPendingCacheR.h"

#include "aios/navi_sched/data/BlockHashUtil.h"
#include "aios/navi_sched/resource/AgentCacheR.h"
#include "navi/engine/NaviRegisterMacro.h"
#include "autil/TimeUtility.h"

#include <algorithm>
#include <stdexcept>

#include "navi/engine/NaviRegisterMacro.h"

namespace navi_sched {

const std::string AgentPendingCacheR::RESOURCE_ID = "navi_sched.agent_pending_cache_r";

AgentPendingCacheR &AgentPendingCacheR::forKeyType(
    NaviSchedCacheKeyType cacheKeyType) const
{
    if (!validCacheKeyType(cacheKeyType)) {
        throw std::invalid_argument("invalid cache key type");
    }
    if (cacheKeyType == _cacheKeyType) {
        return *const_cast<AgentPendingCacheR *>(this);
    }
    // The registered resource is BlockPool; the second instance runs exactly
    // the same cache algorithm with independent locks, revisions and policy.
    if (_cacheKeyType != NaviSchedCacheKeyType::BLOCK_POOL) {
        throw std::invalid_argument("cache type does not belong to this state");
    }
    autil::ScopedLock lock(_typedCacheLock);
    if (!_v6dCache) {
        _v6dCache = std::make_unique<AgentPendingCacheR>();
        _v6dCache->_cacheKeyType = cacheKeyType;
    }
    return *_v6dCache;
}

bool AgentPendingCacheR::hasKeyType(NaviSchedCacheKeyType cacheKeyType) const
{
    if (!validCacheKeyType(cacheKeyType)) {
        return false;
    }
    if (cacheKeyType == _cacheKeyType) {
        return true;
    }
    autil::ScopedLock lock(_typedCacheLock);
    return _v6dCache && _v6dCache->_cacheKeyType == cacheKeyType;
}

void AgentPendingCacheR::def(navi::ResourceDefBuilder &builder) const
{
    builder.name(RESOURCE_ID, navi::RS_BIZ);
}

navi::ErrorCode AgentPendingCacheR::init(navi::ResourceInitContext &)
{
    clear();
    return navi::EC_NONE;
}

bool AgentPendingCacheR::reserve(const NaviSchedPendingCacheReservation &reservation,
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

bool AgentPendingCacheR::replace(const NaviSchedPendingCacheReservation &reservation,
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
    _completed.erase(reservation.batchId);
    ++_revision;
    update.addedReservations.emplace_back(reservation);
    fillStateLocked(update);
    return true;
}

bool AgentPendingCacheR::retain(const NaviSchedPendingCacheReservation &reservation,
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
    if (retiring) {
        _completed.erase(reservation.batchId);
    } else {
        const auto completion = _completed.find(reservation.batchId);
        if (completion != _completed.end()) {
            const auto retainMissing = [](auto &missing, const auto &allowed) {
                for (auto it = missing.begin(); it != missing.end();) {
                    if (std::binary_search(allowed.begin(), allowed.end(), *it)) {
                        ++it;
                    } else {
                        it = missing.erase(it);
                    }
                }
            };
            retainMissing(completion->second.unresolvedFullReadyBlockHashes,
                          remaining.fullReadyBlockHashes);
            retainMissing(completion->second.unresolvedTailReadyBlockHashes,
                          remaining.tailReadyBlockHashes);
        }
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

bool AgentPendingCacheR::retire(int64_t batchId, AgentPendingCacheToMasterData &update,
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
    _completed.erase(batchId);
    if (!changed) {
        return true;
    }
    ++_revision;
    update.removedBatchIds.emplace_back(batchId);
    fillStateLocked(update);
    changed = true;
    return true;
}


bool AgentPendingCacheR::removePublications(int64_t batchId,
                                            const NaviSchedBlockHashes &failedFull,
                                            const NaviSchedBlockHashes &failedTail,
                                            AgentPendingCacheToMasterData &update,
                                            bool &changed, bool &retired)
{
    changed = false;
    retired = false;
    if (batchId <= 0) {
        return false;
    }
    autil::ScopedWriteLock lock(_lock);
    NaviSchedPendingCacheReservation reservation;
    if (!_state.getReservation(batchId, reservation)) {
        retired = true;
        return true;
    }
    const std::unordered_set<uint64_t> full(failedFull.begin(), failedFull.end());
    const std::unordered_set<uint64_t> tail(failedTail.begin(), failedTail.end());
    const auto oldFullSize = reservation.fullReadyBlockHashes.size();
    const auto oldTailSize = reservation.tailReadyBlockHashes.size();
    auto &requiredFull = reservation.fullReadyBlockHashes;
    auto &requiredTail = reservation.tailReadyBlockHashes;
    requiredFull.erase(std::remove_if(requiredFull.begin(), requiredFull.end(),
                                      [&](uint64_t h) { return full.count(h); }),
                       requiredFull.end());
    requiredTail.erase(
        std::remove_if(requiredTail.begin(), requiredTail.end(),
                       [&](uint64_t h) { return full.count(h) || tail.count(h); }),
        requiredTail.end());
    if (requiredFull.size() == oldFullSize && requiredTail.size() == oldTailSize) {
        return true;
    }
    auto completed = _completed.find(batchId);
    if (completed != _completed.end()) {
        for (auto h : full) {
            completed->second.unresolvedFullReadyBlockHashes.erase(h);
            completed->second.unresolvedTailReadyBlockHashes.erase(h);
        }
        for (auto h : tail) {
            completed->second.unresolvedTailReadyBlockHashes.erase(h);
        }
        retired = completed->second.unresolvedFullReadyBlockHashes.empty() &&
                  completed->second.unresolvedTailReadyBlockHashes.empty();
    }
    retired = retired || requiredFull.empty();
    if (retired) {
        if (!_state.erase(batchId, changed)) {
            return false;
        }
        _completed.erase(batchId);
        update.removedBatchIds.emplace_back(batchId);
    } else {
        if (!_state.upsert(reservation, true, changed)) {
            return false;
        }
        update.addedReservations.emplace_back(std::move(reservation));
    }
    ++_revision;
    fillStateLocked(update);
    return true;
}


bool AgentPendingCacheR::complete(int64_t batchId, bool hybridTailRequired,
                                  const NaviSchedBlockHashes &missingFullReadyBlockHashes,
                                  const NaviSchedBlockHashes &missingTailReadyBlockHashes,
                                  bool &readyToRetire)
{
    readyToRetire = false;
    if (batchId <= 0 || (!hybridTailRequired && !missingTailReadyBlockHashes.empty())) {
        return false;
    }
    autil::ScopedWriteLock lock(_lock);
    NaviSchedPendingCacheReservation reservation;
    if (!_state.getReservation(batchId, reservation)) {
        return false;
    }
    auto isSubset = [](const NaviSchedBlockHashes &all,
                       const NaviSchedBlockHashes &part) {
        return std::all_of(part.begin(), part.end(), [&all](uint64_t hash) {
            return std::binary_search(all.begin(), all.end(), hash);
        });
    };
    if (!isSubset(reservation.fullReadyBlockHashes, missingFullReadyBlockHashes) ||
        !isSubset(reservation.tailReadyBlockHashes, missingTailReadyBlockHashes)) {
        return false;
    }
    if (missingFullReadyBlockHashes.empty() && missingTailReadyBlockHashes.empty()) {
        readyToRetire = true;
        return true;
    }
    CompletionState completion;
    completion.completedTimeUs = autil::TimeUtility::monotonicTimeUs();
    completion.hybridTailRequired = hybridTailRequired;
    completion.unresolvedFullReadyBlockHashes.insert(missingFullReadyBlockHashes.begin(),
                                                     missingFullReadyBlockHashes.end());
    completion.unresolvedTailReadyBlockHashes.insert(missingTailReadyBlockHashes.begin(),
                                                     missingTailReadyBlockHashes.end());
    auto it = _completed.find(batchId);
    if (it == _completed.end()) {
        _completed.emplace(batchId, std::move(completion));
        return true;
    }
    return it->second.hybridTailRequired == hybridTailRequired &&
           it->second.unresolvedFullReadyBlockHashes ==
               completion.unresolvedFullReadyBlockHashes &&
           it->second.unresolvedTailReadyBlockHashes ==
               completion.unresolvedTailReadyBlockHashes;
}


std::vector<int64_t> AgentPendingCacheR::resolveCompleted(
    const NaviSchedCacheUpdate &cacheUpdate)
{
    if (!validCacheKeyType(cacheUpdate.cacheKeyType)) {
        return {};
    }
    if (cacheUpdate.cacheKeyType != _cacheKeyType) {
        return forKeyType(cacheUpdate.cacheKeyType).resolveCompleted(cacheUpdate);
    }

    std::vector<int64_t> resolvedBatchIds;
    autil::ScopedWriteLock lock(_lock);
    for (auto &[batchId, completion] : _completed) {
        const auto &addedFull = completion.hybridTailRequired
                                    ? cacheUpdate.addedFullReadyBlockHashes
                                    : cacheUpdate.addedBlockHashes;
        const auto &removedFull = completion.hybridTailRequired
                                      ? cacheUpdate.removedFullReadyBlockHashes
                                      : cacheUpdate.removedBlockHashes;
        for (auto hash : BlockHashUtil::get(addedFull)) {
            completion.unresolvedFullReadyBlockHashes.erase(hash);
        }
        for (auto hash : BlockHashUtil::get(removedFull)) {
            completion.unresolvedFullReadyBlockHashes.erase(hash);
        }
        if (completion.hybridTailRequired) {
            for (auto hash : BlockHashUtil::get(cacheUpdate.addedTailReadyBlockHashes)) {
                completion.unresolvedTailReadyBlockHashes.erase(hash);
            }
            for (auto hash :
                 BlockHashUtil::get(cacheUpdate.removedTailReadyBlockHashes)) {
                completion.unresolvedTailReadyBlockHashes.erase(hash);
            }
        }
        if (completion.unresolvedFullReadyBlockHashes.empty() &&
            completion.unresolvedTailReadyBlockHashes.empty()) {
            resolvedBatchIds.emplace_back(batchId);
        }
    }
    std::sort(resolvedBatchIds.begin(), resolvedBatchIds.end());
    return resolvedBatchIds;
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
    const std::unordered_set<uint64_t> &tailReadyBlockHashes, bool hybridTailRequired,
    size_t matchableBlockCount, int64_t fullAttentionQueryDropBlockCount,
    size_t alignmentBlocks, int64_t *effectiveFullMatchedBlockCount) const
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
    matchedBlockCount =
        std::max<int64_t>(0, matchedBlockCount - fullAttentionQueryDropBlockCount);
    alignmentBlocks = std::max<size_t>(1, alignmentBlocks);
    matchedBlockCount -= matchedBlockCount % alignmentBlocks;
    if (effectiveFullMatchedBlockCount) {
        *effectiveFullMatchedBlockCount = matchedBlockCount;
    }
    if (hybridTailRequired) {
        while (matchedBlockCount > 0) {
            const auto blockHash =
                blockHashes[static_cast<size_t>(matchedBlockCount - 1)];
            if (tailReadyBlockHashes.count(blockHash) > 0 ||
                _state.containsTailReady(blockHash)) {
                break;
            }
            matchedBlockCount -= alignmentBlocks;
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
    State result{_revision, stats.batchCount, stats.uniqueBlockCount,
                 stats.uniqueTailReadyBlockCount, stats.checksum};
    result.completedBatchCount = _completed.size();
    const auto nowUs = autil::TimeUtility::monotonicTimeUs();
    for (const auto &[batchId, completion] : _completed) {
        result.oldestCompletedAgeMs = std::max(
            result.oldestCompletedAgeMs,
            (nowUs - completion.completedTimeUs) / 1000);
    }
    return result;
}

void AgentPendingCacheR::clear()
{
    autil::ScopedWriteLock lock(_lock);
    _state.clear();
    _completed.clear();
    _revision = 0;
}

void AgentPendingCacheR::fillStateLocked(AgentPendingCacheToMasterData &update) const
{
    const auto stats = _state.getStats();
    update.cacheKeyType = _cacheKeyType;
    update.revision = _revision;
    update.batchCount = stats.batchCount;
    update.uniqueBlockCount = stats.uniqueBlockCount;
    update.uniqueTailReadyBlockCount = stats.uniqueTailReadyBlockCount;
    update.checksum = stats.checksum;
}

navi::SummaryErrorCode AgentPendingCacheR::getSummary(const std::string &,
                                                      navi::SummaryBuilder &builder) const
{
    if (_cacheKeyType == NaviSchedCacheKeyType::BLOCK_POOL &&
        hasKeyType(NaviSchedCacheKeyType::V6D)) {
        forKeyType(NaviSchedCacheKeyType::V6D).getSummary("", builder);
    }

    autil::ScopedReadLock lock(_lock);
    const auto stats = _state.getStats();
    auto table = builder.newTable(_cacheKeyType == NaviSchedCacheKeyType::BLOCK_POOL
                                      ? "NaviSchedAgentPendingCacheOverview"
                                      : "NaviSchedAgentPendingCacheOverview_V6D");
    TABLE_ADD_COLUMN_VALUE(table, "cache_key_type", static_cast<int>(_cacheKeyType));
    TABLE_ADD_COLUMN_VALUE(table, "revision", _revision);
    TABLE_ADD_COLUMN_VALUE(table, "batch_count", stats.batchCount);
    TABLE_ADD_COLUMN_VALUE(table, "unique_block_count", stats.uniqueBlockCount);
    TABLE_ADD_COLUMN_VALUE(table, "unique_tail_ready_block_count",
                           stats.uniqueTailReadyBlockCount);
    TABLE_ADD_COLUMN_VALUE(table, "checksum", stats.checksum);
    return navi::SEC_OK;
}

REGISTER_RESOURCE(AgentPendingCacheR);

}
