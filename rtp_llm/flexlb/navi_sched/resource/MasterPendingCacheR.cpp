#include "rtp_llm/flexlb/navi_sched/resource/MasterPendingCacheR.h"

#include "rtp_llm/flexlb/navi_sched/data/BlockHashUtil.h"

#include <algorithm>
#include <atomic>
#include "autil/TimeUtility.h"
#include "autil/legacy/jsonizable.h"

#include "navi/engine/NaviRegisterMacro.h"

namespace navi_sched {

const std::string MasterPendingCacheR::RESOURCE_ID =
    "navi_sched.master_pending_cache_r";

void MasterPendingCacheR::def(navi::ResourceDefBuilder &builder) const
{
    builder.name(RESOURCE_ID, navi::RS_BIZ);
}

navi::ErrorCode MasterPendingCacheR::init(navi::ResourceInitContext &ctx)
{
    clearAll();
    return navi::EC_NONE;
}

MasterPendingCacheR::NodeCachePtr
MasterPendingCacheR::getNodeCache(uint64_t agentIdHash) const
{
    autil::ScopedReadLock lock(_nodeLock);
    auto it = _nodeCaches.find(agentIdHash);
    return _nodeCaches.end() == it ? nullptr : it->second;
}

MasterPendingCacheR::NodeCachePtr
MasterPendingCacheR::getOrCreateNodeCache(uint64_t agentIdHash)
{
    {
        autil::ScopedReadLock lock(_nodeLock);
        auto it = _nodeCaches.find(agentIdHash);
        if (_nodeCaches.end() != it) {
            return it->second;
        }
    }
    autil::ScopedWriteLock lock(_nodeLock);
    auto &cache = _nodeCaches[agentIdHash];
    if (!cache) {
        cache = std::make_shared<NodeCache>();
    }
    return cache;
}

bool MasterPendingCacheR::reserve(uint64_t agentIdHash, int64_t batchId,
                                  int64_t scheduleTimeUs,
                                  const std::vector<ReferencedRequest> &requests,
                                  const NaviSchedCachePolicy &policy)
{
    if (agentIdHash == 0 || batchId <= 0 || scheduleTimeUs <= 0 ||
        requests.empty() || !policy.valid()) {
        return false;
    }
    NaviSchedPendingCacheReservation reservation;
    if (!PendingCacheUtil::makeReservation(batchId, requests, policy,
                                           reservation)) {
        return false;
    }

    auto cache = getOrCreateNodeCache(agentIdHash);
    autil::ScopedWriteLock lock(cache->lock);
    bool changed = false;
    const bool valid = cache->scheduled.upsertGenerated(
        reservation, false, changed, &cache->agent, &cache->overlap);
    if (changed) {
        std::atomic_store(&cache->cachedStats, std::shared_ptr<const Stats>());
    }
    return valid && changed;
}

bool MasterPendingCacheR::updateScheduled(
    uint64_t agentIdHash,
    const NaviSchedPendingCacheReservation &reservation)
{
    if (agentIdHash == 0 || reservation.batchId <= 0 ||
        reservation.batchId != reservation.scheduleBatchId) {
        return false;
    }
    const auto beginUs = autil::TimeUtility::monotonicTimeUs();
    const auto beginCpuUs = autil::TimeUtility::cpuThreadTimeUs();
    auto cache = getOrCreateNodeCache(agentIdHash);
    const auto lockBeginUs = autil::TimeUtility::monotonicTimeUs();
    int64_t acquiredUs = 0;
    int64_t mutatedUs = 0;
    int64_t mutationCpuUs = 0;
    bool changed = false;
    bool valid = false;
    {
        autil::ScopedWriteLock lock(cache->lock);
        acquiredUs = autil::TimeUtility::monotonicTimeUs();
        const auto acquiredCpuUs = autil::TimeUtility::cpuThreadTimeUs();
        valid = cache->scheduled.upsertGenerated(
            reservation, true, changed, &cache->agent, &cache->overlap);
        if (changed) {
            std::atomic_store(&cache->cachedStats, std::shared_ptr<const Stats>());
        }
        mutatedUs = autil::TimeUtility::monotonicTimeUs();
        mutationCpuUs = autil::TimeUtility::cpuThreadTimeUs() - acquiredCpuUs;
    }
    const auto endUs = autil::TimeUtility::monotonicTimeUs();
    if (endUs - beginUs >= 50'000) {
        NAVI_KERNEL_LOG(INFO,
            "[navi-master-pending-mutation-timing] function=updateScheduled agent_hash=%lu "
            "batch_id=%ld full_blocks=%lu tail_blocks=%lu lookup_us=%ld wait_us=%ld "
            "mutation_us=%ld unlock_us=%ld total_us=%ld "
            "mutation_cpu_us=%ld total_cpu_us=%ld",
            agentIdHash, reservation.batchId, reservation.fullReadyBlockHashes.size(),
            reservation.tailReadyBlockHashes.size(), lockBeginUs - beginUs,
            acquiredUs - lockBeginUs, mutatedUs - acquiredUs, endUs - mutatedUs, endUs - beginUs,
            mutationCpuUs, autil::TimeUtility::cpuThreadTimeUs() - beginCpuUs);
    }
    return valid;
}

bool MasterPendingCacheR::removeScheduledBlocks(
    uint64_t agentIdHash, int64_t batchId,
    const std::vector<uint64_t> &fullReadyBlockHashes,
    const std::vector<uint64_t> &tailReadyBlockHashes)
{
    if (agentIdHash == 0 || batchId <= 0) {
        return false;
    }
    const auto beginUs = autil::TimeUtility::monotonicTimeUs();
    const auto beginCpuUs = autil::TimeUtility::cpuThreadTimeUs();
    auto cache = getNodeCache(agentIdHash);
    if (!cache) {
        return false;
    }
    const auto lockBeginUs = autil::TimeUtility::monotonicTimeUs();
    int64_t acquiredUs = 0;
    int64_t mutatedUs = 0;
    int64_t mutationCpuUs = 0;
    bool valid = false;
    {
        autil::ScopedWriteLock lock(cache->lock);
        acquiredUs = autil::TimeUtility::monotonicTimeUs();
        const auto acquiredCpuUs = autil::TimeUtility::cpuThreadTimeUs();
        valid = cache->scheduled.removeGeneratedBlocks(
            batchId, fullReadyBlockHashes, tailReadyBlockHashes,
            &cache->agent, &cache->overlap);
        if (valid) {
            std::atomic_store(&cache->cachedStats, std::shared_ptr<const Stats>());
        }
        mutatedUs = autil::TimeUtility::monotonicTimeUs();
        mutationCpuUs = autil::TimeUtility::cpuThreadTimeUs() - acquiredCpuUs;
    }
    const auto endUs = autil::TimeUtility::monotonicTimeUs();
    if (endUs - beginUs >= 50'000) {
        NAVI_KERNEL_LOG(INFO,
            "[navi-master-pending-mutation-timing] function=removeScheduledBlocks agent_hash=%lu "
            "batch_id=%ld full_blocks=%lu tail_blocks=%lu lookup_us=%ld wait_us=%ld "
            "mutation_us=%ld unlock_us=%ld total_us=%ld "
            "mutation_cpu_us=%ld total_cpu_us=%ld",
            agentIdHash, batchId, fullReadyBlockHashes.size(), tailReadyBlockHashes.size(),
            lockBeginUs - beginUs, acquiredUs - lockBeginUs, mutatedUs - acquiredUs,
            endUs - mutatedUs, endUs - beginUs,
            mutationCpuUs, autil::TimeUtility::cpuThreadTimeUs() - beginCpuUs);
    }
    return valid;
}

bool MasterPendingCacheR::retire(uint64_t agentIdHash, int64_t batchId)
{
    if (agentIdHash == 0 || batchId <= 0) {
        return false;
    }
    auto cache = getNodeCache(agentIdHash);
    if (!cache) {
        return false;
    }
    autil::ScopedWriteLock lock(cache->lock);
    bool changed = false;
    const bool valid = cache->scheduled.erase(
        batchId, changed, &cache->agent, &cache->overlap);
    if (changed) {
        std::atomic_store(&cache->cachedStats, std::shared_ptr<const Stats>());
    }
    return valid && changed;
}

MasterPendingCacheR::ApplyResult MasterPendingCacheR::applyAgentUpdate(
    uint64_t agentIdHash, const AgentPendingCacheToMasterData &update)
{
    ApplyResult result;
    result.fullSnapshot = update.fullSnapshot;
    result.receivedRevision = update.revision;
    result.receivedChecksum = update.checksum;
    result.receivedBatchCount = update.batchCount;
    result.receivedBlockCount = update.uniqueBlockCount;
    result.receivedTailBlockCount = update.uniqueTailReadyBlockCount;
    if (agentIdHash == 0) {
        result.reason = "invalid_agent";
        result.stateMismatch = true;
        return result;
    }
    auto cache = getOrCreateNodeCache(agentIdHash);
    std::unique_ptr<PendingCacheState> preparedSnapshot;
    bool preparedValid = false;
    auto prepareSnapshot = [&]() {
        preparedSnapshot = std::make_unique<PendingCacheState>();
        preparedValid = update.removedBatchIds.empty() &&
                        preparedSnapshot->replace(update.addedReservations);
    };
    if (update.fullSnapshot && update.batchCount >= 0 &&
        update.uniqueBlockCount >= 0 && update.uniqueTailReadyBlockCount >= 0) {
        bool stale;
        {
            autil::ScopedReadLock readLock(cache->lock);
            stale = cache->agentSnapshotReady &&
                    update.revision < cache->agentRevision;
        }
        if (!stale) {
            prepareSnapshot();
        }
    }
    autil::ScopedWriteLock lock(cache->lock);
    std::atomic_store(&cache->cachedStats, std::shared_ptr<const Stats>());
    const auto before = cache->agent.getStats();
    result.snapshotReadyBefore = cache->agentSnapshotReady;
    result.masterRevisionBefore = cache->agentRevision;
    result.expectedRevision = update.fullSnapshot ? update.revision
                                                  : cache->agentRevision + 1;
    result.masterChecksumBefore = before.checksum;
    result.masterBatchCountBefore = static_cast<int64_t>(before.batchCount);
    result.masterBlockCountBefore =
        static_cast<int64_t>(before.uniqueBlockCount);
    result.masterTailBlockCountBefore =
        static_cast<int64_t>(before.uniqueTailReadyBlockCount);
    auto captureCandidateState = [&](const PendingCacheState &state) {
        const auto candidate = state.getStats();
        result.candidateStateComputed = true;
        result.candidateChecksum = candidate.checksum;
        result.candidateBatchCount =
            static_cast<int64_t>(candidate.batchCount);
        result.candidateBlockCount =
            static_cast<int64_t>(candidate.uniqueBlockCount);
        result.candidateTailBlockCount =
            static_cast<int64_t>(candidate.uniqueTailReadyBlockCount);
    };
    auto failState = [&](const char *reason) {
        cache->agent.clear();
        cache->overlap = {};
        cache->agentSnapshotReady = false;
        ++cache->agentStateMismatchCount;
        result.reason = reason;
        result.stateMismatch = true;
    };
    auto failRevision = [&](const char *reason) {
        cache->agent.clear();
        cache->overlap = {};
        cache->agentSnapshotReady = false;
        ++cache->agentRevisionGapCount;
        result.reason = reason;
        result.revisionGap = true;
    };
    if (update.fullSnapshot) {
        if (cache->agentSnapshotReady &&
            update.revision < cache->agentRevision) {
            result.applied = true;
            result.staleSnapshotIgnored = true;
            result.reason = "stale_full_snapshot_ignored";
            result.stats = getStatsLocked(*cache);
            return result;
        }
    }
    if (update.batchCount < 0 || update.uniqueBlockCount < 0 ||
        update.uniqueTailReadyBlockCount < 0) {
        failState("invalid_metadata");
        result.stats = getStatsLocked(*cache);
        return result;
    }
    if (update.fullSnapshot) {
        // A concurrent reset can make a previously stale snapshot eligible.
        if (!preparedSnapshot) {
            prepareSnapshot();
        }
        auto &candidate = *preparedSnapshot;
        if (!preparedValid) {
            failState("invalid_full_snapshot");
        } else {
            captureCandidateState(candidate);
            const auto candidateStats = candidate.getStats();
            const bool candidateValid =
                update.batchCount ==
                    static_cast<int64_t>(candidateStats.batchCount) &&
                update.uniqueBlockCount ==
                    static_cast<int64_t>(candidateStats.uniqueBlockCount) &&
                update.uniqueTailReadyBlockCount ==
                    static_cast<int64_t>(
                        candidateStats.uniqueTailReadyBlockCount) &&
                update.checksum == candidateStats.checksum;
            if (!candidateValid) {
                failState("state_mismatch");
            } else if (cache->agentSnapshotReady &&
                       update.revision == cache->agentRevision &&
                       !candidate.sameReservations(cache->agent)) {
                failState("same_revision_state_mismatch");
            } else {
                // Retire the old index after releasing the node write lock.
                std::swap(cache->agent, candidate);
                cache->overlap.batchCount =
                    cache->scheduled.countBatchOverlap(cache->agent);
                cache->overlap.uniqueBlockCount =
                    cache->scheduled.countFullReadyOverlap(cache->agent);
                cache->overlap.uniqueTailReadyBlockCount =
                    cache->scheduled.countTailReadyOverlap(cache->agent);
                cache->agentRevision = update.revision;
                cache->agentSnapshotReady = true;
                result.applied = true;
            }
        }
    } else {
        const auto operationCount = update.addedReservations.size() +
                                    update.removedBatchIds.size();
        if (!cache->agentSnapshotReady) {
            failRevision("snapshot_not_ready");
        } else if (operationCount != 1) {
            failRevision("invalid_delta_operation_count");
        } else if (update.revision != cache->agentRevision + 1) {
            failRevision("revision_gap");
        } else {
            bool changed = false;
            bool valid = false;
            if (!update.addedReservations.empty()) {
                valid = cache->agent.upsert(update.addedReservations.front(), true,
                                            changed, &cache->scheduled,
                                            &cache->overlap);
            } else {
                valid = cache->agent.erase(update.removedBatchIds.front(), changed,
                                           &cache->scheduled, &cache->overlap);
            }
            cache->agentRevision = update.revision;
            if (!valid) {
                failState("invalid_delta");
            } else {
                captureCandidateState(cache->agent);
                if (!validateAgentStateLocked(*cache, update)) {
                    failState("state_mismatch");
                } else {
                    result.applied = true;
                }
            }
        }
    }
    result.stats = getStatsLocked(*cache);
    return result;
}

void MasterPendingCacheR::clear(uint64_t agentIdHash)
{
    auto cache = getNodeCache(agentIdHash);
    if (!cache) {
        return;
    }
    autil::ScopedWriteLock lock(cache->lock);
    cache->scheduled.clear();
    cache->agent.clear();
    cache->overlap = {};
    cache->agentRevision = 0;
    cache->agentSnapshotReady = false;
    std::atomic_store(&cache->cachedStats, std::shared_ptr<const Stats>());
}

void MasterPendingCacheR::remove(uint64_t agentIdHash)
{
    clear(agentIdHash);
    autil::ScopedWriteLock lock(_nodeLock);
    _nodeCaches.erase(agentIdHash);
}

void MasterPendingCacheR::clearAll()
{
    autil::ScopedWriteLock lock(_nodeLock);
    _nodeCaches.clear();
}

int64_t MasterPendingCacheR::getMatchedBlockCount(
    uint64_t agentIdHash, const NaviSchedBlockHashes &blockHashes,
    const PhysicalBlockSet &cachedBlockHashes,
    const PhysicalBlockSet &fullReadyBlockHashes,
    const PhysicalBlockSet &tailReadyBlockHashes,
    bool hybridTailRequired, size_t matchableBlockCount) const
{
    auto cache = getNodeCache(agentIdHash);
    if (!cache) {
        return getMatchedBlockCountLocked(
            nullptr, blockHashes, cachedBlockHashes, fullReadyBlockHashes,
            tailReadyBlockHashes, hybridTailRequired, matchableBlockCount);
    }
    autil::ScopedReadLock lock(cache->lock);
    return getMatchedBlockCountLocked(
        cache.get(), blockHashes, cachedBlockHashes, fullReadyBlockHashes,
        tailReadyBlockHashes, hybridTailRequired, matchableBlockCount);
}

int64_t MasterPendingCacheR::getMatchedBlockCountExcludingSchedule(
    uint64_t agentIdHash, int64_t excludedScheduleBatchId,
    const NaviSchedBlockHashes &blockHashes,
    const PhysicalBlockSet &cachedBlockHashes,
    const PhysicalBlockSet &fullReadyBlockHashes,
    const PhysicalBlockSet &tailReadyBlockHashes,
    bool hybridTailRequired, size_t matchableBlockCount) const
{
    auto cache = getNodeCache(agentIdHash);
    if (!cache) {
        return getMatchedBlockCountLocked(
            nullptr, blockHashes, cachedBlockHashes, fullReadyBlockHashes,
            tailReadyBlockHashes, hybridTailRequired, matchableBlockCount,
            excludedScheduleBatchId);
    }
    autil::ScopedReadLock lock(cache->lock);
    return getMatchedBlockCountLocked(
        cache.get(), blockHashes, cachedBlockHashes, fullReadyBlockHashes,
        tailReadyBlockHashes, hybridTailRequired, matchableBlockCount,
        excludedScheduleBatchId);
}

MasterPendingCacheR::MatchBreakdown MasterPendingCacheR::getMatchBreakdown(
    uint64_t agentIdHash, const NaviSchedBlockHashes &blockHashes,
    const PhysicalBlockSet &cachedBlockHashes,
    const PhysicalBlockSet &fullReadyBlockHashes,
    const PhysicalBlockSet &tailReadyBlockHashes,
    bool hybridTailRequired, size_t matchableBlockCount) const
{
    MatchBreakdown breakdown;
    breakdown.staticMatchedBlockCount = getMatchedBlockCountLocked(
        nullptr, blockHashes, cachedBlockHashes, fullReadyBlockHashes,
        tailReadyBlockHashes, hybridTailRequired, matchableBlockCount);
    auto cache = getNodeCache(agentIdHash);
    if (!cache) {
        breakdown.matchedBlockCount = breakdown.staticMatchedBlockCount;
        return breakdown;
    }
    autil::ScopedReadLock lock(cache->lock);
    breakdown.matchedBlockCount = getMatchedBlockCountLocked(
        cache.get(), blockHashes, cachedBlockHashes, fullReadyBlockHashes,
        tailReadyBlockHashes, hybridTailRequired, matchableBlockCount);
    if (breakdown.matchedBlockCount > breakdown.staticMatchedBlockCount) {
        const auto firstPendingBlockIndex = static_cast<size_t>(
            breakdown.staticMatchedBlockCount);
        breakdown.firstPendingBlockSourceBatchId = getPendingSourceBatchIdLocked(
            *cache, blockHashes[firstPendingBlockIndex], false);
    }
    if (hybridTailRequired && breakdown.matchedBlockCount > 0) {
        const auto tailBlockIndex = static_cast<size_t>(
            breakdown.matchedBlockCount - 1);
        if (tailReadyBlockHashes.end() ==
            tailReadyBlockHashes.find(blockHashes[tailBlockIndex])) {
            breakdown.pendingTailSourceBatchId = getPendingSourceBatchIdLocked(
                *cache, blockHashes[tailBlockIndex], true);
        }
    }
    return breakdown;
}

int64_t MasterPendingCacheR::getPendingSourceBatchIdLocked(
    const NodeCache &cache, uint64_t blockHash, bool tailReady)
{
    int64_t sourceBatchId = 0;
    const auto scheduled = cache.scheduled.getSourceBatchId(blockHash, tailReady);
    const auto agent = cache.agent.getSourceBatchId(blockHash, tailReady);
    if (scheduled > 0) {
        sourceBatchId = scheduled;
    }
    if (agent > 0 && (sourceBatchId == 0 || agent < sourceBatchId)) {
        sourceBatchId = agent;
    }
    return sourceBatchId;
}

uint64_t MasterPendingCacheR::getMatchedBlockCounts(
    uint64_t agentIdHash,
    const std::vector<NaviSchedScheduleItem> &requests,
    int64_t effectiveBlockSize,
    const PhysicalBlockSet &cachedBlockHashes,
    const PhysicalBlockSet &fullReadyBlockHashes,
    const PhysicalBlockSet &tailReadyBlockHashes,
    bool hybridTailRequired, int64_t *matchedBlockCounts,
    BatchMatchTiming *timing, const std::vector<size_t> *commonPrefixes) const
{
    if (timing) {
        *timing = {};
    }
    if (!matchedBlockCounts) {
        return requests.size();
    }
    const auto lookupBeginUs = timing ? autil::TimeUtility::monotonicTimeUs() : 0;
    auto cache = getNodeCache(agentIdHash);
    const auto lookupEndUs = timing ? autil::TimeUtility::monotonicTimeUs() : 0;
    if (timing) {
        timing->nodeLookupUs = lookupEndUs - lookupBeginUs;
    }
    auto matchRequests = [&](const NodeCache *nodeCache) {
        uint64_t mismatchCount = 0;
        const NaviSchedBlockHashes *previousHashes = nullptr;
        size_t previousFullMatchCount = 0;
        for (size_t requestIndex = 0; requestIndex < requests.size();
             ++requestIndex) {
            const auto &request = requests[requestIndex].request;
            if (effectiveBlockSize <= 0 ||
                request.blockSize != effectiveBlockSize) {
                matchedBlockCounts[requestIndex] = 0;
                ++mismatchCount;
                previousHashes = nullptr;
                previousFullMatchCount = 0;
                continue;
            }
            static const NaviSchedBlockHashes empty;
            const auto &blockHashes =
                request.blockHashes ? *request.blockHashes : empty;
            const auto matchableBlockCount = BlockHashUtil::getMatchableBlockCount(
                request.promptTokenCount, request.blockSize, request.blockHashes);
            size_t initialMatchCount = 0;
            if (previousHashes) {
                const auto sharedLimit = std::min(
                    previousFullMatchCount, std::min(blockHashes.size(), matchableBlockCount));
                // The caller builds these immutable lengths for this exact
                // request order before starting parallel node queries. Cache
                // readiness is still established under this node's read lock.
                if (commonPrefixes && commonPrefixes->size() == requests.size()) {
                    initialMatchCount = std::min(sharedLimit, (*commonPrefixes)[requestIndex]);
                } else if (previousHashes == &blockHashes) {
                    initialMatchCount = sharedLimit;
                } else {
                    while (initialMatchCount < sharedLimit &&
                           (*previousHashes)[initialMatchCount] == blockHashes[initialMatchCount]) {
                        ++initialMatchCount;
                    }
                    if (timing) {
                        timing->prefixComparisonCount += initialMatchCount +
                            (initialMatchCount < sharedLimit ? 1 : 0);
                    }
                }
            }
            // Reuse only within this read-locked batch. Tail readiness is checked
            // separately for every request, even when its full prefix is shared.
            matchedBlockCounts[requestIndex] = getMatchedBlockCountLocked(
                nodeCache, blockHashes, cachedBlockHashes,
                fullReadyBlockHashes, tailReadyBlockHashes,
                hybridTailRequired,
                matchableBlockCount, 0, initialMatchCount, &previousFullMatchCount);
            if (timing) {
                timing->reusedFullPrefixBlockCount += initialMatchCount;
                const auto limit = std::min(blockHashes.size(), matchableBlockCount);
                timing->fullPrefixProbeCount += previousFullMatchCount - initialMatchCount +
                    (previousFullMatchCount < limit ? 1 : 0);
                if (hybridTailRequired) {
                    const auto matched = static_cast<size_t>(matchedBlockCounts[requestIndex]);
                    timing->tailProbeCount += previousFullMatchCount - matched + (matched > 0 ? 1 : 0);
                }
            }
            previousHashes = &blockHashes;
        }
        return mismatchCount;
    };
    if (!cache) {
        const auto result = matchRequests(nullptr);
        if (timing) {
            timing->matchUs = autil::TimeUtility::monotonicTimeUs() - lookupEndUs;
        }
        return result;
    }
    autil::ScopedReadLock lock(cache->lock);
    const auto lockAcquiredUs = timing ? autil::TimeUtility::monotonicTimeUs() : 0;
    const auto result = matchRequests(cache.get());
    if (timing) {
        timing->lockWaitUs = lockAcquiredUs - lookupEndUs;
        timing->matchUs = autil::TimeUtility::monotonicTimeUs() - lockAcquiredUs;
        const auto scheduled = cache->scheduled.getStats();
        const auto agent = cache->agent.getStats();
        timing->scheduledBatches = scheduled.batchCount;
        timing->agentBatches = agent.batchCount;
        timing->scheduledBlocks = scheduled.uniqueBlockCount;
        timing->agentBlocks = agent.uniqueBlockCount;
    }
    return result;
}

int64_t MasterPendingCacheR::getMatchedBlockCountLocked(
    const NodeCache *cache, const NaviSchedBlockHashes &blockHashes,
    const PhysicalBlockSet &cachedBlockHashes,
    const PhysicalBlockSet &fullReadyBlockHashes,
    const PhysicalBlockSet &tailReadyBlockHashes,
    bool hybridTailRequired, size_t matchableBlockCount,
    int64_t excludedScheduleBatchId, size_t initialMatchedBlockCount,
    size_t *fullMatchedBlockCount)
{
    auto pendingFullContains = [cache, excludedScheduleBatchId](uint64_t blockHash) {
        if (!cache) {
            return false;
        }
        if (excludedScheduleBatchId <= 0) {
            return cache->scheduled.containsFullReady(blockHash) ||
                   cache->agent.containsFullReady(blockHash);
        }
        return cache->scheduled.containsFullReadyExcludingSchedule(
                   blockHash, excludedScheduleBatchId) ||
               cache->agent.containsFullReadyExcludingSchedule(
                   blockHash, excludedScheduleBatchId);
    };
    auto pendingTailContains = [cache, excludedScheduleBatchId](uint64_t blockHash) {
        if (!cache) {
            return false;
        }
        if (excludedScheduleBatchId <= 0) {
            return cache->scheduled.containsTailReady(blockHash) ||
                   cache->agent.containsTailReady(blockHash);
        }
        return cache->scheduled.containsTailReadyExcludingSchedule(
                   blockHash, excludedScheduleBatchId) ||
               cache->agent.containsTailReadyExcludingSchedule(
                   blockHash, excludedScheduleBatchId);
    };
    const auto &prefixHashes =
        hybridTailRequired ? fullReadyBlockHashes : cachedBlockHashes;
    const auto limit = std::min(blockHashes.size(), matchableBlockCount);
    int64_t matchedBlockCount = std::min(initialMatchedBlockCount, limit);
    for (size_t blockIndex = matchedBlockCount; blockIndex < limit; ++blockIndex) {
        const auto blockHash = blockHashes[blockIndex];
        if (prefixHashes.end() == prefixHashes.find(blockHash) &&
            !pendingFullContains(blockHash)) {
            break;
        }
        ++matchedBlockCount;
    }
    if (fullMatchedBlockCount) {
        *fullMatchedBlockCount = matchedBlockCount;
    }
    if (hybridTailRequired) {
        while (matchedBlockCount > 0) {
            auto blockHash =
                blockHashes[static_cast<size_t>(matchedBlockCount - 1)];
            if (tailReadyBlockHashes.end() != tailReadyBlockHashes.find(blockHash) ||
                pendingTailContains(blockHash)) {
                break;
            }
            --matchedBlockCount;
        }
    }
    return matchedBlockCount;
}

MasterPendingCacheR::Stats
MasterPendingCacheR::getStats(uint64_t agentIdHash) const
{
    auto cache = getNodeCache(agentIdHash);
    if (!cache) {
        return {};
    }
    autil::ScopedReadLock lock(cache->lock);
    return getStatsLocked(*cache);
}

MasterPendingCacheR::ReplaySnapshot
MasterPendingCacheR::captureReplaySnapshot(uint64_t agentIdHash) const
{
    ReplaySnapshot snapshot;
    auto cache = getNodeCache(agentIdHash);
    if (!cache) return snapshot;
    autil::ScopedReadLock lock(cache->lock);
    snapshot.present = true;
    snapshot.scheduled = cache->scheduled.makeSnapshot();
    snapshot.agent = cache->agent.makeSnapshot();
    snapshot.scheduledStats = cache->scheduled.getStats();
    snapshot.agentStats = cache->agent.getStats();
    snapshot.agentRevision = cache->agentRevision;
    snapshot.agentSnapshotReady = cache->agentSnapshotReady;
    return snapshot;
}

autil::legacy::json::JsonMap MasterPendingCacheR::makeReplayState(uint64_t agentIdHash) const
{
    return captureReplaySnapshot(agentIdHash).toJson();
}

autil::legacy::json::JsonMap MasterPendingCacheR::ReplaySnapshot::toJson() const
{
    using autil::legacy::ToJson;
    autil::legacy::json::JsonMap result;
    if (!present) return result;
    auto encode = [](const std::vector<NaviSchedPendingCacheReservation> &snapshot) {
        std::vector<autil::legacy::json::JsonMap> records;
        records.reserve(snapshot.size());
        for (const auto &reservation : snapshot) {
            autil::legacy::json::JsonMap record;
            record["batch_id"] = ToJson(reservation.batchId);
            record["schedule_batch_id"] = ToJson(reservation.scheduleBatchId);
            record["full_ready_block_hashes"] = ToJson(reservation.fullReadyBlockHashes);
            record["tail_ready_block_hashes"] = ToJson(reservation.tailReadyBlockHashes);
            records.emplace_back(std::move(record));
        }
        return ToJson(records);
    };
    result["scheduled"] = encode(scheduled);
    result["agent"] = encode(agent);
    result["agent_revision"] = ToJson(agentRevision);
    result["agent_snapshot_ready"] = ToJson(agentSnapshotReady);
    result["scheduled_checksum"] = ToJson(scheduledStats.checksum);
    result["agent_checksum"] = ToJson(agentStats.checksum);
    result["scheduled_unique_blocks"] = ToJson(scheduledStats.uniqueBlockCount);
    result["scheduled_unique_tail_blocks"] = ToJson(scheduledStats.uniqueTailReadyBlockCount);
    result["agent_unique_blocks"] = ToJson(agentStats.uniqueBlockCount);
    result["agent_unique_tail_blocks"] = ToJson(agentStats.uniqueTailReadyBlockCount);
    return result;
}

bool MasterPendingCacheR::restoreReplayState(
    uint64_t agentIdHash, const autil::legacy::json::JsonMap &state)
{
    using autil::legacy::FromJson;
    auto candidate = std::make_shared<NodeCache>();
    try {
        auto decode = [](const autil::legacy::Any &value, PendingCacheState &output) {
            std::vector<autil::legacy::json::JsonMap> records;
            FromJson(records, value);
            std::vector<NaviSchedPendingCacheReservation> reservations;
            for (const auto &record : records) {
                NaviSchedPendingCacheReservation reservation;
                FromJson(reservation.batchId, record.at("batch_id"));
                FromJson(reservation.scheduleBatchId, record.at("schedule_batch_id"));
                FromJson(reservation.fullReadyBlockHashes, record.at("full_ready_block_hashes"));
                FromJson(reservation.tailReadyBlockHashes, record.at("tail_ready_block_hashes"));
                reservations.emplace_back(std::move(reservation));
            }
            return output.replace(reservations);
        };
        if (!decode(state.at("scheduled"), candidate->scheduled) ||
            !decode(state.at("agent"), candidate->agent)) return false;
        uint64_t scheduledChecksum = 0, agentChecksum = 0;
        FromJson(scheduledChecksum, state.at("scheduled_checksum"));
        FromJson(agentChecksum, state.at("agent_checksum"));
        if (scheduledChecksum != candidate->scheduled.getStats().checksum ||
            agentChecksum != candidate->agent.getStats().checksum) return false;
        auto matchesCounts = [&state](const PendingCacheState &pending,
                                      const std::string &prefix) {
            size_t blocks = 0, tails = 0;
            FromJson(blocks, state.at(prefix + "_unique_blocks"));
            FromJson(tails, state.at(prefix + "_unique_tail_blocks"));
            const auto stats = pending.getStats();
            return blocks == stats.uniqueBlockCount && tails == stats.uniqueTailReadyBlockCount;
        };
        if (!matchesCounts(candidate->scheduled, "scheduled") ||
            !matchesCounts(candidate->agent, "agent")) return false;
        FromJson(candidate->agentRevision, state.at("agent_revision"));
        FromJson(candidate->agentSnapshotReady, state.at("agent_snapshot_ready"));
        candidate->overlap.batchCount = candidate->scheduled.countBatchOverlap(candidate->agent);
        candidate->overlap.uniqueBlockCount = candidate->scheduled.countFullReadyOverlap(candidate->agent);
        candidate->overlap.uniqueTailReadyBlockCount = candidate->scheduled.countTailReadyOverlap(candidate->agent);
    } catch (const std::exception &) {
        return false;
    }
    autil::ScopedWriteLock lock(_nodeLock);
    _nodeCaches[agentIdHash] = std::move(candidate);
    return true;
}

MasterPendingCacheR::Stats MasterPendingCacheR::getStatsLocked(
    const NodeCache &cache)
{
    if (const auto stats = std::atomic_load(&cache.cachedStats)) {
        return *stats;
    }
    const auto scheduled = cache.scheduled.getStats();
    const auto agent = cache.agent.getStats();
    Stats stats;
    stats.batchCount = scheduled.batchCount + agent.batchCount -
                       cache.overlap.batchCount;
    stats.uniqueBlockCount = scheduled.uniqueBlockCount + agent.uniqueBlockCount -
                             cache.overlap.uniqueBlockCount;
    stats.uniqueTailReadyBlockCount =
        scheduled.uniqueTailReadyBlockCount + agent.uniqueTailReadyBlockCount -
        cache.overlap.uniqueTailReadyBlockCount;
    stats.scheduledBatchCount = scheduled.batchCount;
    stats.agentBatchCount = agent.batchCount;
    stats.agentRevision = cache.agentRevision;
    stats.agentChecksum = agent.checksum;
    stats.agentRevisionGapCount = cache.agentRevisionGapCount;
    stats.agentStateMismatchCount = cache.agentStateMismatchCount;
    stats.agentSnapshotReady = cache.agentSnapshotReady;
    // Readers share an immutable result until the next state mutation.
    std::atomic_store(&cache.cachedStats, std::make_shared<const Stats>(stats));
    return stats;
}

bool MasterPendingCacheR::validateAgentStateLocked(
    const NodeCache &cache, const AgentPendingCacheToMasterData &update)
{
    const auto stats = cache.agent.getStats();
    return update.batchCount == static_cast<int64_t>(stats.batchCount) &&
           update.uniqueBlockCount ==
               static_cast<int64_t>(stats.uniqueBlockCount) &&
           update.uniqueTailReadyBlockCount ==
               static_cast<int64_t>(stats.uniqueTailReadyBlockCount) &&
           update.checksum == stats.checksum;
}

REGISTER_RESOURCE(MasterPendingCacheR);

} // namespace navi_sched
