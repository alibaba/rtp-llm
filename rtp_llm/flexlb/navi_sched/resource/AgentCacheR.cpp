#include "aios/navi_sched/resource/AgentCacheR.h"
#include "aios/navi_sched/data/BlockHashUtil.h"
#include "aios/navi_sched/resource/AgentPendingCacheR.h"
#include "autil/TimeUtility.h"

namespace navi_sched {

const std::string AgentCacheR::RESOURCE_ID = "navi_sched.agent_cache_r";

void AgentCacheR::def(navi::ResourceDefBuilder &builder) const
{
    builder.name(RESOURCE_ID, navi::RS_BIZ);
}

bool AgentCacheR::config(navi::ResourceConfigContext &ctx)
{
    NAVI_JSONIZE(ctx, "total_blocks", _totalBlocks, _totalBlocks);
    return _totalBlocks >= 0;
}

navi::ErrorCode AgentCacheR::init(navi::ResourceInitContext &) { return navi::EC_NONE; }

NaviSchedRequestStats AgentCacheR::makeRequestStats(
    const NaviSchedBatchRequest &request) const
{
    return makeRequestAudit(request).stats;
}

AgentCacheR::RequestAudit AgentCacheR::makeRequestAudit(
    const NaviSchedBatchRequest &request) const
{
    return makeRequestAudit(request, nullptr);
}

AgentCacheR::RequestAudit AgentCacheR::makeRequestAudit(
    const NaviSchedBatchRequest &request,
    const AgentPendingCacheR *pendingCache) const
{
    RequestAudit audit;
    auto &stats = audit.stats;
    stats.requestId = request.requestId;
    stats.totalTokenCount = request.totalTokenCount;
    stats.blockSize = request.blockSize;
    stats.predictedCacheHitTokenCount = request.predictedCacheHitTokenCount;
    stats.predictedQueueLatencyMs = request.predictedQueueLatencyMs;
    stats.predictedExecutionLatencyMs = request.predictedExecutionLatencyMs;
    stats.latencyTrainingPhase = request.latencyTrainingPhase;
    int64_t physicalMatchedBlockCount = 0;
    int64_t matchedBlockCount = 0;
    {
        autil::ScopedReadLock lock(_lock);
        if (!_policyReady) {
            audit.valid = false;
            NAVI_KERNEL_LOG(ERROR,
                            "cache policy is not initialized for request [%lu]",
                            request.requestId);
            return audit;
        }
        if (request.blockSize != _policy.blockSize) {
            audit.valid = false;
            NAVI_KERNEL_LOG(ERROR,
                            "cache policy mismatch for request [%lu], request "
                            "block size [%ld], effective block size [%ld], linear "
                            "step [%ld], hybrid tail required [%d]",
                            request.requestId, request.blockSize, _policy.blockSize,
                            _policy.linearStep, _policy.hybridTailRequired ? 1 : 0);
            return audit;
        }
        const auto &prefixHashes =
            _hybridTailRequired ? _fullReadyBlockHashes : _cachedBlockHashes;
        const auto &blockHashes = BlockHashUtil::get(request.blockHashes);
        const auto matchableBlockCount = BlockHashUtil::getMatchableBlockCount(
            request.totalTokenCount, request.blockSize, request.blockHashes);
        for (size_t blockIndex = 0; blockIndex < matchableBlockCount; ++blockIndex) {
            const auto hash = blockHashes[blockIndex];
            if (prefixHashes.end() == prefixHashes.find(hash)) {
                break;
            }
            ++physicalMatchedBlockCount;
        }
        if (_hybridTailRequired) {
            while (physicalMatchedBlockCount > 0 &&
                   _tailReadyBlockHashes.end() == _tailReadyBlockHashes.find(
                       blockHashes[static_cast<size_t>(physicalMatchedBlockCount - 1)])) {
                --physicalMatchedBlockCount;
            }
        }
        matchedBlockCount = pendingCache
                                ? pendingCache->getMatchedBlockCount(
                                      blockHashes, _cachedBlockHashes,
                                      _fullReadyBlockHashes, _tailReadyBlockHashes,
                                      _hybridTailRequired, matchableBlockCount)
                                : physicalMatchedBlockCount;
        audit.cacheState = {_revision,
                            _cachedBlockHashes.size(),
                            _hashChecksum,
                            _fullReadyBlockHashes.size(),
                            _fullReadyHashChecksum,
                            _tailReadyBlockHashes.size(),
                            _tailReadyHashChecksum,
                            _rtpCachedBlockCount};
        audit.hybridTailRequired = _hybridTailRequired;
        audit.eventTimeUs = autil::TimeUtility::currentTime();
    }
    audit.physicalCacheHitTokenCount =
        physicalMatchedBlockCount * request.blockSize;
    stats.cacheHitTokenCount = matchedBlockCount * request.blockSize;
    return audit;
}

bool AgentCacheR::apply(const NaviSchedCacheUpdate &update, State &state)
{
    autil::ScopedWriteLock lock(_lock);
    const bool hasPolicyFields = update.blockSize >= 0 || update.linearStep >= 0 ||
                                 update.hybridTailRequired >= 0;
    if (hasPolicyFields) {
        if (update.blockSize <= 0 || update.linearStep <= 0 ||
            update.hybridTailRequired < 0) {
            NAVI_KERNEL_LOG(ERROR,
                            "reject incomplete cache policy update, block size [%ld], "
                            "linear step [%ld], hybrid tail required [%d]",
                            update.blockSize, update.linearStep,
                            static_cast<int>(update.hybridTailRequired));
            return false;
        }
        NaviSchedCachePolicy policy{update.blockSize, update.linearStep,
                                    update.hybridTailRequired != 0};
        if (_policyReady &&
            (_policy.blockSize != policy.blockSize ||
             _policy.linearStep != policy.linearStep ||
             _policy.hybridTailRequired != policy.hybridTailRequired)) {
            NAVI_KERNEL_LOG(ERROR,
                            "reject cache policy change, effective block size [%ld], "
                            "linear step [%ld], hybrid tail required [%d], incoming "
                            "block size [%ld], linear step [%ld], hybrid tail required [%d]",
                            _policy.blockSize, _policy.linearStep,
                            _policy.hybridTailRequired ? 1 : 0, policy.blockSize,
                            policy.linearStep, policy.hybridTailRequired ? 1 : 0);
            return false;
        }
        if (!_policyReady) {
            _policy = policy;
            _policyReady = true;
            NAVI_KERNEL_LOG(INFO,
                            "effective cache policy, block size [%ld], linear step [%ld], "
                            "hybrid tail required [%d]",
                            _policy.blockSize, _policy.linearStep,
                            _policy.hybridTailRequired ? 1 : 0);
        }
    }
    if (update.fullSnapshot) {
        _cachedBlockHashes.clear();
        _fullReadyBlockHashes.clear();
        _tailReadyBlockHashes.clear();
        _hashChecksum = 0;
        _fullReadyHashChecksum = 0;
        _tailReadyHashChecksum = 0;
    }
    if (update.totalBlocks >= 0) {
        _totalBlocks = update.totalBlocks;
    }
    if (update.rtpCachedBlockCount >= 0) {
        _rtpCachedBlockCount = update.rtpCachedBlockCount;
    }
    if (update.hybridTailRequired >= 0) {
        _hybridTailRequired = update.hybridTailRequired != 0;
    }
    for (auto hash : BlockHashUtil::get(update.removedBlockHashes)) {
        if (_cachedBlockHashes.erase(hash) > 0) {
            _hashChecksum ^= hash;
        }
    }
    for (auto hash : BlockHashUtil::get(update.addedBlockHashes)) {
        if (_cachedBlockHashes.emplace(hash).second) {
            _hashChecksum ^= hash;
        }
    }
    for (auto hash : BlockHashUtil::get(update.removedFullReadyBlockHashes)) {
        if (_fullReadyBlockHashes.erase(hash) > 0) {
            _fullReadyHashChecksum ^= hash;
        }
    }
    for (auto hash : BlockHashUtil::get(update.addedFullReadyBlockHashes)) {
        if (_fullReadyBlockHashes.emplace(hash).second) {
            _fullReadyHashChecksum ^= hash;
        }
    }
    for (auto hash : BlockHashUtil::get(update.removedTailReadyBlockHashes)) {
        if (_tailReadyBlockHashes.erase(hash) > 0) {
            _tailReadyHashChecksum ^= hash;
        }
    }
    for (auto hash : BlockHashUtil::get(update.addedTailReadyBlockHashes)) {
        if (_tailReadyBlockHashes.emplace(hash).second) {
            _tailReadyHashChecksum ^= hash;
        }
    }
    ++_revision;
    state = {_revision, _cachedBlockHashes.size(), _hashChecksum,
             _fullReadyBlockHashes.size(), _fullReadyHashChecksum,
             _tailReadyBlockHashes.size(), _tailReadyHashChecksum,
             _rtpCachedBlockCount, _policy};
    return true;
}

NaviSchedCacheUpdate AgentCacheR::makeSnapshot() const
{
    NaviSchedCacheUpdate update;
    autil::ScopedReadLock lock(_lock);
    update.totalBlocks = _totalBlocks;
    update.fullSnapshot = true;
    update.addedBlockHashes = std::make_shared<const NaviSchedBlockHashes>(
        _cachedBlockHashes.begin(), _cachedBlockHashes.end());
    update.addedFullReadyBlockHashes = std::make_shared<const NaviSchedBlockHashes>(
        _fullReadyBlockHashes.begin(), _fullReadyBlockHashes.end());
    update.addedTailReadyBlockHashes = std::make_shared<const NaviSchedBlockHashes>(
        _tailReadyBlockHashes.begin(), _tailReadyBlockHashes.end());
    if (_policyReady) {
        update.blockSize = _policy.blockSize;
        update.linearStep = _policy.linearStep;
        update.hybridTailRequired = _policy.hybridTailRequired ? 1 : 0;
    }
    update.revision = _revision;
    update.cachedBlockCount = _cachedBlockHashes.size();
    update.hashChecksum = _hashChecksum;
    update.fullReadyBlockCount = _fullReadyBlockHashes.size();
    update.fullReadyHashChecksum = _fullReadyHashChecksum;
    update.tailReadyBlockCount = _tailReadyBlockHashes.size();
    update.tailReadyHashChecksum = _tailReadyHashChecksum;
    update.rtpCachedBlockCount = _rtpCachedBlockCount;
    return update;
}

size_t AgentCacheR::getCachedBlockCount() const
{
    autil::ScopedReadLock lock(_lock);
    return _cachedBlockHashes.size();
}

AgentCacheR::State AgentCacheR::getState() const
{
    autil::ScopedReadLock lock(_lock);
    return {_revision, _cachedBlockHashes.size(), _hashChecksum,
            _fullReadyBlockHashes.size(), _fullReadyHashChecksum,
            _tailReadyBlockHashes.size(), _tailReadyHashChecksum,
            _rtpCachedBlockCount, _policy};
}

navi::SummaryErrorCode AgentCacheR::getSummary(const std::string &,
                                               navi::SummaryBuilder &builder) const
{
    autil::ScopedReadLock lock(_lock);
    auto table = builder.newTable("NaviSchedAgentCacheOverview");
    TABLE_ADD_COLUMN_VALUE(table, "total_blocks", _totalBlocks);
    TABLE_ADD_COLUMN_VALUE(table, "cached_block_count", _cachedBlockHashes.size());
    TABLE_ADD_COLUMN_VALUE(table, "full_ready_block_count", _fullReadyBlockHashes.size());
    TABLE_ADD_COLUMN_VALUE(table, "tail_ready_block_count", _tailReadyBlockHashes.size());
    TABLE_ADD_COLUMN_VALUE(table, "cache_policy_ready", _policyReady);
    TABLE_ADD_COLUMN_VALUE(table, "block_size", _policy.blockSize);
    TABLE_ADD_COLUMN_VALUE(table, "linear_step", _policy.linearStep);
    TABLE_ADD_COLUMN_VALUE(table, "hybrid_tail_required", _hybridTailRequired);
    return navi::SEC_OK;
}

REGISTER_RESOURCE(AgentCacheR);

}
