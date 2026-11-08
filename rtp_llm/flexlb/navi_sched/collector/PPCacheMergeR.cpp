#include "aios/navi_sched/collector/PPCacheMergeR.h"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace navi_sched {

const std::string PPCacheMergeR::RESOURCE_ID = "navi_sched.pp_cache_merge_r";
PPCacheMergeR::PPCacheMergeR() : ExternalResource(RESOURCE_ID, navi::RS_EXTERNAL) {}

bool PPCacheMergeR::initialize(const std::string &group, uint32_t shardCount,
                               Callback callback)
{
    std::lock_guard<std::mutex> lock(_mutex);
    if (_state || group.empty() || !shardCount || !callback) return false;
    _group = group;
    _shardCount = shardCount;
    _state = std::make_unique<PPCacheCollectorState>(shardCount);
    _callback = std::move(callback);
    return true;
}

PPCacheCollectorChanges PPCacheMergeR::mutate(
    const std::function<PPCacheCollectorChanges(PPCacheCollectorState &)> &operation)
{
    PPCacheCollectorChanges changes;
    bool writer = false;
    {
        std::lock_guard<std::mutex> lock(_mutex);
        if (!_state) return changes;
        changes = operation(*_state);
        if (changes.peerCacheReadAvailable >= 0 || !changes.addedFull.empty() ||
            !changes.removedFull.empty() || !changes.addedTail.empty() ||
            !changes.removedTail.empty() || !changes.completedRequests.empty()) {
            _pending.push_back(changes);
        }
        if (!_pending.empty() && !_draining) {
            _draining = true;
            writer = true;
        }
    }
    if (writer) drain();
    return changes;
}

void PPCacheMergeR::drain()
{
    while (true) {
        PPCacheCollectorChanges changes;
        {
            std::lock_guard<std::mutex> lock(_mutex);
            if (_pending.empty()) {
                _draining = false;
                return;
            }
            changes = std::move(_pending.front());
            _pending.pop_front();
        }
        // Like the Agent report writer, never call downstream under the state
        // lock. The callback must not throw; it may synchronously reenter us.
        _callback(changes);
    }
}

PPCacheCollectorChanges PPCacheMergeR::connect(const std::string &group, uint32_t shardId,
                                               uint32_t shardCount, uint64_t generation)
{
    return mutate([&](PPCacheCollectorState &state) {
        if (group != _group || shardCount != _shardCount || shardId >= _shardCount ||
            !generation)
            return PPCacheCollectorChanges{};
        _nextGeneration = std::max(_nextGeneration, generation);
        return state.connect(shardId, generation);
    });
}

PPCacheCollectorChanges PPCacheMergeR::disconnect(uint32_t shardId, uint64_t generation)
{
    return mutate([&](PPCacheCollectorState &state) {
        return state.disconnect(shardId, generation);
    });
}

PPCacheCollectorChanges PPCacheMergeR::apply(const PPCacheShardUpdate &update)
{
    return mutate([&](PPCacheCollectorState &state) { return state.apply(update); });
}

PPCacheCollectorChanges PPCacheMergeR::beginRequest(const PPRequestAttempt &request)
{
    return mutate(
        [&](PPCacheCollectorState &state) { return state.beginRequest(request); });
}

std::vector<PPRequestShardConnection> PPCacheMergeR::pendingRequestShards(
    const PPRequestAttempt &request)
{
    std::lock_guard<std::mutex> lock(_mutex);
    return _state ? _state->pendingRequestShards(request)
                  : std::vector<PPRequestShardConnection>{};
}

uint64_t PPCacheMergeR::allocateGeneration()
{
    std::lock_guard<std::mutex> lock(_mutex);
    if (_nextGeneration == std::numeric_limits<uint64_t>::max())
        throw std::overflow_error("PP stream generation exhausted");
    return ++_nextGeneration;
}

REGISTER_RESOURCE(PPCacheMergeR);

}
