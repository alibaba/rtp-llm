#include "aios/navi_sched/collector/PPCacheCollectorState.h"

#include <algorithm>
#include <stdexcept>

namespace navi_sched {

PPCacheCollectorState::PPCacheCollectorState(uint32_t shardCount)
    : _shardCount(shardCount), _shards(shardCount)
{
    if (!shardCount) throw std::invalid_argument("PP collector requires shards");
}

void PPCacheCollectorState::checkShard(uint32_t shardId) const
{
    if (shardId >= _shardCount) throw std::invalid_argument("invalid PP shard id");
}

void PPCacheCollectorState::remove(uint64_t hash, HashSet &local, Counts &counts,
                                   std::vector<uint64_t> &removed)
{
    if (!local.erase(hash)) return;
    auto found = counts.find(hash);
    if (found->second == _shardCount) removed.push_back(hash);
    if (!--found->second) counts.erase(found);
}

void PPCacheCollectorState::add(uint64_t hash, HashSet &local, Counts &counts,
                                std::vector<uint64_t> &added)
{
    if (local.insert(hash).second && ++counts[hash] == _shardCount) added.push_back(hash);
}

void PPCacheCollectorState::setPeerCacheReadAvailable(Shard &shard, bool available,
                                                      PPCacheCollectorChanges &changes)
{
    if (shard.peerCacheReadAvailable == available) return;
    const bool previouslyAvailable = _peerReadableShards == _shardCount;
    if (available)
        ++_peerReadableShards;
    else
        --_peerReadableShards;
    shard.peerCacheReadAvailable = available;
    const bool nowAvailable = _peerReadableShards == _shardCount;
    if (previouslyAvailable != nowAvailable)
        changes.peerCacheReadAvailable = nowAvailable;
}

void PPCacheCollectorState::clearCache(Shard &shard, PPCacheCollectorChanges &changes)
{
    while (!shard.full.empty())
        remove(*shard.full.begin(), shard.full, _fullCounts, changes.removedFull);
    while (!shard.tail.empty())
        remove(*shard.tail.begin(), shard.tail, _tailCounts, changes.removedTail);
    setPeerCacheReadAvailable(shard, false, changes);
    shard.synced = false;
}

void PPCacheCollectorState::completeRequestIfReady(const PPRequestAttempt &key,
                                                   PPCacheCollectorChanges &changes)
{
    auto found = _requests.find(key);
    if (found == _requests.end() || found->second.remaining) return;
    PPRequestCompletion completed;
    completed.request = found->first;
    completed.shardStatuses = std::move(found->second.statuses);
    completed.succeeded =
        std::all_of(completed.shardStatuses.begin(), completed.shardStatuses.end(),
                    [](PPShardRequestStatus status) {
                        return status == PPShardRequestStatus::SUCCEEDED;
                    });
    changes.completedRequests.push_back(std::move(completed));
    _requests.erase(found);
}

void PPCacheCollectorState::failShardRequests(uint32_t shardId, uint64_t generation,
                                              PPCacheCollectorChanges &changes)
{
    std::vector<PPRequestAttempt> completed;
    for (auto &entry : _requests) {
        auto &request = entry.second;
        if (request.generations[shardId] == generation &&
            request.statuses[shardId] == PPShardRequestStatus::PENDING) {
            request.statuses[shardId] = PPShardRequestStatus::DISCONNECTED;
            if (!--request.remaining) completed.push_back(entry.first);
        }
    }
    for (const auto &key : completed) completeRequestIfReady(key, changes);
}

PPCacheCollectorChanges PPCacheCollectorState::connect(uint32_t shardId,
                                                       uint64_t generation)
{
    checkShard(shardId);
    if (!generation) throw std::invalid_argument("PP stream generation must be positive");
    std::lock_guard<std::mutex> lock(_mutex);
    auto &shard = _shards[shardId];
    PPCacheCollectorChanges changes;
    if (generation <= shard.generation) return changes;
    clearCache(shard, changes);
    failShardRequests(shardId, shard.generation, changes);
    shard.generation = generation;
    shard.sequence = 0;
    shard.connected = true;
    changes.accepted = true;
    changes.needsFullSnapshot = true;
    return changes;
}

PPCacheCollectorChanges PPCacheCollectorState::disconnect(uint32_t shardId,
                                                          uint64_t generation)
{
    checkShard(shardId);
    std::lock_guard<std::mutex> lock(_mutex);
    auto &shard = _shards[shardId];
    PPCacheCollectorChanges changes;
    if (generation != shard.generation || !shard.connected) return changes;
    clearCache(shard, changes);
    failShardRequests(shardId, generation, changes);
    shard.connected = false;
    changes.accepted = true;
    return changes;
}

PPCacheCollectorChanges PPCacheCollectorState::apply(const PPCacheShardUpdate &update)
{
    checkShard(update.shardId);
    if (!update.streamGeneration || !update.sequence ||
        update.peerCacheReadAvailable < -1 || update.peerCacheReadAvailable > 1 ||
        (update.fullSnapshot &&
         (!update.removedFull.empty() || !update.removedTail.empty()))) {
        throw std::invalid_argument("invalid PP shard update");
    }
    for (const auto &terminal : update.terminals) {
        if (!terminal.request.requestId || !terminal.request.attemptId ||
            terminal.status < PPShardRequestStatus::SUCCEEDED ||
            terminal.status > PPShardRequestStatus::CANCELLED) {
            throw std::invalid_argument("invalid PP shard request terminal");
        }
    }
    std::lock_guard<std::mutex> lock(_mutex);
    auto &shard = _shards[update.shardId];
    PPCacheCollectorChanges changes;
    if (!shard.connected || update.streamGeneration != shard.generation ||
        update.sequence <= shard.sequence)
        return changes;
    if (!update.fullSnapshot &&
        (!shard.synced || update.sequence != shard.sequence + 1)) {
        clearCache(shard, changes);
        failShardRequests(update.shardId, shard.generation, changes);
        changes.needsFullSnapshot = true;
        return changes;
    }
    if (update.fullSnapshot) {
        // Emit only net changes; unchanged blocks must not be removed and
        // re-added when a source replaces its complete view.
        HashSet nextFull(update.addedFull.begin(), update.addedFull.end());
        HashSet nextTail(update.addedTail.begin(), update.addedTail.end());
        const auto oldFull = shard.full;
        const auto oldTail = shard.tail;
        for (auto hash : oldFull)
            if (!nextFull.count(hash))
                remove(hash, shard.full, _fullCounts, changes.removedFull);
        for (auto hash : oldTail)
            if (!nextTail.count(hash))
                remove(hash, shard.tail, _tailCounts, changes.removedTail);
        for (auto hash : nextFull) add(hash, shard.full, _fullCounts, changes.addedFull);
        for (auto hash : nextTail) add(hash, shard.tail, _tailCounts, changes.addedTail);
    } else {
        for (auto hash : update.removedFull)
            remove(hash, shard.full, _fullCounts, changes.removedFull);
        for (auto hash : update.removedTail)
            remove(hash, shard.tail, _tailCounts, changes.removedTail);
        for (auto hash : update.addedFull)
            add(hash, shard.full, _fullCounts, changes.addedFull);
        for (auto hash : update.addedTail)
            add(hash, shard.tail, _tailCounts, changes.addedTail);
    }
    if (update.fullSnapshot || update.peerCacheReadAvailable >= 0)
        setPeerCacheReadAvailable(shard, update.peerCacheReadAvailable == 1, changes);
    shard.sequence = update.sequence;
    shard.synced = true;
    for (const auto &terminal : update.terminals) {
        auto found = _requests.find(terminal.request);
        if (found == _requests.end()) continue;
        auto &request = found->second;
        if (request.generations[update.shardId] != update.streamGeneration ||
            request.statuses[update.shardId] != PPShardRequestStatus::PENDING)
            continue;
        request.statuses[update.shardId] = terminal.status;
        --request.remaining;
        completeRequestIfReady(terminal.request, changes);
    }
    changes.accepted = true;
    return changes;
}

PPCacheCollectorChanges PPCacheCollectorState::beginRequest(const PPRequestAttempt &key)
{
    if (!key.requestId || !key.attemptId)
        throw std::invalid_argument("invalid PP request attempt");
    std::lock_guard<std::mutex> lock(_mutex);
    PPCacheCollectorChanges changes;
    if (_requests.count(key)) return changes;
    Request request;
    for (const auto &shard : _shards) {
        request.generations.push_back(shard.generation);
        const bool available = shard.connected && shard.synced;
        request.statuses.push_back(available ? PPShardRequestStatus::PENDING
                                             : PPShardRequestStatus::DISCONNECTED);
        request.remaining += available;
    }
    _requests.emplace(key, std::move(request));
    completeRequestIfReady(key, changes);
    changes.accepted = true;
    return changes;
}

std::vector<PPRequestShardConnection> PPCacheCollectorState::pendingRequestShards(
    const PPRequestAttempt &key) const
{
    std::lock_guard<std::mutex> lock(_mutex);
    std::vector<PPRequestShardConnection> pending;
    const auto found = _requests.find(key);
    if (found == _requests.end()) return pending;
    const auto &request = found->second;
    pending.reserve(request.remaining);
    for (uint32_t shard = 0; shard < _shardCount; ++shard) {
        if (request.statuses[shard] == PPShardRequestStatus::PENDING)
            pending.push_back({shard, request.generations[shard]});
    }
    return pending;
}

size_t PPCacheCollectorState::pendingRequestCount() const
{
    std::lock_guard<std::mutex> lock(_mutex);
    return _requests.size();
}
bool PPCacheCollectorState::fullReady(uint64_t hash) const
{
    std::lock_guard<std::mutex> lock(_mutex);
    const auto found = _fullCounts.find(hash);
    return found != _fullCounts.end() && found->second == _shardCount;
}
bool PPCacheCollectorState::tailReady(uint64_t hash) const
{
    std::lock_guard<std::mutex> lock(_mutex);
    const auto found = _tailCounts.find(hash);
    return found != _tailCounts.end() && found->second == _shardCount;
}

}
