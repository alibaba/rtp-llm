#include "aios/navi_sched/collector/PPCacheShardReportR.h"
#include <algorithm>

namespace navi_sched {

const std::string PPCacheShardReportR::RESOURCE_ID = "navi_sched.pp_cache_shard_report_r";

PPCacheShardReportR::PPCacheShardReportR()
    : navi::ExternalResource(RESOURCE_ID, navi::RS_EXTERNAL)
{
}

bool PPCacheShardReportR::initializeIdentity(const std::string &group, uint32_t shardId,
                                             uint32_t shardCount)
{
    if (group.empty() || !shardCount || shardId >= shardCount) return false;
    std::lock_guard<std::mutex> lock(_mutex);
    if (_initialized)
        return group == _group && shardId == _shardId && shardCount == _shardCount;
    _group = group;
    _shardId = shardId;
    _shardCount = shardCount;
    _initialized = true;
    return true;
}

void PPCacheShardReportR::enqueueLocked(const PPCacheShardUpdate &change,
                                        const std::shared_ptr<Subscriber> &subscriber)
{
    auto data = std::make_shared<PPCacheShardUpdateData>();
    data->update = change;
    if (change.fullSnapshot)
        data->update.peerCacheReadAvailable = _peerCacheReadAvailable;
    data->update.shardId = _shardId;
    data->update.streamGeneration = subscriber->generation;
    data->update.sequence = ++subscriber->sequence;
    _deliveries.push_back({subscriber, std::move(data)});
}

bool PPCacheShardReportR::subscribe(const std::string &group, uint32_t shardId,
                                    uint32_t shardCount, const std::string &collectorId,
                                    uint64_t generation, const navi::AsyncPipePtr &pipe)
{
    if (collectorId.empty() || !generation || !pipe) return false;
    {
        std::lock_guard<std::mutex> lock(_mutex);
        if (!_initialized || _stopping || group != _group || shardId != _shardId ||
            shardCount != _shardCount ||
            (_subscriber && _subscriber->active.load() && !_subscriber->pipe.expired()))
            return false;
        auto subscriber = std::make_shared<Subscriber>();
        subscriber->collectorId = collectorId;
        subscriber->generation = generation;
        subscriber->pipe = pipe;
        PPCacheShardUpdate snapshot;
        snapshot.fullSnapshot = true;
        snapshot.peerCacheReadAvailable = _peerCacheReadAvailable;
        snapshot.addedFull.assign(_full.begin(), _full.end());
        snapshot.addedTail.assign(_tail.begin(), _tail.end());
        std::sort(snapshot.addedFull.begin(), snapshot.addedFull.end());
        std::sort(snapshot.addedTail.begin(), snapshot.addedTail.end());
        enqueueLocked(snapshot, subscriber);
        _subscriber = std::move(subscriber);
    }
    drain();
    return true;
}

bool PPCacheShardReportR::unsubscribe(const std::string &collectorId,
                                      const navi::AsyncPipePtr &pipe)
{
    std::lock_guard<std::mutex> lock(_mutex);
    if (!_subscriber || _subscriber->collectorId != collectorId ||
        _subscriber->pipe.lock() != pipe)
        return false;
    _subscriber->active.store(false);
    _subscriber.reset();
    return true;
}

bool PPCacheShardReportR::publish(const PPCacheShardUpdate &localChange)
{
    PPCacheShardUpdateData validated;
    validated.update = localChange;
    validated.update.streamGeneration = 1;
    validated.update.sequence = 1;
    if (!validated.valid()) return false;
    {
        std::lock_guard<std::mutex> lock(_mutex);
        if (!_initialized || _stopping) return false;
        if (localChange.peerCacheReadAvailable >= 0)
            _peerCacheReadAvailable = localChange.peerCacheReadAvailable == 1;
        if (localChange.fullSnapshot) {
            _full.clear();
            _full.insert(localChange.addedFull.begin(), localChange.addedFull.end());
            _tail.clear();
            _tail.insert(localChange.addedTail.begin(), localChange.addedTail.end());
        } else {
            for (auto hash : localChange.removedFull) _full.erase(hash);
            for (auto hash : localChange.removedTail) _tail.erase(hash);
            _full.insert(localChange.addedFull.begin(), localChange.addedFull.end());
            _tail.insert(localChange.addedTail.begin(), localChange.addedTail.end());
        }
        if (_subscriber && _subscriber->active.load() && !_subscriber->pipe.expired())
            enqueueLocked(localChange, _subscriber);
    }
    drain();
    return true;
}

bool PPCacheShardReportR::publishRequestFinished(const std::string &group,
                                                 uint32_t shardCount,
                                                 const std::string &collectorId,
                                                 const PPCacheShardUpdateData &data)
{
    const auto &update = data.update;
    // Request outcomes cannot change sealed-resident cache membership.
    if (!data.valid() || update.fullSnapshot || update.peerCacheReadAvailable != -1 ||
        update.terminals.empty() || !update.addedFull.empty() ||
        !update.removedFull.empty() || !update.addedTail.empty() ||
        !update.removedTail.empty())
        return false;
    {
        std::lock_guard<std::mutex> lock(_mutex);
        if (!_initialized || _stopping || group != _group || shardCount != _shardCount ||
            update.shardId != _shardId || !_subscriber || !_subscriber->active.load() ||
            _subscriber->pipe.expired() || _subscriber->collectorId != collectorId ||
            _subscriber->generation != update.streamGeneration ||
            update.sequence != _subscriber->requestFinishedSequence + 1)
            return false;
        _subscriber->requestFinishedSequence = update.sequence;
        PPCacheShardUpdate finished;
        finished.terminals = update.terminals;
        enqueueLocked(finished, _subscriber);
    }
    drain();
    return true;
}

void PPCacheShardReportR::drain()
{
    {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_draining) return;
        _draining = true;
    }
    while (true) {
        Delivery delivery;
        {
            std::lock_guard<std::mutex> lock(_mutex);
            if (_deliveries.empty()) {
                _draining = false;
                return;
            }
            delivery = std::move(_deliveries.front());
            _deliveries.pop_front();
        }
        if (!delivery.subscriber->active.load()) continue;
        auto pipe = delivery.subscriber->pipe.lock();
        if (!pipe || pipe->setData(delivery.data) != navi::EC_NONE) {
            delivery.subscriber->active.store(false);
            if (pipe && pipe->setEof() != navi::EC_NONE) pipe->terminate();
        }
    }
}

void PPCacheShardReportR::stop()
{
    navi::AsyncPipePtr pipe;
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _stopping = true;
        if (_subscriber) {
            _subscriber->active.store(false);
            pipe = _subscriber->pipe.lock();
            _subscriber.reset();
        }
        _deliveries.clear();
    }
    if (pipe) pipe->setEof();
}

REGISTER_RESOURCE(PPCacheShardReportR);

}
