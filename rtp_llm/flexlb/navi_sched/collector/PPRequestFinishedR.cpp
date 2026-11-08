#include "aios/navi_sched/collector/PPRequestFinishedR.h"

namespace navi_sched {

const std::string PPRequestFinishedR::RESOURCE_ID = "navi_sched.pp_request_finished_r";
PPRequestFinishedR::PPRequestFinishedR()
    : ExternalResource(RESOURCE_ID, navi::RS_EXTERNAL)
{
}

bool PPRequestFinishedR::initialize(const std::string &group, uint32_t shardCount,
                                    FailureCallback onStreamFailure)
{
    std::lock_guard<std::mutex> lock(_mutex);
    if (!_slots.empty() || _stopping || group.empty() || shardCount <= 1 ||
        !onStreamFailure)
        return false;
    _group = group;
    _slots.resize(shardCount);
    _onStreamFailure = std::move(onStreamFailure);
    return true;
}

bool PPRequestFinishedR::reserve(uint32_t shardId, uint64_t generation)
{
    std::shared_ptr<Subscriber> previous;
    {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_stopping || shardId >= _slots.size() || !generation) return false;
        auto &slot = _slots[shardId];
        previous = slot.subscriber;
        if (previous && generation <= previous->generation) return false;
        slot = {};
        slot.subscriber = std::make_shared<Subscriber>();
        slot.subscriber->shardId = shardId;
        slot.subscriber->generation = generation;
    }
    if (previous) fail(previous, true);
    return true;
}

bool PPRequestFinishedR::subscribe(const std::string &group, uint32_t shardId,
                                   uint32_t shardCount, uint64_t generation,
                                   const navi::AsyncPipePtr &pipe)
{
    {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_stopping || group != _group || shardCount != _slots.size() ||
            shardId >= _slots.size() || !pipe)
            return false;
        auto &slot = _slots[shardId];
        auto subscriber = slot.subscriber;
        if (!subscriber || !subscriber->active.load() ||
            subscriber->generation != generation || slot.subscribed)
            return false;
        subscriber->pipe = pipe;
        slot.subscribed = true;
        while (!slot.pending.empty()) {
            _deliveries.push_back({subscriber, std::move(slot.pending.front())});
            slot.pending.pop_front();
        }
    }
    drain();
    return true;
}

void PPRequestFinishedR::fail(const std::shared_ptr<Subscriber> &subscriber,
                              bool terminate)
{
    navi::AsyncPipePtr pipe;
    {
        std::lock_guard<std::mutex> lock(_mutex);
        if (!subscriber->active.exchange(false)) return;
        pipe = subscriber->pipe.lock();
        auto &slot = _slots[subscriber->shardId];
        if (slot.subscriber == subscriber) slot.pending.clear();
    }
    if (terminate && pipe)
        if (pipe->setEof() != navi::EC_NONE) pipe->terminate();
    _onStreamFailure(subscriber->shardId, subscriber->generation);
}

bool PPRequestFinishedR::detach(uint32_t shardId, uint64_t generation,
                                const navi::AsyncPipePtr &expectedPipe, bool terminate)
{
    std::shared_ptr<Subscriber> subscriber;
    {
        std::lock_guard<std::mutex> lock(_mutex);
        if (shardId >= _slots.size()) return false;
        auto &slot = _slots[shardId];
        subscriber = slot.subscriber;
        if (!subscriber || !subscriber->active.load() ||
            subscriber->generation != generation ||
            (expectedPipe && subscriber->pipe.lock() != expectedPipe))
            return false;
        slot.pending.clear();
    }
    fail(subscriber, terminate);
    return true;
}

bool PPRequestFinishedR::unsubscribe(uint32_t shardId, uint64_t generation,
                                     const navi::AsyncPipePtr &pipe)
{
    if (!pipe) return false;
    return detach(shardId, generation, pipe, false);
}

bool PPRequestFinishedR::disconnect(uint32_t shardId, uint64_t generation)
{
    return detach(shardId, generation, {}, true);
}

bool PPRequestFinishedR::publish(uint32_t shardId, uint64_t generation,
                                 const PPShardRequestTerminal &finished)
{
    auto data = std::make_shared<PPCacheShardUpdateData>();
    data->update.shardId = shardId;
    data->update.streamGeneration = generation;
    data->update.sequence = 1;
    data->update.terminals.push_back(finished);
    if (!data->valid()) return false;
    {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_stopping || shardId >= _slots.size()) return false;
        auto &slot = _slots[shardId];
        const auto subscriber = slot.subscriber;
        if (!subscriber || !subscriber->active.load() ||
            subscriber->generation != generation)
            return false;
        data->update.sequence = ++slot.sequence;
        if (slot.subscribed)
            _deliveries.push_back({subscriber, data});
        else
            slot.pending.push_back(data);
    }
    drain();
    return true;
}

void PPRequestFinishedR::drain()
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
        if (!pipe || pipe->setData(delivery.data) != navi::EC_NONE)
            fail(delivery.subscriber, true);
    }
}

void PPRequestFinishedR::stop()
{
    std::vector<std::shared_ptr<Subscriber>> subscribers;
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _stopping = true;
        for (auto &slot : _slots) {
            if (slot.subscriber) subscribers.push_back(slot.subscriber);
            slot.pending.clear();
        }
        _deliveries.clear();
    }
    for (const auto &subscriber : subscribers) fail(subscriber, true);
}

REGISTER_RESOURCE(PPRequestFinishedR);

}
