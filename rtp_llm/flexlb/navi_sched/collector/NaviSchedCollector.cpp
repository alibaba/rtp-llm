#include "aios/navi_sched/collector/NaviSchedCollector.h"
#include "aios/navi_sched/config/NaviSchedConfigBuilder.h"
#include "navi/engine/ResourceMap.h"
#include <stdexcept>

namespace navi_sched {

NaviSchedCollector::NaviSchedCollector(const NaviSchedCollectorConfig &config)
    : NaviSchedRole(config.service)
    , _collectorConfig(config)
    , _report(std::make_shared<PPCacheShardReportR>())
{
}

NaviSchedCollector::~NaviSchedCollector() { stop(); }

bool NaviSchedCollector::init()
{
    if (_stopped.load()) return false;
    if (_started.load()) {
        return true;
    }
    const auto &config = _collectorConfig;
    if (config.runtime.bizName.empty() ||
        !_report->initializeIdentity(config.ppGroup, config.shardId, config.shardCount)) {
        return false;
    }
    if (!config.groupRoles.empty() || !config.groupBlockScales.empty()) {
        try {
            _residentState = std::make_unique<PPCacheResidentState>(
                config.groupRoles, config.groupBlockScales);
        } catch (const std::invalid_argument &) {
            return false;
        }
    }
    if (!NaviSchedRole::init()) {
        return false;
    }
    auto resources = std::make_shared<navi::ResourceMap>();
    resources->addResource(_report);
    const auto configStr =
        NaviSchedConfigBuilder::build(config.runtime,
                                      {"^navi_sched\\.pp_cache_shard_source_k$",
                                       "^navi_sched\\.pp_request_finished_sink_k$"},
                                      false);
    if (configStr.empty() || !update(configStr, resources)) {
        stop();
        return false;
    }
    _started.store(true);
    return true;
}

bool NaviSchedCollector::publish(const PPCacheShardUpdate &localChange)
{
    std::lock_guard<std::mutex> lock(_producerMutex);
    // An event-indexed collector accepts raw wire request terminals, but KV
    // membership must come from the object index so counts cannot diverge.
    if (_residentState &&
        (localChange.fullSnapshot || !localChange.addedFull.empty() ||
         !localChange.removedFull.empty() || !localChange.addedTail.empty() ||
         !localChange.removedTail.empty()))
        return false;
    return _started.load() && _report->publish(localChange);
}

bool NaviSchedCollector::publishResidentChange(const PPCacheShardUpdate &change)
{
    if (!change.fullSnapshot && change.addedFull.empty() && change.removedFull.empty() &&
        change.addedTail.empty() && change.removedTail.empty())
        return true;
    return _report->publish(change);
}

bool NaviSchedCollector::replaceResidentObjects(
    const std::vector<PPCacheResidentObject> &objects)
{
    std::lock_guard<std::mutex> lock(_producerMutex);
    return _started.load() && _residentState &&
           publishResidentChange(_residentState->replace(objects));
}

bool NaviSchedCollector::objectResident(const PPCacheResidentObject &object)
{
    std::lock_guard<std::mutex> lock(_producerMutex);
    return _started.load() && _residentState &&
           publishResidentChange(_residentState->resident(object));
}

bool NaviSchedCollector::objectRemoved(const std::string &objectId)
{
    std::lock_guard<std::mutex> lock(_producerMutex);
    return _started.load() && _residentState &&
           publishResidentChange(_residentState->removed(objectId));
}

bool NaviSchedCollector::reportPeerCacheReadAvailable(bool available)
{
    PPCacheShardUpdate change;
    change.peerCacheReadAvailable = available;
    return publish(change);
}

void NaviSchedCollector::stop()
{
    {
        std::lock_guard<std::mutex> lock(_producerMutex);
        _stopped.store(true);
        _started.store(false);
        _report->stop();
    }
    NaviSchedRole::stop();
}

}
