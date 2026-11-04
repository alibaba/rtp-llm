#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/new_heartbeat/HeartbeatClientManager.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/interface/SearchService.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/new_heartbeat/StringTable.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/SearchServiceManager.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/util/FileRecorder.h"

namespace multi_call {
AUTIL_LOG_SETUP(multi_call, HeartbeatClientManager);

void HeartbeatClientManagerNotifier::notifyHeartbeatSuccess() const {
    autil::ScopedLock lock(_managerLock);
    if (_manager) {
        _manager->notifyHeartbeatSuccess();
    }
}

void HeartbeatClientManagerNotifier::stealManager() {
    autil::ScopedLock lock(_managerLock);
    _manager = nullptr;
}

HeartbeatClientManager::HeartbeatClientManager(SearchService *searchService,
                                               SearchServiceManager *manager)
    : _searchService(searchService)
    , _manager(manager)
    , _stringTable(std::make_shared<StringTable>())
    , _notifier(std::make_shared<HeartbeatClientManagerNotifier>(this)) {
}

HeartbeatClientManager::~HeartbeatClientManager() {
    _notifier->stealManager();
}

void HeartbeatClientManager::setThisSpec(const Spec &spec) {
    _thisSpec = std::make_shared<Spec>(spec);
}

bool HeartbeatClientManager::start(
    const MiscConfigPtr &miscConfig,
    const std::shared_ptr<SearchServiceSnapshot> &heartbeatSnapshot) {
    _miscConfig = miscConfig;
    _heartbeatSnapshot = heartbeatSnapshot;
    return initThread();
}

void HeartbeatClientManager::stop() {
    _notifier->stealManager();
    _thread.reset();
    auto hostMap = getHostMap();
    if (hostMap) {
        for (const auto &pair : *hostMap) {
            pair.second->stop();
        }
        setHostMap(nullptr);
    }
}

bool HeartbeatClientManager::initThread() {
    auto thread = autil::LoopThread::createLoopThread(
        std::bind(&HeartbeatClientManager::loop, this), 2 * 1000 * 1000, "GigClientHB");
    if (!thread) {
        AUTIL_LOG(ERROR, "create gig client heartbeat thread failed");
        return false;
    }
    _thread = thread;
    return true;
}

bool HeartbeatClientManager::updateSpecMap(const HeartbeatSpecMap &heartbeatSpecs) {
    bool hasDiff = false;
    auto newHostMapPtr = std::make_shared<HostHeartbeatInfoMap>();
    auto &newHostMap = *newHostMapPtr;
    auto oldHostMap = getHostMap();
    for (const auto &pair : heartbeatSpecs.specMap) {
        const auto &addr = pair.first;
        const auto &spec = pair.second;
        if (!spec || spec->clusterMap.empty()) {
            continue;
        }
        if (oldHostMap) {
            auto it = oldHostMap->find(addr);
            if (oldHostMap->end() != it) {
                if (*spec == *it->second->getSpec()) {
                    newHostMap.emplace(*it);
                    continue;
                }
            }
        }
        if (newHostMap.end() != newHostMap.find(addr)) {
            continue;
        }
        bool isThisHost = false;
        if (_thisSpec) {
            const auto &thisSpec = *_thisSpec;
            const auto &hostSpec = spec->spec;
            isThisHost = (("0.0.0.0" == hostSpec.ip || thisSpec.ip == hostSpec.ip) &&
                          (thisSpec.ports[MC_PROTOCOL_GRPC_STREAM] ==
                           hostSpec.ports[MC_PROTOCOL_GRPC_STREAM]));
        }
        auto info = std::make_shared<HostHeartbeatInfo>(_stringTable, isThisHost);
        info->init(_notifier, _heartbeatSnapshot, _searchService, addr, spec);
        newHostMap.emplace(addr, info);
        hasDiff = true;
    }
    if (!hasDiff) {
        if (oldHostMap) {
            if (oldHostMap->size() != newHostMap.size()) {
                hasDiff = true;
            }
        }
    }
    setHostMap(newHostMapPtr);
    if (hasDiff) {
        if (_thread) {
            _thread->runOnce();
        }
        _notifier->notify();
    }
    return true;
}

void HeartbeatClientManager::notifyHeartbeatSuccess() {
    auto hostMap = getHostMap();
    if (!hostMap) {
        return;
    }
    bool notify = true;
    for (const auto &pair : *hostMap) {
        if (!pair.second->isTopoReady()) {
            notify = false;
            break;
        }
    }
    if (notify && _manager) {
        _manager->notifyUpdate();
    }
}

void HeartbeatClientManager::loop() {
    if (_stringTable) {
        _stringTable->recycle(false);
    }
    auto hostMap = getHostMap();
    if (!hostMap) {
        return;
    }
    for (const auto &pair : *hostMap) {
        pair.second->tick();
    }
}

bool HeartbeatClientManager::fillBizInfoMap(BizInfoMap &bizInfoMap,
                                            multi_call::TopoSignatureSetPtr &localSubSignatures) {
    auto hostMap = getHostMap();
    if (!hostMap) {
        updateFlowControlHeartbeat(nullptr);
        return true;
    }
    bool hasLocalSub = false;
    auto flowControlMap = std::make_shared<BizFlowControlInfoMap>();
    for (const auto &pair : *hostMap) {
        const auto &hostInfo = pair.second;
        hasLocalSub = (hasLocalSub || hostInfo->isThisHost());
        if (!hostInfo->fillBizInfoMap(bizInfoMap, *flowControlMap, localSubSignatures)) {
            return false;
        }
    }
    if (!hasLocalSub) {
        localSubSignatures.reset();
    } else if (!localSubSignatures) {
        localSubSignatures = std::make_shared<TopoSignatureSet>();
    }
    updateFlowControlHeartbeat(flowControlMap);
    return true;
}

void HeartbeatClientManager::updateFlowControlHeartbeat(
    BizFlowControlInfoMapPtr newFlowControlInfoMap) {
    if (newFlowControlInfoMap && newFlowControlInfoMap->empty()) {
        newFlowControlInfoMap.reset();
    }
    auto isSame = (!_flowControlInfoMap && !newFlowControlInfoMap) ||
                  (_flowControlInfoMap && newFlowControlInfoMap &&
                   (*_flowControlInfoMap == *newFlowControlInfoMap));
    if (!isSame) {
        _searchService->updateFlowControlHeartbeat(newFlowControlInfoMap);
        _flowControlInfoMap = newFlowControlInfoMap;
    }
}

void HeartbeatClientManager::setHostMap(const HostHeartbeatInfoMapPtr &newMap) {
    autil::ScopedLock lock(_hostMapLock);
    _hostMap = newMap;
}

HostHeartbeatInfoMapPtr HeartbeatClientManager::getHostMap() const {
    autil::ScopedLock lock(_hostMapLock);
    return _hostMap;
}

void HeartbeatClientManager::log() {
    auto newSnapshotStr = toString();
    if (newSnapshotStr != _snapshotStr) {
        FileRecorder::recordSnapshot(newSnapshotStr, _miscConfig->snapshotLogCount,
                                     "gig_snapshot/heartbeat_client/" + _miscConfig->logPrefix,
                                     _miscConfig->snapshotLogDir);
        _snapshotStr.swap(newSnapshotStr);
    }
}

void HeartbeatClientManager::getSpecSet(std::set<std::string> &specSet) const {
    auto hostMapPtr = getHostMap();
    if (!hostMapPtr) {
        return;
    }
    const auto &hostMap = *hostMapPtr;
    for (const auto &pair : hostMap) {
        specSet.insert(pair.first);
    }
}

std::string HeartbeatClientManager::toString() const {
    auto hostMapPtr = getHostMap();
    if (!hostMapPtr) {
        return "";
    }
    const auto &hostMap = *hostMapPtr;
    std::string ret;
    ret += "hostCount: " + autil::StringUtil::toString(hostMap.size());
    ret += "\n";
    MetasSignatureMap allMetas;
    for (const auto &pair : hostMap) {
        pair.second->toString(ret, allMetas);
        ret += "\n";
    }
    ret += "metasCount: " + autil::StringUtil::toString(allMetas.size());
    ret += "\n";
    for (const auto &pair : allMetas) {
        ret += "  sig: " + autil::StringUtil::toString(pair.first);
        ret += "\n";
        const auto &metaMapPtr = pair.second;
        if (!metaMapPtr || metaMapPtr->empty()) {
            ret += "    (empty)\n";
            continue;
        }
        for (const auto &metaPair : *metaMapPtr) {
            ret += "    ";
            ret += autil::StringUtil::toString(metaPair.first);
            ret += " " + autil::StringUtil::toString(metaPair.second);
            ret += "\n";
        }
        ret += "\n";
    }
    return ret;
}

} // namespace multi_call
