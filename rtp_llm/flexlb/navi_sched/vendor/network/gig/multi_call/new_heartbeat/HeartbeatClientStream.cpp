#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/new_heartbeat/HeartbeatClientStream.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/new_heartbeat/HeartbeatClientManager.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/new_heartbeat/HostHeartbeatInfo.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/new_heartbeat/ServerTopoMap.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/proto/NewHeartbeat.pb.h"

namespace multi_call {
AUTIL_LOG_SETUP(multi_call, HeartbeatClientStream);

HeartbeatClientStream::HeartbeatClientStream(const std::shared_ptr<HostHeartbeatStats> &hostStats,
                                             const ClientTopoInfoMapPtr &clientInfoMap,
                                             const HeartbeatSpecPtr &spec,
                                             const std::shared_ptr<StringTable> &stringTable)
    : GigClientStream(GIG_NEW_HEARTBEAT_METHOD_NAME, GIG_HEARTBEAT_METHOD_NAME)
    , _hostStats(hostStats)
    , _spec(spec)
    , _stringTable(stringTable)
    , _notifier(hostStats->notifier)
    , _clientInfoMap(clientInfoMap) {
    assert(_spec);
    assert(_stringTable);
    setDisableRetry(true);
    setDisableProbe(true);
    setTimeout(INFINITE_TIMEOUT);
    setSpec(_spec->spec);
    setForceStop(true);
    setAllowLack(false);
    AUTIL_LOG(DEBUG, "heartbeat client stream constructed [%p]", this);
}

HeartbeatClientStream::~HeartbeatClientStream() {
    AUTIL_LOG(DEBUG, "heartbeat client stream destructed [%p]", this);
}

google::protobuf::Message *
HeartbeatClientStream::newReceiveMessage(google::protobuf::Arena *arena) const {
    return google::protobuf::Arena::CreateMessage<NewHeartbeatResponse>(arena);
}

bool HeartbeatClientStream::receive(const GigStreamMessage &message) {
    auto response = dynamic_cast<NewHeartbeatResponse *>(message.message);
    if (!response) {
        AUTIL_LOG(ERROR, "null heartbeat msg");
        return false;
    }
    AUTIL_LOG(DEBUG, "message received: %s, netLatencyUs: %ld, host: %s",
              response->DebugString().c_str(), message.netLatencyUs,
              getSpec().getAnetSpec(MC_PROTOCOL_GRPC_STREAM).c_str());
    bool finished = false;
    bool changed = false;
    if (doReceive(*response, message.netLatencyUs, finished, changed)) {
        if (finished) {
            _hostStats->updateLastHeartbeatTime(true, changed);
        }
        return true;
    } else {
        _hostStats->updateLastHeartbeatTime(false, false);
        return false;
    }
}

void HeartbeatClientStream::receiveCancel(const GigStreamMessage &message, MultiCallErrorCode ec) {
    _hostStats->updateLastHeartbeatTime(false, false);
    AUTIL_LOG(WARN, "receive cancel, host [%s] disabled, this [%p], ec [%s]",
              getSpec().getAnetSpec(MC_PROTOCOL_GRPC_STREAM).c_str(), this, translateErrorCode(ec));
    disableProvider();
    _notifier->notify();
}

bool HeartbeatClientStream::tick() {
    if (0 != _sendLock.trylock()) {
        return true;
    }
    auto ret = doTick();
    _sendLock.unlock();
    if (ret) {
        setLackHashSet(nullptr);
    }
    return ret;
}

bool HeartbeatClientStream::doTick() {
    NewHeartbeatRequest request;
    auto id = fillRequest(request);
    if (send(0, false, &request)) {
        _hostStats->updateLastHeartbeatSendTime(id);
        AUTIL_LOG(DEBUG, "send gig heartbeat request success [%p] [%s]", this,
                  request.DebugString().c_str());
        return true;
    } else {
        _hostStats->updateLastHeartbeatTime(false, false);
        AUTIL_LOG(ERROR, "send gig heartbeat request failed, host [%s] [%p] [%s]",
                  getSpec().getAnetSpec(MC_PROTOCOL_GRPC_STREAM).c_str(), this,
                  request.DebugString().c_str());
        return false;
    }
}

void HeartbeatClientStream::disableProvider() {
    auto clientMap = getClientMap();
    if (clientMap) {
        clientMap->disableAllProvider();
        AUTIL_LOG(WARN, "server stopped or timeout, host [%s] disabled, this [%p]",
                  getSpec().getAnetSpec(MC_PROTOCOL_GRPC_STREAM).c_str(), this);
    }
}

int64_t HeartbeatClientStream::fillRequest(NewHeartbeatRequest &request) {
    auto id = _hostStats->totalCount;
    request.set_rpc_version(GIG_STREAM_HEARTBEAT_VERSION);
    request.set_heartbeat_id(id);
    ClientTopoInfoMapPtr clientInfoMap;
    LackHashSetPtr lackHashSet;
    {
        autil::ScopedReadWriteLock lock(_clientMapLock, 'r');
        clientInfoMap = _clientInfoMap;
        lackHashSet = _lackHashSet;
    }
    if (clientInfoMap) {
        clientInfoMap->fillRequest(request);
    }
    fillLackHash(lackHashSet, request);
    if (CNST_CLUSTER_NAME_AS_BIZ_NAME == _spec->clusterNameSubType) {
        request.set_only_default_biz(true);
    }
    return id;
}

void HeartbeatClientStream::fillLackHash(const LackHashSetPtr &lackHashSet,
                                         NewHeartbeatRequest &request) {
    auto hashRequest = request.mutable_hash_request();
    hashRequest->set_support(true);
    if (!lackHashSet || lackHashSet->empty()) {
        return;
    }
    for (auto hash : *lackHashSet) {
        hashRequest->add_hash_list(hash);
    }
}

bool HeartbeatClientStream::doReceive(const NewHeartbeatResponse &response, int64_t netLatencyUs,
                                      bool &finished, bool &changed) {
    auto oldMap = getClientMap();
    bool needNotify = false;
    _hostStats->updateHeartbeatResponseId(response.heartbeat_id());
    std::unordered_set<std::shared_ptr<std::string>> strSet;
    if (response.has_string_table()) {
        updateStringTable(response.string_table(), strSet);
    }
    HashInfoParam hashInfoParam;
    hashInfoParam.stringTable = _stringTable;
    if (oldMap && oldMap->update(response, netLatencyUs, hashInfoParam, needNotify)) {
        if (!hashInfoParam.lackHashSet) {
            if (needNotify) {
                _notifier->notify();
            }
            finished = true;
            changed = false;
            return true;
        }
    }
    auto newClientId = response.client_id();
    auto newMap = std::make_shared<ClientTopoInfoMap>(newClientId);
    if (!newMap->init(getSpec().ip, oldMap, response)) {
        return false;
    }
    const auto &spec = newMap->getServerId()->getSpec();
    auto count = response.topo_size();
    const auto &envDef = response.server_id().env();
    for (int32_t i = 0; i < count; i++) {
        const auto &topoDef = response.topo(i);
        const auto &signature = topoDef.signature();
        auto publishId = signature.publish_id();
        auto topoSig = signature.topo();
        auto oldTopoInfo = getOldInfo(oldMap, topoSig);
        if (oldTopoInfo && (publishId == oldTopoInfo->getPublishId())) {
            newMap->addInfo(topoSig, oldTopoInfo);
        } else {
            if (oldTopoInfo) {
                // disable setTargetWeight in old info destructor
                oldTopoInfo->unBind();
            }
            auto newInfo = std::make_shared<ClientTopoInfo>(_hostStats);
            if (newInfo->init(*_spec, spec, envDef, topoDef, netLatencyUs, hashInfoParam)) {
                newMap->addInfo(topoSig, newInfo);
            } else {
                AUTIL_LOG(ERROR,
                          "invalid topo info, this [%p], host [%s], response [%s], oldMap [%p]",
                          this, getSpec().getAnetSpec(MC_PROTOCOL_GRPC_STREAM).c_str(),
                          response.DebugString().c_str(), oldMap.get());
                return false;
            }
        }
    }
    if (hashInfoParam.lackHashSet) {
        setLackHashSet(hashInfoParam.lackHashSet);
        tick();
        finished = false;
    } else {
        oldMap.reset();
        setClientMap(newMap);
        _notifier->notify();
        finished = true;
        changed = true;
    }
    return true;
}

void HeartbeatClientStream::updateStringTable(
    const StringTableDef &def, std::unordered_set<std::shared_ptr<std::string>> &strSet) {
    auto count = def.item_size();
    auto &stringTable = *_stringTable;
    for (int32_t i = 0; i < count; i++) {
        const auto &itemDef = def.item(i);
        std::shared_ptr<std::string> strPtr;
        stringTable.addHash(itemDef.hash(), itemDef.value(), &strPtr);
        strSet.insert(std::move(strPtr));
    }
}

void HeartbeatClientStream::setClientMap(const ClientTopoInfoMapPtr &newMap) {
    autil::ScopedReadWriteLock lock(_clientMapLock, 'w');
    _clientInfoMap = newMap;
}

ClientTopoInfoMapPtr HeartbeatClientStream::getClientMap() const {
    autil::ScopedReadWriteLock lock(_clientMapLock, 'r');
    return _clientInfoMap;
}

void HeartbeatClientStream::setLackHashSet(const LackHashSetPtr &lackHashSet) {
    autil::ScopedReadWriteLock lock(_clientMapLock, 'w');
    _lackHashSet = lackHashSet;
}

ClientTopoInfoPtr HeartbeatClientStream::getOldInfo(const ClientTopoInfoMapPtr &clientMap,
                                                    SignatureTy topoSig) const {
    if (clientMap) {
        return clientMap->getInfo(topoSig);
    } else {
        return nullptr;
    }
}
} // namespace multi_call
