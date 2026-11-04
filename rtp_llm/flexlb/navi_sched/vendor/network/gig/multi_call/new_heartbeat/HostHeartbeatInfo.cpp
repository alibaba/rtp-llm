#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/new_heartbeat/HostHeartbeatInfo.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/interface/SearchService.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/new_heartbeat/HeartbeatClientManager.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/proto/NewHeartbeat.pb.h"
#include "multi_call/new_heartbeat/HeartbeatClientStream.h"

namespace multi_call {
AUTIL_LOG_SETUP(multi_call, HostHeartbeatStats);
AUTIL_LOG_SETUP(multi_call, HostHeartbeatInfo);

void HostHeartbeatStats::updateLastHeartbeatSendTime(int64_t requestId) {
    if (0 == lastHeartbeatSendTime) {
        lastRequestId = requestId;
        lastHeartbeatSendTime = autil::TimeUtility::currentTime();
    }
}

void HostHeartbeatStats::updateLastResponseTime() {
    lastResponseTime = autil::TimeUtility::currentTime();
}

void HostHeartbeatStats::updateLastHeartbeatTime(bool isSuccess, bool changed) {
    AUTIL_LOG(DEBUG, "heartbeat completed for host [%s]", addr.c_str());
    success = isSuccess;
    bool needNotify = (!topoReady) || (isThisHost && changed);
    topoReady = true;
    subscribeTimeout = false;
    lastHeartbeatTime = autil::TimeUtility::currentTime();
    if (needNotify && notifier) {
        notifier->notifyHeartbeatSuccess();
    }
}

void HostHeartbeatStats::updateHeartbeatResponseId(int64_t responseId) {
    lastResponseId = responseId;
}

bool HostHeartbeatStats::isHeartbeatTimeout() {
    auto lastSend = lastHeartbeatSendTime;
    if (lastSend <= 0) {
        return false;
    }
    auto now = autil::TimeUtility::currentTime();
    if (lastSend + GIG_HEARTBEAT_TIMEOUT > now) {
        // wait
        return false;
    }
    lastHeartbeatSendTime = 0;
    if (lastResponseId > 0) {
        lastCheckSuccess = (success && (lastResponseId >= lastRequestId));
    } else {
        auto lastEcho = lastHeartbeatTime;
        lastCheckSuccess = (success && (lastEcho >= lastSend));
    }
    return !lastCheckSuccess;
}

bool HostHeartbeatStats::checkSubscribeTimeout(int64_t currentTime) {
    if (firstHeartbeatTime > 0) {
        auto diff = currentTime - firstHeartbeatTime;
        subscribeTimeout = (diff > GIG_HEARTBEAT_SUBSCRIBE_TIMEOUT);
    }
    return subscribeTimeout;
}

void HostHeartbeatStats::incTotalCount() {
    if (0 == totalCount) {
        firstHeartbeatTime = autil::TimeUtility::currentTime();
    }
    totalCount++;
}

bool HostHeartbeatStats::skip(int64_t currentTime) {
    if (0 == totalCount) {
        return false;
    }
    if (!lastCheckSuccess) {
        return false;
    }
    if (ignoreNextSkip) {
        ignoreNextSkip = false;
        return false;
    }
    if ((0 == lastHeartbeatTime) || !success) {
        skipCount = 0;
        return false;
    }
    auto heartbeatDiff = currentTime - lastHeartbeatTime;
    if (heartbeatDiff < GIG_HEARTBEAT_SKIP_LOW) {
        return true;
    }
    if (0 == lastResponseTime) {
        return skipByCount();
    }
    auto queryDiff = currentTime - lastResponseTime;
    if (queryDiff < GIG_HEARTBEAT_SKIP_LOW) {
        return skipByCount();
    }
    if (queryDiff > GIG_HEARTBEAT_SKIP_HIGH) {
        return skipByCount();
    }
    skipCount = 0;
    return false;
}

bool HostHeartbeatStats::skipByCount() {
    skipCount++;
    if (skipCount > 30) {
        skipCount = 0;
        return false;
    }
    return true;
}

HostHeartbeatInfo::HostHeartbeatInfo(const std::shared_ptr<StringTable> &stringTable,
                                     bool isThisHost)
    : _stringTable(stringTable)
    , _searchService(nullptr)
    , _stats(std::make_shared<HostHeartbeatStats>(isThisHost))
    , _isThisHost(isThisHost) {
    assert(_stringTable);
}

HostHeartbeatInfo::~HostHeartbeatInfo() {
    stop();
}

void HostHeartbeatInfo::stop() {
    auto stream = getStream();
    if (stream) {
        stream->sendCancel(0, nullptr);
        setStream(nullptr);
    }
}

bool HostHeartbeatInfo::init(const std::shared_ptr<HeartbeatClientManagerNotifier> &notifier,
                             const std::shared_ptr<SearchServiceSnapshot> &heartbeatSnapshot,
                             SearchService *searchService, const std::string &addr,
                             const HeartbeatSpecPtr &spec) {
    if (getStream()) {
        return true;
    }
    if (!searchService) {
        return false;
    }
    _stats->notifier = notifier;
    _heartbeatSnapshot = heartbeatSnapshot;
    _searchService = searchService;
    assert(spec);
    _spec = spec;
    _stats->addr = addr;
    if (!createStream()) {
        _stats->updateLastHeartbeatTime(false, false);
        return false;
    }
    if (!tick()) {
        _stats->updateLastHeartbeatTime(false, false);
        return false;
    }
    return true;
}

bool HostHeartbeatInfo::createStream() {
    auto stream = newClientStream();
    if (!_searchService->bind(stream, _heartbeatSnapshot)) {
        AUTIL_LOG(ERROR, "bind heartbeat stream failed, spec[%s]", _stats->addr.c_str());
        return false;
    }
    setStream(stream);
    return true;
}

std::shared_ptr<HeartbeatClientStream> HostHeartbeatInfo::newClientStream() const {
    ClientTopoInfoMapPtr oldClientMap;
    auto oldStream = getStream();
    if (oldStream) {
        oldClientMap = oldStream->getClientMap();
    }
    return std::make_shared<HeartbeatClientStream>(_stats, oldClientMap, _spec, _stringTable);
}

bool HostHeartbeatInfo::tick() {
    checkHeartbeatTimeout();
    if (_stats->skip(autil::TimeUtility::currentTime())) {
        return true;
    }
    _stats->incTotalCount();
    auto stream = getStream();
    if (!stream) {
        if (!createStream()) {
            AUTIL_LOG(ERROR, "reconnect heartbeat stream failed");
            return false;
        }
    }
    stream = getStream();
    if (!stream) {
        return false;
    }
    if (!stream->tick()) {
        stream->sendCancel(0, nullptr);
        createStream();
        return false;
    }
    return true;
}

void HostHeartbeatInfo::checkHeartbeatTimeout() {
    if (_stats->isHeartbeatTimeout()) {
        auto stream = getStream();
        if (stream) {
            stream->disableProvider();
        }
    }
}

bool HostHeartbeatInfo::fillBizInfoMap(BizInfoMap &bizInfoMap,
                                       BizFlowControlInfoMap &flowControlMap,
                                       multi_call::TopoSignatureSetPtr &localSubSignatures) const {
    if (!_stats->topoReady) {
        if (_stats->checkSubscribeTimeout(autil::TimeUtility::currentTime())) {
            AUTIL_LOG(WARN, "heartbeat subscribe timeout for host [%s], ignored",
                      _stats->addr.c_str());
            return true;
        }
        AUTIL_LOG(INFO, "heartbeat not completed for host [%s]", _stats->addr.c_str());
        return false;
    }
    auto stream = getStream();
    if (!stream) {
        return true;
    }
    auto clientInfoMap = stream->getClientMap();
    if (!clientInfoMap) {
        return true;
    }
    clientInfoMap->fillBizInfoMap(*_spec, bizInfoMap, flowControlMap);
    if (_isThisHost) {
        if (!localSubSignatures) {
            localSubSignatures = std::make_shared<TopoSignatureSet>();
        }
        clientInfoMap->fillTopoSignatures(*localSubSignatures);
    }
    return true;
}

bool HostHeartbeatInfo::isTopoReady() const {
    return _stats->topoReady;
}

void HostHeartbeatInfo::setStream(const HeartbeatClientStreamPtr &stream) {
    autil::ScopedReadWriteLock lock(_streamLock, 'w');
    _stream = stream;
}

HeartbeatClientStreamPtr HostHeartbeatInfo::getStream() const {
    autil::ScopedReadWriteLock lock(_streamLock, 'r');
    return _stream;
}

void HostHeartbeatInfo::toString(std::string &debugStr, MetasSignatureMap &allMetas) const {
    debugStr += _stats->addr;
    debugStr += ", topoReady: " + autil::StringUtil::toString(_stats->topoReady);
    if (_isThisHost) {
        debugStr += ", isThisHost: " + autil::StringUtil::toString(_isThisHost);
    }
    debugStr += ", success: " + autil::StringUtil::toString(_stats->success);
    debugStr += ", subTimeout: " + autil::StringUtil::toString(_stats->subscribeTimeout);
    debugStr += ", lastCheckSuccess: " + autil::StringUtil::toString(_stats->lastCheckSuccess);
    debugStr += ", lastHeartbeatSend: " + getTimeStrFromUs(_stats->lastHeartbeatSendTime);
    debugStr += ", lastHeartbeat: " + getTimeStrFromUs(_stats->lastHeartbeatTime);
    debugStr += ", lastResponse: " + getTimeStrFromUs(_stats->lastResponseTime);
    debugStr += ", lastRequestId: " + autil::StringUtil::toString(_stats->lastRequestId);
    debugStr += ", lastResponseId: " + autil::StringUtil::toString(_stats->lastResponseId);
    debugStr += ", totalCount: " + autil::StringUtil::toString(_stats->totalCount);
    debugStr += ", skipCount: " + autil::StringUtil::toString(_stats->skipCount);
    auto stream = getStream();
    if (!stream) {
        return;
    }
    auto clientMap = stream->getClientMap();
    if (clientMap) {
        clientMap->toString(debugStr, allMetas);
    }
    debugStr += "\n";
}
} // namespace multi_call
