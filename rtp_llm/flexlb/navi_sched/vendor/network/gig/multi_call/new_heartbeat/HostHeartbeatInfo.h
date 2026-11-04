#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/WorkerNodeInfo.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/new_heartbeat/BizTopo.h"

namespace multi_call {

class SearchService;
class SearchServiceSnapshot;
class NewHeartbeatResponse;
class HeartbeatClientManagerNotifier;
class HeartbeatClientStream;

class HostHeartbeatStats
{
public:
    HostHeartbeatStats(bool isThisHost_ = false) : isThisHost(isThisHost_) {
    }

public:
    void updateLastHeartbeatSendTime(int64_t requestId);
    void updateLastResponseTime();
    void updateLastHeartbeatTime(bool isSuccess, bool changed = false);
    void updateHeartbeatResponseId(int64_t responseId);
    bool isHeartbeatTimeout();
    bool checkSubscribeTimeout(int64_t currentTime);
    bool skip(int64_t currentTime);
    void incTotalCount();

private:
    bool skipByCount();

public:
    std::string addr;
    std::shared_ptr<HeartbeatClientManagerNotifier> notifier;
    bool isThisHost = false;
    bool success = false;
    bool topoReady = false;
    bool subscribeTimeout = false;
    bool lastCheckSuccess = false;
    bool ignoreNextSkip = false;
    int64_t firstHeartbeatTime = 0;
    int64_t lastHeartbeatSendTime = 0;
    int64_t lastHeartbeatTime = 0;
    int64_t lastResponseTime = 0;
    int64_t lastRequestId = 0;
    int64_t lastResponseId = 0;
    int64_t skipCount = 0;
    int64_t totalCount = 0;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(HostHeartbeatStats);

class HostHeartbeatInfo
{
public:
    HostHeartbeatInfo(const std::shared_ptr<StringTable> &stringTable, bool isThisHost = false);
    ~HostHeartbeatInfo();

private:
    HostHeartbeatInfo(const HostHeartbeatInfo &);
    HostHeartbeatInfo &operator=(const HostHeartbeatInfo &);

public:
    bool isThisHost() const {
        return _isThisHost;
    }
    bool init(const std::shared_ptr<HeartbeatClientManagerNotifier> &notifier,
              const std::shared_ptr<SearchServiceSnapshot> &heartbeatSnapshot,
              SearchService *searchService, const std::string &addr, const HeartbeatSpecPtr &spec);
    const HeartbeatSpecPtr &getSpec() const {
        return _spec;
    }
    bool tick();
    void stop();
    bool fillBizInfoMap(BizInfoMap &bizInfoMap, BizFlowControlInfoMap &flowControlMap,
                        multi_call::TopoSignatureSetPtr &localSubSignatures) const;
    void toString(std::string &debugStr, MetasSignatureMap &allMetas) const;
    bool isTopoReady() const;

private:
    void checkHeartbeatTimeout();
    bool createStream();
    std::shared_ptr<HeartbeatClientStream> newClientStream() const;
    std::shared_ptr<HeartbeatClientStream> getStream() const;
    void setStream(const std::shared_ptr<HeartbeatClientStream> &stream);

private:
    std::shared_ptr<StringTable> _stringTable;
    std::shared_ptr<SearchServiceSnapshot> _heartbeatSnapshot;
    SearchService *_searchService;
    HeartbeatSpecPtr _spec;
    mutable autil::ReadWriteLock _streamLock;
    std::shared_ptr<HeartbeatClientStream> _stream;
    HostHeartbeatStatsPtr _stats;
    bool _isThisHost = false;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(HostHeartbeatInfo);

} // namespace multi_call
