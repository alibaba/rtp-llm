#pragma once

#include "rtp_llm/flexlb/navi_sched/common/NaviSchedScheduleData.h"
#include "autil/Lock.h"
#include "navi/engine/AsyncPipe.h"
#include "navi/engine/Resource.h"
#include "navi/engine/NaviUserResult.h"
#include <cstdint>
#include <fstream>
#include <unordered_map>
#include <string>
#include <vector>

namespace navi {
class GraphDef;
class Navi;
class RunGraphParams;
}

namespace navi_sched::test {

class TrafficReplayRequestClosure;
class TrafficReplayR;

class TrafficReplayBridgeR : public navi::ExternalResource {
public:
    TrafficReplayBridgeR();

public:
    void setTrafficReplayR(const std::shared_ptr<TrafficReplayR> &resource);
    void clearTrafficReplayR(const TrafficReplayR *resource);
    std::shared_ptr<TrafficReplayR> waitTrafficReplayR(int64_t timeoutUs);

public:
    static const std::string RESOURCE_ID;

private:
    autil::ThreadCond _condition;
    std::shared_ptr<TrafficReplayR> _trafficReplayR;
};

NAVI_TYPEDEF_PTR(TrafficReplayBridgeR);

struct TrafficReplayConfig {
    std::string clientBizName;
    std::string trafficLog;
    std::string targetBizName;
    std::string masterBizName;
    std::string serviceName;
    size_t clientCount = 1;
    size_t clientIndex = 0;
    double timeCompression = 1.0;
    int64_t requestTimeoutMs = 5000;
    int64_t startDelayMs = 2000;
    int64_t blockSize = 256;
    std::string collectReplaySequences;
    std::string replayOutputPath;
    std::string latencyOutputPath;
};

struct TrafficReplayRequest {
    int64_t requestTimeMs = 0;
    size_t sequence = 0;
    size_t clientIndex = 0;
    uint64_t requestId = 0;
    std::string serviceName;
    int64_t promptTokenCount = 0;
    NaviSchedBlockHashesConstPtr blockHashes;
};

class TrafficReplayR : public navi::Resource {
public:
    TrafficReplayR() = default;
    ~TrafficReplayR() override;

public:
    void def(navi::ResourceDefBuilder &builder) const override;
    bool config(navi::ResourceConfigContext &ctx) override;
    navi::ErrorCode init(navi::ResourceInitContext &ctx) override;
    bool takeNext(TrafficReplayRequest &request);
    bool peekNext(TrafficReplayRequest &request) const;
    bool empty() const;
    size_t getRequestCount() const;
    int64_t getFirstRequestTimeMs() const;
    const std::string &getTargetBizName() const;
    const std::string &getMasterBizName() const;
    double getTimeCompression() const;
    int64_t getRequestTimeoutMs() const;
    size_t getClientIndex() const;
    size_t getClientCount() const;
    int64_t getStartDelayMs() const;
    int64_t getBlockSize() const;
    bool shouldCollectReplay(const TrafficReplayRequest &request) const;
    bool saveReplay(const NaviSchedScheduleReplayPtr &replay) const;
    void recordRequestTiming(uint64_t requestId,
                             const NaviSchedRequestTiming &requestTiming,
                             uint64_t scheduleId, double scheduleLoss,
                             int64_t scheduleLatencyUs);
    void logLatencyStats();
    bool setStartPipe(const navi::AsyncPipePtr &startPipe);
    void clearStartPipe(const navi::AsyncPipePtr &startPipe);
    bool start(int64_t startTimeMs, int64_t firstRequestTimeMs);
    bool getStartTime(int64_t &startTimeMs, int64_t &firstRequestTimeMs) const;
    bool setControlPipe(const navi::AsyncPipePtr &controlPipe);
    void clearControlPipe(const navi::AsyncPipePtr &controlPipe);
    bool setCompletionPipe(const navi::AsyncPipePtr &completionPipe);
    void clearCompletionPipe(const navi::AsyncPipePtr &completionPipe);
    void setNavi(navi::Navi *navi);
    void clearNavi(navi::Navi *navi);
    bool startRequest(navi::GraphDef *graphDef, const navi::RunGraphParams &params,
                      uint64_t requestId);
    int64_t getInflightRequestCount() const;
    bool finish();

public:
    static const std::string RESOURCE_ID;

private:
    bool loadTrafficLog();
    bool appendRecord(const std::string &json, size_t sequence);
    void finishRequest(TrafficReplayRequestClosure *closure,
                       const navi::NaviUserResultPtr &userResult);
    static double getPercentile(const std::vector<double> &values, double percentile);
    friend class TrafficReplayRequestClosure;

private:
    RESOURCE_DEPEND_DECLARE();

private:
    RESOURCE_DEPEND_ON_FALSE(TrafficReplayBridgeR, _trafficReplayBridgeR);
    TrafficReplayConfig _config;
    std::vector<TrafficReplayRequest> _requests;
    std::vector<size_t> _collectReplaySequences;
    size_t _nextRequest = 0;
    mutable autil::ThreadMutex _controlLock;
    mutable autil::ThreadMutex _replayOutputLock;
    mutable autil::ThreadMutex _latencyOutputLock;
    navi::AsyncPipePtr _controlPipe;
    navi::AsyncPipePtr _completionPipe;
    navi::AsyncPipePtr _startPipe;
    int64_t _startTimeMs = 0;
    int64_t _firstRequestTimeMs = 0;
    navi::Navi *_navi = nullptr;
    std::vector<std::shared_ptr<TrafficReplayRequestClosure>> _closures;
    std::unordered_map<uint64_t, int64_t> _requestStartTimesUs;
    int64_t _inflightRequestCount = 0;
    std::ofstream _latencyOutput;
    std::vector<double> _clientTotalLatencies;
    bool _latencyStatsLogged = false;
};

} // namespace navi_sched::test
