#pragma once

#include "navi/common.h"
#include "navi/engine/NaviError.h"
#include "navi/log/NaviLogger.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/stream/GigStreamRpcInfo.h"

namespace navi {

class GraphResult;
class GraphDomainUser;
class NaviMessage;
class ComputeScope;

class NaviResult
{
public:
    NaviResult(const NaviLoggerPtr &logger);
public:
    bool hasError() const;
    ErrorCode getErrorCode() const;
    std::string getErrorMessage() const;
    std::string getErrorPrefix() const;
    std::string getErrorBt() const;
    LoggingEventPtr getErrorEvent() const;
    void collectTrace(std::vector<std::string> &traceVec) const;
    std::shared_ptr<GraphVisDef> getVisProto();
    std::shared_ptr<multi_call::GigStreamRpcInfoMap> getRpcInfoMap();
private:
    std::shared_ptr<GraphVisDef>
    getVisProto(const std::shared_ptr<ComputeScope> &computeScope);
    void setGraphResult(const std::shared_ptr<GraphResult> &result);
    const std::shared_ptr<GraphResult> &getGraphResult() const;
    bool toProto(bool partial,
                 NaviMessage &naviMessage,
                 const std::shared_ptr<ComputeScope> &computeScope);
public:
    static const char *getErrorString(ErrorCode ec);
public:
    SessionId id;
    NaviErrorPtr error;
    int64_t beginTime = 0;
    int64_t endTime = 0;
    int64_t queueUs = 0;
    int64_t computeUs = 0;
private:
    friend class NaviUserResult;
    friend class Navi;
    friend class NaviSnapshot;
    friend class DelaySendDestructClosure;
    friend class ArpcGraphClosure;
private:
    DECLARE_LOGGER();
    mutable autil::ThreadMutex _resultMutex;
    std::shared_ptr<GraphResult> _graphResult;
    std::shared_ptr<GraphDomainUser> _userDomain;
    std::shared_ptr<GraphVisDef> _visProto;
    std::shared_ptr<multi_call::GigStreamRpcInfoMap> _rpcInfoMap;
};

NAVI_TYPEDEF_PTR(NaviResult);

}
