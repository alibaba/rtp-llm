#pragma once

#include <functional>

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/rpc/GigRpcController.h"

namespace multi_call {

struct GrpcCallData;
class QuerySession;
class QueryInfo;
MULTI_CALL_TYPEDEF_PTR(QuerySession);
MULTI_CALL_TYPEDEF_PTR(QueryInfo);

class GigClosure : public google::protobuf::Closure
{
public:
    GigClosure();
    ~GigClosure();

public:
    const QuerySessionPtr &getQuerySession() const;
    void setQuerySession(const QuerySessionPtr &session);

    const QueryInfoPtr &getQueryInfo() const;
    void setQueryInfo(const QueryInfoPtr &queryInfo);

    virtual int64_t getStartTime() const;
    GigRpcController &getController();
    float degradeLevel(float level = 1.0) const;
    MultiCallErrorCode getErrorCode() const;
    void setErrorCode(MultiCallErrorCode ec);
    void setTargetWeight(WeightTy targetWeight);
    WeightTy getTargetWeight() const;
    RequestType getRequestType() const;
    virtual ProtocolType getProtocolType() = 0;




protected:
    void setCompatibleFieldInfo(const CompatibleFieldInfo *info);
    void fillCompatibleInfo(google::protobuf::Message *message, MultiCallErrorCode ec,
                            const std::string &responseInfo) const;
    void setNeedFinishQueryInfo(bool needFinish);

private:
    friend struct GrpcCallData;

protected:
    const CompatibleFieldInfo *_compatibleInfo;
    GigRpcController _controller;
    QueryInfoPtr _queryInfo;
    bool _needFinishQueryInfo;
    WeightTy _targetWeight;
    int64_t _startTime;

    QuerySessionPtr _querySession;
};

typedef std::function<void(google::protobuf::RpcController *controller,
                           google::protobuf::Message *request, google::protobuf::Message *response,
                           GigClosure *closure)>
    GigRpcMethod;

} // namespace multi_call

