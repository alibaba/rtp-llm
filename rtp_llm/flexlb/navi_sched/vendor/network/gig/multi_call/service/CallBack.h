#pragma once

#include <functional>

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/ErrorCode.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/interface/Response.h"

namespace multi_call {

class Caller;
typedef std::shared_ptr<Caller> CallerPtr;
class SearchServiceResource;
class QuerySessionContext;

class CallBack
{
public:
    CallBack(const std::shared_ptr<SearchServiceResource> &resource, const CallerPtr &caller,
             bool isRetry);
    virtual ~CallBack();

private:
    CallBack(const CallBack &);
    CallBack &operator=(const CallBack &);

public:
    // virtual for ut
    virtual void run(void *responseData, MultiCallErrorCode ec, const std::string &errorString,
                     const std::string &responseInfo);
    RequestType getRequestType() const;
    bool isCopyRequest() const;
    bool isRetry() const {
        return _isRetry;
    }
    const std::shared_ptr<SearchServiceResource> &getResource() const {
        return _resource;
    }
    
    
    void setProtobufArena(const std::shared_ptr<google::protobuf::Arena> &arena) {
        _arena = arena;
    }
    void setSessionContext(const std::shared_ptr<QuerySessionContext> &sessionContext) {
        _sessionContext = sessionContext;
    }
    void initRequestQueryInfo();
    void rpcBegin();
    void rpcEnd();
    void clearResource() {
        _resource.reset();
        _caller.reset();
    }

private:
    std::shared_ptr<google::protobuf::Arena> _arena;
    std::shared_ptr<QuerySessionContext> _sessionContext;
    std::shared_ptr<SearchServiceResource> _resource;
    CallerPtr _caller;
    bool _isRetry;
    bool _needRpcBegin;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(CallBack);

} // namespace multi_call
