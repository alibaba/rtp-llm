#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/CallBack.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/controlpacket.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/Caller.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/ChildNodeReply.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/SearchServiceResource.h"

using namespace std;

namespace multi_call {
AUTIL_LOG_SETUP(multi_call, CallBack);

CallBack::CallBack(const SearchServiceResourcePtr &resource, const CallerPtr &caller, bool isRetry)
    : _resource(resource)
    , _caller(caller)
    , _isRetry(isRetry)
    , _needRpcBegin(true) {
}

CallBack::~CallBack() {
    _caller.reset();
    _resource.reset();
}

RequestType CallBack::getRequestType() const {
    return _resource->getRequestType();
}

bool CallBack::isCopyRequest() const {
    return _resource->isCopyRequest();
}

void CallBack::initRequestQueryInfo() {
    if (_resource && _sessionContext) {
        _resource->fillRequestQueryInfo(_isRetry, _sessionContext->sessionRequestType());
    }
}

void CallBack::rpcBegin() {
    if (!_needRpcBegin) {
        return;
    }
    _needRpcBegin = false;
    const auto &response = _resource->getResponse(_isRetry);
    response->rpcBegin();
    const auto &provider = _resource->getProvider(_isRetry);
    if (provider) {
        provider->updateRequestCounter(_isRetry, getRequestType());
    }
}

void CallBack::rpcEnd() {
    const auto &response = _resource->getResponse(_isRetry);
    response->callEnd();
    const auto &provider = _resource->getProvider(_isRetry);
    if (provider) {
        provider->updateResponseCounter();
    }
}

void CallBack::run(void *responseData, MultiCallErrorCode ec, const string &errorString,
                   const string &responseInfo) {
    // copy request should drop direct
    if (unlikely(isCopyRequest())) {
        return;
    }

    assert(_resource);
    auto response = _resource->getResponse(_isRetry);
    response->setCallBegTime(_resource->getCallBegTime());
    response->callEnd();
    response->setErrorCode(ec, errorString);
    if (!responseInfo.empty()) {
        response->setAgentInfo(responseInfo);
    }
    response->init(responseData);
    _resource->updateHealthStatus(_isRetry);
    if (unlikely(!_resource->isNormalRequest())) {
        return;
    }
    
    auto isFailed = response->isFailed();
    if (!isFailed) {
        const auto &reply = _caller->getReply();
        reply->updateDetectInfo(_resource->getBizName(), _resource->retryEnabled());
    }
    auto responsePtr = response.get();
    response.reset();
    MULTI_CALL_MEMORY_BARRIER();
    responsePtr->setReturned(); // after this, response can be stealed
    if (_isRetry) {
        responsePtr->setRetried();
    }
    auto index = _resource->getPartIdIndex();
    _resource.reset();
    auto caller = std::move(_caller);
    caller->decNotifier(index);
}

} // namespace multi_call
