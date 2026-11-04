#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/Caller.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/CallBack.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/SearchServiceProvider.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/SearchServiceResource.h"
#include "autil/StringUtil.h"

using namespace std;
using namespace autil;

namespace multi_call {
AUTIL_LOG_SETUP(multi_call, Caller);

Caller::Caller(const ChildNodeReplyPtr &reply,
               const std::shared_ptr<QuerySessionContext> &sessionContext)
    : _hasClosure(false)
    , _stopped(false)
    , _reply(reply)
    , _sessionContext(sessionContext) {
}

Caller::~Caller() {
}

void Caller::call(const SearchServiceResourcePtr &resource, bool isRetry) {
    RequestPtr request = resource->getRequest();
    auto isGrpcStream = (MC_PROTOCOL_GRPC_STREAM == request->getProtocolType());
    if (!isGrpcStream) {
        resource->fillRequestQueryInfo(isRetry, _sessionContext->sessionRequestType());
    }
    PartIdTy partIdIndex = -1;
    PartIdTy partId = resource->getPartId();
    if (_partIdToIndexMap.count(partId) == 0) {
        partIdIndex = _partIdToIndexMap.size();
        _partIdToIndexMap[partId] = partIdIndex;
    } else {
        partIdIndex = _partIdToIndexMap[partId];
    }
    resource->setPartIdIndex(partIdIndex);
    if (!isRetry && resource->isNormalRequest()) {
        _notifier.inc(partIdIndex);
    }
    CallBackPtr callBack(new CallBack(resource, shared_from_this(), isRetry));
    callBack->setProtobufArena(request->getProtobufArena());
    if (isGrpcStream) {
        callBack->setSessionContext(_sessionContext);
    }
    SearchServiceProviderPtr provider = resource->getProvider(isRetry);
    auto ret = provider->post(request, callBack);
    resource->setHasError(!ret);
}

void Caller::earlyTerminate() {
    Closure *closure = _notifier.stealClosure();
    if (closure) {
        closure->Run();
    }
}

} // namespace multi_call
