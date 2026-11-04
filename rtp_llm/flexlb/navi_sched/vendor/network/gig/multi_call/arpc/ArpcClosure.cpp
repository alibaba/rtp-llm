#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/arpc/ArpcClosure.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/RPCServerClosure.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/agent/QueryInfo.h"

using namespace std;

namespace multi_call {
AUTIL_LOG_SETUP(multi_call, ArpcClosure);

int64_t ArpcClosure::getStartTime() const {
    auto tracer = _closure->GetTracer();
    assert(tracer);
    return tracer->GetBeginHandleRequest();
}

void ArpcClosure::Run() {
    auto tracer = _closure->GetTracer();
    assert(tracer);
    auto beginTime = tracer->GetBeginHandleRequest();
    auto latency = (autil::TimeUtility::currentTime() - beginTime) / FACTOR_US_TO_MS;
    auto responseInfoStr =
        _queryInfo->finish(latency, _controller.getErrorCode(), getTargetWeight());
    tracer->setUserPayload(responseInfoStr);
    fillCompatibleInfo(_closure->GetResponseMsg(), _controller.getErrorCode(), responseInfoStr);
    _closure->Run();
    delete this;
}

ProtocolType ArpcClosure::getProtocolType() {
    return MC_PROTOCOL_ARPC;
}

const std::shared_ptr<google::protobuf::Arena> &ArpcClosure::getProtobufArena() const {
    return _closure->getProtobufArena();
}

} // namespace multi_call
