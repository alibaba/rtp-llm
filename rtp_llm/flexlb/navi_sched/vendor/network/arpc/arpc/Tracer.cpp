#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/Tracer.h"

#include <assert.h>
#include <google/protobuf/arena.h>

#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/proto/rpc_extensions.pb.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/util/Log.h"

ARPC_BEGIN_NAMESPACE(arpc);
ARPC_DECLARE_AND_SETUP_LOGGER(Tracer);

Tracer::Tracer(bool enableTrace) {
    _enableTrace = enableTrace;
    _clientTimeout = 0;
    _serverQueueSize = 0;
    _beginHandleRequest = 0;
    _endHandleRequest = 0;
    _beginPostResponse = 0;
    _beginWorkItemProcess = 0;
}

Tracer::~Tracer() {}

TraceInfo *Tracer::ExtractClientTraceInfo(google::protobuf::Arena *arena) const {
    assert(arena);
    TraceInfo *traceInfo = google::protobuf::Arena::CreateMessage<TraceInfo>(arena);
    traceInfo->set_rtt(calcTime(_beginPostRequest, _beginHandleResponse));
    return traceInfo;
}

TraceInfo *Tracer::ExtractServerTraceInfo(google::protobuf::Arena *arena) const {
    assert(arena);
    TraceInfo *traceInfo = google::protobuf::Arena::CreateMessage<TraceInfo>(arena);
    traceInfo->set_serverqueuesize(_serverQueueSize);
    traceInfo->set_handlerequesttime(calcTime(_beginHandleRequest, _endHandleRequest));
    traceInfo->set_workitemwaitprocesstime(calcTime(_endHandleRequest, _beginWorkItemProcess));
    traceInfo->set_workitemprocesstime(calcTime(_beginWorkItemProcess, _beginPostResponse));
    traceInfo->set_requestonservertime(calcTime(_beginHandleRequest, _beginPostResponse));

    traceInfo->set_userpayload(_userPayload);

    return traceInfo;
}

ARPC_END_NAMESPACE(arpc);
