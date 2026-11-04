#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/RPCServerClosure.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/connection.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/ilogger.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/packet.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/runnable.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/MessageCodec.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/MessageSerializable.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/Tracer.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/UtilFun.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/proto/rpc_extensions.pb.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/util/Log.h"

using namespace anet;
ARPC_BEGIN_NAMESPACE(arpc);
ARPC_DECLARE_AND_SETUP_LOGGER(RPCServerClosure);

RPCServerClosure::RPCServerClosure(RPCMessage *reqMsg, RPCMessage *resMsg)
    : _reqMsg(reqMsg), _resMsg(resMsg), _tracer(nullptr) {}

RPCServerClosure::~RPCServerClosure() {
    _reqMsg = NULL;
    if (_tracer) {
        delete _tracer;
        _tracer = NULL;
    }
}

void RPCServerClosure::Run() {
    if (_tracer) {
        int32_t timeout = _tracer->GetClientTimeout();
        _tracer->BeginPostResponse();
        if (timeout > 0 &&
            _tracer->GetBeginPostResponse() - _tracer->GetBeginHandleRequest() > (int64_t)timeout * 1000) {
            ARPC_LOG(ERROR,
                     "Service Timeout: %s request recv: %ld, request process start: %ld, request process end: %ld, "
                     "timeout: %d(ms)",
                     getIpAndPortAddr().c_str(),
                     _tracer->GetBeginHandleRequest(),
                     _tracer->GetBeginWorkItemProcess(),
                     _tracer->GetBeginPostResponse(),
                     _tracer->GetClientTimeout());
        }
    }
    doPostPacket();
    if (_tracer) {
        _tracer->EndHandleRequest(ARPC_ERROR_NONE);
    }
    _arena.reset();
    release();
}

ARPC_END_NAMESPACE(arpc);
