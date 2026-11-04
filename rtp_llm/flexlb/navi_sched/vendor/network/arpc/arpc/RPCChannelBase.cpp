#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/RPCChannelBase.h"

#include <cstddef>
#include <stdint.h>
#include <unistd.h>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/advancepacket.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/connection.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/connectionpriority.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/ilogger.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/ipackethandler.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/packet.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/runnable.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/ANetRPCController.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/CommonMacros.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/MessageCodec.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/PacketArg.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/Tracer.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/anet/SharedClientPacketHandler.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/proto/rpc_extensions.pb.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/util/Log.h"

using namespace std;
using namespace anet;
ARPC_BEGIN_NAMESPACE(arpc);
ARPC_DECLARE_AND_SETUP_LOGGER(RPCChannelBase);

RPCChannelBase::RPCChannelBase() {
    _version = ARPC_VERSION_CURRENT;
    _enableTrace = false;
}

RPCChannelBase::~RPCChannelBase() {}

void RPCChannelBase::SetError(ANetRPCController *pController, ErrorCode errorCode) {
    pController->SetErrorCode(errorCode);
    pController->SetFailed(errorCodeToString(errorCode));
    pController->GetTracer().EndCallMethod(errorCode);
    return;
}

void RPCChannelBase::RunClosure(RPCClosure *pClosure) {
    if (pClosure != NULL) {
        pClosure->Run();
    }
}

void RPCChannelBase::SetTraceFlag(ANetRPCController *controller) {
    if (GetTraceFlag() && controller) {
        controller->SetTraceFlag(true);
    }
}

ARPC_END_NAMESPACE(arpc);
