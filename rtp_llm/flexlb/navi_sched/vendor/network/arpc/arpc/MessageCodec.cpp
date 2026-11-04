#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/MessageCodec.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/MessageSerializable.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/PacketArg.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/RPCMessageSerializable.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/Tracer.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/util/Log.h"

ARPC_BEGIN_NAMESPACE(arpc);
ARPC_DECLARE_AND_SETUP_LOGGER(MessageCodec);

MessageCodec::MessageCodec() {}

MessageCodec::~MessageCodec() {}

CallId MessageCodec::GenerateCallId(const RPCMethodDescriptor *method, version_t version) const {
    CallId callId;
    callId.intId = PacketCodeBuilder()(method);
    return callId;
}

uint32_t MessageCodec::GenerateCallId(uint32_t serviceId, uint32_t methodId) const {
    return PacketCodeBuilder()(serviceId, methodId);
}

ARPC_END_NAMESPACE(arpc);
