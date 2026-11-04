#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/anet/SharedClientPacketHandler.h"

#include <stddef.h>

#include "rtp_llm/flexlb/navi_sched/vendor/autil/autil/Lock.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/controlpacket.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/ilogger.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/packet.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/runnable.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/ANetRPCChannel.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/ANetRPCController.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/MessageCodec.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/PacketArg.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/RPCChannelBase.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/Tracer.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/util/Log.h"

using namespace anet;

ARPC_BEGIN_NAMESPACE(arpc);
ARPC_DECLARE_AND_SETUP_LOGGER(SharedClientPacketHandler);

SharedClientPacketHandler::SharedClientPacketHandler() {
    _ref = 0;
    _func = NULL;
}

SharedClientPacketHandler::~SharedClientPacketHandler() {}

IPacketHandler::HPRetCode SharedClientPacketHandler::handlePacket(Packet *packet, void *args) {
    ARPC_LOG(DEBUG, "handle package in shared client package handler");
    if (_func != NULL) {
        (*_func)();
    }
    IPacketHandler::HPRetCode ret = doHandlePacket(packet, args);
    subRef();
    return ret;
}

IPacketHandler::HPRetCode SharedClientPacketHandler::doHandlePacket(Packet *packet, void *args) {
    RpcReqArg *pArgs = (RpcReqArg *)args;
    if (!packet->isRegularPacket()) {
        ARPC_LOG(DEBUG, "receive control packet[%d]", ((ControlPacket *)packet)->getCommand());
        return handleCmdPacket(packet, pArgs);
    }

    pArgs->sController->GetTracer().BeginHandleResponse();

    ARPC_LOG(TRACE1, "channel pointer: %p, msg chid: %lu", _channel, packet->getChannelId());
    {
        /* Check the error code and remote version.
         * Repost the request if necessary.
         */
        autil::ScopedLock lock(_channelLock);
        if (_channel && !_channel->CheckResponsePacket(packet, pArgs)) {
            return IPacketHandler::FREE_CHANNEL;
        }
    }

    decodePacket(packet, pArgs);
    pArgs->sController->GetTracer().EndCallMethod(ARPC_ERROR_NONE);
    pArgs->sClosure->Run();
    delete pArgs;

    return IPacketHandler::FREE_CHANNEL;
}

void SharedClientPacketHandler::cleanChannel() {
    autil::ScopedLock lock(_channelLock);
    setChannel(NULL);
}

void SharedClientPacketHandler::addRef() {
    autil::ScopedLock lock(_mutex);
    _ref++;
    ARPC_LOG(DEBUG, "add ref, ref count:%ld.", _ref);
}

void SharedClientPacketHandler::subRef() {
    int64_t ref = 0;
    {
        autil::ScopedLock lock(_mutex);
        ref = --_ref;
    }

    ARPC_LOG(DEBUG, "sub ref, ref count:%ld.", ref);
    if (ref == 0) {
        delete this;
    }
}

void SharedClientPacketHandler::setHandlePacketPreFunc(const std::function<void()> *func) { _func = func; }

ARPC_END_NAMESPACE(arpc);
