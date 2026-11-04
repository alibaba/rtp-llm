#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/anet/ClientPacketHandler.h"

#include <assert.h>
#include <cstddef>
#include <google/protobuf/arena.h>
#include <google/protobuf/descriptor.h>
#include <memory>
#include <new>
#include <stdint.h>
#include <string>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/controlpacket.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/databufferserializable.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/delaydecodepacket.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/ilogger.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/ipackethandler.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/packet.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/runnable.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/ANetRPCChannel.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/ANetRPCController.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/CommonMacros.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/MessageCodec.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/MessageSerializable.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/PacketArg.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/RPCChannelBase.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/RPCMessageSerializable.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/RPCRawMessageSerializable.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/Tracer.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/UtilFun.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/proto/rpc_extensions.pb.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/util/Log.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"

using namespace std;
using namespace anet;
using namespace google::protobuf::io;
ARPC_BEGIN_NAMESPACE(arpc);
ARPC_DECLARE_AND_SETUP_LOGGER(ClientPacketHandler);

ClientPacketHandler::ClientPacketHandler() { _channel = NULL; }

ClientPacketHandler::~ClientPacketHandler() { _channel = NULL; }

IPacketHandler::HPRetCode ClientPacketHandler::handlePacket(anet::Packet *packet, void *args) {
    RpcReqArg *pArgs = (RpcReqArg *)args;

    if (!packet->isRegularPacket()) {
        ARPC_LOG(WARN, "receive control packet[%d]", ((ControlPacket *)packet)->getCommand());
        return handleCmdPacket(packet, pArgs);
    }

    pArgs->sController->GetTracer().BeginHandleResponse();

    ARPC_LOG(TRACE1, "channel pointer: %p, msg chid: %lu", _channel, packet->getChannelId());

    /* Check the error code and remote version.
     * Repost the request if necessary.
     */
    if (_channel && !_channel->CheckResponsePacket(packet, pArgs)) {
        return IPacketHandler::FREE_CHANNEL;
    }

    decodePacket(packet, pArgs);
    pArgs->sController->GetTracer().EndCallMethod(ARPC_ERROR_NONE);
    pArgs->sClosure->Run();
    delete pArgs;

    return IPacketHandler::FREE_CHANNEL;
}

IPacketHandler::HPRetCode ClientPacketHandler::handleCmdPacket(Packet *packet, RpcReqArg *pArgs) {
    ErrorCode errCode = ARPC_ERROR_NONE;

    switch (((ControlPacket *)packet)->getCommand()) {
    case ControlPacket::CMD_BAD_PACKET:
        errCode = ARPC_ERROR_BAD_PACKET;
        break;

    case ControlPacket::CMD_TIMEOUT_PACKET:
        errCode = ARPC_ERROR_TIMEOUT;
        break;

    case ControlPacket::CMD_CONNECTION_CLOSED:
        errCode = ARPC_ERROR_CONNECTION_CLOSED;
        break;

    default:
        errCode = ARPC_ERROR_UNKNOWN_PACKET;
        break;
    }

    pArgs->sController->SetErrorCode(errCode);
    pArgs->sController->SetFailed(errorCodeToString(errCode));
    pArgs->sClosure->Run();
    delete pArgs;

    return IPacketHandler::FREE_CHANNEL;
}

void ClientPacketHandler::decodePacket(anet::Packet *packet, RpcReqArg *pArgs) {
    TraceInfo *serverTraceInfo = NULL;
    auto controller = pArgs->sController;
    const Tracer &tracer = controller->GetTracer();

    auto arenaPtr = pArgs->sContext->arena;
    google::protobuf::Arena *arena = nullptr;
    if (pArgs->sContext->IsRawMsg()) {
        arena = arenaPtr.get();
    } else {
        arena = getArena(pArgs->sResponse, arenaPtr);
    }
    assert(arena);
    if (tracer.GetTraceFlag()) {
        serverTraceInfo = google::protobuf::Arena::CreateMessage<TraceInfo>(arena);
    }

    version_t version = packet->getPacketVersion();
    int32_t pcode = packet->getPcode();
    DataBufferSerializable *pSeri = createSerializable(packet, pArgs, serverTraceInfo, arena, arenaPtr);

    if (pSeri == NULL) {
        controller->SetErrorCode(ARPC_ERROR_NEW_NOTHROW);
        controller->SetFailed("create MessageSerializable failed.");
        packet->free();
        return;
    }

    DelayDecodePacket *delayDecodePacket = dynamic_cast<DelayDecodePacket *>(packet);
    assert(delayDecodePacket);
    delayDecodePacket->setContent(pSeri, true);

    bool ret = delayDecodePacket->decodeToContent();

    if (!ret) {
        controller->SetErrorCode(ARPC_ERROR_DECODE_PACKET);
        controller->SetFailed(errorCodeToString(ARPC_ERROR_DECODE_PACKET));
    } else {
        if (pcode == ARPC_ERROR_NO_RESPONSE_SENT) {
            /* This is a special error message sent from server, which means the
             * server does not intentionally send back any message when processing
             * this request.
             * In this case, we will log a warning message in client side. We do
             * not encourage such case. */
            ARPC_LOG(WARN,
                     "Server does not respond to msg chid %lu, which is"
                     " not normal case. Error msg: %s",
                     packet->getChannelId(),
                     errorCodeToString(ARPC_ERROR_NO_RESPONSE_SENT).c_str());
        }

        if (pcode != ARPC_ERROR_NONE) {

            ErrorMsg *errMsg = NULL;

            if (version == ARPC_VERSION_1 && tracer.GetTraceFlag()) {
                RPCMessageSerializable *serializable = dynamic_cast<RPCMessageSerializable *>(pSeri);
                assert(serializable);
                errMsg = (ErrorMsg *)serializable->getBody();
            } else {
                MessageSerializable *serializable = dynamic_cast<MessageSerializable *>(pSeri);
                assert(serializable);
                errMsg = (ErrorMsg *)serializable->getMessage();
            }

            controller->SetFailed(errMsg->error_msg());
            if (errMsg->error_code())
                pcode = errMsg->error_code();
            controller->SetErrorCode((ErrorCode)pcode);
        }
    }

    if (serverTraceInfo != NULL) {
        TraceInfo *clientTraceInfo = tracer.ExtractClientTraceInfo(arena);
        serverTraceInfo->MergeFrom(*clientTraceInfo);
        controller->SetTraceInfo(serverTraceInfo);
    }

    delayDecodePacket->free();
    return;
}

DataBufferSerializable *
ClientPacketHandler::createSerializable(anet::Packet *packet,
                                        RpcReqArg *pArgs,
                                        TraceInfo *traceInfo,
                                        google::protobuf::Arena *arena,
                                        const std::shared_ptr<google::protobuf::Arena> &arenaPtr) {
    DataBufferSerializable *pSeri = NULL;
    version_t version = packet->getPacketVersion();
    int32_t pcode = packet->getPcode();

    if (version == ARPC_VERSION_1 && traceInfo != NULL) {
        if (pcode != ARPC_ERROR_NONE) {
            ErrorMsg *errMsg = google::protobuf::Arena::CreateMessage<ErrorMsg>(arena);

            if (errMsg == NULL) {
                return NULL;
            }

            pSeri = new (nothrow) RPCMessageSerializable(traceInfo, errMsg, arenaPtr);
        } else {
            if (pArgs->sContext->IsRawMsg()) {
                pSeri = new (nothrow) RPCRawMessageSerializable(traceInfo, pArgs->rawResponse);
            } else {
                pSeri = new (nothrow) RPCMessageSerializable(traceInfo, pArgs->sResponse, arenaPtr);
            }
        }
    } else {
        if (pcode != ARPC_ERROR_NONE) {
            ErrorMsg *errMsg = google::protobuf::Arena::CreateMessage<ErrorMsg>(arena);

            if (errMsg == NULL) {
                return NULL;
            }

            pSeri = new (nothrow) MessageSerializable(errMsg, arenaPtr);
        } else {
            pSeri = new (nothrow) MessageSerializable(pArgs->sResponse, arenaPtr);
        }
    }

    return pSeri;
}

void ClientPacketHandler::repostPacket(anet::Packet *packet, void *args) { assert(false); }

RPCChannelBase *ClientPacketHandler::getChannel() { return _channel; }

ARPC_END_NAMESPACE(arpc);
