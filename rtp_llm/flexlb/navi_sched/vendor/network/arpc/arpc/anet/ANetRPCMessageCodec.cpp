#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/anet/ANetRPCMessageCodec.h"

#include <assert.h>
#include <cstddef>
#include <google/protobuf/arena.h>
#include <google/protobuf/message.h>
#include <memory>
#include <new>
#include <stdint.h>
#include <string>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/advancepacket.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/databufferserializable.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/delaydecodepacket.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/ilogger.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/ipacketfactory.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/packet.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/runnable.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/MessageSerializable.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/PacketArg.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/RPCMessageSerializable.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/RPCRawMessageSerializable.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/RPCServer.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/proto/msg_header.pb.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/proto/rpc_extensions.pb.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/util/Log.h"

using namespace anet;
using namespace std;
ARPC_BEGIN_NAMESPACE(arpc);
ARPC_DECLARE_AND_SETUP_LOGGER(ANetRPCMessageCodec);

ANetRPCMessageCodec::ANetRPCMessageCodec(IPacketFactory *packetFactory) : _packetFactory(packetFactory) {}

ANetRPCMessageCodec::~ANetRPCMessageCodec() {}

Packet *ANetRPCMessageCodec::EncodeRequest(const CodecContext *context, version_t version) const {
    assert(context);
    DataBufferSerializable *serializable = NULL;
    uint32_t pcode = 0;

    if (version == ARPC_VERSION_0) {
        pcode = context->callId.intId;
        serializable = new (nothrow) MessageSerializable((RPCMessage *)context->request, context->arena);
    } else if (version == ARPC_VERSION_1) {
        pcode = context->callId.intId;

        if (pcode & ADVANCE_PACKET_MASK) {
            ARPC_LOG(ERROR, "Invalid callId [%ud]", pcode);
            return NULL;
        }

        pcode |= ADVANCE_PACKET_MASK;

        auto arena = getArena(context->request, context->arena);
        assert(arena);
        RpcMsgHeader *msgHeader = google::protobuf::Arena::CreateMessage<RpcMsgHeader>(arena);
        msgHeader->set_enabletrace(context->enableTrace);
        msgHeader->set_userpayload(context->userPayload);

        if (context->qosId != 0)
            msgHeader->set_group(context->qosId);

        if (context->IsRawMsg()) {
            serializable = new (nothrow) RPCRawMessageSerializable(msgHeader, context->rawRequest);
        } else {
            serializable = new (nothrow) RPCMessageSerializable(msgHeader, context->request, context->arena);
        }

    } else {
        ARPC_LOG(ERROR, "Invalid version [%d]", version);
        return NULL;
    }

    if (serializable == NULL) {
        ARPC_LOG(ERROR, "new MessageSerializable return NULL");
        return NULL;
    }

    auto packetDataLen = serializable->getSerializedSize();
    Packet *packet = createPacket(pcode, serializable);

    if (packet == NULL) {
        delete serializable;
        return NULL;
    }

    packet->setPacketVersion(version);
    packet->setDataLen(packetDataLen);
    return packet;
}

CodecContext *ANetRPCMessageCodec::DecodeRequest(Packet *packet, const RPCServer *rpcServer, version_t version) const {
    CodecContext *context = new CodecContext;

    if (version > ARPC_VERSION_CURRENT) {
        context->errorCode = ARPC_ERROR_INVALID_VERSION;
        return context;
    }

    CallId callId;
    callId.intId = packet->getPacketHeader()->_pcode;
    RPCServer::ServiceMethodPair serviceMethodPair = rpcServer->GetRpcCall(callId);
    RPCService *rpcService = serviceMethodPair.first.second;
    RPCMethodDescriptor *rpcMethodDes = serviceMethodPair.second;

    if (rpcService == NULL || rpcMethodDes == NULL) {
        context->errorCode = ARPC_ERROR_RPCCALL_MISMATCH;
        return context;
    }

    context->rpcService = rpcService;
    context->rpcServicePtr = serviceMethodPair.first.first;
    context->rpcMethodDes = rpcMethodDes;

    context->arena.reset(new google::protobuf::Arena());
    DelayDecodePacket *delayDecodePacket = (DelayDecodePacket *)packet;
    RpcMsgHeader *msgHeader = NULL;
    RPCMessage *request = rpcService->GetRequestPrototype(rpcMethodDes).New(context->arena.get());
    DataBufferSerializable *serializable = NULL;

    if (version == ARPC_VERSION_0) {
        serializable = new (nothrow) MessageSerializable(request, context->arena);
    } else if (version == ARPC_VERSION_1) {
        msgHeader = google::protobuf::Arena::CreateMessage<RpcMsgHeader>(context->arena.get());
        serializable = new (nothrow) RPCMessageSerializable(msgHeader, request, context->arena);
    }

    delayDecodePacket->setContent(serializable, true);

    if (!delayDecodePacket->decodeToContent()) {
        ARPC_LOG(ERROR, "decode packet content failed");
        context->errorCode = ARPC_ERROR_DECODE_PACKET;
        return context;
    }

    if (msgHeader) {
        context->enableTrace = msgHeader->enabletrace();
        context->userPayload = msgHeader->userpayload();
        if (msgHeader->has_group())
            context->qosId = msgHeader->group();
    }

    context->request = request;
    return context;
}

Packet *ANetRPCMessageCodec::createPacket(uint32_t pcode, DataBufferSerializable *serializable) const {
    Packet *pPack = _packetFactory->createPacket(pcode);

    if (pPack == NULL) {
        ARPC_LOG(ERROR, "createPacket return NULL, pcode [%ud]", pcode);
        return NULL;
    }

    DelayDecodePacket *packet = dynamic_cast<DelayDecodePacket *>(pPack);
    assert(packet);
    packet->setContent(serializable, true);

    PacketHeader *header = packet->getPacketHeader();
    header->_pcode = pcode;
    return packet;
}

Packet *ANetRPCMessageCodec::EncodeResponse(RPCMessage *response,
                                            Tracer *tracer,
                                            version_t version,
                                            const std::shared_ptr<google::protobuf::Arena> &arena) const {
    assert(tracer);
    uint32_t pcode = 0;
    DataBufferSerializable *pSeri = NULL;

    if (version == ARPC_VERSION_0) {
        pcode = 0;
        pSeri = new MessageSerializable(response, arena);
    } else if (version == ARPC_VERSION_1) {
        pcode |= ADVANCE_PACKET_MASK;

        if (tracer->GetTraceFlag()) {
            TraceInfo *traceInfo = tracer->ExtractServerTraceInfo(getArena(response, arena));
            pSeri = new RPCMessageSerializable(traceInfo, response, arena);
        } else {
            pSeri = new MessageSerializable(response, arena);
        }
    } else {
        assert(false);
        return NULL;
    }

    auto packetDataLen = pSeri->getSerializedSize();
    Packet *packet = createPacket(pcode, pSeri);
    packet->setPacketVersion(version);
    packet->setDataLen(packetDataLen);
    return packet;
}

ARPC_END_NAMESPACE(arpc);
