#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/anet.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/RPCServerAdapter.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/anet/ANetRPCMessageCodec.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/metric/ServerMetricReporter.h"

ARPC_BEGIN_NAMESPACE(arpc)

class ANetRPCServerAdapter : public RPCServerAdapter, public anet::IServerAdapter {
public:
    ANetRPCServerAdapter(RPCServer *pRpcServer, ANetRPCMessageCodec *_messageCodec);
    virtual ~ANetRPCServerAdapter();

public:
    anet::IPacketHandler::HPRetCode handlePacket(anet::Connection *pConnection, anet::Packet *pPacket);
    void SetMetricReporter(const std::shared_ptr<ServerMetricReporter> &metricReporter);

private:
    void handleError(anet::Connection *pConnection,
                     anet::DelayDecodePacket *pReqPacket,
                     ErrorCode errCode,
                     Tracer *tracer,
                     const std::shared_ptr<google::protobuf::Arena> &arena);

    void sendError(anet::Connection *pConnection,
                   channelid_t channelId,
                   version_t requestVersion,
                   const std::string &errMsg,
                   ErrorCode errorCode,
                   Tracer *tracer,
                   const std::shared_ptr<google::protobuf::Arena> &arena);

    anet::IPacketHandler::HPRetCode handleCmdPacket(anet::Packet *pPacket);

    ErrorCode processRequest(CodecContext *pContext,
                             anet::Connection *pConnection,
                             channelid_t channelId,
                             version_t requestVersion,
                             Tracer *tracer);

    ErrorCode decodeRequestPacket(anet::DelayDecodePacket *packet,
                                  RPCService *&service,
                                  RPCMethodDescriptor *&method,
                                  RPCMessage *&request);

    anet::Packet *encodeErrorResponse(channelid_t channelId,
                                      version_t requestVersion,
                                      const std::string &errMsg,
                                      ErrorCode errorCode,
                                      Tracer *tracer,
                                      const std::shared_ptr<google::protobuf::Arena> &arena);
    std::string getClientIpStr(anet::Connection *conn);

    ErrorCode decodePacket(anet::Packet *packet, CodecContext *&context);
    anet::Packet *encodeResponse(RPCMessage *response,
                                 Tracer *tracer,
                                 version_t version,
                                 const std::shared_ptr<google::protobuf::Arena> &arena);

    friend class ANetRPCServerAdapterTest;

protected:
    ANetRPCMessageCodec *_messageCodec;
    std::shared_ptr<ServerMetricReporter> _metricReporter;
};

ARPC_END_NAMESPACE(arpc)
