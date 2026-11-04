#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/anet.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/RPCServerWorkItem.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/anet/ANetRPCMessageCodec.h"

ARPC_BEGIN_NAMESPACE(arpc)

class ANetRPCServerWorkItem : public RPCServerWorkItem {
public:
    ANetRPCServerWorkItem(RPCService *pService,
                          RPCMethodDescriptor *pMethodDes,
                          RPCMessage *pReqMsg,
                          const std::string &userPayload,
                          anet::Connection *pConnection,
                          channelid_t channelId,
                          ANetRPCMessageCodec *messageCodec,
                          version_t version,
                          Tracer *tracer);

    virtual ~ANetRPCServerWorkItem();

public:
    virtual void destroy() override;
    virtual void process() override;
    virtual std::string getClientAddress() override;

private:
    anet::Connection *_pConnection;
    channelid_t _channelId;
    version_t _version;
    ANetRPCMessageCodec *_messageCodec;
};

ARPC_END_NAMESPACE(arpc)
