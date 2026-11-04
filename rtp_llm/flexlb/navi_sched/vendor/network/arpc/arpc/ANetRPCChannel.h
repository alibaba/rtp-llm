#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/anet.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/ANetRPCController.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/CommonMacros.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/RPCChannelBase.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/anet/ANetRPCMessageCodec.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/anet/ClientPacketHandler.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/anet/SharedClientPacketHandler.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/metric/ClientMetricReporter.h"

ARPC_BEGIN_NAMESPACE(arpc);

class ANetRPCChannel : public RPCChannelBase {
public:
    ANetRPCChannel(anet::Connection *pConnection, ANetRPCMessageCodec *messageCode, bool block = false);
    virtual ~ANetRPCChannel();

public:
    virtual void CallMethod(const RPCMethodDescriptor *method,
                            RPCController *controller,
                            const RPCMessage *request,
                            RPCMessage *response,
                            RPCClosure *done) override;
    // for jgig
    void CallMethod(const uint32_t serviceId,
                    uint32_t methodId,
                    RPCController *controller,
                    const std::string *request,
                    std::string *response,
                    RPCClosure *done) override;

    virtual bool ChannelBroken() override;
    virtual bool ChannelConnected() override;
    bool CheckResponsePacket(anet::Packet *packet, RpcReqArg *pArgs);
    std::string getRemoteAddr() const;
    void SetMetricReporter(const std::shared_ptr<ClientMetricReporter> &metricReporter);

public: // for test
    void SyncCall(anet::Packet *pPack, RpcReqArg *pArg);
    bool AsyncCall(anet::Packet *pPack, RpcReqArg *pArg);
    SharedClientPacketHandler *getSharedHandlerFromConnection();

protected:
    virtual bool PostPacket(anet::Packet *packet, anet::IPacketHandler *packetHandler, void *args, bool block);

    bool needRepostPacket(ErrorCode errorCode, version_t remoteVersion, version_t postedPacketVersion) const;

protected:
    ClientPacketHandler _handler;
    anet::Connection *_pConnection;
    bool _block;
    ANetRPCMessageCodec *_messageCodec;
    std::shared_ptr<ClientMetricReporter> _metricReporter;

private:
    friend class ANetRPCChannelTest;
};

ARPC_END_NAMESPACE(arpc);
