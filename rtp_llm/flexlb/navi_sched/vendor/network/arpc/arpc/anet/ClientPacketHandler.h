#pragma once
#include <memory>
#include <stdint.h>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/anet.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/ipackethandler.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/CommonMacros.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/PacketArg.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/proto/rpc_extensions.pb.h"

ARPC_BEGIN_NAMESPACE(arpc);
class RPCChannelBase;
class ANetRPCChannel;

class ClientPacketHandler : public anet::IPacketHandler {
public:
    ClientPacketHandler();
    virtual ~ClientPacketHandler();

public:
    anet::IPacketHandler::HPRetCode handlePacket(anet::Packet *packet, void *args) override;
    void setChannel(ANetRPCChannel *channel) { _channel = channel; }

public:
    // for test
    RPCChannelBase *getChannel();

protected:
    anet::IPacketHandler::HPRetCode handleCmdPacket(anet::Packet *packet, RpcReqArg *pArgs);
    void decodePacket(anet::Packet *packet, RpcReqArg *pArgs);
    bool needRepostPacket(anet::Packet *packet);
    void repostPacket(anet::Packet *packet, void *args);
    anet::DataBufferSerializable *createSerializable(anet::Packet *packet,
                                                     RpcReqArg *pArgs,
                                                     TraceInfo *traceInfo,
                                                     google::protobuf::Arena *arena,
                                                     const std::shared_ptr<google::protobuf::Arena> &arenaPtr);

protected:
    ANetRPCChannel *_channel;

private:
    friend class ClientPacketHandlerTest;
};

ARPC_END_NAMESPACE(arpc);

