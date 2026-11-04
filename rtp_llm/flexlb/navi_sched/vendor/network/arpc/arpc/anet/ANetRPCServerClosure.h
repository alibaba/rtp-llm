#pragma once
#include <memory>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/anet.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/ANetRPCController.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/CommonMacros.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/RPCServerClosure.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/anet/ANetRPCMessageCodec.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/util/Log.h"
#include "google/protobuf/arena.h"

ARPC_BEGIN_NAMESPACE(arpc);

class ANetRPCServerClosure : public RPCServerClosure {
public:
    ANetRPCServerClosure(channelid_t chid,
                         anet::Connection *connection,
                         RPCMessage *reqMsg,
                         RPCMessage *resMsg,
                         ANetRPCMessageCodec *messageCodec,
                         version_t version);

    virtual ~ANetRPCServerClosure();

    std::string getIpAndPortAddr() override;

public:
    // for test
    anet::Connection *GetConnection() const { return _connection; }

    channelid_t GetChid() const { return _chid; }
    anet::Packet *buildResponsePacket();

protected:
    virtual void doPostPacket() override;
    friend class RPCServerClosureTest;

private:
    // need by anet
    channelid_t _chid;
    anet::Connection *_connection;
    ANetRPCMessageCodec *_messageCodec;
    version_t _version;
};

TYPEDEF_PTR(RPCServerClosure);

ARPC_END_NAMESPACE(arpc);
