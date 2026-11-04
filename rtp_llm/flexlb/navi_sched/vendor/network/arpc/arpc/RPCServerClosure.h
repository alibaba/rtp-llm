#pragma once
#include <memory>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/anet.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/ANetRPCController.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/CommonMacros.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/MessageCodec.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/util/Log.h"
#include "google/protobuf/arena.h"

ARPC_BEGIN_NAMESPACE(arpc);

class RPCServerClosure : public RPCClosure {
public:
    RPCServerClosure(RPCMessage *reqMsg, RPCMessage *resMsg);

    virtual ~RPCServerClosure();

    RPCController *GetRpcController() { return &_controller; }

    RPCMessage *GetResponseMsg() { return _resMsg; }

    RPCMessage *GetRequestMsg() { return _reqMsg; }

    void SetResponse(RPCMessage *response) { _resMsg = response; }

    void SetTracer(Tracer *tracer) { _tracer = tracer; }

    Tracer *GetTracer() { return _tracer; }

    void setProtobufArena(const std::shared_ptr<google::protobuf::Arena> &arena) { _arena = arena; }
    const std::shared_ptr<google::protobuf::Arena> &getProtobufArena() {
        return _arena;
    }
public:
    virtual void Run();
    virtual std::string getIpAndPortAddr() = 0;
    virtual void doPostPacket() = 0;

protected:
    void release() { delete this; }
    friend class RPCServerClosureTest;

protected:
    RPCMessage *_reqMsg;
    RPCMessage *_resMsg;
    ANetRPCController _controller;
    Tracer *_tracer;
    std::shared_ptr<google::protobuf::Arena> _arena;
};

TYPEDEF_PTR(RPCServerClosure);
ARPC_END_NAMESPACE(arpc);

