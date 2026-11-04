#pragma once

#include <assert.h>
#include <memory>
#include <string>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/anet.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/CommonMacros.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/Tracer.h"
#include "autil/WorkItem.h"

ARPC_BEGIN_NAMESPACE(arpc);

class RPCServerWorkItem : public autil::WorkItem {
public:
    RPCServerWorkItem(RPCService *pService,
                      RPCMethodDescriptor *pMethodDes,
                      RPCMessage *pReqMsg,
                      const std::string &userPayload,
                      Tracer *tracer);
    virtual ~RPCServerWorkItem();

public:
    virtual void process() { assert(false); }
    virtual void destroy();
    virtual void drop();
    virtual std::string getClientAddress() = 0;
    std::string getMethodName();
    RPCMessage *getRequest();
    Tracer *getTracer() { return _tracer; }
    void setArena(const std::shared_ptr<google::protobuf::Arena> &arena) { _arena = arena; }

protected:
    RPCService *_pService;
    RPCMethodDescriptor *_pMethodDes;
    RPCMessage *_pReqMsg;
    std::string _userPayload;
    Tracer *_tracer;
    std::shared_ptr<google::protobuf::Arena> _arena;
};

ARPC_END_NAMESPACE(arpc);

