#pragma once
#include <google/protobuf/descriptor.pb.h>
#include <google/protobuf/stubs/port.h>
#include <stdint.h>

#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/ANetRPCController.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/CommonMacros.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/MessageCodec.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/proto/rpc_extensions.pb.h"
#include "google/protobuf/descriptor.h"

ARPC_BEGIN_NAMESPACE(arpc)

struct RpcReqArg {
    RpcReqArg(ANetRPCController *controller, RPCMessage *response, RPCClosure *closure, CodecContext *context)
        : sController(controller)
        , sResponse(response)
        , sClosure(closure)
        , sContext(context)
        , sVersion(ARPC_VERSION_CURRENT)
        , rawResponse(nullptr) {}

    RpcReqArg(ANetRPCController *controller, std::string *response, RPCClosure *closure, CodecContext *context)
        : sController(controller)
        , sResponse(nullptr)
        , sClosure(closure)
        , sContext(context)
        , sVersion(ARPC_VERSION_CURRENT)
        , rawResponse(response) {}

    ~RpcReqArg() {
        delete sContext;
        sContext = nullptr;
        sController = nullptr;
        sResponse = nullptr;
        sClosure = nullptr;
        rawResponse = nullptr;
    }

    ANetRPCController *sController;
    RPCMessage *sResponse;
    RPCClosure *sClosure;
    CodecContext *sContext;
    version_t sVersion;
    // for jgig
    std::string *rawResponse;
};

class PacketCodeBuilder {
public:
    uint32_t operator()(const RPCMethodDescriptor *method) {
        const RPCServiceDescriptor *pSerDes = method->service();

        uint32_t serviceId = (uint32_t)(pSerDes->options().GetExtension(global_service_id));
        uint32_t methodId = (uint32_t)(method->options().GetExtension(local_method_id));

        uint32_t ret = (serviceId << 16) | methodId;

        return ret;
    }

    uint32_t operator()(uint32_t serviceId, uint32_t methodId) { return (serviceId << 16) | methodId; }
};

class PacketCodeParser {
public:
    void operator()(uint32_t pCode, uint16_t &serviceId, uint16_t &methodId) {
        serviceId = 0;
        methodId = 0;
        serviceId = pCode >> 16;
        methodId = pCode & (0xFFFF);
        return;
    }
};

ARPC_END_NAMESPACE(arpc)
