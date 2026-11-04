#pragma once
#include <assert.h>
#include <memory>
#include <stddef.h>
#include <stdint.h>
#include <string>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/anet.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/CommonMacros.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/Tracer.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/proto/rpc_extensions.pb.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/util/Log.h"

ARPC_BEGIN_NAMESPACE(arpc);

struct CallId {
    std::string strId;
    uint32_t intId;
};

struct CodecContext {
    CodecContext() {
        request = nullptr;
        rawRequest = nullptr;
        rpcService = nullptr;
        rpcMethodDes = nullptr;
        errorCode = ARPC_ERROR_NONE;
        enableTrace = false;
        ownRequest = false;
        timeout = 0;
        qosId = 0;
        requestId = 0;
    }
    bool IsRawMsg() const { return rawRequest != nullptr; }

    CallId callId;           // for encode
    RPCMessage *request;     // for en/decode
    std::string *rawRequest; // for jgig
    std::string token;       // for en/decode
    RPCService *rpcService;  // for decode
    std::shared_ptr<RPCService> rpcServicePtr;
    RPCMethodDescriptor *rpcMethodDes; // for decode
    ErrorCode errorCode;
    bool enableTrace;
    bool ownRequest;
    int32_t timeout;
    std::string clientName;
    uint32_t qosId;
    std::string userPayload;
    std::shared_ptr<google::protobuf::Arena> arena;
    uint32_t requestId;
};

class RPCServer;

class MessageCodec {
public:
    MessageCodec();
    virtual ~MessageCodec();

public:
    virtual CallId GenerateCallId(const RPCMethodDescriptor *method, version_t version) const;
    virtual uint32_t GenerateCallId(uint32_t serviceId, uint32_t methodId) const;

protected:
    friend class MessageCodecTest;
};

TYPEDEF_PTR(MessageCodec);
ARPC_END_NAMESPACE(arpc);

