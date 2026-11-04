#pragma once
#include <assert.h>
#include <stdint.h>
#include <string>
#include <unistd.h>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/connectionpriority.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/ANetRPCController.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/CommonMacros.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/MessageCodec.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/proto/rpc_extensions.pb.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/util/Log.h"

ARPC_BEGIN_NAMESPACE(arpc);

/**
 * @addtogroup ClientClasses RPC Client Classes
 */

class RPCChannelBase : public RPCChannel {
public:
    RPCChannelBase();
    virtual ~RPCChannelBase();

public:
    virtual void CallMethod(const RPCMethodDescriptor *method,
                            RPCController *controller,
                            const RPCMessage *request,
                            RPCMessage *response,
                            RPCClosure *done) {
        assert(false);
    }

    virtual void CallMethod(const uint32_t serviceId,
                            uint32_t methodId,
                            RPCController *controller,
                            const std::string *request,
                            std::string *response,
                            RPCClosure *done) {
        assert(false);
    }

    /**
     * Check if the underlying socket connection is broken(closed) for
     * some reason.
     * @return Returns true if the connection is broken.
     */
    virtual bool ChannelBroken() = 0;

    /**
     * Check if the underlying socket connection is established.
     * @return Returns true if the connection is already
     * established.
     */
    virtual bool ChannelConnected() = 0;

    version_t GetVersion() { return _version; };
    void SetVersion(version_t version) { _version = version; };

    void SetTraceFlag(bool flag) { _enableTrace = flag; }
    bool GetTraceFlag() const { return _enableTrace; }

    void SetTraceFlag(ANetRPCController *controller);

protected:
    void RunClosure(RPCClosure *pClosure);
    void SetError(ANetRPCController *pController, ErrorCode errorCode);

private:
    friend class RPCChannelBaseTest;
    friend class RPCServerAdapterTest;
    friend class ANetRPCChannelCloseBugTest;

protected:
    version_t _version;
    bool _enableTrace;
};

ARPC_END_NAMESPACE(arpc);

