#pragma once
#include <functional>
#include <stdint.h>

#include "rtp_llm/flexlb/navi_sched/vendor/autil/autil/Lock.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/ipackethandler.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/CommonMacros.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/anet/ClientPacketHandler.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/util/Log.h"

ARPC_BEGIN_NAMESPACE(arpc);

class SharedClientPacketHandler : public ClientPacketHandler {
public:
    SharedClientPacketHandler();

protected:
    virtual ~SharedClientPacketHandler();

public:
    anet::IPacketHandler::HPRetCode handlePacket(anet::Packet *packet, void *args) override;
    void addRef();
    void subRef();
    void cleanChannel();

private:
    anet::IPacketHandler::HPRetCode doHandlePacket(anet::Packet *packet, void *args);

private:
    // for test
    void setHandlePacketPreFunc(const std::function<void()> *func);
    friend class ANetRPCChannelCloseBugTest;

private:
    int64_t _ref;
    mutable autil::ThreadMutex _mutex;
    mutable autil::ThreadMutex _channelLock;
    const std::function<void()> *_func;
};

TYPEDEF_PTR(SharedClientPacketHandler);
ARPC_END_NAMESPACE(arpc);

