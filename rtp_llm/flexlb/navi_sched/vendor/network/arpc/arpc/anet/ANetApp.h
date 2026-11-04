#pragma once
#include <iosfwd>
#include <string>
#include <vector>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/anet.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/connectionpriority.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/defaultpacketstreamer.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/CommonMacros.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/util/Log.h"

ARPC_BEGIN_NAMESPACE(arpc);

class ANetApp {
public:
    ANetApp(anet::Transport *transport);
    ~ANetApp();

public:
    anet::IOComponent *
    Listen(const std::string &address, anet::IServerAdapter *serverAdapter, int timeout, int maxIdleTime, int backlog);

    anet::IOComponent *
    Listen(uint16_t port, anet::IServerAdapter *serverAdapter, int timeout, int maxIdleTime, int backlog);

    anet::Connection *Connect(const std::string &address, bool autoReconn, anet::CONNPRIORITY prio);

    bool OwnTransport() { return _ownTransport; }

    bool StartPrivateTransport();
    bool StartPrivateTransport(const std::string &name);
    bool StopPrivateTransport();
    anet::Transport *GetTransport() { return _transport; }
    anet::IPacketFactory *GetPacketFactory() { return &_factory; }

    anet::Connection *
    Connect(const std::string &address, const std::string &bindAddr, bool autoReconn, anet::CONNPRIORITY prio);
    void dump(std::ostringstream &out);

private:
    bool _ownTransport;
    anet::Transport *_transport{nullptr};
    anet::AdvanceDelayDecodePacketFactory _factory;
    anet::DefaultPacketStreamer _streamer;
};

ARPC_END_NAMESPACE(arpc);

