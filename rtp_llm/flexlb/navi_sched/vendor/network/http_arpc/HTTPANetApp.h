#pragma once

#include <string>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/anet.h"
#if !defined ANET_VERSION || ANET_VERSION < 010300
#error "From this version, HTTP_ARPC requires ANET >= 1.3.0. Please check your ANET version."
#endif

namespace http_arpc {

class HTTPANetApp {
public:
    HTTPANetApp(anet::Transport *transport);
    ~HTTPANetApp();

public:
    anet::IOComponent *
    Listen(const std::string &address, anet::IServerAdapter *serverAdapter, int timeout, int maxIdleTime, int backlog);

    anet::IOComponent *
    Listen(uint16_t port, anet::IServerAdapter *serverAdapter, int timeout, int maxIdleTime, int backlog);

    bool OwnTransport() { return _ownTransport; }

    bool StartPrivateTransport();
    bool StopPrivateTransport();

    anet::Transport *getTransport() { return _transport; }

private:
    bool _ownTransport;
    anet::Transport *_transport;
    anet::HTTPPacketFactory _factory;
    anet::HTTPStreamer _streamer;
};

} // namespace http_arpc
