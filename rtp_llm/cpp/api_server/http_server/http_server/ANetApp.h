#ifndef HTTP_SERVER_ANET_APP_H
#define HTTP_SERVER_ANET_APP_H

#include "aios/network/anet/anet.h"
#include "autil/Log.h"

namespace http_server {

class ANetApp {
public:
    explicit ANetApp(anet::Transport* transport);
    ~ANetApp();

public:
    anet::IOComponent*
    Listen(const std::string& address, anet::IServerAdapter* serverAdapter, int timeout, int maxIdleTime, int backlog);

    bool OwnTransport() {
        return _ownTransport;
    }

    bool                  StartPrivateTransport();
    bool                  StopPrivateTransport();
    anet::IPacketFactory* GetPacketFactory() {
        return &_factory;
    }

private:
    bool                    _ownTransport;
    anet::Transport*        _transport;
    anet::HTTPPacketFactory _factory;
    anet::HTTPStreamer      _streamer;

private:
    AUTIL_LOG_DECLARE();
};

}  // namespace http_server

#endif  // HTTP_SERVER_ANET_APP_H
