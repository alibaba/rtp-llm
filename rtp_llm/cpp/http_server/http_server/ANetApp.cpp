#include "http_server/ANetApp.h"

namespace http_server {
AUTIL_LOG_SETUP(http_server, ANetApp);

ANetApp::ANetApp(anet::Transport* transport): _streamer(&_factory) {
    _ownTransport = false;

    if (transport == NULL) {
        _transport    = new anet::Transport();
        _ownTransport = true;
    } else {
        _transport = transport;
    }
}

ANetApp::~ANetApp() {
    if (_ownTransport) {
        StopPrivateTransport();
        delete _transport;
        _transport = NULL;
    }
}

anet::IOComponent* ANetApp::Listen(
    const std::string& address, anet::IServerAdapter* serverAdapter, int timeout, int maxIdleTime, int backlog) {
    return _transport->listen(address.c_str(), &_streamer, serverAdapter, timeout, maxIdleTime, backlog);
}

bool ANetApp::StartPrivateTransport() {
    if (!_ownTransport) {
        return false;
    }

    _transport->start();
    return true;
}

bool ANetApp::StopPrivateTransport() {
    if (!_ownTransport) {
        return false;
    }
    _transport->stop();
    _transport->wait();

    return true;
}

}  // namespace http_server
