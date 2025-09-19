#include "http_server/HttpServer.h"

#include "http_server/HttpRouter.h"
#include "http_server/HttpServerAdapter.h"

namespace http_server {

AUTIL_LOG_SETUP(http_server, HttpServer);

HttpServer::HttpServer(anet::Transport* transport, size_t threadNum, size_t queueSize): _anetApp(transport) {
    _router        = std::make_shared<HttpRouter>();
    _serverAdapter = std::make_shared<HttpServerAdapter>(_router, threadNum, queueSize);
}

HttpServer::~HttpServer() {
    Stop();
    _serverAdapter.reset();
    _router.reset();
}

bool HttpServer::RegisterRoute(const std::string& method, const std::string& endpoint, const ResponseHandler& func) {
    if (!_router) {
        AUTIL_LOG(
            WARN, "register route failed, router is null, method: %s, endpoint: %s", method.c_str(), endpoint.c_str());
        return false;
    }
    return _router->RegisterRoute(method, endpoint, func);
}

bool HttpServer::Start(const std::string& address, int timeout, int maxIdleTime, int backlog) {
    _listenIoc = _anetApp.Listen(address, _serverAdapter.get(), timeout, maxIdleTime, backlog);

    if (_listenIoc == nullptr) {
        AUTIL_LOG(ERROR, "listen on %s failed", address.c_str());
        return false;
    }

    return _anetApp.StartPrivateTransport();
}

bool HttpServer::Stop() {
    if (_isStopped) {
        return true;
    }
    if (_listenIoc) {
        _listenIoc->close();
        _listenIoc->subRef();
    }
    _anetApp.StopPrivateTransport();
    _isStopped = true;
    return true;
}

}  // namespace http_server
