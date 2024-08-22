#ifndef HTTP_SERVER_HTTPSERVER_H
#define HTTP_SERVER_HTTPSERVER_H

#include "aios/network/anet/transport.h"
#include "http_server/HttpRequest.h"
#include "http_server/HttpResponseWriter.h"
#include "autil/Log.h"
#include "http_server/ANetApp.h"
#include <functional>

namespace http_server {

class HttpRouter;
class HttpServerAdapter;

/**
 * Default time span to break idle connections.
 */
const int MAX_IDLE_TIME = 900000; // 15 minutes

/**
 * Default backlog for server's listen socket
 */
const int LISTEN_BACKLOG = 256;

class HttpServer {
public:
    HttpServer(anet::Transport *transport = nullptr, size_t threadNum = 2, size_t queueSize = 50);
    ~HttpServer();

public:
    using ResponseHandler = std::function<void(std::unique_ptr<HttpResponseWriter>, const HttpRequest &)>;
    bool RegisterRoute(const std::string &method, const std::string &endpoint, const ResponseHandler &func);

    /**
     * Start listening a specific port according to the address string.
     * @todo user want to pass port of 0, then he wants to know which port is
     * actually listen after Listen be called
     * @param address A string to indicate listen address. Format is like this:
     *                <proto>:<addr>:port. A real example: "tcp:0.0.0.0:7893".
     *                Another example for domain socket:
     *                "domainstream:/tmp/ds-chunk.socket"
     * @param timeout default post packet timeout (ms) for all the accepted socket.
     *        will affect all the reply sending back to the client.
     * @param maxIdleTime This parameter determines after how long time idle, server
     *        will break the idle connections intentinally.
     */
    bool Start(const std::string &address,
               int timeout = 5000,
               int maxIdleTime = MAX_IDLE_TIME,
               int backlog = LISTEN_BACKLOG);

    /**
     * Stops HttpServer. Will terminate all the ongoing threads and destroy all
     * the port listening.
     * @return This function will always return true.
     */
    bool Stop();

private:
    ANetApp _anetApp;
    std::shared_ptr<HttpRouter> _router;
    std::shared_ptr<HttpServerAdapter> _serverAdapter;
    anet::IOComponent *_listenIoc{nullptr};

private:
    AUTIL_LOG_DECLARE();
};

} // namespace http_server

#endif // HTTP_SERVER_HTTPSERVER_H
