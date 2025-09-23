#ifndef HTTP_SERVER_HTTPSERVERADAPTER_H
#define HTTP_SERVER_HTTPSERVERADAPTER_H

#include <memory>

#include "aios/network/anet/iserveradapter.h"
#include "autil/Log.h"
#include "http_server/HttpError.h"

namespace autil {
class LockFreeThreadPool;
}

namespace http_server {

class HttpRouter;

class HttpServerAdapter: public anet::IServerAdapter {
public:
    HttpServerAdapter(const std::shared_ptr<HttpRouter>& router, size_t threadNum, size_t queueSize);
    ~HttpServerAdapter() override;

public:
    anet::IPacketHandler::HPRetCode handlePacket(anet::Connection* connection, anet::Packet* packet) override;

private:
    anet::IPacketHandler::HPRetCode handleRegularPacket(anet::Connection* connection, anet::Packet* packet) const;
    anet::IPacketHandler::HPRetCode handleControlPacket(anet::Connection* connection, anet::Packet* packet) const;

    void sendErrorResponse(anet::Connection* connection, HttpError error) const;

private:
    std::shared_ptr<HttpRouter>                _router;
    std::shared_ptr<autil::LockFreeThreadPool> _threadPool;

    AUTIL_LOG_DECLARE();
};

}  // namespace http_server

#endif  // HTTP_SERVER_HTTPSERVERADAPTER_H
