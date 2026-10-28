#ifndef HTTP_SERVER_HTTPSERVERADAPTER_H
#define HTTP_SERVER_HTTPSERVERADAPTER_H

#include <memory>
#include <mutex>
#include <unordered_map>

#include "aios/network/anet/iserveradapter.h"
#include "autil/Log.h"
#include "http_server/HttpError.h"

namespace autil {
class LockFreeThreadPool;
}

namespace http_server {

class HttpRouter;
class HttpResponseOrder;

class HttpServerAdapter: public anet::IServerAdapter {
public:
    HttpServerAdapter(const std::shared_ptr<HttpRouter>& router,
                      size_t                             threadNum,
                      size_t                             queueSize,
                      bool                               orderedResponses   = false,
                      int                                requestAwareIdleMs = 0);
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

    const bool                                                                        _orderedResponses;
    const size_t                                                                      _responseLimit;
    const int                                                                         _requestAwareIdleMs;
    mutable std::mutex                                                                _ordersMutex;
    mutable std::unordered_map<anet::Connection*, std::shared_ptr<HttpResponseOrder>> _orders;

    AUTIL_LOG_DECLARE();
};

}  // namespace http_server

#endif  // HTTP_SERVER_HTTPSERVERADAPTER_H
