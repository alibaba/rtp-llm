#include "http_server/HttpServerAdapter.h"

#include "http_server/HttpRequestWorkItem.h"
#include "http_server/HttpResponse.h"
#include "http_server/HttpRouter.h"
#include "autil/LockFreeThreadPool.h"

namespace http_server {

AUTIL_LOG_SETUP(http_server, HttpServerAdapter);

HttpServerAdapter::HttpServerAdapter(const std::shared_ptr<HttpRouter> &router, size_t threadNum, size_t queueSize)
    : _router(router) {
    _threadPool = std::make_shared<autil::LockFreeThreadPool>(threadNum, queueSize, nullptr, "HttpRequestThreadPool");
    _threadPool->start();
}

HttpServerAdapter::~HttpServerAdapter() {
    if (_threadPool) {
        _threadPool->stop();
        _threadPool.reset();
    }
    _router.reset();
}

anet::IPacketHandler::HPRetCode HttpServerAdapter::handlePacket(anet::Connection *connection, anet::Packet *packet) {
    if (!connection || !packet) {
        AUTIL_LOG(WARN,
                  "http server adapter handle packet failed, connection or packet is null, connection: %p, packet: %p",
                  connection,
                  packet);
        return anet::IPacketHandler::FREE_CHANNEL;
    }

    if (packet->isRegularPacket()) { // handle httpPacket
        return handleRegularPacket(connection, packet);
    } else { // control command received
        return handleControlPacket(connection, packet);
    }
}

anet::IPacketHandler::HPRetCode HttpServerAdapter::handleRegularPacket(anet::Connection *connection,
                                                                       anet::Packet *packet) const {
    anet::HTTPPacket *httpPacket = dynamic_cast<anet::HTTPPacket *>(packet);
    if (!httpPacket) {
        AUTIL_LOG(WARN, "Invalid HTTPPacket received");
        packet->free();
        return anet::IPacketHandler::FREE_CHANNEL;
    }

    auto request = std::make_shared<HttpRequest>();
    const auto parseError = request->parse(httpPacket); // free packet in request
    if (!parseError.IsOK()) {
        AUTIL_LOG(WARN, "parse http request failed. error: %s", parseError.ToString().c_str());
        SendErrorResponse(connection, parseError);
        return anet::IPacketHandler::KEEP_CHANNEL;
    }

    if (!_router) {
        AUTIL_LOG(WARN, "handle packet failed, http router is null");
        auto error = HttpError::InternalError("server http router is null");
        SendErrorResponse(connection, error);
        return anet::IPacketHandler::KEEP_CHANNEL;
    }

    const auto method = request->getMethod();
    const auto endpoint = request->getEndpoint();
    const auto responseHandlerOpt = _router->FindRoute(method, endpoint);
    if (!responseHandlerOpt.has_value()) {
        AUTIL_LOG(WARN,
                  "handle packet failed, http route not found. method: %s, endpoint: %s",
                  method.c_str(),
                  endpoint.c_str());
        auto error = HttpError::NotFound("http route not found: [" + method + ": " + endpoint + "]");
        SendErrorResponse(connection, error);
        return anet::IPacketHandler::KEEP_CHANNEL;
    }
    auto responseHandler = responseHandlerOpt.value();

    connection->addRef();
    auto connectionPtr =
        std::shared_ptr<::anet::Connection>(connection, [](anet::Connection *conn) { conn->subRef(); });

    auto workItem = new HttpRequestWorkItem(responseHandler, connectionPtr, request);
    const auto errorCode = _threadPool->pushWorkItem(workItem, false);
    if (errorCode != autil::ThreadPool::ERROR_NONE) {
        AUTIL_LOG(WARN,
                  "handle packet failed, push work item failed, error code: %d. method: %s, endpoint: %s",
                  errorCode,
                  method.c_str(),
                  endpoint.c_str());
        delete workItem;
        SendErrorResponse(connection, HttpError::InternalError("server push http request work item failed"));
        return anet::IPacketHandler::KEEP_CHANNEL;
    }
    return anet::IPacketHandler::KEEP_CHANNEL;
}

anet::IPacketHandler::HPRetCode HttpServerAdapter::handleControlPacket(anet::Connection *connection,
                                                                       anet::Packet *packet) const {
    anet::ControlPacket *controlPacket = dynamic_cast<anet::ControlPacket *>(packet);
    if (controlPacket) {
        AUTIL_LOG(DEBUG, "Control Packet (%s) received!", controlPacket->what());
    }
    packet->free(); // free packet if finished
    return anet::IPacketHandler::FREE_CHANNEL;
}

void HttpServerAdapter::SendErrorResponse(anet::Connection *connection, HttpError error) const {
    auto response = HttpResponse::make(error);
    if (!response) {
        AUTIL_LOG(
            WARN, "send error response failed, create error response failed, error: %s", error.ToString().c_str());
        return;
    }

    auto packet = response->encode();
    if (!packet) {
        AUTIL_LOG(
            WARN, "send error response failed, response encode to packet failed, error: %s", error.ToString().c_str());
        return;
    }

    if (!connection->postPacket(packet)) {
        AUTIL_LOG(
            ERROR, "send error response failed, post http response packet failed, error: %s", error.ToString().c_str());
        packet->free();
        return;
    }
}

} // namespace http_server
