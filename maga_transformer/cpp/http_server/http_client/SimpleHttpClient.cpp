#include "maga_transformer/cpp/http_server/http_client/SimpleHttpClient.h"

#include "aios/network/anet/httppacket.h"
#include "aios/network/anet/httppacketfactory.h"
#include "aios/network/anet/httpstreamer.h"
#include "aios/network/anet/transport.h"

#include "maga_transformer/cpp/utils/Logger.h"

namespace http_server {

SimpleHttpClient::SimpleHttpClient() {
    connection_pool_ = std::make_shared<ConnectionPool>();
}

SimpleHttpClient::~SimpleHttpClient() {}

bool SimpleHttpClient::get(const std::string&                     address,
                           const std::string&                     route,
                           const std::string&                     body,
                           const HandleHttpPacket::HttpCallBack&& http_call_back) {
    return send(address, route, body, HttpMethodType::GET, http_call_back);
}

bool SimpleHttpClient::send(const std::string&                    address,
                            const std::string&                    route,
                            const std::string&                    body,
                            const HttpMethodType&                 methodType,
                            const HandleHttpPacket::HttpCallBack& http_call_back) {
    ::anet::HTTPPacket* requestPacket = new ::anet::HTTPPacket();
    requestPacket->setURI(route.c_str());
    requestPacket->setBody(body.c_str(), body.size());
    requestPacket->setPacketType(::anet::HTTPPacket::PT_REQUEST);
    requestPacket->setVersion(::anet::HTTPPacket::HTTP_1_1);

    auto anetMethod = ::anet::HTTPPacket::HM_OPTIONS;
    switch (methodType) {
        case HttpMethodType::GET: {
            anetMethod = ::anet::HTTPPacket::HM_GET;
            break;
        }
        case HttpMethodType::POST: {
            anetMethod = ::anet::HTTPPacket::HM_POST;
            break;
        }
        default: {
            FT_LOG_WARNING("send data failed, unknown http method: %d", (int)methodType);
            return false;
        }
    }
    requestPacket->setMethod(anetMethod);

    return sendPacketAsync(address, requestPacket, http_call_back);
}

bool SimpleHttpClient::sendPacketAsync(const std::string&                    address,
                                       ::anet::Packet*                       packet,
                                       const HandleHttpPacket::HttpCallBack& http_call_back) {
    auto conn = connection_pool_->makeHttpConnection(address);
    if (!conn) {
        FT_LOG_WARNING("send packet failed, connection is null, address: %s", address.c_str());
        return false;
    }
    if (conn->isClosed()) {
        FT_LOG_WARNING("send packet failed, connection is closed, address: %s", address.c_str());
        connection_pool_->recycleHttpConnection(address, conn, true);
        return false;
    }
    auto handler = new HandleHttpPacket(http_call_back);
    if (!conn->postPacket(packet, handler)) {
        FT_LOG_WARNING("post packet failed, address: %s", address.c_str());
        packet->free();
        return false;
    }
    connection_pool_->recycleHttpConnection(address, conn, false);
    return true;
}

std::string SimpleHttpClient::httpMethodTypeToString(const HttpMethodType& methodType) const {
    switch (methodType) {
        case HttpMethodType::GET:
            return "GET";
        case HttpMethodType::POST:
            return "POST";
        default:
            break;
    }
    return "UNKNOWN";
}

}  // namespace http_server