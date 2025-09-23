#include "rtp_llm/cpp/api_server/http_server/http_client/SimpleHttpClient.h"

#include "aios/network/anet/httppacket.h"
#include "aios/network/anet/httppacketfactory.h"
#include "aios/network/anet/httpstreamer.h"
#include "aios/network/anet/transport.h"

#include "rtp_llm/cpp/utils/Logger.h"

namespace http_server {

SimpleHttpClient::SimpleHttpClient() {
    connection_pool_ = std::make_shared<ConnectionPool>();
    handler_         = std::make_shared<HandleHttpPacket>();
}

SimpleHttpClient::~SimpleHttpClient() {}

bool SimpleHttpClient::get(const std::string&   address,
                           const std::string&   route,
                           const std::string&   body,
                           const HttpCallBack&& http_call_back) {
    return send(address, route, body, HttpMethodType::GET, http_call_back);
}

bool SimpleHttpClient::post(const std::string&   address,
                            const std::string&   route,
                            const std::string&   body,
                            const HttpCallBack&& http_call_back) {
    return send(address, route, body, HttpMethodType::POST, http_call_back);
}

bool SimpleHttpClient::send(const std::string&    address,
                            const std::string&    route,
                            const std::string&    body,
                            const HttpMethodType& methodType,
                            const HttpCallBack&   http_call_back) {
    ::anet::HTTPPacket* requestPacket = new ::anet::HTTPPacket();
    requestPacket->setURI(route.c_str());
    requestPacket->setBody(body.c_str(), body.size());
    requestPacket->addHeader("host", "unknown");
    requestPacket->setKeepAlive(true);
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
            RTP_LLM_LOG_WARNING("send data failed, unknown http method: %d", (int)methodType);
            return false;
        }
    }
    requestPacket->setMethod(anetMethod);

    return sendPacketAsync(address, requestPacket, http_call_back);
}

bool SimpleHttpClient::sendPacketAsync(const std::string&  address,
                                       ::anet::Packet*     packet,
                                       const HttpCallBack& http_call_back) {
    auto conn = connection_pool_->makeHttpConnection(address);
    if (!conn) {
        RTP_LLM_LOG_WARNING("send packet failed, connection is null, address: %s", address.c_str());
        return false;
    }
    if (conn->isClosed()) {
        RTP_LLM_LOG_WARNING("send packet failed, connection is closed, address: %s", address.c_str());
        connection_pool_->recycleHttpConnection(address, conn, true);
        return false;
    }
    HandlePacketInfo* handle_packet_info = new HandlePacketInfo(address, connection_pool_, conn, http_call_back);
    if (!conn->postPacket(packet, handler_.get(), (void*)(handle_packet_info))) {
        RTP_LLM_LOG_WARNING("post packet failed, address: %s", address.c_str());
        packet->free();
        delete handle_packet_info;
        connection_pool_->recycleHttpConnection(address, conn, true);
        return false;
    }
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
