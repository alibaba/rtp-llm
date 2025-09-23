#pragma once

#include <memory>
#include <optional>
#include <string>
#include <shared_mutex>

#include "rtp_llm/cpp/api_server/http_server/http_client/HandleHttpPacket.h"

namespace anet {
class HTTPPacketFactory;
class HTTPStreamer;
class Transport;
class Connection;
class Packet;
}  // namespace anet

namespace http_server {

class SimpleHttpClient {
public:
    SimpleHttpClient();
    ~SimpleHttpClient();

public:
    bool get(const std::string&   address,
             const std::string&   route,
             const std::string&   body,
             const HttpCallBack&& http_call_back);

    bool post(const std::string&   address,
              const std::string&   route,
              const std::string&   body,
              const HttpCallBack&& http_call_back);

private:
    enum class HttpMethodType {
        GET,
        POST,
    };

    bool        send(const std::string&    address,
                     const std::string&    route,
                     const std::string&    body,
                     const HttpMethodType& methodType,
                     const HttpCallBack&   http_calll_back);
    bool        sendPacketAsync(const std::string& address, ::anet::Packet* packet, const HttpCallBack& http_call_back);
    std::string httpMethodTypeToString(const HttpMethodType& methodType) const;

private:
    std::shared_ptr<ConnectionPool>   connection_pool_;
    std::shared_ptr<HandleHttpPacket> handler_;
};

}  // namespace http_server
