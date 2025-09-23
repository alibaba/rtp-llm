#pragma once

#include <functional>
#include "aios/network/anet/anet.h"
#include "rtp_llm/cpp/api_server/http_server/http_client/ConnectionPool.h"

namespace http_server {
typedef std::function<void(bool ok, const std::string& response_data)> HttpCallBack;
struct HandlePacketInfo {
    const std::string                       address_;
    const std::shared_ptr<ConnectionPool>   connection_pool_;
    const std::shared_ptr<anet::Connection> conn_;
    const HttpCallBack                      http_call_back_;

    HandlePacketInfo(const std::string&                       address,
                     const std::shared_ptr<ConnectionPool>&   connection_pool,
                     const std::shared_ptr<anet::Connection>& conn,
                     const HttpCallBack&                      http_call_back):
        address_(address), connection_pool_(connection_pool), conn_(conn), http_call_back_(http_call_back) {}
};

class HandleHttpPacket: public anet::IPacketHandler {
public:
    HandleHttpPacket()          = default;
    virtual ~HandleHttpPacket() = default;

    virtual anet::IPacketHandler::HPRetCode handlePacket(anet::Packet* packet, void* args);

private:
    void triggerCallBack(bool ok, const std::string& response, const HttpCallBack& http_call_back);
};
}  // namespace http_server