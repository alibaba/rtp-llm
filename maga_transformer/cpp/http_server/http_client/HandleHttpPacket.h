#pragma once

#include <functional>
#include "aios/network/anet/anet.h"

namespace http_server {

class HandleHttpPacket: public anet::IPacketHandler {
public:
    typedef std::function<void(bool ok, const std::string& response_data)> HttpCallBack;

    HandleHttpPacket(const HttpCallBack& http_call_back);
    virtual ~HandleHttpPacket() = default;

    virtual anet::IPacketHandler::HPRetCode handlePacket(anet::Packet* packet, void* args);

private:
    void triggerCallBack(bool ok, const std::string& response);

private:
    HttpCallBack http_call_back_;
};
}  // namespace http_server