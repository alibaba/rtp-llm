#include "maga_transformer/cpp/http_server/http_client/HandleHttpPacket.h"
#include "maga_transformer/cpp/utils/Logger.h"

namespace http_server {
HandleHttpPacket::HandleHttpPacket(const HttpCallBack& http_call_back): http_call_back_(http_call_back){};

void HandleHttpPacket::triggerCallBack(bool ok, const std::string& response) {
    if (http_call_back_) {
        http_call_back_(ok, response);
    }
}
anet::IPacketHandler::HPRetCode HandleHttpPacket::handlePacket(anet::Packet* packet, void* args) {
    if (!packet) {
        FT_LOG_WARNING("decode body failed, packet is null");
        triggerCallBack(false, "");
        return anet::IPacketHandler::FREE_CHANNEL;
    }

    if (!packet->isRegularPacket()) {
        FT_LOG_WARNING("decode body failed, packet is not regular");
        triggerCallBack(false, "");
        packet->free();
        return anet::IPacketHandler::FREE_CHANNEL;
    }

    ::anet::HTTPPacket* httpPacket = dynamic_cast<::anet::HTTPPacket*>(packet);
    if (!httpPacket) {
        FT_LOG_WARNING("decode body failed, packet is not http packet");
        triggerCallBack(false, "");
        packet->free();
        return anet::IPacketHandler::FREE_CHANNEL;
    }

    if (200 != httpPacket->getStatusCode()) {
        FT_LOG_WARNING("decode body failed, packet get error status, status: %d", httpPacket->getStatusCode());
        triggerCallBack(false, "");
        packet->free();
        return anet::IPacketHandler::FREE_CHANNEL;
    }

    triggerCallBack(true, std::string(httpPacket->getBody(), httpPacket->getBodyLen()));
    packet->free();
    return anet::IPacketHandler::FREE_CHANNEL;
}
}  // namespace http_server