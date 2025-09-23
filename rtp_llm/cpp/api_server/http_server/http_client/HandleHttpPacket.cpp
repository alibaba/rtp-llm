#include "rtp_llm/cpp/api_server/http_server/http_client/HandleHttpPacket.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace http_server {
void HandleHttpPacket::triggerCallBack(bool ok, const std::string& response, const HttpCallBack& http_call_back) {
    if (http_call_back) {
        http_call_back(ok, response);
    }
}
anet::IPacketHandler::HPRetCode HandleHttpPacket::handlePacket(anet::Packet* packet, void* args) {
    HandlePacketInfo* handle_packet_info = static_cast<HandlePacketInfo*>(args);
    auto              deleter            = [&packet](HandlePacketInfo* handle_packet_info_ptr) {
        if (handle_packet_info_ptr) {
            packet->free();
            delete handle_packet_info_ptr;
        }
    };
    auto handle_packet_unique_ptr = std::unique_ptr<HandlePacketInfo, decltype(deleter)>(handle_packet_info, deleter);
    handle_packet_unique_ptr->connection_pool_->recycleHttpConnection(
        handle_packet_unique_ptr->address_, handle_packet_unique_ptr->conn_, false);

    if (!packet) {
        RTP_LLM_LOG_WARNING("decode body failed, packet is null");
        triggerCallBack(false, "", handle_packet_unique_ptr->http_call_back_);
        return anet::IPacketHandler::FREE_CHANNEL;
    }

    if (!packet->isRegularPacket()) {
        RTP_LLM_LOG_WARNING("decode body failed, packet is not regular");
        triggerCallBack(false, "", handle_packet_unique_ptr->http_call_back_);
        return anet::IPacketHandler::FREE_CHANNEL;
    }

    ::anet::HTTPPacket* httpPacket = dynamic_cast<::anet::HTTPPacket*>(packet);
    if (!httpPacket) {
        RTP_LLM_LOG_WARNING("decode body failed, packet is not http packet");
        triggerCallBack(false, "", handle_packet_unique_ptr->http_call_back_);
        return anet::IPacketHandler::FREE_CHANNEL;
    }

    if (200 != httpPacket->getStatusCode()) {
        size_t      length;
        std::string response = "";
        if (httpPacket->getBody(length)) {
            response = std::string(httpPacket->getBody(), length);
        }
        RTP_LLM_LOG_WARNING("decode body failed, packet get error status, status: %d, response: %s, length: %d",
                            httpPacket->getStatusCode(),
                            response.c_str(),
                            length);
        triggerCallBack(false, response, handle_packet_unique_ptr->http_call_back_);
        return anet::IPacketHandler::FREE_CHANNEL;
    }

    triggerCallBack(
        true, std::string(httpPacket->getBody(), httpPacket->getBodyLen()), handle_packet_unique_ptr->http_call_back_);
    return anet::IPacketHandler::FREE_CHANNEL;
}
}  // namespace http_server
