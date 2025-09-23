#include "http_server/HttpResponse.h"

#include "aios/network/anet/httppacket.h"

namespace http_server {

AUTIL_LOG_SETUP(http_server, HttpResponse);

anet::Packet* HttpResponse::Encode() const {
    auto packet = _isHttpPacket ? new anet::HTTPPacket() : new anet::DefaultPacket();
    packet->setBody(_body.c_str(), _body.size());

    if (_isHttpPacket) {
        auto httpPacket = static_cast<anet::HTTPPacket*>(packet);
        httpPacket->setVersion(anet::HTTPPacket::HTTP_1_1);
        httpPacket->setStatusCode(_statusCode);
        if (_statusMessage)
            httpPacket->setReasonPhrase(_statusMessage.value().c_str());
        httpPacket->setPacketType(anet::HTTPPacket::PT_RESPONSE);
        for (auto& header : _headers) {
            httpPacket->addHeader(header.first.c_str(), header.second.c_str());
        }
        if (_disableContentLengthHeader) {
            httpPacket->disableContentLengthHeader();
        }
    }

    return packet;
}

}  // namespace http_server
