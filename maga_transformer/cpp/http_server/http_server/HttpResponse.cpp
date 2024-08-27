#include "http_server/HttpResponse.h"

#include "aios/network/anet/httppacket.h"

namespace http_server {

AUTIL_LOG_SETUP(http_server, HttpResponse);

anet::Packet *HttpResponse::encode() const {
    auto packet = _isHttpPacket ? new anet::HTTPPacket() : new anet::DefaultPacket();
    packet->setBody(_body.c_str(), _body.size());

    if (_isHttpPacket) {
        auto httpPacket = static_cast<anet::HTTPPacket *>(packet);
        httpPacket->setVersion(anet::HTTPPacket::HTTP_1_1);
        httpPacket->setStatusCode(_statusCode);
        httpPacket->setPacketType(anet::HTTPPacket::PT_RESPONSE);
        for (auto &header : _headers) {
            httpPacket->addHeader(header.first.c_str(), header.second.c_str());
        }
        // TODO: move this to anet HTTPPacket::encodeHeaders()
        if (auto value = httpPacket->getHeader("Transfer-Encoding"); value && strcasecmp(value, "chunked") == 0) {
            httpPacket->disableContentLengthHeader();
        }
    }

    return packet;
}

std::unique_ptr<HttpResponse> HttpResponse::make(const std::string &body, const HeadersType &headers) {
    auto response = std::unique_ptr<HttpResponse>(new HttpResponse());
    response->_body = body;
    response->_headers = headers;
    return response;
}

std::unique_ptr<HttpResponse> HttpResponse::make(const HttpError &error) {
    auto response = std::unique_ptr<HttpResponse>(new HttpResponse());
    response->_statusCode = error.code;
    response->_body = error.message;
    return response;
}

std::unique_ptr<HttpResponse> HttpResponse::makeChunkedResponseData(const std::string &body) {
    auto response = std::unique_ptr<HttpResponse>(new HttpResponse());
    response->_isHttpPacket = false;
    response->_body = body;
    return response;
}

} // namespace http_server
