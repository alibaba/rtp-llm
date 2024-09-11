#pragma once

#include <map>
#include <string>
#include <optional>

#include "autil/Log.h"
#include "http_server/HttpError.h"

namespace anet {
class Packet;
}

namespace http_server {

class HttpResponse {
public:
    ~HttpResponse() {}

    using HeadersType = std::map<std::string, std::string>;
    static std::unique_ptr<HttpResponse> make(const std::string &body, const HeadersType &headers = {});
    static std::unique_ptr<HttpResponse> make(const HttpError &error);
    static std::unique_ptr<HttpResponse> makeChunkedResponseData(const std::string &body);

    anet::Packet *encode() const;
    void setStatusCode(int code) { _statusCode = code; }
    void setStatusMessage(const std::string message) { _statusMessage = message; }

private:
    HttpResponse() : _isHttpPacket(true), _statusCode(200) {}

private:
    bool _isHttpPacket;
    int _statusCode;
    std::optional<std::string> _statusMessage;
    HeadersType _headers;
    std::string _body;

    AUTIL_LOG_DECLARE();
};

} // namespace http_server
