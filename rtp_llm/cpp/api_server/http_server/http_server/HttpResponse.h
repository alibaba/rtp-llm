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
    HttpResponse(const std::string& body): _body(body) {}
    HttpResponse(const HttpError& error): _body(error.message), _statusCode(error.code) {}
    ~HttpResponse() = default;

public:
    anet::Packet* Encode() const;
    void          SetIsHttpPacket(bool isHttpPacket) {
        _isHttpPacket = isHttpPacket;
    }
    void SetDisableContentLengthHeader(bool disable) {
        _disableContentLengthHeader = disable;
    }
    void setStatusCode(int code) {
        _statusCode = code;
    }
    void setStatusMessage(const std::string message) {
        _statusMessage = message;
    }
    void SetHeaders(const std::map<std::string, std::string>& headers) {
        _headers = headers;
    }

private:
    std::string                        _body;
    std::map<std::string, std::string> _headers;
    int32_t                            _statusCode{200};
    bool                               _isHttpPacket{true};
    bool                               _disableContentLengthHeader{false};
    std::optional<std::string>         _statusMessage;

    AUTIL_LOG_DECLARE();
};

}  // namespace http_server
