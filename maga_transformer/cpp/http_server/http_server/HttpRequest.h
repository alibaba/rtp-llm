#pragma once

#include <map>
#include <string>

#include "aios/network/anet/httppacket.h"
#include "http_server/HttpError.h"

namespace anet {
class HTTPPacket;
}

namespace http_server {

class HttpRequest {
public:
    HttpRequest() = default;
    ~HttpRequest();

public:
    HttpError parse(anet::HTTPPacket *request);
    std::string getMethod() const { return _request->getMethodString(); }
    std::string getBody() const { return std::string(_request->getBody(), _request->getBodyLen()); }
    std::string getUri() const { return _request->getURI(); }
    const std::string &getEndpoint() const { return _endpoint; }
    const std::map<std::string, std::string> &getUriParams() const { return _uriParams; }
    anet::HTTPPacket *getRawRequest() const { return _request; }

private:
    std::string _endpoint;
    std::map<std::string, std::string> _uriParams;
    anet::HTTPPacket *_request{nullptr};
};

} // namespace http_server
