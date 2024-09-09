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
    HttpError Parse(anet::HTTPPacket *request);
    std::string GetMethod() const { return _request->getMethodString(); }
    std::string GetBody() const { return std::string(_request->getBody(), _request->getBodyLen()); }
    std::string GetUri() const { return _request->getURI(); }
    const std::string &GetEndpoint() const { return _endpoint; }
    const std::map<std::string, std::string> &GetUriParams() const { return _uriParams; }
    anet::HTTPPacket *GetRawRequest() const { return _request; }

private:
    std::string _endpoint;
    std::map<std::string, std::string> _uriParams;
    anet::HTTPPacket *_request{nullptr};
};

} // namespace http_server
