#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <string_view>

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
    HttpError   Parse(std::unique_ptr<::anet::HTTPPacket, std::function<void(::anet::HTTPPacket*)>> request);
    std::string GetMethod() const {
        return _request ? _request->getMethodString() : "";
    }
    std::string GetBody() const {
        return (_request && (_request->getBodyLen() > 0)) ? std::string(_request->getBody(), _request->getBodyLen()) :
                                                            "";
    }
    size_t GetBodySize() const {
        return _request && _request->getBodyLen() > 0 ? static_cast<size_t>(_request->getBodyLen()) : 0;
    }
    // Borrowed packet storage: parse during the route callback, or copy before
    // returning from it. An asynchronous callback must not retain this view.
    std::string_view GetBodyView() const {
        return GetBodySize() ? std::string_view(_request->getBody(), GetBodySize()) : std::string_view{};
    }
    bool HasHeader(const std::string& name) const {
        return _request && _request->getHeader(name.c_str()) != nullptr;
    }
    std::string GetHeader(const std::string& name) const {
        const char* value = _request ? _request->getHeader(name.c_str()) : nullptr;
        return value ? value : "";
    }
    std::string GetUri() const {
        return _request ? _request->getURI() : "";
    }
    const std::string& GetEndpoint() const {
        return _endpoint;
    }
    const std::map<std::string, std::string>& GetUriParams() const {
        return _uriParams;
    }
    int64_t getRecvTime() const {
        return recv_time_us_;
    }

private:
    int64_t                                                                       recv_time_us_;
    std::string                                                                   _endpoint;
    std::map<std::string, std::string>                                            _uriParams;
    std::unique_ptr<::anet::HTTPPacket, std::function<void(::anet::HTTPPacket*)>> _request;
};

}  // namespace http_server
