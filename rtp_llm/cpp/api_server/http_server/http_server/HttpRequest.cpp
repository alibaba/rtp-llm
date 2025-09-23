#include "http_server/HttpRequest.h"

#include "aios/network/anet/httppacket.h"
#include "autil/StringUtil.h"
#include "autil/TimeUtility.h"

namespace http_server {

HttpRequest::~HttpRequest() {
    _request.reset();
}

HttpError HttpRequest::Parse(std::unique_ptr<::anet::HTTPPacket, std::function<void(::anet::HTTPPacket*)>> request) {
    if (!request) {
        return HttpError::BadRequest("http packet is nullptr");
    }
    _request      = std::move(request);
    recv_time_us_ = autil::TimeUtility::currentTimeInMicroSeconds();

    std::string              uri   = _request->getURI();
    std::vector<std::string> items = autil::StringUtil::split(uri, "?");
    if (items.size() == 0 || items.size() > 2) {
        return HttpError::BadRequest("invalid uri: " + uri);
    }
    _endpoint = items[0];

    if (items.size() == 2) {
        std::vector<std::string> paramItems = autil::StringUtil::split(items[1], "&");
        for (size_t i = 0; i < paramItems.size(); i++) {
            std::string              paramKvStr = paramItems[i];
            std::vector<std::string> kvItems    = autil::StringUtil::split(paramKvStr, "=", false);
            if (kvItems.size() != 2) {
                return HttpError::BadRequest("uri has invalid param: " + uri);
            }
            _uriParams[kvItems[0]] = kvItems[1];
        }
    }
    return HttpError::OK();
}

}  // namespace http_server
