
#include "DefaultUriParser.h"
#include "HTTPRPCServerAdapter.h"
#include "autil/URLUtil.h"

using namespace autil;
using namespace anet;

namespace http_arpc {

void DefaultUriParser::doGetParse(const std::string &uri, std::string &serviceAndMethod, std::string &request) {
    if (_decodeUri) {
        std::string decodedUri = URLUtil::decode(uri);
        fillInfoFromUri(decodedUri, serviceAndMethod, request, _haCompatible);
    } else {
        fillInfoFromUri(uri, serviceAndMethod, request, _haCompatible);
    }
}

void DefaultUriParser::doPostParse(HTTPPacket *httpPacket, std::string &serviceAndMethod, std::string &request) {
    const char *uri = httpPacket->getURI();
    size_t len = 0;
    char *buf = httpPacket->getBody(len);
    request = std::string(buf, len);
    serviceAndMethod = uri;
}

void DefaultUriParser::doOptionsParse(anet::Connection *connection, HTTPPacket *httpPacket) {
    auto responsePacket = _serverAdapter->buildErrorPacket(httpPacket->isKeepAlive(), 200, "");
    responsePacket->addHeader("Access-Control-Allow-Origin", "*");
    auto allowHeader = httpPacket->getHeader("Access-Control-Request-Headers");
    if (allowHeader) {
        responsePacket->addHeader("Access-Control-Allow-Headers", allowHeader);
    }
    _serverAdapter->sendError(connection, responsePacket);
}

void DefaultUriParser::fillInfoFromUri(const std::string &uri,
                                       std::string &serviceAndMethod,
                                       std::string &request,
                                       bool haCompatible) {
    size_t pos = uri.find("?");
    if (pos != std::string::npos) {
        serviceAndMethod = uri.substr(0, pos);
        request = uri.substr(pos + 1);
        return;
    }
    if (haCompatible && processHaCompatible(uri, serviceAndMethod, request)) {
        return;
    }
    size_t serviceBegin = uri.find("/");
    if (serviceBegin != std::string::npos) {
        size_t methodBegin = uri.find("/", serviceBegin + 1);
        if (methodBegin != std::string::npos) {
            size_t requestBegin = uri.find("/", methodBegin + 1);
            if (requestBegin != std::string::npos) {
                request = uri.substr(requestBegin + 1);
                serviceAndMethod = uri.substr(0, requestBegin);
            } else {
                serviceAndMethod = uri;
                request = "";
            }
        } else {
            serviceAndMethod = uri;
            request = "";
        }
    } else {
        serviceAndMethod = uri;
        request = "";
    }
}

bool DefaultUriParser::processHaCompatible(const std::string &uri,
                                           std::string &serviceAndMethod,
                                           std::string &request) {
    static const std::string STATUS_CHECK_PREFIX = "status";
    static const std::string HTTP_SUPPORT_PREFIX = "httpsupport:file";
    static const std::string HA_QUERY_PREFIX = "query=";
    static const std::string HA_CONFIG_PREFIX = "config=";
    if (uri.empty()) {
        return false;
    }
    if (uri.find(HA_CONFIG_PREFIX) != std::string::npos && uri.find(HA_QUERY_PREFIX) != std::string::npos) {
        serviceAndMethod = "/";
        if (uri[0] == '/') {
            request = uri.substr(1);
        } else {
            request = uri;
        }
        return true;
    }
    size_t pos = uri.find(STATUS_CHECK_PREFIX);
    if (pos == 0 || (pos == 1 && uri[0] == '/')) {
        serviceAndMethod = "/status_check";
        request = uri;
        return true;
    }
    pos = uri.find(HTTP_SUPPORT_PREFIX);
    if (pos == 0 || (pos == 1 && uri[0] == '/')) {
        serviceAndMethod = "/status_check";
        request = uri;
        return true;
    }
    return false;
}

} // namespace http_arpc
