#pragma once
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/anet.h"

namespace http_arpc {

class HTTPRPCServerAdapter;

class UriParserBase {
public:
    UriParserBase() = default;
    virtual ~UriParserBase() = default;
    UriParserBase(const UriParserBase &) = delete;
    UriParserBase &operator=(const UriParserBase &) = delete;

public:
    bool Parse(anet::Connection *connection,
               anet::HTTPPacket *httpPacket,
               std::string &serviceAndMethod,
               std::string &request) {
        anet::HTTPPacket::HTTPMethod httpMethod = httpPacket->getMethod();
        if (httpMethod == anet::HTTPPacket::HM_GET) {
            const char *uri = httpPacket->getURI();
            doGetParse(uri, serviceAndMethod, request);
        } else if (httpMethod == anet::HTTPPacket::HM_POST) {
            doPostParse(httpPacket, serviceAndMethod, request);
        } else if (httpMethod == anet::HTTPPacket::HM_OPTIONS) {
            doOptionsParse(connection, httpPacket);
            return false;
        } else {
            return false;
        }
        return true;
    }

private:
    virtual void doGetParse(const std::string &uri, std::string &serviceAndMethod, std::string &request) = 0;
    virtual void doPostParse(anet::HTTPPacket *httpPacket, std::string &serviceAndMethod, std::string &request) = 0;
    virtual void doOptionsParse(anet::Connection *connection, anet::HTTPPacket *httpPacket) = 0;

protected:
    HTTPRPCServerAdapter *_serverAdapter;
};
} // namespace http_arpc
