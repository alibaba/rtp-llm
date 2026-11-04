#pragma once

#include "UriParserBase.h"

namespace http_arpc {

class DefaultUriParser : public UriParserBase {
public:
    DefaultUriParser() = default;
    DefaultUriParser(bool decodeUri, bool hacompatible) : _decodeUri(decodeUri), _haCompatible(hacompatible){};
    DefaultUriParser(const DefaultUriParser &) = delete;
    DefaultUriParser &operator=(const DefaultUriParser &) = delete;

private:
    virtual void doGetParse(const std::string &uri, std::string &serviceAndMethod, std::string &request) override;
    virtual void
    doPostParse(anet::HTTPPacket *httpPacket, std::string &serviceAndMethod, std::string &request) override;
    virtual void doOptionsParse(anet::Connection *connection, anet::HTTPPacket *httpPacket) override;
    bool processHaCompatible(const std::string &uri, std::string &serviceAndMethod, std::string &request);

protected:
    void fillInfoFromUri(const std::string &uri,
                         std::string &serviceAndMethod,
                         std::string &request,
                         bool haCompatible = false);

private:
    bool _decodeUri = false;
    bool _haCompatible = false;
};
} // namespace http_arpc
