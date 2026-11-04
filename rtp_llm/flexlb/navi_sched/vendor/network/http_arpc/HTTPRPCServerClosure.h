#pragma once

#include <map>

#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/CommonMacros.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/http_arpc/HTTPRPCController.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/http_arpc/ProtoJsonizer.h"

namespace anet {
class Connection;
class HTTPPacket;
} // namespace anet

namespace http_arpc {
class HTTPRPCStats;

struct EagleInfo {
    std::string traceid;
    std::string rpcid;
    std::string udata;
    std::string gigdata;

public:
    bool isEmpty() const { return traceid.empty() && rpcid.empty() && udata.empty(); }
};

class HTTPRPCServerClosure : public RPCClosure {
public:
    HTTPRPCServerClosure(anet::Connection *connection,
                         RPCMessage *requestMessage,
                         RPCMessage *responseMessage,
                         bool keepAlive,
                         const std::string &encoding,
                         const ProtoJsonizerPtr &protoJsonizer,
                         const EagleInfo &eagleInfo,
                         const std::string &aiosDebugType);
    ~HTTPRPCServerClosure();

public:
    /* override */ void Run();
    HTTPRPCController *getController() { return &_httpController; }

    const std::string &getAiosDebugType() const { return _aiosDebugType; }

    void setServiceAndMethod(const std::string &serviceAndMethod) { _serviceAndMethod = serviceAndMethod; }

    const std::string &getServiceAndMethod() const { return _serviceAndMethod; }

    // for case
    void setProtoJsonizer(ProtoJsonizerPtr protoJsonizer);
    ProtoJsonizerPtr getProtoJsonizer() const;

    const EagleInfo &getEagleInfo() const { return _eagleInfo; }

    std::string getResponseHeader(const std::string &key) {
        auto iter = _responseHeader.find(key);
        if (iter != _responseHeader.end()) {
            return iter->second;
        }
        return "";
    }

    void addResponseHeader(const std::string &key, const std::string &value) {
        if (key.empty()) {
            return;
        }
        _responseHeader[key] = value;
    }

    RPCMessage *stealRequest() {
        auto tmp = _requestMessage;
        _requestMessage = NULL;
        return tmp;
    }

    void setHTTPRPCStats(const std::shared_ptr<HTTPRPCStats> &stats) { _httpRPCStats = stats; }

private:
    anet::HTTPPacket *buildPacket();

private:
    anet::Connection *_connection;
    RPCMessage *_requestMessage;
    RPCMessage *_responseMessage;
    HTTPRPCController _httpController;
    bool _keepAlive;
    std::string _encoding;
    ProtoJsonizerPtr _protoJsonizer;
    EagleInfo _eagleInfo;
    std::map<std::string, std::string> _responseHeader;
    std::string _aiosDebugType;
    std::string _serviceAndMethod;
    std::shared_ptr<HTTPRPCStats> _httpRPCStats;
};

} // namespace http_arpc
