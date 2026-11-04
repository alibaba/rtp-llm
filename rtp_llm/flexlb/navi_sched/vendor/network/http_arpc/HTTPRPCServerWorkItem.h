#pragma once

#include <memory>

#include "HTTPRPCServerClosure.h"
#include "ProtoJsonizer.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/CommonMacros.h"
#include "autil/WorkItem.h"

namespace anet {
class Connection;
}

namespace http_arpc {
class HTTPRPCStats;

class HTTPRPCServerWorkItem : public autil::WorkItem {
public:
    HTTPRPCServerWorkItem(RPCService *pService,
                          RPCMethodDescriptor *pMethodDes,
                          anet::Connection *pConnection,
                          bool keepAlive,
                          const std::string &encoding,
                          const ProtoJsonizerPtr &protoJsonizer,
                          const EagleInfo &eagleInfo,
                          std::string requestStr);
    ~HTTPRPCServerWorkItem();

public:
    virtual void process();
    virtual void destroy();
    virtual void drop();

public:
    void SetBeginTime(int64_t beginTime);
    void SetAddr(const std::string &addr);
    void SetAiosDebugType(const char *aiosDebugType);

    void SetRPCServicePtr(const std::shared_ptr<RPCService> &servicePtr) { _servicePtr = servicePtr; }

    void setHttpMethod(int32_t httpMethod) { _httpMethod = httpMethod; }

    void setServiceAndMethod(const std::string &serviceAndMethod) { _serviceAndMethod = serviceAndMethod; }

    void setUri(const std::string &uri) { _uri = uri; }

    void addHeader(const std::string &key, const std::string &value) { _headers[key] = value; }

    void setHTTPRPCStats(const std::shared_ptr<HTTPRPCStats> &stats) { _httpRPCStats = stats; }

    void setThreadPoolItemCount(size_t itemCount) { _threadPoolItemCount = itemCount; }

private:
    bool parseRequst(RPCMessage *requestMessage);

private:
    std::shared_ptr<RPCService> _servicePtr;
    RPCService *_service;
    RPCMethodDescriptor *_method;
    anet::Connection *_connection;
    bool _keepAlive;
    std::string _encoding;
    ProtoJsonizerPtr _protoJsonizer;
    EagleInfo _eagleInfo;
    std::string _requestStr;
    int64_t _beginTime;
    std::string _addr;
    std::string _aiosDebugType;
    int32_t _httpMethod;
    std::string _serviceAndMethod;
    std::string _uri;
    std::map<std::string, std::string> _headers;
    std::shared_ptr<HTTPRPCStats> _httpRPCStats;
    size_t _threadPoolItemCount{0};
};

} // namespace http_arpc
