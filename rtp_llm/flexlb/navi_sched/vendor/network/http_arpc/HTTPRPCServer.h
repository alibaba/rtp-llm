#pragma once

#include <list>
#include <string>
#include <unordered_map>

#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/CommonMacros.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/RPCServer.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/http_arpc/ProtoJsonizer.h"
#include "autil/Lock.h"

namespace anet {
class Transport;
}

namespace http_arpc {
class KMonitorHTTPRPCServerMetricReporter;
}

namespace arpc {
class ANetRPCServer;
}

namespace http_arpc {

/**
 * Default timeout to break idle connections.
 */
const int TIMEOUT = 5000;

/**
 * Default time span to break idle connections.
 */
const int MAX_IDLE_TIME = 900000; // 15 minutes

/**
 * Default backlog for server's listen socket
 */
const int LISTEN_BACKLOG = 256;

/**
 * Http packet header for authentication
 */
constexpr char HTTP_AUTH_TOKEN[] = "Auth-Token";
constexpr char HTTP_AUTH_SIGN[] = "Auth-Sign";

class HTTPRPCServerWorkItem;
class HTTPRPCServerImpl;
class UriParserBase;

typedef std::pair<std::pair<std::shared_ptr<RPCService>, RPCService *>, RPCMethodDescriptor *> ServiceMethodPair;
typedef std::unordered_map<std::string, ServiceMethodPair> RPCNameMap;
typedef std::map<std::string, std::string> AliasMap;
typedef std::unordered_map<std::string, std::string> TokenSecretMap;

class HTTPRPCServer {
public:
    HTTPRPCServer(arpc::ANetRPCServer *rpcServer,
                  anet::Transport *transport = NULL,
                  size_t threadNum = 1,
                  size_t queueSize = 50);

    HTTPRPCServer(arpc::ANetRPCServer *rpcServer,
                  anet::Transport *transport,
                  size_t threadNum,
                  size_t queueSize,
                  bool decodeUri,
                  bool haCompatible,
                  bool passThrough = false);
    virtual ~HTTPRPCServer();

public:
    virtual bool StartThreads();
    virtual bool StartPrivateTransport();
    virtual bool StopPrivateTransport();
    virtual bool Listen(const std::string &spec,
                        int timeout = TIMEOUT,
                        int maxIdleTime = MAX_IDLE_TIME,
                        int backlog = LISTEN_BACKLOG);
    virtual bool
    Listen(uint16_t port, int timeout = TIMEOUT, int maxIdleTime = MAX_IDLE_TIME, int backlog = LISTEN_BACKLOG);
    virtual void Close();
    int getPort();
    void setProtoJsonizer(const ProtoJsonizerPtr &protoJsonizer);
    ProtoJsonizerPtr getProtoJsonizer();
    bool setProtoJsonizer(RPCService *service, const std::string &method, const ProtoJsonizerPtr &protoJsonizer);
    void setHttpUriParser(const std::shared_ptr<UriParserBase> &uriParser);
    void setHttpAuthMap(const std::unordered_map<std::string, std::string> &httpAuthMap);
    void setDefaultGigHeader(const std::string &header);
    bool needDecodeUri();
    bool haCompatible();
    bool passThrough();
    bool httpAuth();

public:
    bool PushItem(RPCService *service, HTTPRPCServerWorkItem *workItem);
    void registerService();
    bool addAlias(const AliasMap &aliasMap);
    void removeAlias(const AliasMap &aliasMap);
    std::vector<std::string> getRPCNames();
    ServiceMethodPair getMethod(const std::string &serviceAndMethod);
    bool addMethod(const std::string &methodName, ServiceMethodPair &methodPair);
    bool addMethod(const std::string &methodName,
                   const ServiceMethodPair &methodPair,
                   const arpc::ThreadPoolDescriptor &threadPoolDescriptor);
    bool hasMethod(const std::string &methodName);

    size_t getQueueSize();
    size_t getQueueSize(const std::string &name);
    size_t getItemCount();
    size_t getItemCount(const std::string &name);

    autil::ThreadPoolBasePtr getThreadPool(const std::string &name);
    std::string getHttpAuthSecret(const std::string &token) const;

    void setHTTPRPCMetricReporter(const std::shared_ptr<KMonitorHTTPRPCServerMetricReporter> &reporter);
    std::shared_ptr<KMonitorHTTPRPCServerMetricReporter> getHTTPRPCMetricReporter() const;
    anet::Transport *getTransport();

private:
    bool refreshAddMethod();

private:
    HTTPRPCServerImpl *_impl = nullptr;
    autil::ReadWriteLock _methodMapMutex;
    AliasMap _aliasMap;
    std::map<std::string, std::pair<ServiceMethodPair, arpc::ThreadPoolDescriptor>> _addMethods;
    int _listenPort = -1;
    std::shared_ptr<KMonitorHTTPRPCServerMetricReporter> _httpRPCMetricReporter;
};

} // namespace http_arpc
