#pragma once

#include "HTTPRPCServerClosure.h"
#include "ProtoJsonizer.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/anet.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/RPCServer.h"
#include "autil/Lock.h"

namespace http_arpc {

class HTTPRPCServer;
class HTTPRPCServerWorkItem;
class HTTPRPCServerAdapter;
class UriParserBase;

class HTTPRPCServerAdapter : public anet::IServerAdapter {
public:
    HTTPRPCServerAdapter(HTTPRPCServer *server);
    ~HTTPRPCServerAdapter();

public:
    /* override */ anet::IPacketHandler::HPRetCode handlePacket(anet::Connection *connection, anet::Packet *packet);
    void init();
    void setProtoJsonizer(const ProtoJsonizerPtr &protoJsonizer);
    ProtoJsonizerPtr &getProtoJsonizer();
    bool setProtoJsonizer(RPCService *service, const std::string &method, const ProtoJsonizerPtr &protoJsonizer);
    ProtoJsonizerPtr getProtoJsonizer(RPCService *service, RPCMethodDescriptor *method);
    void setHttpUriParser(const std::shared_ptr<UriParserBase> &uriParser);
    anet::HTTPPacket *buildErrorPacket(bool isKeepAlive, int statusCode, std::string reason);
    void setDefaultGigHeader(const std::string &header);
    void sendError(anet::Connection *connection, anet::HTTPPacket *packet);

private:
    anet::IPacketHandler::HPRetCode handleCmdPacket(anet::Connection *connection, anet::Packet *packet);
    anet::IPacketHandler::HPRetCode handleRegularPacket(anet::Connection *connection, anet::Packet *packet);
    bool parseRequest(anet::Connection *connection,
                      anet::HTTPPacket *httpPacket,
                      std::string &serviceAndMethod,
                      std::string &request);
    bool fillInfo(anet::Connection *connection,
                  anet::HTTPPacket *httpPacket,
                  std::string &serviceAndMethod,
                  std::string &request);
    bool fillInfoForPassThrough(anet::Connection *connection,
                                anet::HTTPPacket *httpPacket,
                                std::string &serviceAndMethod,
                                std::string &request);
    void handleError(anet::Connection *connection, anet::HTTPPacket *httpPacket, int statusCode, std::string reason);
    void setHttpHeader(anet::HTTPPacket *httpPacket, HTTPRPCServerWorkItem *workItem);
    static EagleInfo getEagleInfo(anet::HTTPPacket *httpPacket);
    bool checkHttpRequestAuth(anet::HTTPPacket *httpPacket, const std::string &request);

private:
    HTTPRPCServer *_server;
    ProtoJsonizerPtr _protoJsonizer;
    std::string _defaultGigHeader;
    std::unordered_map<std::string, std::unordered_map<std::string, ProtoJsonizerPtr>> _serviceProtoJsonizerMap;
    autil::ReadWriteLock _mutex;
    std::shared_ptr<UriParserBase> _uriParser;
};

} // namespace http_arpc
