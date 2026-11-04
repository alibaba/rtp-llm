#include "HTTPRPCServerAdapter.h"

#include <cstring>
#include <sstream>

#include "DefaultUriParser.h"
#include "HTTPRPCServer.h"
#include "HTTPRPCServerWorkItem.h"
#include "Log.h"
#include "ProtoJsonizer.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/ANetRPCServer.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/http_arpc/metric/HTTPRPCStats.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/http_arpc/metric/KMonitorHTTPRPCServerMetricReporter.h"
#include "autil/StringTokenizer.h"
#include "autil/URLUtil.h"
#include "autil/legacy/md5.h"

using namespace std;
using namespace arpc;
using namespace anet;
using namespace autil;
using namespace std;

namespace http_arpc {

HTTP_ARPC_DECLARE_AND_SETUP_LOGGER(HTTPRPCServerAdapter);

class PacketFreeGuard {
public:
    PacketFreeGuard(Packet *packet) : _packet(packet) {}

    ~PacketFreeGuard() { _packet->free(); }

private:
    Packet *_packet;
};

HTTPRPCServerAdapter::HTTPRPCServerAdapter(HTTPRPCServer *server) : _server(server) {}

HTTPRPCServerAdapter::~HTTPRPCServerAdapter() { _protoJsonizer.reset(); }

void HTTPRPCServerAdapter::init() {
    _protoJsonizer.reset(new ProtoJsonizer());
    _uriParser = std::make_shared<DefaultUriParser>(_server->needDecodeUri(), _server->haCompatible());
}

IPacketHandler::HPRetCode HTTPRPCServerAdapter::handlePacket(Connection *connection, Packet *packet) {
    PacketFreeGuard freePacket(packet);
    if (packet->isRegularPacket()) {
        return handleRegularPacket(connection, packet);
    } else {
        return handleCmdPacket(connection, packet);
    }
}

IPacketHandler::HPRetCode HTTPRPCServerAdapter::handleCmdPacket(Connection *connection, Packet *packet) {
    ControlPacket *cmd = dynamic_cast<ControlPacket *>(packet);
    assert(cmd);
    HTTP_ARPC_LOG(DEBUG, "Control Packet (%s) received!", cmd->what());
    return IPacketHandler::FREE_CHANNEL;
}

bool HTTPRPCServerAdapter::parseRequest(anet::Connection *connection,
                                        HTTPPacket *httpPacket,
                                        string &serviceAndMethod,
                                        string &request) {
    bool passThrough = _server->passThrough();
    if (!passThrough) {
        return fillInfo(connection, httpPacket, serviceAndMethod, request);
    } else {
        return fillInfoForPassThrough(connection, httpPacket, serviceAndMethod, request);
    }
}

bool HTTPRPCServerAdapter::fillInfo(anet::Connection *connection,
                                    HTTPPacket *httpPacket,
                                    string &serviceAndMethod,
                                    string &request) {
    if (!_uriParser->Parse(connection, httpPacket, serviceAndMethod, request)) {
        HTTPPacket::HTTPMethod httpMethod = httpPacket->getMethod();
        if (anet::HTTPPacket::HM_OPTIONS != httpMethod) {
            stringstream ss;
            ss << "invalid http method " << int(httpMethod);
            handleError(connection, httpPacket, 400, ss.str());
        }
        return false;
    }
    return true;
}

bool HTTPRPCServerAdapter::fillInfoForPassThrough(anet::Connection *connection,
                                                  HTTPPacket *httpPacket,
                                                  string &serviceAndMethod,
                                                  string &request) {
    const char *uri = httpPacket->getURI();
    HTTPPacket::HTTPMethod httpMethod = httpPacket->getMethod();
    serviceAndMethod = "/";
    if (httpMethod == HTTPPacket::HM_GET) {
        request = uri;
    } else if (httpMethod == HTTPPacket::HM_POST) {
        size_t len = 0;
        char *buf = httpPacket->getBody(len);
        request = string(buf, len);
    } else {
        stringstream ss;
        ss << "invalid http method " << int(httpMethod);
        handleError(connection, httpPacket, 400, ss.str());
        return false;
    }
    return true;
}

void HTTPRPCServerAdapter::setHttpHeader(HTTPPacket *httpPacket, HTTPRPCServerWorkItem *workItem) {
    for (auto it = httpPacket->headerBegin(); it != httpPacket->headerEnd(); ++it) {
        workItem->addHeader(it->first, it->second);
    }
}

IPacketHandler::HPRetCode HTTPRPCServerAdapter::handleRegularPacket(Connection *connection, Packet *packet) {
    int64_t beginTime = TimeUtility::currentTime();
    char addr[256];
    if (!connection->getIpAndPortAddr(addr, 256)) {
        HTTP_ARPC_LOG(WARN, "get ip addr failed!");
        addr[0] = 0;
    }
    HTTPPacket *httpPacket = dynamic_cast<HTTPPacket *>(packet);
    assert(httpPacket);
    const char *uri = httpPacket->getURI();
    HTTPPacket::HTTPMethod httpMethod = httpPacket->getMethod();

    string serviceAndMethod;
    string request;

    if (!parseRequest(connection, httpPacket, serviceAndMethod, request)) {
        return IPacketHandler::KEEP_CHANNEL;
    }

    if (uri && unlikely(strcmp(uri, "/__method__") == 0)) {
        stringstream ss;
        vector<string> names = _server->getRPCNames();
        ss << "__method__ call:\n";
        for (size_t i = 0; i < names.size(); i++) {
            ss << names[i] << '\n';
        }
        handleError(connection, httpPacket, 200, ss.str());
        return IPacketHandler::KEEP_CHANNEL;
    }

    if (request.empty()) {
        request = "{}";
    }
    ServiceMethodPair serviceMethodPair = _server->getMethod(serviceAndMethod);
    RPCService *service = serviceMethodPair.first.second;
    RPCMethodDescriptor *method = serviceMethodPair.second;

    if (!service || !method) {
        stringstream ss;
        ss << "service or method not found for " << serviceAndMethod << ", host: " << addr;
        handleError(connection, httpPacket, 404, ss.str());
        return IPacketHandler::KEEP_CHANNEL;
    }

    if (_server->httpAuth()) {
        if (!checkHttpRequestAuth(httpPacket, request)) {
            HTTP_ARPC_LOG(ERROR, "find invaild auth info in http request: [%s]", request.c_str());
            return IPacketHandler::KEEP_CHANNEL;
        }
    }

    const char *encoding = httpPacket->getHeader("Accept-Encoding");
    string encodingStr;
    if (encoding) {
        encodingStr = encoding;
    }
    auto protoJsonizer = getProtoJsonizer(service, method);
    HTTPRPCServerWorkItem *workItem = new HTTPRPCServerWorkItem(service,
                                                                method,
                                                                connection,
                                                                httpPacket->isKeepAlive(),
                                                                encodingStr,
                                                                protoJsonizer,
                                                                getEagleInfo(httpPacket),
                                                                std::move(request));
    workItem->SetBeginTime(beginTime);
    workItem->SetAddr(string(addr));
    workItem->SetAiosDebugType(httpPacket->getHeader("aios-debug"));
    workItem->SetRPCServicePtr(serviceMethodPair.first.first);
    workItem->setHttpMethod(httpMethod);
    workItem->setServiceAndMethod(serviceAndMethod);

    // 创建HTTP RPC统计
    auto metricReporter = _server->getHTTPRPCMetricReporter();
    if (metricReporter) {
        auto httpRPCStats = metricReporter->makeHTTPRPCStats();
        if (httpRPCStats) {
            httpRPCStats->markRequestBegin();
            httpRPCStats->setRequestUri(uri);
            httpRPCStats->setHttpMethod(httpMethod == HTTPPacket::HM_GET    ? "GET"
                                        : httpMethod == HTTPPacket::HM_POST ? "POST"
                                                                            : "OTHER");
            httpRPCStats->markRequestPendingBegin();
            workItem->setHTTPRPCStats(httpRPCStats);
        }
    }
    if (_server->passThrough()) {
        if (httpMethod == HTTPPacket::HM_POST) {
            workItem->setUri(uri);
        }
        setHttpHeader(httpPacket, workItem);
    }

    if (!_server->PushItem(service, workItem)) {
        handleError(connection,
                    httpPacket,
                    503,
                    method->service()->name() + string(" push to threadpool fail, maybe queue full"));
        return IPacketHandler::KEEP_CHANNEL;
    }
    return IPacketHandler::KEEP_CHANNEL;
}

HTTPPacket *HTTPRPCServerAdapter::buildErrorPacket(bool isKeepAlive, int statusCode, string reason)

{
    HTTPPacket *packet = new HTTPPacket();
    packet->setPacketType(HTTPPacket::PT_RESPONSE);
    packet->setStatusCode(statusCode);
    packet->setBody(reason.c_str(), reason.size());
    packet->setKeepAlive(isKeepAlive);
    return packet;
}

void HTTPRPCServerAdapter::handleError(Connection *connection, HTTPPacket *httpPacket, int statusCode, string reason) {
    HTTP_ARPC_LOG(ERROR, "code[%d] reason[%s]", statusCode, reason.c_str());
    HTTPPacket *packet = buildErrorPacket(httpPacket->isKeepAlive(), statusCode, reason);
    if (httpPacket->getHeader("aios-debug")) {
        packet->addHeader("Access-Control-Allow-Origin", "*");
    }
    if (!_defaultGigHeader.empty()) {
        auto eagleInfo = getEagleInfo(httpPacket);
        if (!eagleInfo.gigdata.empty()) {
            packet->addHeader("GigData", _defaultGigHeader.c_str());
        }
    }
    sendError(connection, packet);
}

void HTTPRPCServerAdapter::setDefaultGigHeader(const std::string &header) { _defaultGigHeader = header; }

void HTTPRPCServerAdapter::sendError(Connection *connection, HTTPPacket *packet) {
    if (!packet->isKeepAlive()) {
        connection->setWriteFinishClose(true);
    }
    if (!connection->postPacket(packet)) {
        packet->free();
    }
}

void HTTPRPCServerAdapter::setProtoJsonizer(const ProtoJsonizerPtr &protoJsonizer) {
    ScopedWriteLock lock(_mutex);
    _protoJsonizer = protoJsonizer;
}

ProtoJsonizerPtr &HTTPRPCServerAdapter::getProtoJsonizer() {
    ScopedReadLock lock(_mutex);
    return _protoJsonizer;
}

void HTTPRPCServerAdapter::setHttpUriParser(const std::shared_ptr<UriParserBase> &uriParser) {
    ScopedWriteLock lock(_mutex);
    _uriParser = uriParser;
}

bool HTTPRPCServerAdapter::setProtoJsonizer(RPCService *service,
                                            const std::string &method,
                                            const ProtoJsonizerPtr &protoJsonizer) {
    auto serviceDesc = service->GetDescriptor();
    const auto &serviceName = serviceDesc->full_name();
    auto methodDesc = serviceDesc->FindMethodByName(method);
    if (!methodDesc) {
        HTTP_ARPC_LOG(ERROR,
                      "set protoJsonizer failed, method [%s] not found in service [%s]",
                      method.c_str(),
                      serviceName.c_str());
        return false;
    }
    ScopedWriteLock lock(_mutex);
    auto &methodMap = _serviceProtoJsonizerMap[serviceName];
    methodMap[method] = protoJsonizer;
    return true;
}

ProtoJsonizerPtr HTTPRPCServerAdapter::getProtoJsonizer(RPCService *service, RPCMethodDescriptor *method) {
    const auto &name = service->GetDescriptor()->full_name();
    const auto &methodName = method->name();
    ScopedReadLock lock(_mutex);
    auto it = _serviceProtoJsonizerMap.find(name);
    if (it == _serviceProtoJsonizerMap.end()) {
        return _protoJsonizer;
    } else {
        const auto &methodMap = it->second;
        auto methodIt = methodMap.find(methodName);
        if (methodMap.end() == methodIt) {
            return _protoJsonizer;
        } else {
            return methodIt->second;
        }
    }
}

EagleInfo HTTPRPCServerAdapter::getEagleInfo(anet::HTTPPacket *httpPacket) {
    EagleInfo eagleInfo;
    const char *traceId = httpPacket->getHeader("EagleEye-TraceId");
    if (traceId) {
        eagleInfo.traceid = traceId;
    }
    const char *rpcId = httpPacket->getHeader("EagleEye-RpcId");
    if (rpcId) {
        eagleInfo.rpcid = rpcId;
    }
    const char *udata = httpPacket->getHeader("EagleEye-UserData");
    if (udata) {
        eagleInfo.udata = udata;
    }
    const char *gdata = httpPacket->getHeader("GigData");
    if (!gdata) {
        gdata = httpPacket->getHeader("gig_data");
    }
    if (gdata) {
        eagleInfo.gigdata = gdata;
    }
    return eagleInfo;
}

bool HTTPRPCServerAdapter::checkHttpRequestAuth(HTTPPacket *httpPacket, const std::string &request) {
    std::string token = httpPacket->getHeader(HTTP_AUTH_TOKEN);
    std::string sign = httpPacket->getHeader(HTTP_AUTH_SIGN);
    if (token.empty() || sign.empty()) {
        return false;
    }
    auto secret = _server->getHttpAuthSecret(token);
    if (secret.empty()) {
        return false;
    }
    autil::legacy::Md5Stream stream;
    stream.Put((const uint8_t *)request.c_str(), request.length());
    stream.Put((const uint8_t *)secret.c_str(), secret.length());
    return stream.GetMd5String() == sign;
}

} // namespace http_arpc
