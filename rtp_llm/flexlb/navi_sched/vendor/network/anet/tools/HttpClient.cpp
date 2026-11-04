#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/tools/HttpClient.h"

#include <assert.h>
#include <cstddef>
#include <memory>
#include <stdint.h>
#include <string>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/anet.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/connection.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/controlpacket.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/httppacket.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/log.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/packet.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/runnable.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/tools/ConnectionPool.h"
#include "alog/Logger.h"
#include "autil/Log.h"

namespace anet {
namespace tools {

using namespace std;
using namespace anet;

AUTIL_LOG_SETUP(anet.tools, HttpClient);

constexpr int64_t kDefaultTimeoutMs = 1000;
constexpr int64_t kDefaultQueueLimited = 1024;

string HttpClient::_kDefaultContentType = "application/x-www-form-urlencoded";

bool HttpClient::init(int64_t timeoutMs, int64_t queueLimited) {
    _connectionPool = make_shared<ConnectionPool>(&_stream);
    timeoutMs = timeoutMs > 0 ? timeoutMs : kDefaultTimeoutMs;
    int64_t checkoutTimeoutMs = timeoutMs / 2;
    _connectionPool->setTimeoutCheckInterval(checkoutTimeoutMs);
    _connectionPool->setQueueTimeout(timeoutMs);

    if (queueLimited > 0) {
        _connectionPool->setQueueLimit(queueLimited);
    } else {
        _connectionPool->setQueueLimit(kDefaultQueueLimited);
    }
    return _connectionPool->open();
}

bool HttpClient::get(const string &host, const string &path, string &content, const int timeoutMs) {
    Connection *connection = NULL;
    string spec = "tcp:" + host;
    connection = _connectionPool->getConnection(spec);
    if (NULL == connection) {
        AUTIL_LOG(ERROR, "Failed to connect server[%s] path[%s] \n", host.c_str(), path.c_str());
        return false;
    }

    HTTPPacket *requestPacket = createGetRequest(host, path, timeoutMs);
    Packet *responsePacket = connection->sendPacket(requestPacket);
    if (nullptr == responsePacket) {
        requestPacket->free();
        connection->subRef();
        AUTIL_LOG(WARN, "Failed to sendRequest to [%s]", host.c_str());
        return false;
    }

    bool ret = handleSendResponsePacket(host, responsePacket, content);
    _connectionPool->recycleConnection(spec, connection);
    return ret;
}

bool HttpClient::post(const string &host,
                      const string &path,
                      const string &requestBody,
                      string &content,
                      const int timeoutMs,
                      const std::string &contentType) {
    Connection *connection = nullptr;
    string spec = "tcp:" + host;
    connection = _connectionPool->getConnection(spec);
    if (nullptr == connection) {
        AUTIL_LOG(ERROR, "Failed to connect server[%s].", host.c_str());
        return false;
    }

    HTTPPacket *requestPacket = createPostRequest(host, path, requestBody, timeoutMs, contentType);
    Packet *responsePacket = connection->sendPacket(requestPacket);
    if (nullptr == responsePacket) {
        requestPacket->free();
        connection->subRef();
        AUTIL_LOG(ERROR, "Failed to sendRequest to server [%s%s]", host.c_str(), path.c_str());
        return false;
    }

    bool ret = handleSendResponsePacket(host, responsePacket, content);
    _connectionPool->recycleConnection(spec, connection);
    return ret;
}

string HttpClient::getHostFromSpec(const string &spec) {
    size_t pos = spec.find(":");
    if (pos == string::npos) {
        return spec;
    }

    string protocolPrefix = spec.substr(0, pos);
    if (protocolPrefix != "tcp") {
        return spec;
    }

    return spec.substr(pos + 1);
}

HTTPPacket *HttpClient::createBaseHttpRequest(const string &host,
                                              const string &path,
                                              HTTPPacket::HTTPMethod method,
                                              const int timeoutMs) {
    HTTPPacket *requestPacket = new HTTPPacket;
    requestPacket->setMethod(method);
    requestPacket->setURI(path.c_str());
    requestPacket->addHeader("Accept", "*/*");
    requestPacket->addHeader("Connection", "Keep-Alive");
    // string host = getHostFromSpec(spec);
    requestPacket->addHeader("Host", host.c_str());
    requestPacket->setExpireTime(timeoutMs);

    return requestPacket;
}

HTTPPacket *HttpClient::createGetRequest(const string &host, const string &path, const int timeoutMs) {
    AUTIL_LOG(DEBUG, "get uri:[%s] from spec:[%s].", path.c_str(), host.c_str());
    HTTPPacket *request = createBaseHttpRequest(host, path, HTTPPacket::HM_GET, timeoutMs);
    AUTIL_LOG(DEBUG, "request len: %zd.", request->getBodyLen());
    return request;
}

HTTPPacket *HttpClient::createPostRequest(const string &host,
                                          const string &path,
                                          const string &requestBody,
                                          const int timeoutMs,
                                          const std::string &contentType) {
    HTTPPacket *request = createBaseHttpRequest(host, path, HTTPPacket::HM_POST, timeoutMs);
    request->addHeader("Content-Type", contentType.c_str());
    request->setBody(requestBody.c_str(), requestBody.size());
    return request;
}

bool HttpClient::handleSendResponsePacket(const string &addr, Packet *responsePacket, string &responseData) {
    bool ret = true;
    if (responsePacket->isRegularPacket()) {
        HTTPPacket *replyPacket = dynamic_cast<HTTPPacket *>(responsePacket);
        assert(replyPacket);
        responseData.assign(replyPacket->getBody(), replyPacket->getBodyLen());
    } else {
        ret = false;
        ControlPacket *controlPacket = dynamic_cast<ControlPacket *>(responsePacket);
        assert(controlPacket);
        AUTIL_LOG(ERROR, "%s : address [%s]", controlPacket->what(), addr.c_str());
    }
    responsePacket->free();
    return ret;
}

} // namespace tools
} // namespace anet
