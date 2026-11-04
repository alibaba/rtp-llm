#pragma once

#include <memory>
#include <stdint.h>
#include <string>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/httppacket.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/httppacketfactory.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/httpstreamer.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/tools/ConnectionPool.h"
#include "autil/Log.h"

namespace anet {
namespace tools {

class HttpClient {
public:
    HttpClient() : _stream(&_factory), _connectionPool(nullptr){};
    ~HttpClient(){};

private:
    HttpClient(const HttpClient &httpClient) = delete;
    HttpClient &operator=(const HttpClient &httpClient) = delete;

public:
    bool init(int64_t timeoutMs = 0, int64_t queueLimited = 0);

    bool get(const std::string &host, const std::string &path, std::string &content, const int timeoutMs = 5000);

    bool post(const std::string &host,
              const std::string &path,
              const std::string &requestBody,
              std::string &content,
              const int timeoutMs = 5000,
              const std::string &contentType = _kDefaultContentType);

private:
    std::string getHostFromSpec(const std::string &spec);
    anet::HTTPPacket *createBaseHttpRequest(const std::string &host,
                                            const std::string &path,
                                            anet::HTTPPacket::HTTPMethod method,
                                            const int timeoutMs);

    anet::HTTPPacket *createGetRequest(const std::string &host, const std::string &path, const int timeoutMs);

    anet::HTTPPacket *createPostRequest(const std::string &host,
                                        const std::string &path,
                                        const std::string &requestBody,
                                        const int timeoutMs,
                                        const std::string &contentType);

    bool handleSendResponsePacket(const std::string &addr, anet::Packet *responsPacket, std::string &responsData);

private:
    anet::HTTPPacketFactory _factory;
    anet::HTTPStreamer _stream;
    ConnectionPoolPtr _connectionPool;

    static std::string _kDefaultContentType;

private:
    AUTIL_LOG_DECLARE();
};

typedef std::shared_ptr<HttpClient> HttpClientPtr;

} // namespace tools
} // namespace anet

