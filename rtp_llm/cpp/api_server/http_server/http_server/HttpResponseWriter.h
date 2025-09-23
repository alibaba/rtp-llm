#pragma once

#include <mutex>
#include <string>
#include <optional>

#include "aios/network/anet/connection.h"
#include "autil/Log.h"

namespace http_server {

class HttpResponse;

class HttpResponseWriter {
public:
    explicit HttpResponseWriter(const std::shared_ptr<anet::Connection>& conn): _connection(conn) {}
    virtual ~HttpResponseWriter();

public:
    enum class WriteType {
        Undefined,
        Normal,  // 普通响应 (一问一答)
        Stream,  // 流式响应
    };
    void SetWriteType(WriteType type) {
        _type = type;
    }
    // not thread safe. `virtual` for test.
    virtual bool Write(const std::string& data);
    virtual bool WriteDone();
    virtual bool isConnected();

    void AddHeader(const std::string& key, const std::string& value) {
        _headers[key] = value;
    }
    void SetStatus(int code, const std::string message) {
        _statusCode    = code;
        _statusMessage = message;
    }

private:
    bool                          WriteNormal(const std::string& data);
    bool                          WriteStream(const std::string& data, bool isWriteDone);
    std::shared_ptr<HttpResponse> Chunk(const std::string& data, bool isWriteDone);
    bool                          PostHttpResponse(const std::shared_ptr<HttpResponse>& response) const;

private:
    std::shared_ptr<anet::Connection> _connection;
    WriteType                         _type{WriteType::Undefined};

    int                        _statusCode{200};
    std::optional<std::string> _statusMessage;

    // for normal response
    bool _alreadyWrite{false};

    // for stream response
    bool                               _firstPacket{true};
    bool                               _calledDone{false};
    std::map<std::string, std::string> _headers;

    AUTIL_LOG_DECLARE();
};

}  // namespace http_server
