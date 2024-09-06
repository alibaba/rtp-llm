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
    explicit HttpResponseWriter(const std::shared_ptr<anet::Connection> &conn) : _connection(conn) {}
    ~HttpResponseWriter();

public:
    bool Write(const std::string &data);
    bool WriteStream(const std::string &data);
    // is valid for write stream, call this method when write complete
    bool WriteDone();
    bool WriteError(int errorCode, const std::string &errorMsg) const;

    void AddHeader(const std::string &key, const std::string &value) { _headers[key] = value; }
    void SetStatus(int code, const std::string message) {
        _statusCode = code;
        _statusMessage = message;
    }

private:
    bool DoWrite(const std::string &data);
    bool DoWriteStream(const std::string &data, bool isWriteDone);
    std::unique_ptr<HttpResponse> Chunk(const std::string &data, bool isWriteDone);

    void SendErrorResponse();

private:
    enum class WriteStatus {
        Undefined,
        Normal, // 普通响应 (一问一答)
        Stream, // 流式响应
    };

    std::shared_ptr<anet::Connection> _connection;
    WriteStatus _status{WriteStatus::Undefined};

    int _statusCode;
    std::optional<std::string> _statusMessage;

    // for normal response
    bool _sent{false};

    // for stream response
    bool _firstPacket{true};
    bool _calledDone{false};
    std::map<std::string, std::string> _headers;

    AUTIL_LOG_DECLARE();
};

} // namespace http_server
