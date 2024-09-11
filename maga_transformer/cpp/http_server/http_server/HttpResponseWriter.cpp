#include "http_server/HttpResponseWriter.h"

#include <sstream>

#include "http_server/HttpError.h"
#include "http_server/HttpResponse.h"

namespace http_server {

AUTIL_LOG_SETUP(http_server, HttpResponseWriter);

HttpResponseWriter::~HttpResponseWriter() {
    if (_status == WriteStatus::Stream) {
        WriteDone();
    }
    _connection.reset();
}

bool HttpResponseWriter::Write(const std::string &data) {
    if (_status == WriteStatus::Stream) {
        AUTIL_LOG(WARN, "write failed, already called write stream, cannot write");
        SendErrorResponse();
        return false;
    }
    _status = WriteStatus::Normal;

    if (!DoWrite(data)) {
        SendErrorResponse();
        return false;
    }
    return true;
}

bool HttpResponseWriter::DoWrite(const std::string &data) {
    if (_sent) {
        AUTIL_LOG(ERROR, "write failed, already write data, cannot write more than once");
        return false;
    }
    if (!_connection) {
        AUTIL_LOG(WARN, "write failed, connection is null");
        return false;
    }

    auto response = HttpResponse::make(data, _headers);
    if (!response) {
        AUTIL_LOG(WARN, "write failed, genarate http response failed");
        return false;
    }

    response->setStatusCode(_statusCode);
    if (_statusMessage) response->setStatusMessage(_statusMessage.value());
    auto packet = response->encode();
    if (!packet) {
        AUTIL_LOG(WARN, "write failed, http response encode failed");
        return false;
    }

    if (!_connection->postPacket(packet)) {
        AUTIL_LOG(ERROR, "write failed, connection post http response packet failed");
        packet->free();
        return false;
    }
    _sent = true;
    return true;
}

bool HttpResponseWriter::WriteStream(const std::string &data) {
    if (_status == WriteStatus::Normal) {
        AUTIL_LOG(WARN, "write stream failed, already called write, cannot write stream");
        SendErrorResponse();
        return false;
    }
    _status = WriteStatus::Stream;

    if (!DoWriteStream(data, false)) {
        AUTIL_LOG(WARN, "write stream failed");
        SendErrorResponse();
        return false;
    }
    return true;
}

bool HttpResponseWriter::WriteDone() {
    if (_status != WriteStatus::Stream) {
        return true;
    }
    if (_calledDone) {
        return true;
    }
    if (!DoWriteStream("", true)) {
        AUTIL_LOG(WARN, "write done failed");
        SendErrorResponse();
        return false;
    }
    _calledDone = true;
    return true;
}

bool HttpResponseWriter::DoWriteStream(const std::string &data, bool isWriteDone) {
    if (_calledDone) {
        AUTIL_LOG(WARN, "write stream failed, already called write done, cannot write data any more");
        return false;
    }
    if (!_connection) {
        AUTIL_LOG(WARN, "write stream failed, connection is null");
        return false;
    }

    auto chunkResponse = Chunk(data, isWriteDone);
    if (!chunkResponse) {
        AUTIL_LOG(WARN, "write stream failed, chunk http response failed");
        return false;
    }

    auto packet = chunkResponse->encode();
    if (!packet) {
        AUTIL_LOG(WARN, "write stream failed, http response encode failed");
        return false;
    }

    if (!_connection->postPacket(packet)) {
        AUTIL_LOG(ERROR, "write strem failed, post chunked response packet failed");
        packet->free();
        return false;
    }
    _firstPacket = false;
    return true;
}

std::unique_ptr<HttpResponse> HttpResponseWriter::Chunk(const std::string &data, bool isWriteDone) {
    if (_firstPacket) {
        AddHeader("Transfer-Encoding", "chunked");
        std::stringstream ss;
        ss << std::hex << data.size();
        const auto body = ss.str() + "\r\n" + data + "\r\n";
        return HttpResponse::make(body, _headers);
    }

    if (isWriteDone) {
        const auto body = "0\r\n\r\n";
        return HttpResponse::makeChunkedResponseData(body);
    } else {
        std::stringstream ss;
        ss << std::hex << data.size();
        const auto body = ss.str() + "\r\n" + data + "\r\n";
        return HttpResponse::makeChunkedResponseData(body);
    }
}

void HttpResponseWriter::SendErrorResponse() {
    if (!_connection) {
        AUTIL_LOG(WARN, "send error response failed, connection is null");
        return;
    }
    auto response = HttpResponse::make(HttpError::InternalError("server recvd request but send response failed"));
    if (!response) {
        AUTIL_LOG(WARN, "send error response failed, genarate http response failed");
        return;
    }
    auto packet = response->encode();
    if (!packet) {
        AUTIL_LOG(WARN, "send error response failed, http response encode failed");
        return;
    }
    if (!_connection->postPacket(packet)) {
        AUTIL_LOG(ERROR, "send error response failed, post http response packet failed");
        packet->free();
    }
}

} // namespace http_server
