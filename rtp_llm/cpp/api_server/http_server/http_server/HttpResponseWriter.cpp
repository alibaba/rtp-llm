#include "http_server/HttpResponseWriter.h"

#include <sstream>

#include "http_server/HttpError.h"
#include "http_server/HttpResponse.h"

namespace http_server {

AUTIL_LOG_SETUP(http_server, HttpResponseWriter);

HttpResponseWriter::~HttpResponseWriter() {
    if (_type == WriteType::Stream) {
        WriteDone();
    }
    _connection.reset();
}

bool HttpResponseWriter::Write(const std::string& data) {
    if (_type == WriteType::Undefined) {
        AUTIL_LOG(WARN, "write failed, write type is undefined, call SetWriteType first");
        return false;
    }
    if (_type == WriteType::Normal) {
        return WriteNormal(data);
    } else {
        return WriteStream(data, false);
    }
}

bool HttpResponseWriter::WriteDone() {
    if (_type != WriteType::Stream || _calledDone) {
        return true;
    }
    if (!WriteStream("", true)) {
        AUTIL_LOG(WARN, "write done failed");
        return false;
    }
    _calledDone = true;
    return true;
}

bool HttpResponseWriter::isConnected() {
    return _connection->isConnected();
}

bool HttpResponseWriter::WriteNormal(const std::string& data) {
    if (_alreadyWrite) {
        AUTIL_LOG(ERROR, "write failed, already write data, cannot write more than once");
        return false;
    }
    std::shared_ptr<HttpResponse> response;
    if (_statusCode == 200) {
        response = std::make_shared<HttpResponse>(data);
    } else {
        HttpError error;
        error.code    = _statusCode;
        error.message = data;
        response      = std::make_shared<HttpResponse>(error);
    }
    response->SetHeaders(_headers);
    if (_statusMessage)
        response->setStatusMessage(_statusMessage.value());
    if (!PostHttpResponse(response)) {
        AUTIL_LOG(WARN, "write normal failed, post http response failed");
        return false;
    }
    _alreadyWrite = true;
    return true;
}

bool HttpResponseWriter::WriteStream(const std::string& data, bool isWriteDone) {
    if (_calledDone) {
        AUTIL_LOG(WARN, "write stream failed, already called write done, cannot write data any more");
        return false;
    }

    auto chunkResponse = Chunk(data, isWriteDone);
    if (!chunkResponse) {
        AUTIL_LOG(WARN, "write stream failed, chunk http response failed");
        return false;
    }
    if (_statusMessage)
        chunkResponse->setStatusMessage(_statusMessage.value());
    if (!PostHttpResponse(chunkResponse)) {
        AUTIL_LOG(WARN, "write stream failed, post http response failed");
        return false;
    }
    _firstPacket = false;
    return true;
}

bool HttpResponseWriter::PostHttpResponse(const std::shared_ptr<HttpResponse>& response) const {
    if (!response) {
        AUTIL_LOG(WARN, "post http response failed, http response is null");
        return false;
    }
    if (!_connection) {
        AUTIL_LOG(WARN, "post http response failed, connection is null");
        return false;
    }

    auto packet = response->Encode();
    if (!packet) {
        AUTIL_LOG(WARN, "post http response failed, http response encode failed");
        return false;
    }

    if (!_connection->postPacket(packet)) {
        AUTIL_LOG(ERROR, "post http response failed, post chunked response packet failed");
        packet->free();
        return false;
    }
    return true;
}

std::shared_ptr<HttpResponse> HttpResponseWriter::Chunk(const std::string& data, bool isWriteDone) {
    if (_firstPacket) {
        AddHeader("Transfer-Encoding", "chunked");
        std::stringstream ss;
        ss << std::hex << data.size();
        const auto body     = ss.str() + "\r\n" + data + "\r\n";
        auto       response = std::make_shared<HttpResponse>(body);
        if (response) {
            response->SetHeaders(_headers);
            response->SetDisableContentLengthHeader(true);
        }
        return response;
    }

    std::string body;
    if (isWriteDone) {
        body = "0\r\n\r\n";
    } else {
        std::stringstream ss;
        ss << std::hex << data.size();
        body = ss.str() + "\r\n" + data + "\r\n";
    }
    auto response = std::make_shared<HttpResponse>(body);
    if (response) {
        response->SetIsHttpPacket(false);
    }
    return response;
}

}  // namespace http_server
