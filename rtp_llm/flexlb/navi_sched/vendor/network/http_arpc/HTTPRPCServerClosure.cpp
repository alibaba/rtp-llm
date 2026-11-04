#include "rtp_llm/flexlb/navi_sched/vendor/network/http_arpc/HTTPRPCServerClosure.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/anet.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/http_arpc/Log.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/http_arpc/ProtoJsonizer.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/http_arpc/metric/HTTPRPCStats.h"
#include "autil/ZlibCompressor.h"

using namespace std;
using namespace anet;
using namespace autil;

namespace http_arpc {

HTTPRPCServerClosure::HTTPRPCServerClosure(Connection *connection,
                                           RPCMessage *requestMessage,
                                           RPCMessage *responseMessage,
                                           bool keepAlive,
                                           const string &encoding,
                                           const ProtoJsonizerPtr &protoJsonizer,
                                           const EagleInfo &eagleInfo,
                                           const std::string &aiosDebugType)
    : _connection(connection)
    , _requestMessage(requestMessage)
    , _responseMessage(responseMessage)
    , _keepAlive(keepAlive)
    , _encoding(encoding)
    , _protoJsonizer(protoJsonizer)
    , _eagleInfo(eagleInfo)
    , _aiosDebugType(aiosDebugType) {
    if (_connection != NULL) {
        _connection->addRef();
    }
}

HTTPRPCServerClosure::~HTTPRPCServerClosure() {
    delete _requestMessage;
    _requestMessage = NULL;
    delete _responseMessage;
    _responseMessage = NULL;
    if (_connection != NULL) {
        _connection->subRef();
    }
}

HTTPPacket *HTTPRPCServerClosure::buildPacket() {
    if (_httpRPCStats) {
        _httpRPCStats->markRequestCallDone();
        _httpRPCStats->markResponsePackBegin();
    }

    HTTPPacket *reply = new HTTPPacket();
    reply->setPacketType(HTTPPacket::PT_RESPONSE);
    if (_httpController.Failed()) {
        HTTP_ARPC_LOG(WARN, "[%s]", _httpController.ErrorText().c_str());
        reply->setStatusCode(400);
        reply->setBody(_httpController.ErrorText().c_str(), _httpController.ErrorText().size());
    } else {
        string response = _protoJsonizer->toJson(*_responseMessage);
        int code = _protoJsonizer->getStatusCode(*_responseMessage);
        reply->setStatusCode(code);
        if (string::npos == _encoding.find("deflate")) {
            reply->setBody(response.c_str(), response.size());
        } else {
            string compressResult;
            ZlibCompressor compressor;
            if (compressor.compress(response, compressResult)) {
                reply->addHeader("Content-Encoding", "deflate");
                reply->setBody(compressResult.c_str(), compressResult.size());
            } else {
                reply->setBody(response.c_str(), response.size());
            }
        }
    }
    for (auto iter = _responseHeader.begin(); iter != _responseHeader.end(); ++iter) {
        reply->addHeader(iter->first.c_str(), iter->second.c_str());
    }
    reply->setKeepAlive(_keepAlive);

    if (_httpRPCStats) {
        _httpRPCStats->markResponsePackDone();
        _httpRPCStats->markResponseSize(reply->getBodyLen());
    }

    return reply;
}

void HTTPRPCServerClosure::Run() {
    if (_connection != NULL) {
        if (_httpRPCStats) {
            _httpRPCStats->markResponseSendBegin();
        }

        HTTPPacket *reply = buildPacket();
        if (!reply->isKeepAlive()) {
            _connection->setWriteFinishClose(true);
        }
        if (!_connection->postPacket(reply)) {
            reply->free();
        }

        if (_httpRPCStats) {
            _httpRPCStats->markResponseSendDone();
            int statusCode = reply ? reply->getStatusCode() : 500;
            _httpRPCStats->markRequestDone(statusCode);
        }
    }
    delete this;
}

// for case
void HTTPRPCServerClosure::setProtoJsonizer(ProtoJsonizerPtr protoJsonizer) { _protoJsonizer = protoJsonizer; }

ProtoJsonizerPtr HTTPRPCServerClosure::getProtoJsonizer() const { return _protoJsonizer; }

} // namespace http_arpc
