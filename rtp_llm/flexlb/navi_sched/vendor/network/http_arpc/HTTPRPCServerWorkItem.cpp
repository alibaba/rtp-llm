#include "HTTPRPCServerWorkItem.h"

#include "HTTPRPCController.h"
#include "HTTPRPCServerClosure.h"
#include "Log.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/anet.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/httppacket.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/http_arpc/metric/HTTPRPCStats.h"
#include "autil/TimeUtility.h"

using namespace std;

namespace http_arpc {

HTTPRPCServerWorkItem::HTTPRPCServerWorkItem(RPCService *service,
                                             RPCMethodDescriptor *method,
                                             anet::Connection *connection,
                                             bool keepAlive,
                                             const string &encoding,
                                             const ProtoJsonizerPtr &protoJsonizer,
                                             const EagleInfo &eagleInfo,
                                             string requestStr)
    : _service(service)
    , _method(method)
    , _connection(NULL)
    , _keepAlive(keepAlive)
    , _encoding(encoding)
    , _protoJsonizer(protoJsonizer)
    , _eagleInfo(eagleInfo)
    , _requestStr(requestStr)
    , _beginTime(0)
    , _httpMethod(0) {
    connection->addRef();
    _connection = connection;
}

HTTPRPCServerWorkItem::~HTTPRPCServerWorkItem() { _connection->subRef(); }

void HTTPRPCServerWorkItem::process() {
    if (_httpRPCStats) {
        _httpRPCStats->markRequestPendingDone();
        _httpRPCStats->markThreadPoolItemCount(_threadPoolItemCount);
        _httpRPCStats->markRequestParseBegin();
    }

    RPCMessage *requestMessage = _service->GetRequestPrototype(_method).New();
    RPCMessage *responseMessage = _service->GetResponsePrototype(_method).New();
    unique_ptr<HTTPRPCServerClosure> closure(new HTTPRPCServerClosure(_connection,
                                                                      requestMessage,
                                                                      responseMessage,
                                                                      _keepAlive,
                                                                      _encoding,
                                                                      _protoJsonizer,
                                                                      _eagleInfo,
                                                                      _aiosDebugType));
    closure->setServiceAndMethod(_serviceAndMethod);
    if (_httpRPCStats) {
        closure->setHTTPRPCStats(_httpRPCStats);
    }
    HTTPRPCController *controller = closure->getController();

    if (!parseRequst(requestMessage)) {
        if (_httpRPCStats) {
            _httpRPCStats->markRequestParseDone();
            _httpRPCStats->markRequestDone(400);
        }
        controller->SetFailed("parse request: " + _requestStr + "failed");
        closure.release()->Run();
        return;
    }

    if (_httpRPCStats) {
        _httpRPCStats->markRequestParseDone();
        _httpRPCStats->markRequestCallBegin();
    }

    int64_t queueEnd = autil::TimeUtility::currentTime();
    controller->SetQueueTime(queueEnd - _beginTime);
    controller->SetAddr(_addr);
    _service->CallMethod(_method, controller, requestMessage, responseMessage, closure.release());
}

bool HTTPRPCServerWorkItem::parseRequst(RPCMessage *requestMessage) {
    if (_httpMethod == anet::HTTPPacket::HM_GET) {
        return _protoJsonizer->fromJson(_requestStr, _httpMethod, _requestStr, _headers, requestMessage);
    } else {
        return _protoJsonizer->fromJson(_requestStr, _httpMethod, _uri, _headers, requestMessage);
    }
    return true;
}

void HTTPRPCServerWorkItem::destroy() { delete this; }

void HTTPRPCServerWorkItem::drop() {
    HTTP_ARPC_LOG(ERROR, "drop work item");
    delete this;
}

void HTTPRPCServerWorkItem::SetBeginTime(int64_t beginTime) { _beginTime = beginTime; }

void HTTPRPCServerWorkItem::SetAddr(const std::string &addr) { _addr = addr; }

void HTTPRPCServerWorkItem::SetAiosDebugType(const char *aiosDebugType) {
    if (aiosDebugType) {
        _aiosDebugType = aiosDebugType;
    }
}

} // namespace http_arpc
