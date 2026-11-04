#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/stream/GigStreamRequest.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/grpc/client/GrpcClientStreamHandler.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/CallBack.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/GrpcStreamConnection.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/SearchServiceResource.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/stream/GigClientStream.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/stream/GigStreamResponse.h"

using namespace std;
using namespace google::protobuf;

namespace multi_call {
AUTIL_LOG_SETUP(multi_call, GigStreamRequest);

void SingleCallInfo::abort() {
    connection.reset();
    callBack.reset();
    if (handler) {
        handler->abort();
    }
}

GigStreamRequest::GigStreamRequest()
    : Request(MC_PROTOCOL_GRPC_STREAM, std::make_shared<google::protobuf::Arena>())
    , _disableRetry(false)
    , _disableProbe(false)
    , _useRetryResult(false)
    , _forceStop(false)
    , _stream(nullptr)
    , _normalCallInfo(nullptr)
    , _retryReceived(false)
    , _retryCallInfo(nullptr) {
}

GigStreamRequest::~GigStreamRequest() {
    if (_normalCallInfo) {
        DELETE_AND_SET_NULL(_normalCallInfo);
    }
    if (_retryCallInfo) {
        DELETE_AND_SET_NULL(_retryCallInfo);
    }
    for (auto callInfo : _otherCallInfos) {
        DELETE_AND_SET_NULL(callInfo);
    }
}

ResponsePtr GigStreamRequest::newResponse() {
    return ResponsePtr(new GigStreamResponse());
}

std::shared_ptr<GigStreamBase> GigStreamRequest::getStream() const {
    if (_normalCallInfo && _normalCallInfo->handler) {
        return _normalCallInfo->handler->getStream();
    } else {
        auto stream = _stream.load();
        return stream->shared_from_this();
    }
}

bool GigStreamRequest::addHandler(const std::shared_ptr<GrpcStreamConnection> &connection,
                                  const std::shared_ptr<CallBack> &callBack) {
    if (!connection) {
        AUTIL_LOG(ERROR, "add handler failed, null connection");
        return false;
    }
    if (!callBack) {
        AUTIL_LOG(ERROR, "add handler failed, null callback");
        return false;
    }
    const auto &resource = callBack->getResource();
    auto requestType = resource->getRequestType();
    auto callInfo = new SingleCallInfo();
    callInfo->requestType = requestType;
    callInfo->connection = connection;
    callInfo->callBack = callBack;
    if (callBack->isRetry()) {
        _retryCallInfo = callInfo;
        MULTI_CALL_MEMORY_BARRIER();
        flushRetryMessage();
        return true;
    }
    if (RT_NORMAL == requestType) {
        _normalCallInfo = callInfo;
    } else {
        _otherCallInfos.emplace_back(callInfo);
    }
    return true;
}

bool GigStreamRequest::initHandler(SingleCallInfo &callInfo) {
    if (callInfo.handlerInited) {
        return true;
    }
    auto callBack = callInfo.callBack;
    auto connection = callInfo.connection;
    if (unlikely(!connection || !callBack)) {
        AUTIL_LOG(ERROR, "connection or callBack is null");
        return false;
    }
    const auto &resource = callBack->getResource();
    if (!resource) {
        AUTIL_LOG(ERROR, "resource is null");
        return false;
    }
    callBack->initRequestQueryInfo();
    auto partId = resource->getPartId();
    auto handler = connection->createGrpcHandler(resource->getBizName(), this, partId);
    if (unlikely(!handler)) {
        AUTIL_LOG(ERROR, "create grpc client handler failed");
        return false;
    }
    if (!handler->init()) {
        AUTIL_LOG(ERROR, "init grpc client handler failed");
        return false;
    }
    callInfo.handler = handler;
    handler->setRequestInfo(this, callBack);
    handler->setCallInfo(&callInfo);
    callInfo.handlerInited = true;
    callInfo.callBack.reset();
    callInfo.connection.reset();
    return true;
}

bool GigStreamRequest::post(bool cancel, const GigStreamMessage &message) {
    if (!cancel) {
        return postNormal(message);
    } else {
        return postCancel(message);
    }
}

bool GigStreamRequest::postNormal(const GigStreamMessage &message) {
    auto eof = message.eof;
    bool normalOk = false;
    auto sendMessage = GigStreamHandlerBase::createSendBufferMessage(eof, false, message.message);
    if (_normalCallInfo && postSingleMessage(*_normalCallInfo, sendMessage)) {
        normalOk = true;
    }
    if (!_disableRetry) {
        {
            autil::ScopedLock lock(_retrySendLock);
            _retryRequestVec.push_back(sendMessage);
        }
        if (flushRetryMessage()) {
            normalOk = true;
        }
    }
    for (const auto &ptr : _otherCallInfos) {
        postSingleMessage(*ptr, sendMessage);
    }
    if (_normalCallInfo) {
        return normalOk;
    } else {
        return !eof;
    }
}

bool GigStreamRequest::flushRetryMessage() {
    if (!_retryCallInfo) {
        return false;
    }
    std::vector<SendBufferMessage> clearRequestVec;
    if (0 != _retrySendLock.trylock()) {
        return true;
    }
    if (_retryRequestVec.empty()) {
        _retrySendLock.unlock();
        return true;
    }
    auto callInfo = _retryCallInfo;
    clearRequestVec.swap(_retryRequestVec);
    for (const auto &message : clearRequestVec) {
        if (!message.cancel) {
            if (!postSingleMessage(*callInfo, message)) {
                _retrySendLock.unlock();
                return false;
            }
        } else {
            postSingleCancelMessage(*callInfo, message, true);
        }
    }
    _retrySendLock.unlock();
    return true;
}

bool GigStreamRequest::postSingleMessage(SingleCallInfo &callInfo,
                                         const SendBufferMessage &message) {
    if (!callInfo.handlerInited) {
        if (!initHandler(callInfo)) {
            return false;
        }
    }
    const auto &handler = callInfo.handler;
    if (!handler) {
        return false;
    }
    AUTIL_LOG(DEBUG, "handler [%p], send message eof [%d]", handler.get(), message.eof);
    if (!handler->send(message)) {
        return false;
    } else {
        callInfo.sendEof = message.eof;
        return true;
    }
}

bool GigStreamRequest::postCancel(const GigStreamMessage &message) {
    auto sendMessage = GigStreamHandlerBase::createSendBufferMessage(true, true, message.message);
    sendMessage.forceStop = _forceStop;
    auto ignoreIfSendEof = _disableProbe ? false : true;
    if (_normalCallInfo) {
        postSingleCancelMessage(*_normalCallInfo, sendMessage, ignoreIfSendEof);
    }
    if (!_disableRetry) {
        {
            autil::ScopedLock lock(_retrySendLock);
            _retryRequestVec.push_back(sendMessage);
        }
        flushRetryMessage();
    }
    for (const auto &ptr : _otherCallInfos) {
        postSingleCancelMessage(*ptr, sendMessage, true);
    }
    return true;
}

void GigStreamRequest::postSingleCancelMessage(SingleCallInfo &callInfo,
                                               const SendBufferMessage &sendMessage,
                                               bool ignoreIfSendEof) {
    bool ignore = ignoreIfSendEof && callInfo.sendEof;
    if (!callInfo.handlerInited) {
        callInfo.abort();
    } else if (callInfo.handler && !ignore) {
        AUTIL_LOG(DEBUG, "handler [%p] requestType [%d], send cancel", callInfo.handler.get(),
                  callInfo.requestType);
        callInfo.handler->send(sendMessage);
    }
}

bool GigStreamRequest::receive(SingleCallInfo *callInfo,
                               const std::shared_ptr<GigStreamBase> &stream,
                               const GigStreamMessage &message) {
    if (RT_NORMAL != callInfo->requestType) {
        return true;
    }
    auto clientStream = static_cast<GigClientStream *>(stream.get());
    if (_disableRetry) {
        if (clientStream) {
            return clientStream->innerReceive(message, MULTI_CALL_ERROR_NONE);
        } else {
            return false;
        }
    }
    callInfo->responseVec.push_back(message);
    if (!message.eof) {
        return true;
    }
    if (!beginRetryReceive()) {
        return true;
    }
    if (callInfo == _retryCallInfo) {
        _useRetryResult = true;
    }
    return receiveBufferedMessage(clientStream, callInfo);
}

void GigStreamRequest::receiveCancel(SingleCallInfo *callInfo,
                                     const std::shared_ptr<GigStreamBase> &stream,
                                     const GigStreamMessage &message, MultiCallErrorCode ec) {
    if (RT_NORMAL != callInfo->requestType) {
        return;
    }
    auto clientStream = static_cast<GigClientStream *>(stream.get());
    if (_disableRetry) {
        if (clientStream) {
            clientStream->innerReceive(message, ec);
        }
        return;
    }
    if (!beginRetryReceive()) {
        return;
    }
    if (callInfo == _retryCallInfo) {
        _useRetryResult = true;
    }
    if (!receiveBufferedMessage(clientStream, callInfo)) {
        return;
    }
    clientStream->innerReceive(message, ec);
}

bool GigStreamRequest::receiveBufferedMessage(GigClientStream *stream, SingleCallInfo *callInfo) {
    auto &responseVec = callInfo->responseVec;
    for (const auto &oldMessage : responseVec) {
        if (!stream->innerReceive(oldMessage, MULTI_CALL_ERROR_NONE)) {
            return false;
        }
    }
    responseVec.clear();
    return true;
}

void GigStreamRequest::abort() {
    if (_normalCallInfo) {
        _normalCallInfo->abort();
    }
    if (_retryCallInfo) {
        _retryCallInfo->abort();
    }
    for (const auto &callInfo : _otherCallInfos) {
        callInfo->abort();
    }
}

std::shared_ptr<GrpcClientStreamHandler> GigStreamRequest::getResultHandler() const {
    if (!_useRetryResult && _normalCallInfo) {
        return _normalCallInfo->handler;
    } else if (_retryCallInfo) {
        return _retryCallInfo->handler;
    } else {
        return nullptr;
    }
}

} // namespace multi_call
