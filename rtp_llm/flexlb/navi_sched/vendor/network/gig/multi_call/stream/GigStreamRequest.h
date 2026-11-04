#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/interface/Request.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/stream/GigStreamHandlerBase.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/stream/GigStreamMessage.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/stream/GigStreamRpcInfo.h"
#include "autil/Lock.h"

namespace multi_call {

class GigClientStream;
class GrpcClientStreamHandler;
class GrpcStreamConnection;
class CallBack;

struct SingleCallInfo {
public:
    SingleCallInfo() {
        handlerInited = false;
        sendEof = false;
    }

public:
    void abort();

public:
    std::shared_ptr<GrpcStreamConnection> connection;
    std::shared_ptr<CallBack> callBack;
    RequestType requestType;
    bool handlerInited : 1;
    bool sendEof       : 1;
    std::shared_ptr<GrpcClientStreamHandler> handler;
    std::vector<GigStreamMessage> responseVec;
};

MULTI_CALL_TYPEDEF_PTR(SingleCallInfo);

class GigStreamRequest : public Request
{
public:
    GigStreamRequest();
    ~GigStreamRequest();

private:
    GigStreamRequest(const GigStreamRequest &);
    GigStreamRequest &operator=(const GigStreamRequest &);

public:
    ResponsePtr newResponse() override;
    bool serialize() override {
        return true;
    }
    size_t size() const override {
        return 0;
    }

public:
    void setStream(GigClientStream *stream) {
        _stream.store(stream);
    }
    std::shared_ptr<GigStreamBase> getStream() const;
    bool beginRetryReceive() {
        return !_retryReceived.exchange(true);
    }
    void setDisableRetry(bool disable) {
        _disableRetry = disable;
    }
    void disableProbe(bool disable) {
        _disableProbe = disable;
    }
    void setForceStop(bool force) {
        _forceStop = force;
    }
    bool addHandler(const std::shared_ptr<GrpcStreamConnection> &connection,
                    const std::shared_ptr<CallBack> &callBack);
    bool post(bool cancel, const GigStreamMessage &message);
    bool receive(SingleCallInfo *callInfo, const std::shared_ptr<GigStreamBase> &stream,
                 const GigStreamMessage &message);
    void receiveCancel(SingleCallInfo *callInfo, const std::shared_ptr<GigStreamBase> &stream,
                       const GigStreamMessage &message, MultiCallErrorCode ec);
    void abort();
    std::shared_ptr<GrpcClientStreamHandler> getResultHandler() const;

private:
    bool initHandler(SingleCallInfo &callInfo);
    bool postNormal(const GigStreamMessage &message);
    bool flushRetryMessage();
    bool postSingleMessage(SingleCallInfo &callInfo, const SendBufferMessage &message);
    bool postCancel(const GigStreamMessage &message);
    void postSingleCancelMessage(SingleCallInfo &callInfo, const SendBufferMessage &sendMessage,
                                 bool ignoreIfSendEof);

private:
    static bool receiveBufferedMessage(GigClientStream *stream, SingleCallInfo *callInfo);

private:
    bool _disableRetry   : 1;
    bool _disableProbe   : 1;
    bool _useRetryResult : 1;
    bool _forceStop      : 1;
    std::atomic<GigClientStream *> _stream;
    SingleCallInfo *_normalCallInfo;
    std::vector<SingleCallInfo *> _otherCallInfos;
    std::atomic<bool> _retryReceived;
    autil::ThreadMutex _retrySendLock;
    SingleCallInfo *_retryCallInfo;
    std::vector<SendBufferMessage> _retryRequestVec;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(GigStreamRequest);

} // namespace multi_call
