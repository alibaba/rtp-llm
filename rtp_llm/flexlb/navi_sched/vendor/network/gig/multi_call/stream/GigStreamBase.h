#pragma once

#include <list>

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/stream/GigStreamMessage.h"

namespace multi_call {

class QueryInfo;

class PartMessageQueue
{
public:
    PartMessageQueue();
    ~PartMessageQueue();

private:
    PartMessageQueue(const PartMessageQueue &);
    PartMessageQueue &operator=(const PartMessageQueue &);

public:
    bool nextMessage(GigStreamMessage &message, MultiCallErrorCode &ec);
    bool peekMessage(GigStreamMessage &message, MultiCallErrorCode &ec);
    bool empty() const {
        return (0 == _count);
    }
    void appendMessage(const GigStreamMessage &message, MultiCallErrorCode ec);
    bool beginReceive();
    bool endReceive();

private:
    autil::ThreadMutex _lock;
    std::atomic<bool> _receiving;
    size_t _count;
    std::list<GigStreamMessage> _messageList;
    MultiCallErrorCode _ec;
};

typedef std::function<uint64_t(PartIdTy)> CollectFlagsFunc;

struct StreamRpcParam {
    bool asyncMode = true;
    uint64_t collectFlags = 0;
    int64_t globalStartTimeUs = 0;
    CollectFlagsFunc collectFlagsFunc;
    std::string mallocTraceStr;
};

class GigStreamBase : public std::enable_shared_from_this<GigStreamBase>
{
public:
    GigStreamBase();
    virtual ~GigStreamBase();

private:
    GigStreamBase(const GigStreamBase &);
    GigStreamBase &operator=(const GigStreamBase &);

public:
    virtual google::protobuf::Message *newReceiveMessage(google::protobuf::Arena *arena) const = 0;

public:
    virtual bool send(PartIdTy partId, bool eof, google::protobuf::Message *message) = 0;
    virtual void sendCancel(PartIdTy partId, google::protobuf::Message *message) = 0;

private:
    virtual bool receive(const GigStreamMessage &message) = 0;
    virtual void receiveCancel(const GigStreamMessage &message, MultiCallErrorCode ec) = 0;

public:
    virtual bool hasError() const;

public:
    void setParam(const StreamRpcParam &param) {
        _param = param;
    }
    const StreamRpcParam &getParam() const {
        return _param;
    }
    virtual std::shared_ptr<QueryInfo> getQueryInfo() const {
        return nullptr;
    }
    virtual std::string getPeer() const {
        return std::string();
    }

public:
    // async
    void setAsyncMode(bool async) {
        _param.asyncMode = async;
    }
    bool isAsyncMode() const {
        return _param.asyncMode;
    }
    // override to implement async receive
    virtual bool notifyReceive(multi_call::PartIdTy partId) {
        return asyncReceive(partId);
    }
    bool asyncReceive(PartIdTy partId);
    // A bounded turn requires notifyReceive() to enqueue the continuation.
    bool asyncReceive(PartIdTy partId, size_t maxMessageCount);
    bool peekMessage(PartIdTy partId, GigStreamMessage &message, MultiCallErrorCode &ec);

public:
    // metrics
    virtual void *beginReceive(PartIdTy partId) {
        return nullptr;
    }
    virtual void endReceive(void *handle, bool fake) {
    }
    virtual void *beginSend(PartIdTy partId) {
        return nullptr;
    }
    virtual void endSend(void *handle, bool fake) {
    }

private:
    bool innerReceive(const GigStreamMessage &message, MultiCallErrorCode ec);
    friend class GigStreamHandlerBase;
    friend class GigStreamRequest;

private:
    bool appendMessage(const GigStreamMessage &message, MultiCallErrorCode ec);
    PartMessageQueue &getPartQueue(PartIdTy partId);
    bool tryNotify(PartIdTy partId);
    bool doReceive(PartMessageQueue &partQueue, size_t &receivedCount, size_t maxMessageCount);

private:
    StreamRpcParam _param;
    mutable autil::ReadWriteLock _asyncQueueLock;
    std::unordered_map<PartIdTy, PartMessageQueue> _partAsyncQueues;
};

MULTI_CALL_TYPEDEF_PTR(GigStreamBase);

class StreamProcessScope
{
public:
    StreamProcessScope(bool send, PartIdTy partId, GigStreamBase *stream)
        : _send(send)
        , _partId(partId)
        , _stream(stream) {
        if (_stream) {
            if (_send) {
                _handle = _stream->beginSend(_partId);
            } else {
                _handle = _stream->beginReceive(_partId);
            }
        }
    }
    ~StreamProcessScope() {
        if (_stream) {
            if (_send) {
                _stream->endSend(_handle, _fakeEvent);
            } else {
                _stream->endReceive(_handle, _fakeEvent);
            }
        }
    }
    void setFakeEvent(bool fake) {
        _fakeEvent = fake;
    }

private:
    bool _send;
    bool _fakeEvent = false;
    PartIdTy _partId;
    void *_handle = nullptr;
    GigStreamBase *_stream = nullptr;
};

} // namespace multi_call
