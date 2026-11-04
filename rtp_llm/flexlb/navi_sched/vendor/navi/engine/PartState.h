#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/atomic.h"
#include "autil/Lock.h"
#include "navi/common.h"
#include "navi/proto/GraphDef.pb.h"
#include "navi/engine/PartInfo.h"
#include "navi/util/ReadyBitMap.h"

namespace navi {

class MessageCounter
{
public:
    MessageCounter();
public:
    void incMessageCount();
    void decMessageCount();
    int64_t messageCount() const;
private:
    atomic64_t _messageCount;
};

class BorderPartState : public MessageCounter
{
public:
    BorderPartState(size_t edgeCount);
    ~BorderPartState();
public:
    void setEof(size_t edgeId);
    bool eof() const;
private:
    InlineReadyBitMap _portEofMap;
};

class BorderState
{
public:
    BorderState();
    ~BorderState();
public:
    bool init(const PartInfo &partInfo, size_t edgeCount);
    void setEof(NaviPartId partId, size_t edgeId);
    bool eof(NaviPartId partId) const;
    bool eof() const;
    void incMessageCount(NaviPartId partId);
    void decMessageCount(NaviPartId partId);
    int64_t messageCount(NaviPartId partId) const;
private:
    PartInfo _partInfo;
    BorderPartState *_partStates = nullptr;
};


class StreamPartState : public MessageCounter
{
public:
    StreamPartState();
    ~StreamPartState();
public:
    bool inited() const;
    void setInited();
public:
    autil::RecursiveThreadMutex _lock;
    bool _sending;
    bool _collected;
private:
    bool _inited;
};

class StreamState
{
public:
    StreamState();
    ~StreamState();
public:
    bool init();
    bool streamInited() const;
    StreamPartState &getPartState(NaviPartId partId);
    void lock(NaviPartId partId);
    void unlock(NaviPartId partId);
    void setPartCount(NaviPartId partCount);
    NaviPartId getPartCount() const;
    bool inited(NaviPartId partId) const;
    void setInited(NaviPartId partId);
    void incMessageCount(NaviPartId partId);
    void decMessageCount(NaviPartId partId);
    bool noMessage(NaviPartId partId) const;
    bool setCollected(NaviPartId partId);
    void setSendEof(NaviPartId partId);
    bool sendEof(NaviPartId partId) const;
    bool sendEof() const;
    void setReceiveEof(NaviPartId partId);
    bool receiveEof(NaviPartId partId) const;
    bool receiveEof() const;
    void receiveCancel(NaviPartId partId);
    bool cancelled(NaviPartId partId) const;
private:
    NaviPartId _partCount;
    StreamPartState *_partStates;
    InlineReadyBitMap _partSendEofMap;
    InlineReadyBitMap _partReceiveEofMap;
    InlineReadyBitMap _partCancelMap;
};

class StreamStatePartScopedLock
{
public:
    explicit StreamStatePartScopedLock(StreamState &streamState, NaviPartId partId)
        : _streamState(streamState)
        , _partId(partId) {
        _needLock = _streamState.streamInited();
        if (_needLock) {
            _streamState.lock(_partId);
        }
    }
    StreamStatePartScopedLock(StreamPartState &state) = delete;
    StreamStatePartScopedLock& operator= (const StreamStatePartScopedLock &) = delete;
    ~StreamStatePartScopedLock() {
        if (_needLock) {
            _streamState.unlock(_partId);
        }
    }
private:
    StreamState &_streamState;
    NaviPartId _partId;
    bool _needLock;
};

class SendScope {
public:
    SendScope(StreamState &state, NaviPartId partId)
        : _state(state.getPartState(partId))
        , _success(false)
    {
        if (0 == _state._lock.trylock()) {
            if (!_state._sending) {
                _state._sending = true;
                _success = true;
            } else {
                _state._lock.unlock();
            }
        }
    }
    bool success() const {
        return _success;
    }
    ~SendScope() {
        if (_success) {
            _state._sending = false;
            _state._lock.unlock();
        }
    }
private:
    StreamPartState &_state;
    bool _success;
};

}
