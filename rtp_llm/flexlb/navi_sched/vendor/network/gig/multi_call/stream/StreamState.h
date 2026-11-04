#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "autil/Lock.h"

namespace multi_call {

enum HandlerOp {
    HO_RUN,
    HO_CANCEL,
    HO_SEND_CANCEL,
    HO_SEND,
    HO_FINISH,
    HO_IDLE,
    HO_FINISHED,
};

enum FinishStatus {
    FS_INIT,
    FS_RUNNING,
    FS_CANCEL_TRIGGER,
    FS_CANCELLING,
    FS_FINISHING,
    FS_FINISHED,
};

class StreamState
{
public:
    StreamState();
    ~StreamState();

private:
    StreamState(const StreamState &);
    StreamState &operator=(const StreamState &);

public:
    // state change
    HandlerOp next();
    void initFailed();
    void endInit();
    bool startCancel();
    void endCancel();
    void endFinish();
    void disableReceive();
    bool isDisableReceive() const;

public:
    // input change
    void setSendEof();
    bool sendEof() const;
    void setSendFinished();
    void setReceiveEof();
    bool receiveEof() const;
    void setReceiveFinished();
    void setReceiveCancel();
    bool bidiEof() const;
    bool running() const;
    void setErrorCode(MultiCallErrorCode ec);
    MultiCallErrorCode getErrorCode() const;
    bool hasError() const;
    bool isCancelled() const;
    void logState(bool useError = false) const;
    void setHandlerId(int64_t handlerId);

private:
    HandlerOp nextRunning();

public:
    static const char *getOpString(HandlerOp op);
    static const char *getStatusString(FinishStatus status);

private:
    mutable autil::ReadWriteLock _lock;
    FinishStatus _finishStatus;
    MultiCallErrorCode _ec;
    bool _sendEof;
    bool _sendFinished;
    bool _receiveEof;
    bool _receiveFinished;
    bool _initFailed;
    bool _cancelled;
    bool _finished;
    bool _receiveCancel;
    bool _disableReceive;
    int64_t _handlerId;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(StreamState);

} // namespace multi_call

