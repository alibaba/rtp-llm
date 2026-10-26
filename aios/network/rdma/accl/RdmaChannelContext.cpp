#include "aios/network/rdma/accl/RdmaChannelContext.h"

#include "aios/network/accl-barex/include/accl/barex/barex.h"
#include "aios/network/accl-barex/include/accl/barex/highlevel/xconnection.h"
#include "aios/network/accl-barex/include/accl/barex/highlevel/xmessage.h"
#include "aios/network/accl-barex/include/accl/barex/xsimple_mempool.h"
#include "aios/network/rdma/accl/RdmaChannelClient.h"
#include "aios/network/rdma/accl/RdmaMessage.h"
#include "aios/network/rdma/accl/RdmaRequest.h"
#include "aios/network/rdma/accl/TimerManager.h"
#include "autil/TimeUtility.h"

ARPC_BEGIN_NAMESPACE(arpc);
AUTIL_LOG_SETUP(arpc, RdmaChannelContext);

uint64_t RdmaChannelContext::sDelayReleaseMessageMs = 5000;  // 5s

RdmaChannelContext::RdmaChannelContext(const std::shared_ptr<RdmaRequest>&        req,
                                       const std::shared_ptr<RdmaChannelClosure>& closure,
                                       const std::shared_ptr<TimerManager>&       timerManager,
                                       const std::shared_ptr<ClientRPCStats>&     stats):
    _request(req), _stats(stats), _done(closure), _timerManager(timerManager), _messageDeadlineMs(UINT64_MAX) {}

RdmaChannelContext::~RdmaChannelContext() {}

void RdmaChannelContext::RunSuccess() {
    StopRequestTimer();
    if (_done) {
        _done->Run(RdmaChannelClosure::ERROR_NONE);
    }
    CheckMessageReleased();
}

void RdmaChannelContext::RunFailed(RdmaChannelClosure::ErrorState errorState) {
    StopRequestTimer();
    if (_done) {
        _done->Run(errorState);
    }
    CheckMessageReleased();
}

void RdmaChannelContext::StopRequestTimer() {
    if (_timer) {
        _timer->stop();
        _timer.reset();
    }
}

void RdmaChannelContext::ReleaseMessageMemory() {
    if (!_message) {
        return;
    }

    autil::ScopedLock lock(_messageMutex);
    if (_message) {
        _message->ReleaseMessage();
        _message.reset();
    }

    if (_messageTimer) {
        _messageTimer->stop();
        _messageTimer.reset();
    }
}

uint64_t RdmaChannelContext::GetMessageDeadlineMs() {
    if (_request && _messageDeadlineMs == UINT64_MAX) {
        _messageDeadlineMs = autil::TimeUtility::currentTimeInMilliSeconds() + _request->GetTimeoutMs() * 10;
    }
    return _messageDeadlineMs;
}

void RdmaChannelContext::CheckMessageReleased() {
    if (_message == nullptr || _messageDeadlineMs == UINT64_MAX) {
        return;
    }

    autil::ScopedLock lock(_messageMutex);
    if (_message == nullptr || _messageDeadlineMs == UINT64_MAX) {
        return;
    }

    // iov已经超时, 可以直接释放
    auto releaseTimeMs = _messageDeadlineMs + sDelayReleaseMessageMs;
    auto currentTimeMs = autil::TimeUtility::currentTimeInMilliSeconds();
    if (currentTimeMs >= releaseTimeMs) {
        _message->ReleaseMessage();
        _message.reset();
        return;
    }

    // iov 没有超时, 加个超时检查, 超时后强制释放
    auto context = shared_from_this();
    _messageTimer =
        _timerManager->addTimer(releaseTimeMs - currentTimeMs, [message = _message] { message->ReleaseMessage(); });
    if (_messageTimer == nullptr) {
        AUTIL_LOG(WARN, "add release message timer failed");
    }
}

ARPC_END_NAMESPACE(arpc);