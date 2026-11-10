#pragma once

#include "aios/network/accl-barex/include/accl/barex/barex_types.h"
#include "aios/network/arpc/arpc/CommonMacros.h"
#include "aios/network/rdma/accl/RdmaChannelClosure.h"
#include "autil/Log.h"
namespace accl {
namespace barex {
class XMessage;
class XSimpleMempool;
}  // namespace barex
}  // namespace accl

ARPC_BEGIN_NAMESPACE(arpc);

class RdmaChannelClient;
class RdmaRequest;
class RdmaMessage;
class Timer;
class TimerManager;
class ClientRPCStats;

class RdmaChannelContext: public std::enable_shared_from_this<RdmaChannelContext> {
public:
    RdmaChannelContext(const std::shared_ptr<RdmaRequest>&        req,
                       const std::shared_ptr<RdmaChannelClosure>& closure,
                       const std::shared_ptr<TimerManager>&       timerManager,
                       const std::shared_ptr<ClientRPCStats>&     stats);
    ~RdmaChannelContext();

public:
    void RunSuccess();
    void RunFailed(RdmaChannelClosure::ErrorState errorState);

    void SetTimer(const std::shared_ptr<Timer>& timer) {
        _timer = timer;
    }
    void SetMessage(const std::shared_ptr<RdmaMessage>& msg) {
        _message = msg;
    }
    const std::shared_ptr<RdmaMessage>& GetMessage() {
        return _message;
    }
    const std::shared_ptr<RdmaRequest>& GetRequest() {
        return _request;
    }

    void ReleaseMessageMemory();

    std::shared_ptr<ClientRPCStats> stats() {
        return _stats;
    };

    uint64_t GetMessageDeadlineMs();

private:
    void                       StopRequestTimer();
    ::accl::barex::BarexResult ReleaseRdmaMemory();
    void                       CheckMessageReleased();

private:
    std::shared_ptr<RdmaRequest>    _request;
    std::shared_ptr<ClientRPCStats> _stats;

    std::shared_ptr<RdmaChannelClosure> _done;

    std::shared_ptr<TimerManager> _timerManager;
    std::shared_ptr<Timer>        _timer;

    autil::ThreadMutex           _messageMutex;
    std::shared_ptr<RdmaMessage> _message;
    uint64_t                     _messageDeadlineMs;
    std::shared_ptr<Timer>       _messageTimer;

private:
    static uint64_t sDelayReleaseMessageMs;

private:
    AUTIL_LOG_DECLARE();
};

ARPC_END_NAMESPACE(arpc);