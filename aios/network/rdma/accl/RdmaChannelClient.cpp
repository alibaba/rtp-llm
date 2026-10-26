#include "aios/network/rdma/accl/RdmaChannelClient.h"

#include "aios/network/accl-barex/include/accl/barex/barex.h"
#include "aios/network/arpc/arpc/metric/ClientRPCStats.h"
#include "aios/network/rdma/accl/AcclClientWrapper.h"
#include "aios/network/rdma/accl/RdmaChannelClientCallback.h"
#include "aios/network/rdma/accl/RdmaChannelContext.h"
#include "aios/network/rdma/accl/RdmaMessage.h"
#include "aios/network/rdma/accl/RdmaRequest.h"
#include "aios/network/rdma/accl/TimerManager.h"
#include "alog/Logger.h"
#include "autil/NetUtil.h"

ARPC_BEGIN_NAMESPACE(arpc)
AUTIL_LOG_SETUP(arpc, RdmaChannelClient);

RdmaChannelClient::RdmaChannelClient(::accl::barex::TcpProtoType               protocol,
                                     const std::string&                        ip,
                                     int32_t                                   port,
                                     const std::shared_ptr<TimerManager>&      timerManager,
                                     const std::shared_ptr<AcclClientWrapper>& client):
    _protocol(protocol),
    _ip(ip),
    _port(port),
    _timerManager(timerManager),
    _client(client),
    _requestId(0),
    _state(CHANNEL_INIT) {
    if (_ip == "localhost") {
        autil::NetUtil::GetDefaultIp(_ip);
    }
    if (!client) {
        _state.store(CHANNEL_BROKEN);
        return;
    }
    _mempool = client->GetMempool();
    if (!_mempool) {
        _state.store(CHANNEL_BROKEN);
    }
}

RdmaChannelClient::RdmaChannelClient(const std::shared_ptr<TimerManager>&      timerManager,
                                     const std::shared_ptr<AcclClientWrapper>& client):
    _timerManager(timerManager), _client(client), _requestId(0), _state(CHANNEL_INIT) {}

RdmaChannelClient::~RdmaChannelClient() {
    Close();
    _callback.reset();
}

void RdmaChannelClient::DoInit() {
    auto client = _client.lock();
    if (!client) {
        _state.store(CHANNEL_BROKEN);
        return;
    }

    _callback = std::make_shared<RdmaChannelClientCallback>(shared_from_this(), _mempool);
    auto ret  = client->Connect(_protocol, _ip, _port, _callback);
    if (ret) {
        auto curState = _state.load();
        if (curState == CHANNEL_INIT && _state.compare_exchange_strong(curState, CHANNEL_CONNECTING)) {
            return;
        }
        AUTIL_LOG(WARN, "accl connect to %s:%d failed, curState is %d", _ip.c_str(), _port, curState);
    }
    _state = CHANNEL_BROKEN;
    AUTIL_LOG(WARN, "accl connect to %s:%d failed", _ip.c_str(), _port);
}

void RdmaChannelClient::Init() {
    std::call_once(_initOnce, [this]() { this->DoInit(); });
}

void RdmaChannelClient::Close() {
    _state.store(CHANNEL_BROKEN);

    std::shared_ptr<::accl::barex::XConnection> connection;
    {
        autil::ScopedWriteLock lock(_connectionLock);
        connection = _connection;
        _connection.reset();
    }
    if (connection) {
        connection->Close();
    }

    FailInflightRequests();
    FailPendingSendTasks();
}

void RdmaChannelClient::Disconnect() {
    _state.store(CHANNEL_BROKEN);

    std::shared_ptr<::accl::barex::XConnection> connection;
    {
        autil::ScopedReadLock lock(_connectionLock);
        connection = _connection;
    }

    auto client = _client.lock();
    if (client && connection) {
        client->Disconnect(connection);
    }
}

void RdmaChannelClient::ConnectFailed(const std::shared_ptr<::accl::barex::XConnection>& connection) {
    AUTIL_LOG(INFO, "client connect failed or disconnect, %s:%d", _ip.c_str(), _port);
    Close();
}

void RdmaChannelClient::ConnectSuccess(const std::shared_ptr<::accl::barex::XConnection>& connection) {
    {
        autil::ScopedWriteLock lock(_connectionLock);
        _connection = connection;
    }
    _state.store(CHANNEL_CONNECTED);
    SendPendingSendTasks();
    AUTIL_LOG(INFO, "client connect %s:%d success", _ip.c_str(), _port);
}

void RdmaChannelClient::Post(const std::shared_ptr<RdmaRequest>&        request,
                             const std::shared_ptr<RdmaChannelClosure>& done,
                             const std::shared_ptr<ClientRPCStats>&     stats) {
    if (!done) {
        AUTIL_LOG(WARN, "rdma channel client post failed, closure nullptr");
        return;
    }

    if (!request) {
        AUTIL_LOG(WARN, "rdma channel client post failed, request nullptr");
        done->Run(RdmaChannelClosure::ERROR_BAD_PACKET);
        return;
    }

    uint32_t requestId = _requestId++;
    request->SetRequestId(requestId);
    if (stats) {
        stats->setRequestId(requestId);
    }

    auto context = std::make_shared<RdmaChannelContext>(request, done, _timerManager, stats);
    if (!context) {
        AUTIL_LOG(WARN, "rdma channel client post failed, new context failed");
        done->Run(RdmaChannelClosure::ERROR_CONNECTION_FAILED);
        return;
    }

    // 已经broken 或是 还没有触发连接操作的请求直接返回
    auto curState = _state.load();
    if (curState == CHANNEL_BROKEN || curState == CHANNEL_INIT) {
        AUTIL_INTERVAL_LOG(100, WARN, "rdma channel client post failed, channel broken");
        context->RunFailed(RdmaChannelClosure::ERROR_CONNECTION_FAILED);
        return;
    }

    // timer
    auto strongThis    = shared_from_this();
    auto timeoutMs     = request->GetTimeoutMs();
    auto timerCallback = [strongThis, requestId, timeoutMs] { strongThis->HandleTimeoutRequest(requestId, timeoutMs); };
    auto timer         = _timerManager->addTimer(timeoutMs, std::move(timerCallback));
    if (timer == nullptr) {
        AUTIL_LOG(WARN, "rdma channel client post failed, add timer failed");
        context->RunFailed(RdmaChannelClosure::ERROR_UNKNOWN);
        return;
    }
    context->SetTimer(timer);

    // 已经连接的请求直接发出去, 即使发出去的过程中触发了ConnectFailed, 也会在Send之前控制住.
    if (curState == CHANNEL_CONNECTED) {
        SendRequest(requestId, context);
        return;
    }

    // CHANNEL_CONNECTING状态的请求加入Pending队列
    // 可能存在在加入队列的过程中, state变成BROKEN或是CONNECTTED的情况.
    if (stats) {
        stats->markRequestPendingBegin();
    }
    SetPendingSendTasks(requestId, context);
}

void RdmaChannelClient::SendRequest(uint32_t requestId, const std::shared_ptr<RdmaChannelContext>& context) {
    auto stats = context ? context->stats() : nullptr;
    if (stats) {
        stats->markRequestPendingDone();
        stats->markRequestPackBegin();
    }

    if (!GenerateChannelContext(context)) {
        return;
    }

    auto message = context->GetMessage();
    if (!message) {
        context->RunFailed(RdmaChannelClosure::ERROR_BAD_PACKET);
        return;
    }

    auto xmessage = message->GetMessage();
    xmessage.SetIovsDeadlineMs(context->GetMessageDeadlineMs());
    if (stats) {
        stats->markRequestPackDone();
        stats->markRequestSendBegin();
        message->reportClientRPCStats(stats, xmessage);
    }

    auto strongThis = shared_from_this();
    auto sendDone   = [strongThis, requestId, context, stats](::accl::barex::Status status) {
        if (stats) {
            stats->markRequestSendDone();
        }

        if (!status.IsOk()) {
            auto channelError = RdmaChannelClosure::ERROR_BAD_PACKET;
            if (status.ErrCode() == ::accl::barex::BAREX_ERR_RATE_LIMITED) {
                channelError = RdmaChannelClosure::ERROR_ACCL_SERVER_RATE_LIMIT;
            }
            strongThis->HandleSendRequestFailed(requestId, channelError);
        }
        context->ReleaseMessageMemory();
    };

    _connectionLock.rdlock();
    if (!_connection || _state.load() != CHANNEL_CONNECTED) {
        _connectionLock.unlock();
        context->RunFailed(RdmaChannelClosure::ERROR_CONNECTION_FAILED);
        return;
    }

    // 先Set再Send保证即使SendDone直接被回调也能找到context
    SetInflightRequest(requestId, context);
    _connection->Send(xmessage, sendDone);
    _connectionLock.unlock();
}

bool RdmaChannelClient::GenerateChannelContext(const std::shared_ptr<RdmaChannelContext>& context) {
    auto& request = context->GetRequest();
    request->SetConnection(_connection);
    request->SetMempool(_mempool);

    // message & request Mem
    auto message = request->Encode();
    if (!message) {
        AUTIL_LOG(WARN, "rdma channel client encode request to RdmaMessage failed");
        context->RunFailed(RdmaChannelClosure::ERROR_BAD_PACKET);
        return false;
    }
    context->SetMessage(message);
    return true;
}

void RdmaChannelClient::FailInflightRequests() {
    _inflightRequestsLock.lock();
    auto temp = _inflightRequests;
    _inflightRequests.clear();
    _inflightRequestsLock.unlock();

    auto request = temp.begin();
    while (request != temp.end()) {
        if (request->second) {
            request->second->RunFailed(RdmaChannelClosure::ERROR_CONNECTION_FAILED);
        }
        request++;
    }
}

void RdmaChannelClient::HandleRecvResponse(const std::shared_ptr<::accl::barex::XConnection>& connection,
                                           const ::accl::barex::XMessage&                     msgs) {
    auto beginTimeUs = autil::TimeUtility::currentTimeInMicroSeconds();

    auto message = std::make_shared<RdmaMessage>();
    message->SetConnection(connection);
    message->SetMempool(_mempool);

    // ParseRecvBuffer, XMessage mem lifecycle to rdmaMessage
    if (!message->SetRecvMessage(msgs)) {
        AUTIL_LOG(WARN, "HandleRecvResponse, decode from response failed");
        return;
    }

    uint32_t requestId = message->GetRequestId();

    // 检测是否超时
    auto context = PopInflightRequest(requestId);
    if (!context) {
        AUTIL_LOG(WARN, "call handle recv response timeout, requestId %d, this = %p", requestId, this);
        return;
    }
    auto stats = context->stats();
    if (stats) {
        stats->markResponsePopBegin(beginTimeUs);
        stats->markResponsePopDone();
    }

    if (stats) {
        stats->markResponseUnpackBegin();
    }
    bool decodeSuccess = context->GetRequest()->Decode(message);
    if (stats) {
        stats->markResponseUnpackDone();
    }

    if (!message->IsFailedMessage()) {
        if (decodeSuccess) {
            context->RunSuccess();
        } else {
            AUTIL_LOG(WARN, "HandleRecvResponse request decode failed");
            context->RunFailed(RdmaChannelClosure::ERROR_BAD_PACKET);
        }
    } else {
        auto errorState = decodeSuccess ? RdmaChannelClosure::ERROR_SERVER_ERROR : RdmaChannelClosure::ERROR_BAD_PACKET;
        AUTIL_LOG(WARN,
                  "HandleRecvResponse message status failed, peer: %s, error state: %d",
                  connection->GetPeerAddrAndPort().first.c_str(),
                  errorState);
        context->RunFailed(errorState);
    }
    return;
}

void RdmaChannelClient::HandleSendRequestFailed(uint32_t requestId, RdmaChannelClosure::ErrorState channelError) {
    auto context = PopInflightRequest(requestId);
    if (!context) {
        AUTIL_LOG(INFO, "failed request, can not find context, requestId = %d, this = %p", requestId, this);
        return;
    }
    AUTIL_LOG(WARN, "rdma channel handle client send request failed, erroCode %d", channelError);
    context->RunFailed(channelError);
}

void RdmaChannelClient::HandleTimeoutRequest(uint32_t requestId, double timeoutMs) {
    auto context = PopInflightRequest(requestId);
    if (!context) {
        context = PopPendingRequest(requestId);
        if (!context) {
            AUTIL_LOG(INFO, "timeout, can not find context, requestId = %d, this = %p", requestId, this);
            return;
        }
        if (auto stats = context->stats(); stats) {
            stats->markRequestPendingDone();
        }
    }
    AUTIL_LOG(WARN,
              "rdma channel handle client send request timeout, requestId = %d, this = %p, timeout is %fms",
              requestId,
              this,
              timeoutMs);
    context->RunFailed(RdmaChannelClosure::ERROR_TIMEOUT);
}

void RdmaChannelClient::SetInflightRequest(uint32_t requestId, const std::shared_ptr<RdmaChannelContext>& context) {
    _inflightRequestsLock.lock();
    _inflightRequests[requestId] = context;
    _inflightRequestsLock.unlock();
}

std::shared_ptr<RdmaChannelContext> RdmaChannelClient::PopInflightRequest(uint32_t requestId) {
    std::shared_ptr<RdmaChannelContext> context;
    _inflightRequestsLock.lock();
    auto iter = _inflightRequests.find(requestId);
    if (iter != _inflightRequests.end()) {
        context = iter->second;
        _inflightRequests.erase(requestId);
    }
    _inflightRequestsLock.unlock();
    return context;
}

void RdmaChannelClient::SetPendingSendTasks(uint32_t requestId, const std::shared_ptr<RdmaChannelContext>& context) {
    _pendingSendTasksLock.lock();
    _pendingSendTasks[requestId] = context;
    _pendingSendTasksLock.unlock();

    // set新的pending task后检查一下, 防止执行set的过程中状态变化
    if (_state.load() == CHANNEL_CONNECTED) {
        SendPendingSendTasks();
    }

    if (_state.load() == CHANNEL_BROKEN) {
        FailPendingSendTasks();
    }
}

void RdmaChannelClient::SendPendingSendTasks() {
    // 加锁拷贝一份副本，防止SendRequestMem阻塞
    std::map<uint32_t, std::shared_ptr<RdmaChannelContext>> temporaryPendingSendTasks;
    _pendingSendTasksLock.lock();
    temporaryPendingSendTasks = _pendingSendTasks;
    _pendingSendTasks.clear();
    _pendingSendTasksLock.unlock();

    for (auto iter = temporaryPendingSendTasks.begin(); iter != temporaryPendingSendTasks.end(); iter++) {
        auto requestId = iter->first;
        auto context   = iter->second;
        SendRequest(requestId, context);
    }
}

void RdmaChannelClient::FailPendingSendTasks() {
    _pendingSendTasksLock.lock();
    auto temp = _pendingSendTasks;
    _pendingSendTasks.clear();
    _pendingSendTasksLock.unlock();

    auto request = temp.begin();
    while (request != temp.end()) {
        if (request->second) {
            request->second->RunFailed(RdmaChannelClosure::ERROR_CONNECTION_FAILED);
        }
        request++;
    }
}

std::shared_ptr<RdmaChannelContext> RdmaChannelClient::PopPendingRequest(uint32_t requestId) {
    std::shared_ptr<RdmaChannelContext> context;
    _pendingSendTasksLock.lock();
    auto iter = _pendingSendTasks.find(requestId);
    if (iter != _pendingSendTasks.end()) {
        context = iter->second;
        _pendingSendTasks.erase(requestId);
    }
    _pendingSendTasksLock.unlock();
    return context;
}

ARPC_END_NAMESPACE(arpc)