#pragma once

#include <list>

#include "aios/network/accl-barex/include/accl/barex/barex_types.h"
#include "aios/network/arpc/arpc/CommonMacros.h"
#include "aios/network/arpc/arpc/SyncClosure.h"
#include "aios/network/rdma/accl/RdmaChannelClosure.h"
#include "autil/Lock.h"
#include "autil/Log.h"

namespace accl {
namespace barex {
class XConnection;
class XConnectionCallback;
class XMessage;
}  // namespace barex
}  // namespace accl

ARPC_BEGIN_NAMESPACE(arpc);

class RdmaChannelContext;
class RdmaRequest;
class RdmaChannelClientCallback;
class RdmaMempool;
class TimerManager;
class AcclClientWrapper;
class ClientRPCStats;

enum ChannelState {
    CHANNEL_CONNECTING = 0,
    CHANNEL_CONNECTED  = 1,
    CHANNEL_INIT       = 2,
    CHANNEL_BROKEN     = 3,
};

class RdmaChannelClient: public std::enable_shared_from_this<RdmaChannelClient> {
public:
    RdmaChannelClient(::accl::barex::TcpProtoType               protocol,
                      const std::string&                        ip,
                      int32_t                                   port,
                      const std::shared_ptr<TimerManager>&      timerManager,
                      const std::shared_ptr<AcclClientWrapper>& client);

    virtual ~RdmaChannelClient();

public:
    void         Init();
    virtual void Disconnect();
    virtual void Close();

    bool ChannelConnected() {
        return _state == ChannelState::CHANNEL_CONNECTED;
    }
    bool ChannelBroken() {
        return _state == ChannelState::CHANNEL_BROKEN;
    }

    virtual void ConnectFailed(const std::shared_ptr<::accl::barex::XConnection>& connection);
    virtual void ConnectSuccess(const std::shared_ptr<::accl::barex::XConnection>& connection);

    virtual void Post(const std::shared_ptr<RdmaRequest>&        request,
                      const std::shared_ptr<RdmaChannelClosure>& done,
                      const std::shared_ptr<ClientRPCStats>&     stats);

    virtual void HandleRecvResponse(const std::shared_ptr<::accl::barex::XConnection>& connection,
                                    const ::accl::barex::XMessage&                     msgs);

private:
    void DoInit();
    void SendRequest(uint32_t requestId, const std::shared_ptr<RdmaChannelContext>& context);
    bool GenerateChannelContext(const std::shared_ptr<RdmaChannelContext>& context);

    void HandleSendRequestFailed(uint32_t requestId, RdmaChannelClosure::ErrorState channelError);
    void HandleTimeoutRequest(uint32_t requestId, double timeoutMs);

    void FailInflightRequests();
    void SetInflightRequest(uint32_t requestId, const std::shared_ptr<RdmaChannelContext>& context);
    std::shared_ptr<RdmaChannelContext> PopInflightRequest(uint32_t requestId);

    void SetPendingSendTasks(uint32_t requestId, const std::shared_ptr<RdmaChannelContext>& context);
    void SendPendingSendTasks();
    void FailPendingSendTasks();
    std::shared_ptr<RdmaChannelContext> PopPendingRequest(uint32_t requestId);

protected:
    // const data after init
    ::accl::barex::TcpProtoType                         _protocol{::accl::barex::TCP_V4};
    std::string                                         _ip;
    uint16_t                                            _port;
    std::shared_ptr<TimerManager>                       _timerManager;
    std::weak_ptr<AcclClientWrapper>                    _client;
    std::shared_ptr<::accl::barex::XConnectionCallback> _callback;
    std::shared_ptr<RdmaMempool>                        _mempool;

    // data plane, change when call post or recv
    std::atomic<uint32_t> _requestId;

    autil::ThreadMutex                                      _inflightRequestsLock;
    std::map<uint32_t, std::shared_ptr<RdmaChannelContext>> _inflightRequests;

    autil::ThreadMutex                                      _pendingSendTasksLock;
    std::map<uint32_t, std::shared_ptr<RdmaChannelContext>> _pendingSendTasks;

    // control plane, change when connection state change
    std::atomic<ChannelState> _state;

    autil::ReadWriteLock                        _connectionLock;
    std::shared_ptr<::accl::barex::XConnection> _connection;

    std::once_flag _initOnce;

private:
    // for mock test
    RdmaChannelClient(const std::shared_ptr<TimerManager>&      timerManager,
                      const std::shared_ptr<AcclClientWrapper>& client);

private:
    AUTIL_LOG_DECLARE();
    friend class RdmaChannelClientTest;
    friend class RdmaChannelClientOnMockTest_testConnectionStateChange_Test;
    friend class RdmaChannelClientOnMockTest_testPost_ChannelStateError_Test;
};

ARPC_END_NAMESPACE(arpc)