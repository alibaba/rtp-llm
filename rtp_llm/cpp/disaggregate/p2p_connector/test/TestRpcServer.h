#pragma once

#include "grpc++/grpc++.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"
#include <atomic>

namespace rtp_llm {

// 测试用RpcService，用于模拟RPC服务
class TestRpcService final: public RpcService::Service {
public:
    ::grpc::Status BroadcastTp(::grpc::ServerContext*        context,
                               const ::BroadcastTpRequestPB* request,
                               ::BroadcastTpResponsePB*      response);
    ::grpc::Status StartLoad(::grpc::ServerContext*                  context,
                             const ::P2PConnectorStartLoadRequestPB* request,
                             ::P2PConnectorStartLoadResponsePB*      response);
    void           setSleepMillis(int ms);
    void           setP2PResponseSuccess(bool success);
    void           setStartLoadResponseSuccess(bool success);
    void           setRpcResponseStatus(const ::grpc::Status& status);

    // 调用计数相关方法
    int  getBroadcastTpCallCount() const;
    int  getBroadcastTpCancelCallCount() const;
    int  getStartLoadCallCount() const;
    void resetCallCounts();

private:
    int              sleep_millis_{0};
    bool             p2p_response_success_{true};
    bool             start_load_response_success_{true};
    ::grpc::Status   rpc_response_status_{::grpc::Status::OK};
    std::atomic<int> broadcast_tp_call_count_{0};
    std::atomic<int> broadcast_tp_cancel_call_count_{0};
    std::atomic<int> start_load_call_count_{0};
};

class TestRpcServer {
public:
    TestRpcServer(std::unique_ptr<TestRpcService> service): service_(std::move(service)) {}
    ~TestRpcServer() {
        shutdown();
    }

public:
    bool start();

    int listenPort() const;

    TestRpcService* service() const;

private:
    void shutdown();

private:
    std::unique_ptr<TestRpcService> service_;
    std::unique_ptr<grpc::Server>   server_;
    int                             listen_port_{0};
};

}  // namespace rtp_llm