#pragma once

#include <thread>
#include <gtest/gtest.h>
#include "grpc++/grpc++.h"

#include "rtp_llm/cpp/model_rpc/RPCPool.h"

namespace rtp_llm::test {

// 测试用RpcService，用于模拟RPC服务
class TestRpcService final: public RpcService::Service {
public:
    ::grpc::Status BroadcastTp(::grpc::ServerContext*        context,
                               const ::BroadcastTpRequestPB* request,
                               ::BroadcastTpResponsePB*      response) override {
        if (sleep_millis_ > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_millis_));
        }
        if (context->IsCancelled()) {
            return ::grpc::Status(grpc::StatusCode::CANCELLED, "request cancelled");
        }
        response->mutable_mem_response()->set_success(mem_response_success_);
        return rpc_response_status_;
    }
    void setSleepMillis(int ms) {
        sleep_millis_ = ms;
    }
    void setMemResponseSuccess(bool success) {
        mem_response_success_ = success;
    }
    void setRpcResponseStatus(const ::grpc::Status& status) {
        rpc_response_status_ = status;
    }

private:
    int            sleep_millis_{0};
    bool           mem_response_success_{true};
    ::grpc::Status rpc_response_status_{::grpc::Status::OK};
};

class TestRpcServer {
public:
    TestRpcServer(std::unique_ptr<TestRpcService> service): service_(std::move(service)) {}
    ~TestRpcServer() {
        shutdown();
    }

public:
    bool start() {
        if (!service_) {
            return false;
        }

        std::string         bind_addr = "0.0.0.0:0";
        grpc::ServerBuilder builder;
        builder.AddListeningPort(bind_addr, grpc::InsecureServerCredentials(), &listen_port_);
        builder.RegisterService(service_.get());
        server_ = builder.BuildAndStart();
        if (!server_ || listen_port_ == 0) {
            return false;
        }
        return true;
    }

    int listenPort() const {
        return listen_port_;
    }

private:
    void shutdown() {
        if (server_) {
            server_->Shutdown();
            server_->Wait();
            server_.reset();
        }
    }

private:
    std::unique_ptr<TestRpcService> service_;
    std::unique_ptr<grpc::Server>   server_;
    int                             listen_port_{0};
};

}  // namespace rtp_llm::test