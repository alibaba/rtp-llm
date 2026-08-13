#pragma once

#include <atomic>
#include <chrono>
#include <thread>
#include <gtest/gtest.h>
#include "grpc++/grpc++.h"

#include "rtp_llm/cpp/model_rpc/RPCPool.h"

namespace rtp_llm::test {

// 测试用RpcService，用于模拟RPC服务
class TestRpcService final: public RpcService::Service {
public:
    ::grpc::Status ExecuteFunction(::grpc::ServerContext*     context,
                                   const ::FunctionRequestPB* request,
                                   ::FunctionResponsePB*      response) override {
        if (request->has_mem_request()
            && request->mem_request().operation_kind() == MemoryOperationRequestPB::CAPABILITY) {
            ++capability_call_count_;
            if (capability_sleep_millis_ > 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(capability_sleep_millis_));
            }
            if (context->IsCancelled()) {
                return ::grpc::Status(grpc::StatusCode::CANCELLED, "request cancelled");
            }
            if (!capability_supported_) {
                response->mutable_mem_response()->set_success(true);
                return ::grpc::Status::OK;
            }
            auto* mem_response = response->mutable_mem_response();
            mem_response->set_operation_id(request->mem_request().operation_id());
            mem_response->set_success(true);
            mem_response->set_quiesced(true);
            mem_response->set_protocol_version(request->mem_request().protocol_version());
            return capability_rpc_status_;
        }
        if (request->has_mem_request()
            && request->mem_request().operation_kind() == MemoryOperationRequestPB::QUIESCE) {
            const bool quiesce_succeeded = quiesce_response_success_.load();
            ++quiesce_call_count_;
            if (!quiesce_succeeded) {
                ++quiesce_failure_count_;
            }
            auto* mem_response = response->mutable_mem_response();
            mem_response->set_operation_id(request->mem_request().operation_id());
            mem_response->set_success(quiesce_succeeded);
            mem_response->set_quiesced(quiesce_succeeded);
            mem_response->set_protocol_version(request->mem_request().protocol_version());
            return ::grpc::Status::OK;
        }
        if (request->has_mem_request()
            && request->mem_request().operation_kind() == MemoryOperationRequestPB::COPY) {
            ++copy_call_count_;
        }
        if (sleep_millis_ > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_millis_));
        }
        if (context->IsCancelled()) {
            return ::grpc::Status(grpc::StatusCode::CANCELLED, "request cancelled");
        }
        auto* mem_response = response->mutable_mem_response();
        mem_response->set_success(mem_response_success_);
        if (request->has_mem_request()) {
            mem_response->set_operation_id(request->mem_request().operation_id());
            mem_response->set_quiesced(true);
            mem_response->set_protocol_version(request->mem_request().protocol_version());
        }
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
    void setCapabilitySleepMillis(int ms) {
        capability_sleep_millis_ = ms;
    }
    void setCapabilitySupported(bool supported) {
        capability_supported_ = supported;
    }
    void setCapabilityRpcStatus(const ::grpc::Status& status) {
        capability_rpc_status_ = status;
    }
    void setQuiesceResponseSuccess(bool success) {
        quiesce_response_success_.store(success);
    }
    int capabilityCallCount() const {
        return capability_call_count_.load();
    }
    int copyCallCount() const {
        return copy_call_count_.load();
    }
    int quiesceCallCount() const {
        return quiesce_call_count_.load();
    }
    int quiesceFailureCount() const {
        return quiesce_failure_count_.load();
    }

private:
    int               sleep_millis_{0};
    int               capability_sleep_millis_{0};
    bool              mem_response_success_{true};
    bool              capability_supported_{true};
    ::grpc::Status    rpc_response_status_{::grpc::Status::OK};
    ::grpc::Status    capability_rpc_status_{::grpc::Status::OK};
    std::atomic<bool> quiesce_response_success_{true};
    std::atomic<int>  capability_call_count_{0};
    std::atomic<int>  copy_call_count_{0};
    std::atomic<int>  quiesce_call_count_{0};
    std::atomic<int>  quiesce_failure_count_{0};
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

    TestRpcService* service() const {
        return service_.get();
    }

private:
    void shutdown() {
        if (server_) {
            // Use a bounded shutdown to avoid rare hangs in tests if there are still in-flight RPCs.
            server_->Shutdown(std::chrono::system_clock::now() + std::chrono::seconds(1));
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
