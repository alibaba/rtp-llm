#include <atomic>
#include <chrono>
#include <thread>
#include <gtest/gtest.h>
#include "grpc++/grpc++.h"

#include "autil/NetUtil.h"
#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"

namespace rtp_llm::test {

// 测试用RpcService，用于模拟RPC服务
class TestRpcService final: public RpcService::Service {
public:
    ::grpc::Status ExecuteFunction(::grpc::ServerContext*     context,
                                   const ::FunctionRequestPB* request,
                                   ::FunctionResponsePB*      response) override {
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

// ---------------------------- BroadcastManagerTest ----------------------------

class BroadcastManagerTest: public ::testing::Test {
protected:
    void SetUp() override {
        for (int i = 0; i < 2; ++i) {
            ports_.push_back(autil::NetUtil::randomPort());
        }

        std::vector<std::string> worker_addrs;
        for (int i = 0; i < 2; ++i) {
            worker_addrs.push_back("127.0.0.1:" + std::to_string(ports_[i]));
        }

        manager_ = std::make_unique<BroadcastManager>(worker_addrs);
        ASSERT_TRUE(manager_->init());
    }
    void TearDown() override {
        manager_.reset();
    }

private:
    std::unique_ptr<BroadcastManager> manager_;
    std::vector<int>                  ports_;
};

// ---------------------------- init ----------------------------

TEST_F(BroadcastManagerTest, Init_ReturnFalse_EmptyWorkerAddrs) {
    std::vector<std::string> empty_addrs;
    auto                     manager = std::make_unique<BroadcastManager>(empty_addrs);
    ASSERT_FALSE(manager->init());
}

TEST_F(BroadcastManagerTest, Init_ReturnTrue_ValidWorkerAddrs) {
    std::vector<std::string> worker_addrs;
    worker_addrs.push_back("127.0.0.1:12345");
    auto manager = std::make_unique<BroadcastManager>(worker_addrs);
    ASSERT_TRUE(manager->init());
    ASSERT_EQ(manager->workerNum(), 1u);
    ASSERT_NE(manager->rpc_pool_, nullptr);
}

// ---------------------------- broadcast ----------------------------

TEST_F(BroadcastManagerTest, Broadcast_ReturnNull_RequestsSizeMismatch) {
    std::vector<FunctionRequestPB> requests(1);
    auto                           rpc_call = [](const std::shared_ptr<RpcService::Stub>&    stub,
                       const std::shared_ptr<grpc::ClientContext>& ctx,
                       const FunctionRequestPB&                    req,
                       grpc::CompletionQueue* cq) { return stub->AsyncExecuteFunction(ctx.get(), req, cq); };
    auto result = manager_->broadcast<FunctionRequestPB, FunctionResponsePB>(requests, /*timeout_ms=*/100, rpc_call);
    EXPECT_EQ(result, nullptr);
}

TEST_F(BroadcastManagerTest, Broadcast_ReturnNull_GetConnectionFailed) {
    std::vector<std::string> empty_addrs;
    auto                     manager = std::make_unique<BroadcastManager>(empty_addrs);

    std::vector<FunctionRequestPB> requests(3);
    auto                           rpc_call = [](const std::shared_ptr<RpcService::Stub>&    stub,
                       const std::shared_ptr<grpc::ClientContext>& ctx,
                       const FunctionRequestPB&                    req,
                       grpc::CompletionQueue* cq) { return stub->AsyncExecuteFunction(ctx.get(), req, cq); };
    auto result = manager->broadcast<FunctionRequestPB, FunctionResponsePB>(requests, /*timeout_ms=*/500, rpc_call);
    ASSERT_EQ(result, nullptr);
}

TEST_F(BroadcastManagerTest, Broadcast_ReturnNotNull_AllRequestsSuccess) {
    std::vector<std::unique_ptr<TestRpcServer>> servers;
    std::vector<std::string>                    server_addrs;
    for (int i = 0; i < 3; ++i) {
        auto service = std::make_unique<TestRpcService>();
        auto server  = std::make_unique<TestRpcServer>(std::move(service));
        ASSERT_TRUE(server->start());
        server_addrs.push_back("127.0.0.1:" + std::to_string(server->listenPort()));
        servers.push_back(std::move(server));
    }

    auto manager = std::make_unique<BroadcastManager>(server_addrs);
    ASSERT_TRUE(manager->init());
    ASSERT_EQ(manager->workerNum(), server_addrs.size());

    std::vector<FunctionRequestPB> requests(manager->workerNum());
    auto                           rpc_call = [](const std::shared_ptr<RpcService::Stub>&    stub,
                       const std::shared_ptr<grpc::ClientContext>& ctx,
                       const FunctionRequestPB&                    req,
                       grpc::CompletionQueue* cq) { return stub->AsyncExecuteFunction(ctx.get(), req, cq); };
    auto result = manager->broadcast<FunctionRequestPB, FunctionResponsePB>(requests, /*timeout_ms=*/500, rpc_call);
    ASSERT_NE(result, nullptr);

    result->waitDone();
    EXPECT_TRUE(result->success());

    const auto& responses = result->responses();
    EXPECT_EQ(responses.size(), server_addrs.size());
    for (size_t i = 0; i < responses.size(); ++i) {
        EXPECT_TRUE(responses[i].has_mem_response());
        EXPECT_TRUE(responses[i].mem_response().success());
    }

    manager.reset();
    for (auto& server : servers) {
        server->shutdown();
    }
}

TEST_F(BroadcastManagerTest, Broadcast_ReturnNotNull_AllRequestsTimeout) {
    std::vector<std::unique_ptr<TestRpcServer>> servers;
    std::vector<std::string>                    server_addrs;
    for (int i = 0; i < 3; ++i) {
        auto service = std::make_unique<TestRpcService>();
        // set sleep time to 100ms, so the request should timeout
        service->setSleepMillis(100);
        auto server = std::make_unique<TestRpcServer>(std::move(service));
        ASSERT_TRUE(server->start());
        server_addrs.push_back("127.0.0.1:" + std::to_string(server->listenPort()));
        servers.push_back(std::move(server));
    }

    auto manager = std::make_unique<BroadcastManager>(server_addrs);
    ASSERT_TRUE(manager->init());
    ASSERT_EQ(manager->workerNum(), server_addrs.size());

    std::vector<FunctionRequestPB> requests(manager->workerNum());
    // set timeout to 50ms, so the request should timeout
    auto rpc_call = [](const std::shared_ptr<RpcService::Stub>&    stub,
                       const std::shared_ptr<grpc::ClientContext>& ctx,
                       const FunctionRequestPB&                    req,
                       grpc::CompletionQueue* cq) { return stub->AsyncExecuteFunction(ctx.get(), req, cq); };
    auto result   = manager->broadcast<FunctionRequestPB, FunctionResponsePB>(requests, /*timeout_ms=*/50, rpc_call);
    ASSERT_NE(result, nullptr);

    EXPECT_THROW(result->waitDone(), rtp_llm::RTPException);
}

TEST_F(BroadcastManagerTest, Broadcast_ReturnNotNull_PartialRequestsTimeout) {
    std::vector<std::unique_ptr<TestRpcServer>> servers;
    std::vector<std::string>                    server_addrs;
    for (int i = 0; i < 3; ++i) {
        auto service = std::make_unique<TestRpcService>();
        if (i == 0) {
            // set sleep time to 100ms, so the request should timeout
            service->setSleepMillis(100);
        }
        auto server = std::make_unique<TestRpcServer>(std::move(service));
        ASSERT_TRUE(server->start());
        server_addrs.push_back("127.0.0.1:" + std::to_string(server->listenPort()));
        servers.push_back(std::move(server));
    }

    auto manager = std::make_unique<BroadcastManager>(server_addrs);
    ASSERT_TRUE(manager->init());
    ASSERT_EQ(manager->workerNum(), server_addrs.size());

    std::vector<FunctionRequestPB> requests(manager->workerNum());
    // set timeout to 50ms, so the request should timeout
    auto rpc_call = [](const std::shared_ptr<RpcService::Stub>&    stub,
                       const std::shared_ptr<grpc::ClientContext>& ctx,
                       const FunctionRequestPB&                    req,
                       grpc::CompletionQueue* cq) { return stub->AsyncExecuteFunction(ctx.get(), req, cq); };
    auto result   = manager->broadcast<FunctionRequestPB, FunctionResponsePB>(requests, /*timeout_ms=*/50, rpc_call);
    ASSERT_NE(result, nullptr);

    EXPECT_THROW(result->waitDone(), rtp_llm::RTPException);
}

TEST_F(BroadcastManagerTest, Broadcast_ReturnNotNull_PartialResponseRpcStatusFailed) {
    std::vector<std::unique_ptr<TestRpcServer>> servers;
    std::vector<std::string>                    server_addrs;
    for (int i = 0; i < 3; ++i) {
        auto service = std::make_unique<TestRpcService>();
        // set the first request to rpc status failed, so the final result should be false
        if (i == 0) {
            // 当返回的rpc状态为失败时, 这次请求的响应消息会被丢弃，response中的oneof字段不会被填充
            service->setSleepMillis(50);
            service->setRpcResponseStatus(::grpc::Status(grpc::StatusCode::INTERNAL, "internal error"));
        }
        auto server = std::make_unique<TestRpcServer>(std::move(service));
        ASSERT_TRUE(server->start());
        server_addrs.push_back("127.0.0.1:" + std::to_string(server->listenPort()));
        servers.push_back(std::move(server));
    }

    auto manager = std::make_unique<BroadcastManager>(server_addrs);
    ASSERT_TRUE(manager->init());
    ASSERT_EQ(manager->workerNum(), server_addrs.size());

    std::vector<FunctionRequestPB> requests(manager->workerNum());
    auto                           rpc_call = [](const std::shared_ptr<RpcService::Stub>&    stub,
                       const std::shared_ptr<grpc::ClientContext>& ctx,
                       const FunctionRequestPB&                    req,
                       grpc::CompletionQueue* cq) { return stub->AsyncExecuteFunction(ctx.get(), req, cq); };
    auto result = manager->broadcast<FunctionRequestPB, FunctionResponsePB>(requests, /*timeout_ms=*/100, rpc_call);
    ASSERT_NE(result, nullptr);

    result->waitDone();
    EXPECT_FALSE(result->success());  // because the first request is failed, so the final result should be false

    const auto& responses = result->responses();
    EXPECT_EQ(responses.size(), server_addrs.size());
    for (size_t i = 0; i < responses.size(); ++i) {
        const auto& ctx = result->worker_contexts_[i];
        if (i == 0) {
            EXPECT_EQ(ctx->status.error_code(), grpc::StatusCode::INTERNAL);
            EXPECT_FALSE(responses[i].has_mem_response());
            EXPECT_FALSE(responses[i].mem_response().success());
        } else {
            EXPECT_EQ(ctx->status.error_code(), grpc::StatusCode::OK);
            EXPECT_TRUE(responses[i].has_mem_response());
            EXPECT_TRUE(responses[i].mem_response().success());
        }
    }

    manager.reset();
    for (auto& server : servers) {
        server->shutdown();
    }
}

TEST_F(BroadcastManagerTest, Broadcast_ReturnNotNull_ResponseStatusOkButMemResponseNotSuccess) {
    std::vector<std::unique_ptr<TestRpcServer>> servers;
    std::vector<std::string>                    server_addrs;
    for (int i = 0; i < 3; ++i) {
        auto service = std::make_unique<TestRpcService>();
        // set the first request to mem response not success, so the final result should be false
        if (i == 0) {
            service->setMemResponseSuccess(false);
        }
        auto server = std::make_unique<TestRpcServer>(std::move(service));
        ASSERT_TRUE(server->start());
        server_addrs.push_back("127.0.0.1:" + std::to_string(server->listenPort()));
        servers.push_back(std::move(server));
    }

    auto manager = std::make_unique<BroadcastManager>(server_addrs);
    ASSERT_TRUE(manager->init());
    ASSERT_EQ(manager->workerNum(), server_addrs.size());

    std::vector<FunctionRequestPB> requests(manager->workerNum());
    auto                           rpc_call = [](const std::shared_ptr<RpcService::Stub>&    stub,
                       const std::shared_ptr<grpc::ClientContext>& ctx,
                       const FunctionRequestPB&                    req,
                       grpc::CompletionQueue* cq) { return stub->AsyncExecuteFunction(ctx.get(), req, cq); };
    auto result = manager->broadcast<FunctionRequestPB, FunctionResponsePB>(requests, /*timeout_ms=*/100, rpc_call);
    ASSERT_NE(result, nullptr);

    result->waitDone();
    EXPECT_TRUE(result->success());

    const auto& responses = result->responses();
    EXPECT_EQ(responses.size(), server_addrs.size());
    for (size_t i = 0; i < responses.size(); ++i) {
        const auto& ctx = result->worker_contexts_[i];
        EXPECT_EQ(ctx->status.error_code(), grpc::StatusCode::OK);
        EXPECT_TRUE(responses[i].has_mem_response());
        if (i == 0) {
            EXPECT_FALSE(responses[i].mem_response().success());
        } else {
            EXPECT_TRUE(responses[i].mem_response().success());
        }
    }

    manager.reset();
    for (auto& server : servers) {
        server->shutdown();
    }
}

TEST_F(BroadcastManagerTest, Broadcast_WaitDone_IsIdempotent_AndThreadSafe) {
    std::vector<std::unique_ptr<TestRpcServer>> servers;
    std::vector<std::string>                    server_addrs;
    for (int i = 0; i < 2; ++i) {
        auto service = std::make_unique<TestRpcService>();
        auto server  = std::make_unique<TestRpcServer>(std::move(service));
        ASSERT_TRUE(server->start());
        server_addrs.push_back("127.0.0.1:" + std::to_string(server->listenPort()));
        servers.push_back(std::move(server));
    }

    auto manager = std::make_unique<BroadcastManager>(server_addrs);
    ASSERT_TRUE(manager->init());
    ASSERT_EQ(manager->workerNum(), server_addrs.size());

    std::vector<FunctionRequestPB> requests(manager->workerNum());
    auto                           rpc_call = [](const std::shared_ptr<RpcService::Stub>&    stub,
                       const std::shared_ptr<grpc::ClientContext>& ctx,
                       const FunctionRequestPB&                    req,
                       grpc::CompletionQueue* cq) { return stub->AsyncExecuteFunction(ctx.get(), req, cq); };
    auto result = manager->broadcast<FunctionRequestPB, FunctionResponsePB>(requests, /*timeout_ms=*/500, rpc_call);
    ASSERT_NE(result, nullptr);

    // Concurrent waitDone should be safe and callback should effectively run once.
    std::thread t1([&]() { result->waitDone(); });
    std::thread t2([&]() { result->waitDone(); });
    result->waitDone();
    t1.join();
    t2.join();

    EXPECT_TRUE(result->success());
    EXPECT_EQ(result->responses().size(), server_addrs.size());
}

TEST_F(BroadcastManagerTest, Broadcast_WaitsAllRequests_WhenAnyRpcStatusFailed) {
    // Even if one request fails (e.g. INTERNAL), waitDone() should wait for all other requests to finish.
    // This ensures callers can always observe every response/status deterministically.
    std::vector<std::string> server_addrs;

    // server0: fails with INTERNAL (fast)
    auto service0 = std::make_unique<TestRpcService>();
    service0->setRpcResponseStatus(::grpc::Status(grpc::StatusCode::INTERNAL, "internal error"));
    auto server0 = std::make_unique<TestRpcServer>(std::move(service0));
    ASSERT_TRUE(server0->start());
    server_addrs.push_back("127.0.0.1:" + std::to_string(server0->listenPort()));

    // server1: succeeds but takes some time (should not be cancelled)
    auto service1 = std::make_unique<TestRpcService>();
    service1->setSleepMillis(300);
    auto server1 = std::make_unique<TestRpcServer>(std::move(service1));
    ASSERT_TRUE(server1->start());
    server_addrs.push_back("127.0.0.1:" + std::to_string(server1->listenPort()));

    auto manager = std::make_unique<BroadcastManager>(server_addrs);
    ASSERT_TRUE(manager->init());

    std::vector<FunctionRequestPB> requests(manager->workerNum());
    auto                           rpc_call = [](const std::shared_ptr<RpcService::Stub>&    stub,
                       const std::shared_ptr<grpc::ClientContext>& ctx,
                       const FunctionRequestPB&                    req,
                       grpc::CompletionQueue* cq) { return stub->AsyncExecuteFunction(ctx.get(), req, cq); };
    auto result = manager->broadcast<FunctionRequestPB, FunctionResponsePB>(requests, /*timeout_ms=*/2000, rpc_call);
    ASSERT_NE(result, nullptr);

    const auto start = std::chrono::steady_clock::now();
    result->waitDone();
    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();

    EXPECT_FALSE(result->success());
    EXPECT_GE(elapsed_ms, 250) << "waitDone should wait for the slow request to finish";
    EXPECT_LE(elapsed_ms, 1500) << "slow request should finish before broadcast timeout";

    const auto& ctx0 = result->worker_contexts_.at(0);
    const auto& ctx1 = result->worker_contexts_.at(1);
    ASSERT_NE(ctx0, nullptr);
    ASSERT_NE(ctx1, nullptr);
    EXPECT_EQ(ctx0->status.error_code(), grpc::StatusCode::INTERNAL);
    EXPECT_EQ(ctx1->status.error_code(), grpc::StatusCode::OK);

    server1->shutdown();
    server0->shutdown();
}

// ---------------------------- workerNum ----------------------------

TEST_F(BroadcastManagerTest, WorkerNum) {
    EXPECT_EQ(manager_->workerNum(), 2u);
}

}  // namespace rtp_llm::test

int main(int argc, char** argv) {
    rtp_llm::initLogger();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}