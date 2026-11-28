#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <future>
#include <stdexcept>
#include <thread>
#include <gtest/gtest.h>
#include "grpc++/grpc++.h"

#include "autil/NetUtil.h"
#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"

namespace rtp_llm::test {

TEST(BroadcastDiagnosticsTest, ProgressCompletionDoesNotCopyRegisteredCallable) {
    using Result = BroadcastResult<int, int>;
    struct Callback {
        std::shared_ptr<std::atomic<bool>> throw_on_copy;
        std::shared_ptr<std::atomic<int>>  calls;
        Callback(std::shared_ptr<std::atomic<bool>> flag, std::shared_ptr<std::atomic<int>> count):
            throw_on_copy(std::move(flag)), calls(std::move(count)) {}
        Callback(const Callback& other): throw_on_copy(other.throw_on_copy), calls(other.calls) {
            if (throw_on_copy->load()) {
                throw std::runtime_error("copy after progress registration");
            }
        }
        Callback(Callback&&) = default;
        void operator()() const {
            ++*calls;
        }
    };
    auto   flag    = std::make_shared<std::atomic<bool>>(false);
    auto   calls   = std::make_shared<std::atomic<int>>(0);
    auto   context = std::make_shared<Result::WorkerRpcContext>();
    Result result({context});
    result.setProgressCallback(Callback(flag, calls));
    ASSERT_EQ(calls->load(), 1);
    int completed = 0;
    result.onDone([&] { ++completed; });
    flag->store(true);
    EXPECT_NO_THROW(result.finishRank(0, true));
    EXPECT_TRUE(result.done());
    EXPECT_TRUE(result.success());
    EXPECT_EQ(calls->load(), 2);
    EXPECT_EQ(completed, 1);
}

TEST(BroadcastDiagnosticsTest, ConnectFailureKeepsQueryDirectionAndReportsEachRankOnce) {
    using Result = BroadcastResult<FunctionRequestPB, FunctionResponsePB>;
    std::vector<std::shared_ptr<Result::WorkerRpcContext>> contexts;
    for (size_t rank = 0; rank < 2; ++rank) {
        auto context         = std::make_shared<Result::WorkerRpcContext>();
        context->server_addr = "worker-" + std::to_string(rank);
        context->timeout_ms  = 1000;
        context->status      = grpc::Status(grpc::StatusCode::UNAVAILABLE, "Connect Failed");
        contexts.push_back(std::move(context));
    }
    std::vector<std::string> directions;
    Result result(contexts, 42, "H2D", [&](const auto& direction) { directions.push_back(direction); });
    result.finishRank(0, true);
    result.finishRank(0, true);
    result.finishRank(1, true);
    EXPECT_EQ(directions, (std::vector<std::string>{"H2D", "H2D"}));
    EXPECT_NE(result.failureSummary().find("request_id=42"), std::string::npos);
    EXPECT_NE(result.failureSummary().find("direction=H2D"), std::string::npos);
    EXPECT_FALSE(result.success());
    EXPECT_NO_THROW(result.waitDone());
}

TEST(BroadcastDiagnosticsTest, CompletionTimesAreRecordedOncePerRankAndOnlyWhenFinished) {
    using Result  = BroadcastResult<FunctionRequestPB, FunctionResponsePB>;
    auto   first  = std::make_shared<Result::WorkerRpcContext>();
    auto   second = std::make_shared<Result::WorkerRpcContext>();
    Result result({first, second});
    EXPECT_EQ(result.completionTimesUs(), (std::vector<int64_t>{0, 0}));
    result.finishRank(1, true);
    auto times = result.completionTimesUs();
    EXPECT_EQ(times[0], 0);
    EXPECT_GT(times[1], 0);
    const auto walls = result.completionWallTimesUs();
    EXPECT_GT(walls[1], 0);
    result.finishRank(1, true);
    EXPECT_EQ(result.completionTimesUs(), times);
    result.finishRank(0, true);
    EXPECT_GE(result.completionTimesUs()[0], times[1]);
    EXPECT_TRUE(result.done());
}

namespace {
struct CopyObservedResponse {
    std::shared_ptr<std::atomic<size_t>> copies;
    CopyObservedResponse() = default;
    CopyObservedResponse(const CopyObservedResponse& other): copies(other.copies) {
        if (copies) {
            ++*copies;
        }
    }
    CopyObservedResponse& operator=(const CopyObservedResponse& other) {
        copies = other.copies;
        if (copies) {
            ++*copies;
        }
        return *this;
    }
};
}  // namespace

TEST(BroadcastDiagnosticsTest, PendingAndLocallyFinishedRanksNeverReadGrpcResponseOrDefaultStatus) {
    using Result             = BroadcastResult<int, CopyObservedResponse>;
    auto context             = std::make_shared<Result::WorkerRpcContext>();
    auto copies              = std::make_shared<std::atomic<size_t>>(0);
    context->response.copies = copies;
    Result result({context});
    result.markDispatchStarted(0);

    auto pending = result.rankSnapshots();
    ASSERT_EQ(pending.size(), 1u);
    EXPECT_EQ(pending[0].dispatch_state, BroadcastDispatchState::POSSIBLY_SUBMITTED);
    EXPECT_FALSE(pending[0].cq_received);
    EXPECT_FALSE(pending[0].grpc_code.has_value());
    EXPECT_FALSE(pending[0].response.has_value());
    EXPECT_EQ(copies->load(), 0u);

    result.finishDispatchFailure(0, "Finish failed after the RPC may have started");
    EXPECT_TRUE(result.done());
    EXPECT_FALSE(result.success());
    const auto failed = result.rankSnapshots();
    EXPECT_FALSE(failed[0].cq_received);
    EXPECT_FALSE(failed[0].grpc_code.has_value());
    EXPECT_FALSE(failed[0].response.has_value());
    EXPECT_EQ(failed[0].dispatch_error, "Finish failed after the RPC may have started");
    EXPECT_EQ(result.responses().size(), 1u);
    EXPECT_EQ(copies->load(), 0u);

    // A real, late CQ event makes the response readable without reversing the
    // previously published aggregate failure or mutating an earlier snapshot.
    result.finishRank(0, true);
    const auto completed = result.rankSnapshots();
    EXPECT_TRUE(completed[0].cq_received);
    EXPECT_TRUE(completed[0].cq_ok);
    EXPECT_EQ(completed[0].dispatch_state, BroadcastDispatchState::SUBMITTED);
    ASSERT_TRUE(completed[0].grpc_code.has_value());
    EXPECT_EQ(*completed[0].grpc_code, grpc::StatusCode::OK);
    EXPECT_TRUE(completed[0].response.has_value());
    EXPECT_GT(copies->load(), 0u);
    EXPECT_FALSE(failed[0].cq_received);
    EXPECT_FALSE(result.success());
}

TEST(BroadcastDiagnosticsTest, FailedCqDoesNotInventGrpcOkStatus) {
    using Result   = BroadcastResult<FunctionRequestPB, FunctionResponsePB>;
    auto   context = std::make_shared<Result::WorkerRpcContext>();
    Result result({context});
    result.finishRank(0, false);
    const auto snapshots = result.rankSnapshots();
    ASSERT_EQ(snapshots.size(), 1u);
    EXPECT_TRUE(snapshots[0].cq_received);
    EXPECT_FALSE(snapshots[0].cq_ok);
    EXPECT_FALSE(snapshots[0].grpc_code.has_value());
    EXPECT_FALSE(snapshots[0].response.has_value());
}

TEST(BroadcastDiagnosticsTest, RankSnapshotsKeepEveryRawGrpcFailure) {
    using Result        = BroadcastResult<FunctionRequestPB, FunctionResponsePB>;
    auto first          = std::make_shared<Result::WorkerRpcContext>();
    auto second         = std::make_shared<Result::WorkerRpcContext>();
    first->server_addr  = "worker-0";
    first->status       = grpc::Status(grpc::StatusCode::UNAVAILABLE, "Connect Failed");
    second->server_addr = "worker-1";
    second->status      = grpc::Status(grpc::StatusCode::DEADLINE_EXCEEDED, "slow transfer");
    Result result({first, second});
    result.finishRank(1, true);
    result.finishRank(0, true);
    const auto snapshots = result.rankSnapshots();
    ASSERT_EQ(snapshots.size(), 2u);
    ASSERT_TRUE(snapshots[0].grpc_code.has_value());
    ASSERT_TRUE(snapshots[1].grpc_code.has_value());
    EXPECT_EQ(*snapshots[0].grpc_code, grpc::StatusCode::UNAVAILABLE);
    EXPECT_EQ(snapshots[0].grpc_message, "Connect Failed");
    EXPECT_EQ(snapshots[0].server_address, "worker-0");
    EXPECT_EQ(*snapshots[1].grpc_code, grpc::StatusCode::DEADLINE_EXCEEDED);
    EXPECT_EQ(snapshots[1].grpc_message, "slow transfer");
    EXPECT_TRUE(result.deadlineExceeded());
}

TEST(BroadcastDiagnosticsTest, OtherRpcFailuresAreNotConnectFailures) {
    for (const auto& status : {grpc::Status(grpc::StatusCode::UNAVAILABLE, "server temporarily busy"),
                               grpc::Status(grpc::StatusCode::DEADLINE_EXCEEDED, "Connect Failed"),
                               grpc::Status(grpc::StatusCode::CANCELLED, "request cancelled"),
                               grpc::Status(grpc::StatusCode::INTERNAL, "Connect Failed")}) {
        EXPECT_FALSE(isGrpcConnectFailure(status));
    }
    EXPECT_TRUE(isGrpcConnectFailure(grpc::Status(grpc::StatusCode::UNAVAILABLE, "Connect Failed")));
    EXPECT_TRUE(isGrpcConnectFailure(
        grpc::Status(grpc::StatusCode::UNAVAILABLE, "failed to connect to all addresses; connection refused")));
}

TEST(BroadcastDiagnosticsTest, MixedFailuresKeepConnectFailureAndTimeoutPriority) {
    using Result = BroadcastResult<FunctionRequestPB, FunctionResponsePB>;
    for (const bool with_timeout : {false, true}) {
        auto first          = std::make_shared<Result::WorkerRpcContext>();
        auto second         = std::make_shared<Result::WorkerRpcContext>();
        first->server_addr  = "worker-0";
        second->server_addr = "worker-1";
        first->status  = grpc::Status(grpc::StatusCode::UNAVAILABLE, with_timeout ? "Connect Failed" : "server busy");
        second->status = with_timeout ? grpc::Status(grpc::StatusCode::DEADLINE_EXCEEDED, "deadline exceeded") :
                                        grpc::Status(grpc::StatusCode::UNAVAILABLE, "Connect Failed");
        Result result({first, second}, 73, "H2D");
        result.finishRank(0, true);
        result.finishRank(1, true);
        const auto summary = result.failureSummary();
        EXPECT_NE(summary.find("Connect Failed"), std::string::npos);
        EXPECT_NE(summary.find(with_timeout ? "target_rank=0" : "target_rank=1"), std::string::npos);
        EXPECT_NE(summary.find("request_id=73"), std::string::npos);
        EXPECT_EQ(result.deadlineExceeded(), with_timeout);
        if (with_timeout) {
            EXPECT_NE(summary.find("broadcast rpc timeout"), std::string::npos);
        }
    }
}

TEST(BroadcastDiagnosticsTest, ThrowingObserverDoesNotPreventCompletion) {
    using Result         = BroadcastResult<FunctionRequestPB, FunctionResponsePB>;
    auto context         = std::make_shared<Result::WorkerRpcContext>();
    context->status      = grpc::Status(grpc::StatusCode::UNAVAILABLE, "Connect Failed");
    context->timeout_ms  = 1000;
    bool   callback_done = false;
    Result result({context}, std::nullopt, "H2D", [](const auto&) { throw std::runtime_error("observer failure"); });
    result.onDone([&] { callback_done = true; });
    EXPECT_NO_THROW(result.finishRank(0, true));
    EXPECT_TRUE(result.done());
    EXPECT_TRUE(callback_done);
}

TEST(BroadcastDiagnosticsTest, NonConnectFailureDoesNotReportAndBackgroundHasNoQuery) {
    using Result    = BroadcastResult<FunctionRequestPB, FunctionResponsePB>;
    auto context    = std::make_shared<Result::WorkerRpcContext>();
    context->status = grpc::Status(grpc::StatusCode::UNAVAILABLE, "server busy");
    size_t reports  = 0;
    Result result({context}, std::nullopt, "D2H", [&](const auto&) { ++reports; });
    result.finishRank(0, true);
    EXPECT_EQ(reports, 0u);
    EXPECT_NE(result.failureSummary().find("request_id=none"), std::string::npos);
    EXPECT_NE(result.failureSummary().find("direction=D2H"), std::string::npos);
}

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
        response->mutable_mem_response()->set_code(mem_response_code_);
        return rpc_response_status_;
    }
    void setSleepMillis(int ms) {
        sleep_millis_ = ms;
    }
    void setMemResponseCode(MemoryOperationResponsePB::Code code) {
        mem_response_code_ = code;
    }
    void setRpcResponseStatus(const ::grpc::Status& status) {
        rpc_response_status_ = status;
    }

private:
    int                             sleep_millis_{0};
    MemoryOperationResponsePB::Code mem_response_code_{MemoryOperationResponsePB::OK};
    ::grpc::Status                  rpc_response_status_{::grpc::Status::OK};
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
        EXPECT_EQ(responses[i].mem_response().code(), MemoryOperationResponsePB::OK);
    }

    manager.reset();
    for (auto& server : servers) {
        server->shutdown();
    }
}

TEST_F(BroadcastManagerTest, BroadcastCommonRequestKeepsIndependentPayloadsAfterCallerClearsRequest) {
    std::vector<std::unique_ptr<TestRpcServer>> servers;
    std::vector<std::string>                    addresses;
    for (size_t rank = 0; rank < 2; ++rank) {
        auto server = std::make_unique<TestRpcServer>(std::make_unique<TestRpcService>());
        ASSERT_TRUE(server->start());
        addresses.push_back("127.0.0.1:" + std::to_string(server->listenPort()));
        servers.push_back(std::move(server));
    }
    BroadcastManager manager(addresses);
    ASSERT_TRUE(manager.init());
    FunctionRequestPB request;
    auto*             memory = request.mutable_mem_request();
    memory->set_transfer_trace_id(73);
    memory->set_copy_direction(MemoryOperationRequestPB::H2D);
    auto* item = memory->add_copy_items();
    item->set_mem_block(21);
    item->add_group_tags("common-payload");
    const auto expected = request.SerializeAsString();
    auto       rpc_call = [](const std::shared_ptr<RpcService::Stub>&    stub,
                       const std::shared_ptr<grpc::ClientContext>& context,
                       const FunctionRequestPB&                    payload,
                       grpc::CompletionQueue* cq) { return stub->AsyncExecuteFunction(context.get(), payload, cq); };
    auto       result   = manager.broadcast<FunctionRequestPB, FunctionResponsePB>(request, 1000, rpc_call);
    ASSERT_NE(result, nullptr);
    request.Clear();

    ASSERT_EQ(result->worker_contexts_.size(), 2u);
    for (const auto& context : result->worker_contexts_) {
        EXPECT_EQ(context->request.SerializeAsString(), expected);
        EXPECT_NE(&context->request, &request);
    }
    EXPECT_NE(&result->worker_contexts_[0]->request, &result->worker_contexts_[1]->request);
    result->waitDone();
    EXPECT_TRUE(result->success());
}

TEST_F(BroadcastManagerTest, Broadcast_OnDoneCompletesWithoutWaitDone) {
    auto service = std::make_unique<TestRpcService>();
    auto server  = std::make_unique<TestRpcServer>(std::move(service));
    ASSERT_TRUE(server->start());
    auto manager = std::make_unique<BroadcastManager>(
        std::vector<std::string>{"127.0.0.1:" + std::to_string(server->listenPort())});
    ASSERT_TRUE(manager->init());

    std::vector<FunctionRequestPB> requests(manager->workerNum());
    auto                           rpc_call = [](const std::shared_ptr<RpcService::Stub>&    stub,
                       const std::shared_ptr<grpc::ClientContext>& ctx,
                       const FunctionRequestPB&                    req,
                       grpc::CompletionQueue* cq) { return stub->AsyncExecuteFunction(ctx.get(), req, cq); };
    auto result = manager->broadcast<FunctionRequestPB, FunctionResponsePB>(requests, 500, rpc_call);
    ASSERT_NE(result, nullptr);
    std::promise<void> done;
    auto               future = done.get_future();
    result->onDone([&] { done.set_value(); });

    ASSERT_EQ(future.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    EXPECT_TRUE(result->done());
    EXPECT_TRUE(result->success());
}

TEST_F(BroadcastManagerTest, CachePartialSubmissionFailureSettlesSubmittedRankWithoutFatal) {
    auto server = std::make_unique<TestRpcServer>(std::make_unique<TestRpcService>());
    ASSERT_TRUE(server->start());
    BroadcastManager manager(std::vector<std::string>(3, "127.0.0.1:" + std::to_string(server->listenPort())));
    ASSERT_TRUE(manager.init());
    std::vector<FunctionRequestPB> requests(manager.workerNum());
    size_t                         submitted = 0;
    auto                           rpc_call  = [&](const std::shared_ptr<RpcService::Stub>&    stub,
                        const std::shared_ptr<grpc::ClientContext>& context,
                        const FunctionRequestPB&                    request,
                        grpc::CompletionQueue*                      queue) {
        if (++submitted == 2) {
            throw std::runtime_error("injected partial dispatch failure");
        }
        return stub->AsyncExecuteFunction(context.get(), request, queue);
    };
    auto result = manager.broadcast<FunctionRequestPB, FunctionResponsePB>(
        requests, 1000, rpc_call, 42, "REMOTE2D", /*fatal_on_error=*/false);
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->waitDone(2000, /*fatal_on_error=*/false));
    EXPECT_EQ(submitted, 2u);
    EXPECT_TRUE(result->rpcSucceeded(0));
    EXPECT_FALSE(result->rpcSucceeded(1));
    EXPECT_FALSE(result->rpcSucceeded(2));
    EXPECT_FALSE(result->success());
    EXPECT_NE(result->failureSummary().find("request_id=42"), std::string::npos);
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
    EXPECT_TRUE(result->deadlineExceeded());
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
    EXPECT_TRUE(result->deadlineExceeded());
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
        } else {
            EXPECT_EQ(ctx->status.error_code(), grpc::StatusCode::OK);
            EXPECT_TRUE(responses[i].has_mem_response());
            EXPECT_EQ(responses[i].mem_response().code(), MemoryOperationResponsePB::OK);
        }
    }

    manager.reset();
    for (auto& server : servers) {
        server->shutdown();
    }
}

TEST_F(BroadcastManagerTest, Broadcast_ReturnNotNull_ResponseStatusOkButMemResponseFailed) {
    std::vector<std::unique_ptr<TestRpcServer>> servers;
    std::vector<std::string>                    server_addrs;
    for (int i = 0; i < 3; ++i) {
        auto service = std::make_unique<TestRpcService>();
        // set the first request to report a transfer business failure over a gRPC OK response
        if (i == 0) {
            service->setMemResponseCode(MemoryOperationResponsePB::COPY_FAILED);
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
            EXPECT_EQ(responses[i].mem_response().code(), MemoryOperationResponsePB::COPY_FAILED);
        } else {
            EXPECT_EQ(responses[i].mem_response().code(), MemoryOperationResponsePB::OK);
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
}

TEST_F(BroadcastManagerTest, Broadcast_WaitDone_WithOverallTimeout_CanBeResumed) {
    std::vector<std::unique_ptr<TestRpcServer>> servers;
    std::vector<std::string>                    server_addrs;
    for (int i = 0; i < 2; ++i) {
        auto service = std::make_unique<TestRpcService>();
        service->setSleepMillis(200);
        auto server = std::make_unique<TestRpcServer>(std::move(service));
        ASSERT_TRUE(server->start());
        server_addrs.push_back("127.0.0.1:" + std::to_string(server->listenPort()));
        servers.push_back(std::move(server));
    }

    auto manager = std::make_unique<BroadcastManager>(server_addrs);
    ASSERT_TRUE(manager->init());

    std::vector<FunctionRequestPB> requests(manager->workerNum());
    auto                           rpc_call = [](const std::shared_ptr<RpcService::Stub>&    stub,
                       const std::shared_ptr<grpc::ClientContext>& ctx,
                       const FunctionRequestPB&                    req,
                       grpc::CompletionQueue* cq) { return stub->AsyncExecuteFunction(ctx.get(), req, cq); };
    auto result = manager->broadcast<FunctionRequestPB, FunctionResponsePB>(requests, /*timeout_ms=*/2000, rpc_call);
    ASSERT_NE(result, nullptr);

    EXPECT_FALSE(result->waitDone(/*timeout_ms=*/10));
    EXPECT_FALSE(result->done());

    EXPECT_TRUE(result->waitDone(/*timeout_ms=*/0));
    EXPECT_TRUE(result->done());
    EXPECT_TRUE(result->success());
}

TEST_F(BroadcastManagerTest, Broadcast_WaitDone_FailureStatePersistsAcrossTimeoutResume) {
    std::vector<std::string> server_addrs;

    // server0: fails immediately
    auto service0 = std::make_unique<TestRpcService>();
    service0->setRpcResponseStatus(::grpc::Status(grpc::StatusCode::INTERNAL, "internal error"));
    auto server0 = std::make_unique<TestRpcServer>(std::move(service0));
    ASSERT_TRUE(server0->start());
    server_addrs.push_back("127.0.0.1:" + std::to_string(server0->listenPort()));

    // server1: slow success, so the first short waitDone should timeout
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

    EXPECT_FALSE(result->waitDone(/*timeout_ms=*/10));
    EXPECT_FALSE(result->done());
    EXPECT_FALSE(result->success());

    EXPECT_TRUE(result->waitDone(/*timeout_ms=*/0));
    EXPECT_TRUE(result->done());
    EXPECT_FALSE(result->success());

    const auto& ctx0 = result->worker_contexts_.at(0);
    ASSERT_NE(ctx0, nullptr);
    EXPECT_EQ(ctx0->status.error_code(), grpc::StatusCode::INTERNAL);

    server1->shutdown();
    server0->shutdown();
}

// ---------------------------- dispatch invariants ----------------------------

namespace {

using AsyncFunctionReader = std::unique_ptr<grpc::ClientAsyncResponseReader<FunctionResponsePB>>;

using FunctionBroadcastResult = BroadcastResult<FunctionRequestPB, FunctionResponsePB>;

std::shared_ptr<FunctionBroadcastResult::WorkerRpcContext> makeIdleContext() {
    auto ctx         = std::make_shared<FunctionBroadcastResult::WorkerRpcContext>();
    ctx->server_addr = "127.0.0.1:0";
    ctx->timeout_ms  = 1000;
    return ctx;
}

}  // namespace

// ---------------------------- completion queue invariants ----------------------------

TEST_F(BroadcastManagerTest, PartialDispatchSeparatesSubmittedUncertainAndUnsubmittedRanks) {
    auto service = std::make_unique<TestRpcService>();
    auto server  = std::make_unique<TestRpcServer>(std::move(service));
    ASSERT_TRUE(server->start());
    const auto address = "127.0.0.1:" + std::to_string(server->listenPort());
    auto       manager = std::make_unique<BroadcastManager>(std::vector<std::string>{address, address, address});
    ASSERT_TRUE(manager->init());
    size_t     calls    = 0;
    const auto rpc_call = [&](const std::shared_ptr<RpcService::Stub>&    stub,
                              const std::shared_ptr<grpc::ClientContext>& context,
                              const FunctionRequestPB&                    request,
                              grpc::CompletionQueue*                      cq) -> AsyncFunctionReader {
        if (calls++ == 1) {
            throw std::runtime_error("injected dispatch failure");
        }
        return stub->AsyncExecuteFunction(context.get(), request, cq);
    };
    auto result = manager->broadcast<FunctionRequestPB, FunctionResponsePB>(
        std::vector<FunctionRequestPB>(3), 1000, rpc_call, std::nullopt, "H2D", false);
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->waitDone(5000, false));
    EXPECT_FALSE(result->success());
    EXPECT_EQ(calls, 2u);
    const auto snapshots = result->rankSnapshots();
    ASSERT_EQ(snapshots.size(), 3u);
    EXPECT_EQ(snapshots[0].dispatch_state, BroadcastDispatchState::SUBMITTED);
    EXPECT_TRUE(snapshots[0].cq_received);
    EXPECT_TRUE(snapshots[0].cq_ok);
    EXPECT_EQ(snapshots[1].dispatch_state, BroadcastDispatchState::POSSIBLY_SUBMITTED);
    EXPECT_EQ(snapshots[1].dispatch_error, "injected dispatch failure");
    EXPECT_FALSE(snapshots[1].cq_received);
    EXPECT_FALSE(snapshots[1].grpc_code.has_value());
    EXPECT_EQ(snapshots[2].dispatch_state, BroadcastDispatchState::NOT_SUBMITTED);
    EXPECT_FALSE(snapshots[2].cq_received);
    EXPECT_FALSE(snapshots[2].grpc_code.has_value());
    EXPECT_FALSE(snapshots[2].response.has_value());
    std::weak_ptr<FunctionBroadcastResult> weak_result = result;
    result.reset();
    // Joining the CQ thread also destroys the last completed real RPC tag.
    manager.reset();
    EXPECT_TRUE(weak_result.expired());
}

TEST_F(BroadcastManagerTest, WaitDone_Fatal_CqEventNotOk) {
    auto ctx    = makeIdleContext();
    auto result = std::make_shared<FunctionBroadcastResult>(
        std::vector<std::shared_ptr<FunctionBroadcastResult::WorkerRpcContext>>{ctx});

    result->finishRank(/*rank=*/0, /*cq_event_ok=*/false);

    EXPECT_THROW(result->waitDone(), rtp_llm::RTPException);
    EXPECT_FALSE(result->deadlineExceeded());
}

TEST_F(BroadcastManagerTest, CqFailureStillPublishesCompletionCallback) {
    auto ctx    = makeIdleContext();
    auto result = std::make_shared<FunctionBroadcastResult>(
        std::vector<std::shared_ptr<FunctionBroadcastResult::WorkerRpcContext>>{ctx});
    std::promise<void> callback_done;
    auto               callback_future = callback_done.get_future();
    result->onDone([&] { callback_done.set_value(); });

    result->finishRank(/*rank=*/0, /*cq_event_ok=*/false);

    EXPECT_EQ(callback_future.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    EXPECT_THROW(result->waitDone(), rtp_llm::RTPException);
    EXPECT_TRUE(result->done());
    EXPECT_FALSE(result->success());
}

TEST_F(BroadcastManagerTest, ApplicationFailureDoesNotFailBroadcastTransport) {
    auto ctx = makeIdleContext();
    ctx->response.mutable_mem_response()->set_code(MemoryOperationResponsePB::COPY_FAILED);
    FunctionBroadcastResult result({ctx});
    EXPECT_FALSE(result.done());
    result.finishRank(0, true);
    EXPECT_TRUE(result.done());
    EXPECT_TRUE(result.success());
    EXPECT_NO_THROW(result.waitDone());
    EXPECT_EQ(result.responses().front().mem_response().code(), MemoryOperationResponsePB::COPY_FAILED);
}

TEST_F(BroadcastManagerTest, DeadlineFailsBroadcastAndThrowsOnWait) {
    auto ctx    = makeIdleContext();
    ctx->status = grpc::Status(grpc::StatusCode::DEADLINE_EXCEEDED, "deadline");
    FunctionBroadcastResult result({ctx});
    result.finishRank(0, true);
    EXPECT_TRUE(result.done());
    EXPECT_FALSE(result.success());
    EXPECT_THROW(result.waitDone(), RTPException);
}

TEST_F(BroadcastManagerTest, DeadlineSurvivesOtherRankFailuresInEitherArrivalOrder) {
    for (bool timeout_first : {false, true}) {
        for (bool cq_ok : {false, true}) {
            auto timeout    = makeIdleContext();
            auto other      = makeIdleContext();
            timeout->status = grpc::Status(grpc::StatusCode::DEADLINE_EXCEEDED, "deadline");
            other->status   = grpc::Status(grpc::StatusCode::INTERNAL, "other failure");
            FunctionBroadcastResult result({timeout, other});
            int                     callbacks = 0;
            result.onDone([&] { ++callbacks; });
            EXPECT_FALSE(result.deadlineExceeded());
            result.finishRank(timeout_first ? 0 : 1, timeout_first || cq_ok);
            EXPECT_FALSE(result.done());
            EXPECT_EQ(result.deadlineExceeded(), timeout_first);
            result.finishRank(timeout_first ? 1 : 0, !timeout_first || cq_ok);
            EXPECT_THROW(result.waitDone(), RTPException);
            EXPECT_FALSE(result.success());
            EXPECT_TRUE(result.deadlineExceeded());
            EXPECT_EQ(callbacks, 1);
            result.finishRank(0, true);
            result.finishRank(1, true);
            EXPECT_EQ(callbacks, 1);
            EXPECT_TRUE(result.deadlineExceeded());
        }
    }
}

TEST_F(BroadcastManagerTest, RejectedBroadcastNeverDispatchesAnyRank) {
    for (bool initialized : {false, true}) {
        BroadcastManager manager({"127.0.0.1:1", "127.0.0.1:2"});
        if (initialized) {
            ASSERT_TRUE(manager.init());
        }
        for (size_t request_count : {1u, 2u}) {
            if (initialized && request_count == 2) {
                continue;
            }
            int  calls    = 0;
            auto rpc_call = [&](const std::shared_ptr<RpcService::Stub>&,
                                const std::shared_ptr<grpc::ClientContext>&,
                                const FunctionRequestPB&,
                                grpc::CompletionQueue*) -> AsyncFunctionReader {
                ++calls;
                return nullptr;
            };
            EXPECT_EQ((manager.broadcast<FunctionRequestPB, FunctionResponsePB>(
                          std::vector<FunctionRequestPB>(request_count), 100, rpc_call)),
                      nullptr);
            EXPECT_EQ(calls, 0);
        }
    }
}

TEST_F(BroadcastManagerTest, CancellationBeforeRankDispatchKeepsEarlierRpcTracked) {
    auto server = std::make_unique<TestRpcServer>(std::make_unique<TestRpcService>());
    ASSERT_TRUE(server->start());
    const auto       address = "127.0.0.1:" + std::to_string(server->listenPort());
    BroadcastManager manager({address, address});
    ASSERT_TRUE(manager.init());

    int  calls    = 0;
    auto rpc_call = [&](const std::shared_ptr<RpcService::Stub>&    stub,
                        const std::shared_ptr<grpc::ClientContext>& context,
                        const FunctionRequestPB&                    request,
                        grpc::CompletionQueue*                      queue) {
        ++calls;
        return stub->AsyncExecuteFunction(context.get(), request, queue);
    };
    auto before_dispatch = [&](FunctionRequestPB&) -> grpc::Status {
        return calls == 0 ? grpc::Status::OK : grpc::Status(grpc::StatusCode::CANCELLED, "test cancellation");
    };
    auto result = manager.broadcast<FunctionRequestPB, FunctionResponsePB>(
        std::vector<FunctionRequestPB>(2), 1000, rpc_call, before_dispatch);
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->waitDone(2000));
    EXPECT_EQ(calls, 1);
    EXPECT_TRUE(result->rpcSucceeded(0));
    EXPECT_FALSE(result->rpcSucceeded(1));
    EXPECT_FALSE(result->success());
}

TEST_F(BroadcastManagerTest, PartialDispatchExceptionKeepsLegacyPropagation) {
    auto server = std::make_unique<TestRpcServer>(std::make_unique<TestRpcService>());
    ASSERT_TRUE(server->start());
    const auto       address = "127.0.0.1:" + std::to_string(server->listenPort());
    BroadcastManager manager({address, address});
    ASSERT_TRUE(manager.init());
    int  calls    = 0;
    auto rpc_call = [&](const std::shared_ptr<RpcService::Stub>&    stub,
                        const std::shared_ptr<grpc::ClientContext>& context,
                        const FunctionRequestPB&                    request,
                        grpc::CompletionQueue*                      queue) -> AsyncFunctionReader {
        if (++calls == 2) {
            throw std::runtime_error("second rank dispatch failed");
        }
        return stub->AsyncExecuteFunction(context.get(), request, queue);
    };
    EXPECT_THROW(
        (manager.broadcast<FunctionRequestPB, FunctionResponsePB>(std::vector<FunctionRequestPB>(2), 1000, rpc_call)),
        RTPException);
    EXPECT_EQ(calls, 2);

    calls = 0;
    BroadcastManager p2p_manager({address, address, address});
    ASSERT_TRUE(p2p_manager.init());
    auto result = p2p_manager.broadcast<FunctionRequestPB, FunctionResponsePB>(
        std::vector<FunctionRequestPB>(3), 1000, rpc_call, {}, BroadcastManager::DispatchFailurePolicy::kStop);
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->waitDone(2000, /*fatal_on_error=*/false));
    EXPECT_THROW(result->waitDone(2000), RTPException);
    EXPECT_EQ(calls, 2);
    EXPECT_TRUE(result->rpcSucceeded(0));
    EXPECT_FALSE(result->rpcSucceeded(1));
    EXPECT_FALSE(result->rpcSucceeded(2));
    EXPECT_NE(result->firstError().error.ToString().find("second rank dispatch failed"), std::string::npos);
}

TEST_F(BroadcastManagerTest, WaitDoneReportsTransportErrors) {
    for (auto code : {grpc::StatusCode::OK,
                      grpc::StatusCode::INTERNAL,
                      grpc::StatusCode::UNAVAILABLE,
                      grpc::StatusCode::CANCELLED}) {
        auto       ctx   = makeIdleContext();
        const bool cq_ok = code != grpc::StatusCode::OK;
        if (cq_ok) {
            ctx->status = grpc::Status(code, "completion confirmation lost");
        }
        FunctionBroadcastResult result({ctx});
        result.finishRank(0, cq_ok);
        EXPECT_TRUE(result.done());
        if (cq_ok) {
            EXPECT_NO_THROW(result.waitDone());
        } else {
            EXPECT_THROW(result.waitDone(), RTPException);
        }
        EXPECT_FALSE(result.success());
    }
}

TEST_F(BroadcastManagerTest, FailedBroadcastCompletesInEitherRankOrder) {
    for (bool failure_first : {false, true}) {
        auto failed                       = makeIdleContext();
        failed->status                    = grpc::Status(grpc::StatusCode::UNAVAILABLE, "connection lost");
        auto                    completed = makeIdleContext();
        FunctionBroadcastResult result({failed, completed});
        result.finishRank(failure_first ? 0 : 1, true);
        EXPECT_FALSE(result.done());
        result.finishRank(failure_first ? 1 : 0, true);
        EXPECT_NO_THROW(result.waitDone());
        EXPECT_TRUE(result.done());
        EXPECT_FALSE(result.success());
    }
}

// ---------------------------- workerNum ----------------------------

TEST_F(BroadcastManagerTest, WorkerNum) {
    EXPECT_EQ(manager_->workerNum(), 2u);
}

TEST(BroadcastDiagnosticsTest, LocalWorkersReuseCompletionWithoutFabricatingRpcEvidence) {
    using Result = BroadcastResult<FunctionRequestPB, FunctionResponsePB>;
    auto first = std::make_shared<Result::WorkerRpcContext>();
    auto second = std::make_shared<Result::WorkerRpcContext>();
    first->status = grpc::Status(grpc::StatusCode::UNAVAILABLE, "not an RPC");
    second->status = first->status;
    first->response.mutable_remote_response()->set_payload_buffer_safe(true);
    second->response.mutable_remote_response()->set_payload_buffer_safe(false);
    Result result({first, second});
    size_t callbacks = 0;
    result.finishLocalRank(0);
    EXPECT_FALSE(result.done());
    result.finishLocalRank(0);
    EXPECT_FALSE(result.done());
    result.finishLocalRank(1); // Both finish inline before driver registration.
    result.onDone([&] { ++callbacks; });
    EXPECT_EQ(callbacks, 1u);
    EXPECT_TRUE(result.done());
    EXPECT_TRUE(result.success()); // Delivery only; payload proof remains separate.
    ASSERT_EQ(result.responses().size(), 2u);
    EXPECT_FALSE(result.responses()[1].remote_response().payload_buffer_safe());
    for (const auto& rank : result.rankSnapshots()) {
        EXPECT_EQ(rank.dispatch_state, BroadcastDispatchState::LOCAL_COMPLETED);
        EXPECT_FALSE(rank.cq_received);
        EXPECT_FALSE(rank.cq_ok);
        EXPECT_FALSE(rank.grpc_code);
        EXPECT_TRUE(rank.response);
        EXPECT_FALSE(result.rpcSucceeded(rank.rank));
    }
    result.finishLocalRank(1);
    EXPECT_EQ(callbacks, 1u);
    auto owned = std::make_shared<Result>(std::vector<std::shared_ptr<Result::WorkerRpcContext>>{first});
    const std::weak_ptr<Result> lifetime = owned;
    owned->onDone([owned] {}); // Same result/callback cycle as the actual payload collector.
    owned->finishLocalRank(0);
    owned.reset();
    EXPECT_TRUE(lifetime.expired()); // finishLocked swaps out callbacks before invoking them.
}

}  // namespace rtp_llm::test

int main(int argc, char** argv) {
    rtp_llm::initLogger();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
