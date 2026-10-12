#include <chrono>
#include <condition_variable>
#include <mutex>
#include <vector>

#include "grpc++/grpc++.h"
#include "gtest/gtest.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"
#include "rtp_llm/cpp/multimodal_processor/transport/grpc/MMGrpcTransport.h"

namespace rtp_llm {
namespace {

class ReleaseService: public MultimodalRpcService::Service {
public:
    grpc::StatusCode                      failure    = grpc::StatusCode::DEADLINE_EXCEEDED;
    int                                   fail_count = 1;
    std::mutex                            mutex;
    std::condition_variable               cv;
    std::vector<std::vector<std::string>> calls;

    grpc::Status ReleaseRdmaLease(grpc::ServerContext*, const ReleaseLeasePB* request, EmptyPB*) override {
        std::lock_guard<std::mutex> lock(mutex);
        calls.emplace_back(request->lease_id().begin(), request->lease_id().end());
        cv.notify_all();
        return calls.size() <= fail_count ? grpc::Status(failure, "injected failure") : grpc::Status::OK;
    }

    bool waitCalls(size_t count, int timeout_ms = 3000) {
        std::unique_lock<std::mutex> lock(mutex);
        return cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] { return calls.size() >= count; });
    }
};

class MMGrpcReleaseTest: public ::testing::Test {
protected:
    void SetUp() override {
        grpc::ServerBuilder builder;
        int                 port = 0;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
        builder.RegisterService(&service);
        server = builder.BuildAndStart();
        ASSERT_NE(server, nullptr);
        endpoint = "127.0.0.1:" + std::to_string(port);
        client   = createGrpcMMControlClient(std::make_shared<MMTransportMetrics>(nullptr), 100);
    }
    void TearDown() override {
        client.reset();
        server->Shutdown();
    }
    ReleaseService                   service;
    std::unique_ptr<grpc::Server>    server;
    std::unique_ptr<MMControlClient> client;
    std::string                      endpoint;
};

TEST_F(MMGrpcReleaseTest, retriesTimedOutReleaseAndRetainsDedupKeys) {
    client->releaseAsync(endpoint, {"lease", "lease", ""});
    ASSERT_TRUE(service.waitCalls(1));
    client->releaseAsync(endpoint, {"lease"});
    ASSERT_TRUE(service.waitCalls(2));
    EXPECT_FALSE(service.waitCalls(3, 300));
    std::lock_guard<std::mutex> lock(service.mutex);
    ASSERT_EQ(service.calls.size(), 2u);
    EXPECT_EQ(service.calls[0], std::vector<std::string>({"lease"}));
    EXPECT_EQ(service.calls[1], service.calls[0]);
}

TEST_F(MMGrpcReleaseTest, exhaustedRequestBudgetStillReleasesAsynchronously) {
    service.fail_count = 0;
    DeadlineBudget expired(0);
    client->release(endpoint, {"lease"}, expired);
    ASSERT_TRUE(service.waitCalls(1));
}

TEST_F(MMGrpcReleaseTest, retryCountIsBounded) {
    service.fail_count = 100;
    client->releaseAsync(endpoint, {"lease"});
    ASSERT_TRUE(service.waitCalls(6, 6000));
    EXPECT_FALSE(service.waitCalls(7, 500));
}

TEST_F(MMGrpcReleaseTest, permanentFailureDoesNotRetry) {
    service.failure = grpc::StatusCode::INVALID_ARGUMENT;
    client->releaseAsync(endpoint, {"lease"});
    ASSERT_TRUE(service.waitCalls(1));
    EXPECT_FALSE(service.waitCalls(2, 300));
}

}  // namespace
}  // namespace rtp_llm
