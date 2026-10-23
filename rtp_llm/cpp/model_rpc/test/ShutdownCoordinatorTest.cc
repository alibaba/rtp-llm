#include "rtp_llm/cpp/model_rpc/ShutdownCoordinator.h"

#include <array>
#include <atomic>
#include <future>
#include <grpcpp/grpcpp.h>
#include <gtest/gtest.h>

namespace rtp_llm {
namespace {
using namespace std::chrono_literals;

class ShutdownCoordinatorTest: public ::testing::Test {
protected:
    struct Peer: RpcService::Service {
        std::atomic<bool>                    draining{true};
        std::atomic<size_t>                  onflight{0};
        std::atomic<bool>                    fail_first_release{false};
        std::unique_ptr<ShutdownCoordinator> coordinator;
        std::unique_ptr<grpc::Server>        server;
        grpc::Status
        ExecuteFunction(grpc::ServerContext*, const FunctionRequestPB* request, FunctionResponsePB* response) override {
            if (request->shutdown_request().drained_instances_size() && fail_first_release.exchange(false)) {
                return {grpc::StatusCode::UNAVAILABLE, "transient release failure"};
            }
            return coordinator->handle(request->shutdown_request(), response->mutable_shutdown_response());
        }
    };

    void SetUp() override {
        for (auto& peer : peers_) {
            int                 port = 0;
            grpc::ServerBuilder builder;
            builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
            builder.RegisterService(&peer);
            peer.server = builder.BuildAndStart();
            ASSERT_NE(peer.server, nullptr);
            addresses_.push_back("127.0.0.1:" + std::to_string(port));
        }
        for (size_t i = 0; i < peers_.size(); ++i) {
            resetCoordinator(i);
        }
    }

    void resetCoordinator(size_t rank) {
        peers_[rank].coordinator = std::make_unique<ShutdownCoordinator>(
            addresses_,
            rank,
            [this, rank] {
                return ShutdownCoordinator::State{peers_[rank].draining.load(), peers_[rank].onflight.load()};
            },
            [this, rank] { peers_[rank].draining = true; });
    }

    void TearDown() override {
        for (auto& peer : peers_) {
            peer.server->Shutdown(std::chrono::system_clock::now());
        }
    }

    ShutdownRequestPB statusRequest() {
        ShutdownRequestPB request;
        std::string       group;
        for (const auto& address : addresses_) {
            group += std::to_string(address.size()) + ":" + address;
        }
        request.set_group_id(group);
        return request;
    }

    std::array<Peer, 4>      peers_;
    std::vector<std::string> addresses_;
};

TEST_F(ShutdownCoordinatorTest, IdleRanksWaitForStaggeredBusyRank) {
    peers_[1].draining = false;
    peers_[1].onflight = 1;
    auto deadline      = std::chrono::system_clock::now() + 3s;
    auto idle          = std::async(std::launch::async, [&] { return peers_[0].coordinator->wait(deadline); });
    EXPECT_EQ(idle.wait_for(80ms), std::future_status::timeout);
    EXPECT_TRUE(peers_[1].draining) << "peer BEGIN closes admission before its local stop signal";
    EXPECT_EQ(idle.wait_for(80ms), std::future_status::timeout);
    peers_[1].onflight = 0;
    EXPECT_TRUE(idle.get());
    for (size_t i = 1; i < peers_.size(); ++i) {
        EXPECT_TRUE(peers_[i].coordinator->wait(deadline));
    }
}

TEST_F(ShutdownCoordinatorTest, ReleaseSurvivesObserverExit) {
    auto deadline = std::chrono::system_clock::now() + 2s;
    ASSERT_TRUE(peers_[0].coordinator->wait(deadline));
    peers_[0].server->Shutdown();
    EXPECT_TRUE(peers_[3].coordinator->wait(deadline));
}

TEST_F(ShutdownCoordinatorTest, MissingPeerIsNotDrainedAndWaitIsBounded) {
    peers_[1].server->Shutdown();
    auto start = std::chrono::system_clock::now();
    EXPECT_FALSE(peers_[0].coordinator->wait(start + 300ms));
    EXPECT_LT(std::chrono::system_clock::now() - start, 1s);
}

TEST_F(ShutdownCoordinatorTest, OldReleaseCannotDrainRestartedProcess) {
    ASSERT_TRUE(peers_[0].coordinator->wait(std::chrono::system_clock::now() + 2s));
    ShutdownResponsePB response;
    auto               request = statusRequest();
    ASSERT_TRUE(peers_[0].coordinator->handle(request, &response).ok());
    ASSERT_EQ(response.drained_instances_size(), 4);
    for (const auto& id : response.drained_instances()) {
        request.add_drained_instances(id);
    }
    resetCoordinator(1);
    EXPECT_EQ(peers_[1].coordinator->handle(request, &response).error_code(), grpc::StatusCode::FAILED_PRECONDITION);
}

TEST_F(ShutdownCoordinatorTest, RetriesReleaseBeforeObserverExits) {
    peers_[2].fail_first_release = true;
    ASSERT_TRUE(peers_[0].coordinator->wait(std::chrono::system_clock::now() + 2s));
    peers_[0].server->Shutdown();
    auto               request = statusRequest();
    ShutdownResponsePB response;
    ASSERT_TRUE(peers_[2].coordinator->handle(request, &response).ok());
    EXPECT_EQ(response.drained_instances_size(), 4);
}

TEST_F(ShutdownCoordinatorTest, LaterStopUsesEarliestGroupDeadline) {
    peers_[1].onflight = 1;
    EXPECT_FALSE(peers_[0].coordinator->wait(std::chrono::system_clock::now() + 100ms));
    auto start = std::chrono::system_clock::now();
    EXPECT_FALSE(peers_[1].coordinator->wait(start + 2s));
    EXPECT_LT(std::chrono::system_clock::now() - start, 200ms);
}

TEST_F(ShutdownCoordinatorTest, RejectsDifferentGroup) {
    ShutdownRequestPB request;
    request.set_group_id("other group");
    ShutdownResponsePB response;
    EXPECT_EQ(peers_[0].coordinator->handle(request, &response).error_code(), grpc::StatusCode::FAILED_PRECONDITION);
}
}  // namespace
}  // namespace rtp_llm
