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
        std::atomic<int>                     fail_release_attempts{0};
        std::atomic<int>                     release_requests{0};
        std::atomic<bool>                    drop_release_reply_once{false};
        std::atomic<bool>                    corrupt_release_reply_once{false};
        std::unique_ptr<ShutdownCoordinator> coordinator;
        std::unique_ptr<grpc::Server>        server;
        grpc::Status
        ExecuteFunction(grpc::ServerContext*, const FunctionRequestPB* request, FunctionResponsePB* response) override {
            const bool release = request->shutdown_request().drained_instances_size() != 0;
            if (release) {
                ++release_requests;
                int remaining = fail_release_attempts.load();
                while (remaining > 0 && !fail_release_attempts.compare_exchange_weak(remaining, remaining - 1)) {}
                if (remaining > 0 || fail_first_release.exchange(false)) {
                    return {grpc::StatusCode::UNAVAILABLE, "transient release failure"};
                }
            }
            auto status = coordinator->handle(request->shutdown_request(), response->mutable_shutdown_response());
            if (release && status.ok() && drop_release_reply_once.exchange(false)) {
                return {grpc::StatusCode::UNAVAILABLE, "release accepted but reply lost"};
            }
            if (release && status.ok() && corrupt_release_reply_once.exchange(false)) {
                response->mutable_shutdown_response()->clear_drained_instances();
            }
            return status;
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

    ShutdownRequestPB releaseRequest(size_t sender) {
        auto request = statusRequest();
        for (const auto& peer : peers_) {
            ShutdownResponsePB response;
            EXPECT_TRUE(peer.coordinator->handle(statusRequest(), &response).ok());
            request.add_drained_instances(response.instance_id());
        }
        request.set_sender_rank(sender);
        request.set_sender_instance_id(request.drained_instances(sender));
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
    auto deadline = std::chrono::system_clock::now() + 5s;
    ASSERT_TRUE(peers_[0].coordinator->wait(deadline));
    peers_[0].server->Shutdown();
    auto start = std::chrono::steady_clock::now();
    EXPECT_TRUE(peers_[3].coordinator->wait(deadline));
    EXPECT_LT(std::chrono::steady_clock::now() - start, 1500ms);
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

TEST_F(ShutdownCoordinatorTest, SenderProofSkipsExitedIssuerAndStillRelaysOtherPeers) {
    auto               release = releaseRequest(0);
    ShutdownResponsePB response;
    ASSERT_TRUE(peers_[3].coordinator->handle(release, &response).ok());
    peers_[0].server->Shutdown();
    auto start = std::chrono::steady_clock::now();
    EXPECT_TRUE(peers_[3].coordinator->wait(std::chrono::system_clock::now() + 5s));
    EXPECT_LT(std::chrono::steady_clock::now() - start, 1500ms);
    for (size_t rank : {1, 2}) {
        response.Clear();
        ASSERT_TRUE(peers_[rank].coordinator->handle(statusRequest(), &response).ok());
        EXPECT_EQ(response.drained_instances_size(), 4);
    }
}

TEST_F(ShutdownCoordinatorTest, LegacyCertificateDoesNotInventDefaultRankZeroSender) {
    auto release = releaseRequest(0);
    release.clear_sender_instance_id();
    release.clear_sender_rank();
    ShutdownResponsePB response;
    ASSERT_TRUE(peers_[3].coordinator->handle(release, &response).ok());
    peers_[0].server->Shutdown();
    auto start = std::chrono::steady_clock::now();
    EXPECT_TRUE(peers_[3].coordinator->wait(std::chrono::system_clock::now() + 350ms));
    EXPECT_GE(std::chrono::steady_clock::now() - start, 250ms);
    EXPECT_LT(std::chrono::steady_clock::now() - start, 1500ms);
}

TEST_F(ShutdownCoordinatorTest, RetriesMoreThanTwoTransientReleaseFailures) {
    peers_[2].fail_release_attempts = 4;
    ASSERT_TRUE(peers_[0].coordinator->wait(std::chrono::system_clock::now() + 3s));
    EXPECT_GE(peers_[2].release_requests.load(), 5);
    EXPECT_EQ(peers_[2].fail_release_attempts.load(), 0);
    ShutdownResponsePB response;
    ASSERT_TRUE(peers_[2].coordinator->handle(statusRequest(), &response).ok());
    EXPECT_EQ(response.drained_instances_size(), 4);
}

TEST_F(ShutdownCoordinatorTest, RejectsMissingOutOfRangeAndMismatchedSenderProof) {
    auto               release = releaseRequest(0);
    ShutdownResponsePB response;
    release.set_sender_rank(4);
    EXPECT_EQ(peers_[3].coordinator->handle(release, &response).error_code(), grpc::StatusCode::FAILED_PRECONDITION);
    release.set_sender_rank(0);
    release.set_sender_instance_id("old-or-wrong-instance");
    EXPECT_EQ(peers_[3].coordinator->handle(release, &response).error_code(), grpc::StatusCode::FAILED_PRECONDITION);
    release.clear_drained_instances();
    EXPECT_EQ(peers_[3].coordinator->handle(release, &response).error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    response.Clear();
    ASSERT_TRUE(peers_[3].coordinator->handle(statusRequest(), &response).ok());
    EXPECT_EQ(response.drained_instances_size(), 0);
}

TEST_F(ShutdownCoordinatorTest, RejectsConflictingCertificateWithoutMergingSenderEvidence) {
    auto               release = releaseRequest(0);
    ShutdownResponsePB response;
    ASSERT_TRUE(peers_[3].coordinator->handle(release, &response).ok());
    auto conflicting = release;
    conflicting.set_drained_instances(1, "different-incarnation");
    conflicting.set_sender_rank(1);
    conflicting.set_sender_instance_id("different-incarnation");
    EXPECT_EQ(peers_[3].coordinator->handle(conflicting, &response).error_code(),
              grpc::StatusCode::FAILED_PRECONDITION);
    response.Clear();
    ASSERT_TRUE(peers_[3].coordinator->handle(statusRequest(), &response).ok());
    for (int rank = 0; rank < 4; ++rank) {
        EXPECT_EQ(response.drained_instances(rank), release.drained_instances(rank));
    }
}

TEST_F(ShutdownCoordinatorTest, SenderProofCannotReleaseRestartedReceiver) {
    auto release = releaseRequest(0);
    resetCoordinator(3);
    ShutdownResponsePB response;
    EXPECT_EQ(peers_[3].coordinator->handle(release, &response).error_code(), grpc::StatusCode::FAILED_PRECONDITION);
    response.Clear();
    ASSERT_TRUE(peers_[3].coordinator->handle(statusRequest(), &response).ok());
    EXPECT_EQ(response.drained_instances_size(), 0);
}

TEST_F(ShutdownCoordinatorTest, ConcurrentIssuersKeepRetryingUnknownHolder) {
    // Two local certificates with no remote provenance model simultaneous
    // observers. Neither may infer delivery to rank 2 before it accepts.
    auto release = releaseRequest(0);
    release.clear_sender_instance_id();
    release.clear_sender_rank();
    ShutdownResponsePB response;
    ASSERT_TRUE(peers_[0].coordinator->handle(release, &response).ok());
    response.Clear();
    ASSERT_TRUE(peers_[1].coordinator->handle(release, &response).ok());
    peers_[2].fail_release_attempts = 6;
    const auto deadline             = std::chrono::system_clock::now() + 5s;
    auto       first  = std::async(std::launch::async, [&] { return peers_[0].coordinator->wait(deadline); });
    auto       second = std::async(std::launch::async, [&] { return peers_[1].coordinator->wait(deadline); });
    ASSERT_EQ(first.wait_for(1500ms), std::future_status::ready);
    ASSERT_TRUE(first.get());
    peers_[0].server->Shutdown();
    ASSERT_EQ(second.wait_for(1500ms), std::future_status::ready);
    EXPECT_TRUE(second.get());
    EXPECT_GE(peers_[2].release_requests.load(), 7);
    response.Clear();
    ASSERT_TRUE(peers_[2].coordinator->handle(statusRequest(), &response).ok());
    EXPECT_EQ(response.drained_instances_size(), 4);
}

TEST_F(ShutdownCoordinatorTest, LostAcknowledgementRemainsUnknownAndIsRetried) {
    peers_[2].drop_release_reply_once = true;
    EXPECT_TRUE(peers_[0].coordinator->wait(std::chrono::system_clock::now() + 3s));
    EXPECT_GE(peers_[2].release_requests.load(), 2);
}

TEST_F(ShutdownCoordinatorTest, SuccessfulRpcWithoutMatchingCertificateIsNotDeliveryProof) {
    peers_[2].corrupt_release_reply_once = true;
    EXPECT_TRUE(peers_[0].coordinator->wait(std::chrono::system_clock::now() + 3s));
    EXPECT_GE(peers_[2].release_requests.load(), 2);
}

TEST_F(ShutdownCoordinatorTest, RepeatedWaitRetainsAcknowledgedCertificate) {
    const auto deadline = std::chrono::system_clock::now() + 5s;
    ASSERT_TRUE(peers_[0].coordinator->wait(deadline));
    for (size_t rank : {1, 2, 3}) {
        peers_[rank].server->Shutdown();
    }
    auto start = std::chrono::steady_clock::now();
    EXPECT_TRUE(peers_[0].coordinator->wait(deadline));
    EXPECT_LT(std::chrono::steady_clock::now() - start, 1500ms);
}

TEST_F(ShutdownCoordinatorTest, AnyRankCanInitiateShutdownAndWaitForBusyPeer) {
    peers_[2].draining  = false;
    peers_[2].onflight  = 1;
    const auto deadline = std::chrono::system_clock::now() + 3s;
    auto       issuer   = std::async(std::launch::async, [&] { return peers_[3].coordinator->wait(deadline); });
    EXPECT_EQ(issuer.wait_for(80ms), std::future_status::timeout);
    EXPECT_TRUE(peers_[2].draining);
    peers_[2].onflight = 0;
    ASSERT_EQ(issuer.wait_for(1500ms), std::future_status::ready);
    EXPECT_TRUE(issuer.get());
}

TEST_F(ShutdownCoordinatorTest, ConcurrentIncomingSenderProofUnblocksExistingWait) {
    auto release = releaseRequest(0);
    auto legacy  = release;
    legacy.clear_sender_instance_id();
    legacy.clear_sender_rank();
    ShutdownResponsePB response;
    ASSERT_TRUE(peers_[1].coordinator->handle(legacy, &response).ok());
    peers_[0].server->Shutdown();
    const auto deadline = std::chrono::system_clock::now() + 5s;
    auto       receiver = std::async(std::launch::async, [&] { return peers_[1].coordinator->wait(deadline); });
    EXPECT_EQ(receiver.wait_for(80ms), std::future_status::timeout);
    // Model an already-in-flight issuer message arriving after that issuer's
    // server has closed. Only this exact message supplies the missing proof.
    response.Clear();
    ASSERT_TRUE(peers_[1].coordinator->handle(release, &response).ok());
    ASSERT_EQ(receiver.wait_for(1500ms), std::future_status::ready);
    EXPECT_TRUE(receiver.get());
}
}  // namespace
}  // namespace rtp_llm
