#include "rtp_llm/cpp/model_rpc/PDWriteLease.h"

#include <grpcpp/grpcpp.h>
#include <gtest/gtest.h>
#include <atomic>
#include <future>
#include <thread>

namespace rtp_llm {

TEST(PDWriteLeaseTest, ClosesAdmissionAndTracksLastPhysicalWriter) {
    auto                         registry = std::make_shared<PDWriteLeaseRegistry>();
    PreparePDWriteLeaseRequestPB prepare;
    prepare.set_request_key("first");
    prepare.set_timeout_ms(10000);
    BroadcastLoadRequestPB load;
    *load.mutable_write_lease() = registry->prepare(prepare);
    EXPECT_EQ(registry->activeWriters(), 0u);
    EXPECT_FALSE(registry->claim(load));  // Wrong request identity cannot consume the token.
    load.set_request_key(prepare.request_key());
    auto writer = registry->claim(load);
    ASSERT_TRUE(writer);
    EXPECT_EQ(registry->activeWriters(), 1u);
    EXPECT_FALSE(registry->claim(load));
    bool stopped = false;
    EXPECT_TRUE(registry->seal(load.write_lease(), stopped));
    EXPECT_FALSE(stopped);
    auto observer = writer;
    writer.reset();
    EXPECT_TRUE(registry->seal(load.write_lease(), stopped));
    EXPECT_FALSE(stopped);
    observer.reset();
    EXPECT_EQ(registry->activeWriters(), 0u);
    EXPECT_TRUE(registry->seal(load.write_lease(), stopped));
    EXPECT_TRUE(stopped);
    EXPECT_FALSE(registry->claim(load));
    // An old token cannot be certified or claimed via a replacement incarnation,
    // even if that incarnation's first sequence number is numerically identical.
    auto replacement = std::make_shared<PDWriteLeaseRegistry>();
    auto another     = replacement->prepare(prepare);
    EXPECT_EQ(another.token(), load.write_lease().token());
    EXPECT_FALSE(replacement->seal(load.write_lease(), stopped));
    EXPECT_FALSE(stopped);
    EXPECT_FALSE(replacement->claim(load));

    *load.mutable_write_lease() = another;
    EXPECT_TRUE(replacement->seal(another, stopped));
    EXPECT_TRUE(stopped);
    EXPECT_FALSE(replacement->claim(load));  // Seal won the race against RemoteLoad.
    prepare.set_timeout_ms(1);
    *load.mutable_write_lease() = replacement->prepare(prepare);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    EXPECT_FALSE(replacement->claim(load));
    EXPECT_TRUE(replacement->seal(load.write_lease(), stopped));
    EXPECT_TRUE(stopped);
}

TEST(PDWriteLeaseTest, SealArmsRecoveryOnceForEachContextAndWaitsForPhysicalProof) {
    auto                         registry = std::make_shared<PDWriteLeaseRegistry>();
    PreparePDWriteLeaseRequestPB prepare;
    prepare.set_timeout_ms(10000);
    BroadcastLoadRequestPB request;
    const auto             lease   = registry->prepare(prepare);
    *request.mutable_write_lease() = lease;
    auto writer                    = registry->claim(request);
    ASSERT_TRUE(writer);
    int registered_before = 0;
    int registered_after  = 0;
    int attempts          = 0;
    registry->registerRecovery(lease, [&] {
        EXPECT_EQ(registry->activeWriters(), 1u);
        ++registered_before;
    });
    registry->registerRecovery(lease, [&] {
        if (++attempts == 1)
            throw std::runtime_error("recovery temporarily unavailable");
    });
    EXPECT_EQ(registered_before, 0);
    bool stopped = true;
    ASSERT_TRUE(registry->seal(lease, stopped));
    EXPECT_FALSE(stopped);
    EXPECT_EQ(registered_before, 1);
    EXPECT_EQ(attempts, 1);
    registry->registerRecovery(lease, [&] { ++registered_after; });
    EXPECT_EQ(registered_after, 1);
    ASSERT_TRUE(registry->seal(lease, stopped));
    EXPECT_FALSE(stopped);
    EXPECT_EQ(registered_before, 1);
    EXPECT_EQ(registered_after, 1);
    EXPECT_EQ(attempts, 2);
    writer.reset();
    ASSERT_TRUE(registry->seal(lease, stopped));
    EXPECT_TRUE(stopped);
}

namespace {
class LeaseService final: public RpcService::Service {
public:
    grpc::Status PreparePDWriteLease(grpc::ServerContext*,
                                     const PreparePDWriteLeaseRequestPB* request,
                                     PDWriteLeasePB*                     response) override {
        *response = registry->prepare(*request);
        response->set_supports_protected_positions(supports_protected_positions.load());
        return grpc::Status::OK;
    }
    grpc::Status
    SealPDWriteLease(grpc::ServerContext*, const PDWriteLeasePB* lease, SealPDWriteLeaseResponsePB* response) override {
        if (++seal_calls == 1) {
            lost_reply.set_value();
            return grpc::Status(grpc::StatusCode::UNAVAILABLE, "simulated lost seal response");
        }
        if (lease->token() <= unavailable_through.load()) {
            return grpc::Status(grpc::StatusCode::UNAVAILABLE, "simulated unreachable peer");
        }
        bool stopped = false;
        if (!registry->seal(*lease, stopped))
            return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "foreign lease");
        *response->mutable_lease() = *lease;
        response->set_stopped(stopped);
        return grpc::Status::OK;
    }

    std::shared_ptr<PDWriteLeaseRegistry> registry = std::make_shared<PDWriteLeaseRegistry>();
    std::atomic<bool>                     supports_protected_positions{true};
    std::atomic<int>                      seal_calls{0};
    std::atomic<uint64_t>                 unavailable_through{0};
    std::promise<void>                    lost_reply;
};
}  // namespace

TEST(PDWriteLeaseTest, RetriesLostStopReceiptAndWaitsForEveryWriter) {
    using namespace std::chrono_literals;
    LeaseService        service;
    grpc::ServerBuilder builder;
    int                 port = 0;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&service);
    auto server = builder.BuildAndStart();
    ASSERT_TRUE(server);
    std::shared_ptr<RpcService::Stub> stub = RpcService::NewStub(
        grpc::CreateChannel("127.0.0.1:" + std::to_string(port), grpc::InsecureChannelCredentials()));
    std::vector<PDWriteLeasePeer> peers{{stub, {}}, {stub, {}}};
    PreparePDWriteLeaseRequestPB  request;
    request.set_request_id(23);
    request.set_request_key("delayed-stop");
    request.set_timeout_ms(10000);
    service.supports_protected_positions.store(false);
    const auto unsupported =
        preparePDWriteLeases(peers, request, std::chrono::system_clock::now() + 5s, [] { return false; });
    EXPECT_EQ(unsupported.error_code(), grpc::StatusCode::FAILED_PRECONDITION);
    service.supports_protected_positions.store(true);
    ASSERT_TRUE(preparePDWriteLeases(peers, request, std::chrono::system_clock::now() + 5s, [] { return false; }).ok());
    ASSERT_NE(peers[0].lease.token(), peers[1].lease.token());
    BroadcastLoadRequestPB load;
    load.set_request_id(request.request_id());
    load.set_request_key(request.request_key());
    *load.mutable_write_lease() = peers[0].lease;
    auto first_writer           = service.registry->claim(load);
    *load.mutable_write_lease() = peers[1].lease;
    auto last_writer            = service.registry->claim(load);
    ASSERT_TRUE(first_writer);
    ASSERT_TRUE(last_writer);
    auto released = std::make_shared<std::promise<void>>();
    auto proof    = released->get_future();
    // Each peer owns a reference to the same destination.
    std::shared_ptr<void>        pin(released.get(), [released](void*) { released->set_value(); });
    std::list<PDWriteLeaseDrain> pending{{std::move(peers[0]), pin}, {std::move(peers[1]), pin}};
    auto                         lost = service.lost_reply.get_future();
    PDWriteLeaseRecovery::instance().adopt(pending);
    pin.reset();
    first_writer.reset();
    ASSERT_EQ(lost.wait_for(5s), std::future_status::ready);
    EXPECT_EQ(proof.wait_for(200ms), std::future_status::timeout);
    last_writer.reset();
    EXPECT_EQ(proof.wait_for(5s), std::future_status::ready);
    EXPECT_GE(service.seal_calls.load(), 3);
}

TEST(PDWriteLeaseTest, ShutdownKeepsClaimedWriterWhileRecoveryIsInProgress) {
    using namespace std::chrono_literals;
    auto registry = std::make_shared<PDWriteLeaseRegistry>();
    PreparePDWriteLeaseRequestPB prepare;
    prepare.set_request_key("shutdown-drain");
    prepare.set_timeout_ms(10000);
    BroadcastLoadRequestPB request;
    request.set_request_key(prepare.request_key());
    *request.mutable_write_lease() = registry->prepare(prepare);
    auto writer = registry->claim(request);
    ASSERT_TRUE(writer);
    auto unclaimed_lease = registry->prepare(prepare);

    std::promise<void> recovery_started;
    std::promise<void> finish_recovery;
    auto started = recovery_started.get_future();
    auto finish = finish_recovery.get_future();
    registry->registerRecovery(request.write_lease(), [&] {
        recovery_started.set_value();
        finish.wait();
    });
    auto seal = std::async(std::launch::async, [&] {
        bool stopped = false;
        return registry->seal(request.write_lease(), stopped) && !stopped;
    });

    EXPECT_EQ(started.wait_for(5s), std::future_status::ready);
    registry->closeAdmission();
    EXPECT_EQ(registry->activeWriters(), 1u);
    EXPECT_EQ(registry->prepare(prepare).token(), 0u);
    BroadcastLoadRequestPB late_request = request;
    *late_request.mutable_write_lease() = unclaimed_lease;
    EXPECT_FALSE(registry->claim(late_request));

    finish_recovery.set_value();
    EXPECT_TRUE(seal.get());
    writer.reset();
    EXPECT_EQ(registry->activeWriters(), 0u);
    bool stopped = false;
    EXPECT_TRUE(registry->seal(request.write_lease(), stopped));
    EXPECT_TRUE(stopped);
}

TEST(PDWriteLeaseTest, RecoversReachableLeasesPastFirstBatchWhileRetainingUnavailablePins) {
    using namespace std::chrono_literals;
    constexpr size_t unavailable_count = 33;
    constexpr size_t reachable_count = 7;

    LeaseService service;
    service.unavailable_through.store(unavailable_count);
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&service);
    auto server = builder.BuildAndStart();
    ASSERT_TRUE(server);
    std::shared_ptr<RpcService::Stub> stub = RpcService::NewStub(
        grpc::CreateChannel("127.0.0.1:" + std::to_string(port), grpc::InsecureChannelCredentials()));

    struct PinGroup {
        explicit PinGroup(size_t count): remaining(count) {}
        std::atomic<size_t> remaining;
        std::promise<void> drained;
    };
    auto unavailable = std::make_shared<PinGroup>(unavailable_count);
    auto reachable = std::make_shared<PinGroup>(reachable_count);
    auto unavailable_proof = unavailable->drained.get_future();
    auto reachable_proof = reachable->drained.get_future();
    PreparePDWriteLeaseRequestPB prepare;
    prepare.set_request_key("large-drain");
    prepare.set_timeout_ms(10000);
    std::list<PDWriteLeaseDrain> pending;
    for (size_t i = 0; i < unavailable_count + reachable_count; ++i) {
        auto lease = service.registry->prepare(prepare);
        auto group = i < unavailable_count ? unavailable : reachable;
        std::shared_ptr<void> pin(group.get(), [group](void*) {
            if (group->remaining.fetch_sub(1) == 1) {
                group->drained.set_value();
            }
        });
        pending.push_back({{stub, std::move(lease)}, std::move(pin)});
    }
    PDWriteLeaseRecovery::instance().adopt(pending);

    EXPECT_EQ(reachable_proof.wait_for(5s), std::future_status::ready);
    EXPECT_EQ(reachable->remaining.load(), 0u);
    EXPECT_EQ(unavailable_proof.wait_for(0s), std::future_status::timeout);
    EXPECT_EQ(unavailable->remaining.load(), unavailable_count);
    service.unavailable_through.store(0);
    EXPECT_EQ(unavailable_proof.wait_for(5s), std::future_status::ready);
    EXPECT_EQ(unavailable->remaining.load(), 0u);
}

}  // namespace rtp_llm
