#include "rtp_llm/cpp/model_rpc/LocalRpcServiceImpl.h"
#include "rtp_llm/cpp/model_rpc/RemoteRpcServiceImpl.h"

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <thread>

#include "gtest/gtest.h"

namespace rtp_llm {
namespace {

using namespace std::chrono_literals;

void expectDraining(const grpc::Status& status) {
    EXPECT_EQ(status.error_code(), grpc::StatusCode::UNAVAILABLE);
    EXPECT_EQ(status.error_message(), "server is draining");
}

TEST(RequestAdmissionServiceTest, LocalEndpointsRejectBeforeDereferencingDelegate) {
    LocalRpcServiceImpl service;
    service.beginDrain();
    service.beginDrain();

    EXPECT_TRUE(service.waitForRequestDrain(std::chrono::steady_clock::now()));
    expectDraining(service.GenerateStreamCall(nullptr, nullptr, nullptr));
    expectDraining(service.BatchGenerateCall(nullptr, nullptr, nullptr));
    expectDraining(service.GetCacheStatus(nullptr, nullptr, nullptr));
    expectDraining(service.ExecuteFunction(nullptr, nullptr, nullptr));
}

TEST(RequestAdmissionServiceTest, RemoteGenerateEndpointsUseTheSameGate) {
    RemoteRpcServiceImpl service;
    service.beginDrain();

    EXPECT_TRUE(service.waitForRequestDrain(std::chrono::steady_clock::now()));
    expectDraining(service.GenerateStreamCall(nullptr, nullptr, nullptr));
    expectDraining(service.BatchGenerateCall(nullptr, nullptr, nullptr));
    expectDraining(service.RemoteGenerate(nullptr, nullptr));
}

TEST(RequestAdmissionServiceTest, LocalBatchGenerateRemainsAvailable) {
    LocalRpcServiceImpl    service;
    BatchGenerateInputPB   request;
    BatchGenerateOutputsPB response;
    service.local_server_ = std::make_shared<LocalRpcServer>();

    EXPECT_TRUE(service.BatchGenerateCall(nullptr, &request, &response).ok());
}

TEST(RequestAdmissionServiceTest, RemoteLoadRejectsAsSoonAsDrainBegins) {
    RemoteRpcServiceImpl service;
    service.beginDrain();

    const auto status = service.RemoteLoad(nullptr, nullptr, nullptr);
    EXPECT_EQ(status.error_code(), grpc::StatusCode::UNAVAILABLE);
    EXPECT_EQ(status.error_message(), "server is quiescing remote loads");
    EXPECT_TRUE(service.prepareStop(std::chrono::milliseconds::zero()));
}

TEST(RequestAdmissionServiceTest, RemoteCleanupEndpointsRemainAvailableWhileDraining) {
    RemoteRpcServiceImpl service;
    service.beginDrain();
    EXPECT_TRUE(service.prepareStop(std::chrono::milliseconds::zero()));

    EXPECT_EQ(service.RemoteFinish(nullptr, nullptr, nullptr).error_code(), grpc::StatusCode::INTERNAL);
}

TEST(RequestAdmissionServiceTest, BoundedPrepareStopKeepsUnresolvedRemoteLoadLeaseAlive) {
    std::atomic<bool> allow_quiesce{false};
    auto              lease = std::make_shared<int>(1);
    std::weak_ptr<int> weak_lease = lease;
    RemoteRpcServiceImpl service;
    service.decode_server_ = std::make_shared<DecodeRpcServer>();

    auto ticket = service.decode_server_->remote_load_leases_.reserve(
        "allocation-a", lease, [&]() { return allow_quiesce.load(); });
    ASSERT_TRUE(ticket.ok()) << ticket.status();
    ASSERT_TRUE((*ticket)->markStarted());
    lease.reset();

    EXPECT_FALSE(service.prepareStop(std::chrono::milliseconds::zero()));
    EXPECT_FALSE(weak_lease.expired());
    EXPECT_EQ(service.decode_server_->remote_load_leases_.activeJobsForTest(), 1);
    const auto rejected = service.RemoteLoad(nullptr, nullptr, nullptr);
    EXPECT_EQ(rejected.error_code(), grpc::StatusCode::UNAVAILABLE);
    EXPECT_EQ(rejected.error_message(), "server is quiescing remote loads");

    allow_quiesce = true;
    ticket->reset();
    EXPECT_TRUE(service.prepareStop(std::chrono::milliseconds(100)));
    EXPECT_TRUE(weak_lease.expired());
}

TEST(RequestAdmissionServiceTest, ActiveRemoteLoadAdmissionDrainsBeforeExistingLeases) {
    std::atomic<bool> allow_quiesce{false};
    RemoteRpcServiceImpl service;
    service.decode_server_ = std::make_shared<DecodeRpcServer>();

    auto ticket = service.decode_server_->remote_load_leases_.reserve(
        "allocation-a", std::make_shared<int>(1), [&]() { return allow_quiesce.load(); });
    ASSERT_TRUE(ticket.ok()) << ticket.status();
    ASSERT_TRUE((*ticket)->markStarted());
    auto permit = service.remote_load_admission_gate_.tryAcquire();
    ASSERT_TRUE(permit);
    service.beginDrain();
    EXPECT_TRUE(service.remote_load_admission_gate_.isClosed());
    EXPECT_FALSE(service.remote_load_admission_gate_.tryAcquire());

    auto stop_result = std::async(std::launch::async, [&]() { return service.prepareStop(2s); });
    EXPECT_EQ(stop_result.wait_for(20ms), std::future_status::timeout);

    permit.reset();
    EXPECT_EQ(stop_result.wait_for(20ms), std::future_status::timeout);

    allow_quiesce = true;
    ticket->reset();
    EXPECT_TRUE(stop_result.get());
}

TEST(RequestAdmissionServiceTest, PrepareStopSharesOneDeadlineAcrossAdmissionAndLeaseDrain) {
    constexpr auto kGrace = 800ms;
    constexpr auto kAdmissionDelay = 500ms;
    std::atomic<bool> allow_quiesce{false};
    RemoteRpcServiceImpl service;
    service.decode_server_ = std::make_shared<DecodeRpcServer>();

    auto ticket = service.decode_server_->remote_load_leases_.reserve(
        "allocation-a", std::make_shared<int>(1), [&]() { return allow_quiesce.load(); });
    ASSERT_TRUE(ticket.ok()) << ticket.status();
    ASSERT_TRUE((*ticket)->markStarted());
    auto permit = service.remote_load_admission_gate_.tryAcquire();
    ASSERT_TRUE(permit);
    service.beginDrain();

    auto release_admission = std::async(std::launch::async, [&]() {
        std::this_thread::sleep_for(kAdmissionDelay);
        permit.reset();
    });
    const auto started = std::chrono::steady_clock::now();
    EXPECT_FALSE(service.prepareStop(kGrace));
    const auto elapsed = std::chrono::steady_clock::now() - started;
    release_admission.get();

    EXPECT_GE(elapsed, 650ms);
    EXPECT_LT(elapsed, 1100ms);
    EXPECT_TRUE(service.remote_load_admission_gate_.isClosed());
    EXPECT_FALSE(service.remote_load_admission_gate_.tryAcquire());

    allow_quiesce = true;
    ticket->reset();
    EXPECT_TRUE(service.prepareStop(1s));
}

}  // namespace
}  // namespace rtp_llm
