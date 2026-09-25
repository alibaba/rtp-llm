#include "rtp_llm/cpp/engine_base/sleep/test/BoundSleepLifecycleController.h"
#include "gtest/gtest.h"

#include "rtp_llm/cpp/engine_base/sleep/AdmissionGate.h"
#include "rtp_llm/cpp/engine_base/sleep/SleepLifecycleController.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

namespace {

constexpr int64_t kEngineUnavailable = 8600;

SleepHooks successHooks() {
    SleepHooks hooks;
    hooks.drain = [](const SleepOptions&) { return true; };
    return hooks;
}

}  // namespace

class AdmissionGateTest: public ::testing::Test {
protected:
    BoundSleepLifecycleController controller_{true};
    AdmissionGate            gate_{&controller_, "test_instance_0"};
};

TEST_F(AdmissionGateTest, RunningAdmits) {
    ASSERT_EQ(controller_.state(), SleepState::RUNNING);
    EXPECT_TRUE(gate_.check().ok());
    const auto detail = gate_.checkDetail();
    EXPECT_TRUE(detail.admitted);
    EXPECT_EQ(detail.error_code, 0);
    EXPECT_EQ(detail.state, "RUNNING");
    EXPECT_EQ(detail.instance_id, "test_instance_0");
}

TEST_F(AdmissionGateTest, CompletionNotificationIsIdempotentAndDoesNotOwnLease) {
    auto acquired = gate_.acquire();
    ASSERT_TRUE(acquired.detail.admitted);
    ASSERT_TRUE(static_cast<bool>(acquired.complete));
    EXPECT_EQ(controller_.activeAdmissionCount(), 1);

    auto duplicate = acquired.complete;
    EXPECT_EQ(controller_.activeAdmissionCount(), 1);

    acquired.complete();
    duplicate();
    EXPECT_EQ(controller_.activeAdmissionCount(), 0);
}

TEST_F(AdmissionGateTest, SuccessfulAcquireDoesNotBuildStatusStrings) {
    auto acquired = gate_.acquire();
    ASSERT_TRUE(acquired.detail.admitted);
    EXPECT_TRUE(acquired.detail.instance_id.empty());
    EXPECT_TRUE(acquired.detail.state.empty());
    EXPECT_TRUE(acquired.detail.message.empty());
    EXPECT_TRUE(acquired.detail.error_code_str.empty());
    EXPECT_TRUE(static_cast<bool>(acquired.complete));
    acquired.complete();
}

TEST(AdmissionGateDisabledTest, DisabledSleepStillTracksAdmission) {
    BoundSleepLifecycleController controller(false);
    AdmissionGate gate(&controller, "long-instance-id-that-would-require-a-string-allocation");
    auto acquired = gate.acquire();
    EXPECT_TRUE(acquired.detail.admitted);
    EXPECT_TRUE(static_cast<bool>(acquired.complete));
    EXPECT_EQ(controller.activeAdmissionCount(), 1);
    acquired.complete();
    EXPECT_EQ(controller.activeAdmissionCount(), 0);
    EXPECT_TRUE(acquired.detail.instance_id.empty());
    EXPECT_TRUE(admitAndComplete(controller.admission()->admit()));
}

TEST_F(AdmissionGateTest, MissingAdmissionFailsClosed) {
    AdmissionGate null_gate(nullptr, "no_controller");
    EXPECT_FALSE(null_gate.check().ok());
    EXPECT_FALSE(null_gate.checkDetail().admitted);
}

TEST_F(AdmissionGateTest, DrainingRejects) {
    SleepHooks hooks;
    // Drain "timeout": controller stays in DRAINING per design.
    hooks.drain = [](const SleepOptions&) { return false; };
    controller_.setHooks(hooks);
    EXPECT_FALSE(controller_.sleep(SleepOptions{}).ok);
    ASSERT_EQ(controller_.state(), SleepState::DRAINING);

    const auto status = gate_.check();
    EXPECT_EQ(status.error_code(), grpc::StatusCode::UNAVAILABLE);
    const auto detail = gate_.checkDetail();
    EXPECT_FALSE(detail.admitted);
    EXPECT_EQ(detail.state, "DRAINING");

    auto acquired = gate_.acquire();
    EXPECT_FALSE(acquired.detail.admitted);
    EXPECT_FALSE(static_cast<bool>(acquired.complete));
    EXPECT_EQ(acquired.detail.state, "DRAINING");
    EXPECT_EQ(controller_.activeAdmissionCount(), 0);
}

TEST_F(AdmissionGateTest, SuspendingRejects) {
    SleepHooks hooks = successHooks();
    // Observe gate behavior while the controller is mid-SUSPENDING.
    hooks.releaseKvMemoryBacking = [this](const SleepOptions&) {
        EXPECT_EQ(controller_.state(), SleepState::SUSPENDING);
        const auto status = gate_.check();
        EXPECT_EQ(status.error_code(), grpc::StatusCode::UNAVAILABLE);
        EXPECT_EQ(gate_.checkDetail().state, "SUSPENDING");
        return true;
    };
    controller_.setHooks(hooks);
    EXPECT_TRUE(controller_.sleep(SleepOptions{}).ok);
}

TEST_F(AdmissionGateTest, SleepingRejects) {
    controller_.setHooks(successHooks());
    ASSERT_TRUE(controller_.sleep(SleepOptions{}).ok);
    ASSERT_EQ(controller_.state(), SleepState::SLEEPING);

    const auto status = gate_.check();
    EXPECT_EQ(status.error_code(), grpc::StatusCode::UNAVAILABLE);
    const auto detail = gate_.checkDetail();
    EXPECT_FALSE(detail.admitted);
    EXPECT_EQ(detail.state, "SLEEPING");
    EXPECT_EQ(detail.error_code, kEngineUnavailable);
}

TEST_F(AdmissionGateTest, WakingUpRejects) {
    SleepHooks hooks = successHooks();
    // Observe gate behavior while the controller is mid-WAKING_UP.
    hooks.restoreKvMemoryBackingAndResetMetadata = [this]() {
        EXPECT_EQ(controller_.state(), SleepState::WAKING_UP);
        const auto status = gate_.check();
        EXPECT_EQ(status.error_code(), grpc::StatusCode::UNAVAILABLE);
        EXPECT_EQ(gate_.checkDetail().state, "WAKING_UP");
        return true;
    };
    controller_.setHooks(hooks);
    ASSERT_TRUE(controller_.sleep(SleepOptions{}).ok);
    EXPECT_TRUE(controller_.wakeUp().ok);
}

TEST_F(AdmissionGateTest, ErrorRejects) {
    SleepHooks hooks             = successHooks();
    hooks.releaseKvMemoryBacking = [](const SleepOptions&) { return false; };
    controller_.setHooks(hooks);
    EXPECT_FALSE(controller_.sleep(SleepOptions{}).ok);
    ASSERT_EQ(controller_.state(), SleepState::ERROR);

    const auto status = gate_.check();
    EXPECT_EQ(status.error_code(), grpc::StatusCode::UNAVAILABLE);
    const auto detail = gate_.checkDetail();
    EXPECT_FALSE(detail.admitted);
    EXPECT_EQ(detail.state, "ERROR");
}

TEST_F(AdmissionGateTest, ErrorBodyFieldsComplete) {
    controller_.setHooks(successHooks());
    ASSERT_TRUE(controller_.sleep(SleepOptions{}).ok);
    ASSERT_EQ(controller_.state(), SleepState::SLEEPING);

    // Structured detail carries the full RPC error body.
    const auto detail = gate_.checkDetail();
    EXPECT_EQ(detail.error_code, kEngineUnavailable);
    EXPECT_EQ(detail.error_code, static_cast<int64_t>(ErrorCode::ENGINE_UNAVAILABLE));
    EXPECT_EQ(detail.error_code_str, "ENGINE_UNAVAILABLE");
    EXPECT_EQ(detail.instance_id, "test_instance_0");
    EXPECT_EQ(detail.sleep_epoch, controller_.sleepEpoch());
    EXPECT_GE(detail.sleep_epoch, 1);
    EXPECT_EQ(detail.state, "SLEEPING");
    EXPECT_FALSE(detail.message.empty());
    EXPECT_EQ(detail.message.find("test_instance_0"), std::string::npos);

    // grpc::Status error_details round-trips through ErrorDetailsPB.
    const auto status = gate_.check();
    EXPECT_EQ(status.error_code(), grpc::StatusCode::UNAVAILABLE);
    EXPECT_EQ(status.error_message(), detail.message);
    EXPECT_EQ(status.error_message().find("test_instance_0"), std::string::npos);
    ErrorDetailsPB details;
    ASSERT_TRUE(details.ParseFromString(status.error_details()));
    EXPECT_EQ(details.error_code(), kEngineUnavailable);
    EXPECT_EQ(details.error_code_str(), "ENGINE_UNAVAILABLE");
    EXPECT_EQ(details.error_message(), detail.message);
    EXPECT_EQ(details.instance_id(), "test_instance_0");
    EXPECT_EQ(details.sleep_epoch(), detail.sleep_epoch);
    EXPECT_EQ(details.state(), "SLEEPING");
}

TEST_F(AdmissionGateTest, AdmitsAgainAfterWakeUp) {
    controller_.setHooks(successHooks());
    ASSERT_TRUE(controller_.sleep(SleepOptions{}).ok);
    EXPECT_EQ(gate_.check().error_code(), grpc::StatusCode::UNAVAILABLE);

    ASSERT_TRUE(controller_.wakeUp().ok);
    ASSERT_EQ(controller_.state(), SleepState::RUNNING);
    EXPECT_TRUE(gate_.check().ok());
    const auto detail = gate_.checkDetail();
    EXPECT_TRUE(detail.admitted);
    EXPECT_EQ(detail.state, "RUNNING");
    // Epoch from the completed sleep cycle is still visible.
    EXPECT_GE(detail.sleep_epoch, 1);
}

TEST_F(AdmissionGateTest, KvContinuationIsCountedDuringDrainButRootAdmissionStaysClosed) {
    SleepHooks hooks;
    hooks.drain = [](const SleepOptions&) { return false; };
    controller_.setHooks(hooks);
    ASSERT_FALSE(controller_.sleep(SleepOptions{}).ok);
    auto child = gate_.acquireCacheTransfer();
    EXPECT_TRUE(child.detail.admitted);
    EXPECT_TRUE(static_cast<bool>(child.complete));
    EXPECT_EQ(controller_.activeAdmissionCount(), 1);
    EXPECT_FALSE(gate_.acquire().detail.admitted);
    EXPECT_EQ(gate_.check().error_code(), grpc::StatusCode::UNAVAILABLE);
    EXPECT_TRUE(child.detail.message.empty());
    child.complete();
    EXPECT_EQ(controller_.activeAdmissionCount(), 0);
}

TEST_F(AdmissionGateTest, ClosedContinuationGateKeepsStructuredErrorAndReopensAfterWake) {
    controller_.setHooks(successHooks());
    ASSERT_TRUE(controller_.sleep(SleepOptions{}).ok);
    auto child = gate_.acquireCacheTransfer();
    ASSERT_FALSE(child.detail.admitted);
    EXPECT_FALSE(static_cast<bool>(child.complete));
    const auto status = AdmissionGate::toGrpcStatus(child.detail);
    EXPECT_EQ(status.error_code(), grpc::StatusCode::UNAVAILABLE);
    ErrorDetailsPB details;
    ASSERT_TRUE(details.ParseFromString(status.error_details()));
    EXPECT_EQ(details.error_code(), kEngineUnavailable);
    EXPECT_EQ(details.state(), "SLEEPING");
    EXPECT_EQ(details.sleep_epoch(), 1);
    EXPECT_EQ(details.instance_id(), "test_instance_0");
    // Later lifecycle states retain their normal state-specific explanation.
    EXPECT_EQ(status.error_message(), gate_.acquire().detail.message);
    ASSERT_TRUE(controller_.wakeUp().ok);
    auto next_child = gate_.acquireCacheTransfer();
    auto next_root  = gate_.acquire();
    EXPECT_TRUE(next_child.detail.admitted);
    EXPECT_TRUE(next_root.detail.admitted);
    next_child.complete();
    next_root.complete();
}

TEST_F(AdmissionGateTest, FrozenContinuationExplainsPhaseWithoutChangingErrorContract) {
    controller_.setHooks(successHooks());
    SleepOptions options;
    options.prepare_only = true;
    ASSERT_TRUE(controller_.sleep(options).ok);
    ASSERT_EQ(controller_.state(), SleepState::DRAINING);

    const auto root  = gate_.acquire();
    const auto child = gate_.acquireCacheTransfer();
    ASSERT_FALSE(root.detail.admitted);
    ASSERT_FALSE(child.detail.admitted);
    EXPECT_FALSE(static_cast<bool>(child.complete));
    EXPECT_EQ(controller_.activeAdmissionCount(), 0);
    const auto epoch = std::to_string(controller_.sleepEpoch());
    EXPECT_EQ(root.detail.message,
              "engine unavailable: DRAINING (sleep_epoch=" + epoch + "), request can be retried elsewhere");
    EXPECT_EQ(child.detail.message,
              "engine unavailable: cache-transfer continuation admission is frozen in DRAINING (sleep_epoch=" + epoch
                  + "), retry the inference request after wake or on another engine");

    const auto status = AdmissionGate::toGrpcStatus(child.detail);
    EXPECT_EQ(status.error_code(), grpc::StatusCode::UNAVAILABLE);
    EXPECT_EQ(status.error_message(), child.detail.message);
    ErrorDetailsPB details;
    ASSERT_TRUE(details.ParseFromString(status.error_details()));
    EXPECT_EQ(details.error_code(), kEngineUnavailable);
    EXPECT_EQ(details.error_code_str(), "ENGINE_UNAVAILABLE");
    EXPECT_EQ(details.error_message(), child.detail.message);
    EXPECT_EQ(details.state(), "DRAINING");
    EXPECT_EQ(details.instance_id(), "test_instance_0");
    EXPECT_EQ(details.sleep_epoch(), controller_.sleepEpoch());
}

TEST(AdmissionGateDisabledTest, MissingLedgerRejectsAndDisabledSleepStillTracksContinuations) {
    AdmissionGate null_gate(nullptr, "none");
    auto          null_child = null_gate.acquireCacheTransfer();
    EXPECT_FALSE(null_child.detail.admitted);
    EXPECT_FALSE(static_cast<bool>(null_child.complete));
    BoundSleepLifecycleController disabled(false);
    AdmissionGate            disabled_gate(&disabled, "disabled");
    auto                     child = disabled_gate.acquireCacheTransfer();
    EXPECT_TRUE(child.detail.admitted);
    EXPECT_TRUE(static_cast<bool>(child.complete));
    EXPECT_EQ(disabled.activeAdmissionCount(), 1);
    child.complete();
    EXPECT_EQ(disabled.activeAdmissionCount(), 0);
}

}  // namespace rtp_llm
