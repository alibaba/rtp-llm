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
    SleepLifecycleController controller_{true};
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

TEST_F(AdmissionGateTest, LeaseMoveTransfersOwnershipAndReleasesOnce) {
    auto acquired = gate_.acquire();
    ASSERT_TRUE(acquired.detail.admitted);
    ASSERT_TRUE(static_cast<bool>(acquired.lease));
    EXPECT_EQ(controller_.activeAdmissionCount(), 1);

    AdmissionLease moved(std::move(acquired.lease));
    EXPECT_FALSE(static_cast<bool>(acquired.lease));
    EXPECT_TRUE(static_cast<bool>(moved));
    EXPECT_EQ(controller_.activeAdmissionCount(), 1);

    moved = AdmissionLease{};
    EXPECT_EQ(controller_.activeAdmissionCount(), 0);
}

TEST_F(AdmissionGateTest, SuccessfulAcquireDoesNotBuildStatusStrings) {
    auto acquired = gate_.acquire();
    ASSERT_TRUE(acquired.detail.admitted);
    EXPECT_TRUE(acquired.detail.instance_id.empty());
    EXPECT_TRUE(acquired.detail.state.empty());
    EXPECT_TRUE(acquired.detail.message.empty());
    EXPECT_TRUE(acquired.detail.error_code_str.empty());
    EXPECT_TRUE(static_cast<bool>(acquired.lease));
}

TEST(AdmissionGateDisabledTest, DisabledAdmissionHasNoLeaseOrCounter) {
    SleepLifecycleController controller(false);
    AdmissionGate gate(&controller, "long-instance-id-that-would-require-a-string-allocation");
    auto acquired = gate.acquire();
    EXPECT_TRUE(acquired.detail.admitted);
    EXPECT_FALSE(static_cast<bool>(acquired.lease));
    EXPECT_EQ(controller.activeAdmissionCount(), 0);
    EXPECT_TRUE(acquired.detail.instance_id.empty());
    EXPECT_TRUE(controller.acquireAdmission().admitted());
}

TEST_F(AdmissionGateTest, NullControllerAdmits) {
    AdmissionGate null_gate(nullptr, "no_controller");
    EXPECT_TRUE(null_gate.check().ok());
    EXPECT_TRUE(null_gate.checkDetail().admitted);
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
    EXPECT_FALSE(static_cast<bool>(acquired.lease));
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

    // Structured detail carries the full M4 error body.
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

    // JSON body for the HTTP layer contains the same fields.
    const auto json = AdmissionGate::toJson(detail);
    EXPECT_NE(json.find("\"error_code\":8600"), std::string::npos);
    EXPECT_NE(json.find("\"error_code_str\":\"ENGINE_UNAVAILABLE\""), std::string::npos);
    EXPECT_NE(json.find("\"instance_id\":\"test_instance_0\""), std::string::npos);
    EXPECT_NE(json.find("\"sleep_epoch\":" + std::to_string(detail.sleep_epoch)), std::string::npos);
    EXPECT_NE(json.find("\"state\":\"SLEEPING\""), std::string::npos);
}

TEST_F(AdmissionGateTest, JsonEscapesEveryControlByteAndPreservesUtf8) {
    auto detail = gate_.checkDetail();
    detail.message.clear();
    for (int byte = 0; byte < 0x20; ++byte) {
        detail.message.push_back(static_cast<char>(byte));
    }
    const std::string utf8 = u8"休眠";
    detail.message += "\"\\" + utf8;
    const std::string escaped = "\\u0000\\u0001\\u0002\\u0003\\u0004\\u0005\\u0006\\u0007"
                                "\\u0008\\t\\n\\u000b\\u000c\\r\\u000e\\u000f"
                                "\\u0010\\u0011\\u0012\\u0013\\u0014\\u0015\\u0016\\u0017"
                                "\\u0018\\u0019\\u001a\\u001b\\u001c\\u001d\\u001e\\u001f"
                                "\\\"\\\\"
                                + utf8;

    const auto json = AdmissionGate::toJson(detail);
    EXPECT_NE(json.find("\"message\":\"" + escaped + "\""), std::string::npos);
    for (const unsigned char byte : json) {
        EXPECT_GE(byte, 0x20);
    }
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

}  // namespace rtp_llm
