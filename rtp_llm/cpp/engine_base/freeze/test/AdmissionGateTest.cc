#include "gtest/gtest.h"

#include "rtp_llm/cpp/engine_base/freeze/AdmissionGate.h"
#include "rtp_llm/cpp/engine_base/freeze/FreezeLifecycleController.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

namespace {

constexpr int64_t kEngineUnavailable = 8600;

FreezeHooks successHooks() {
    FreezeHooks hooks;
    hooks.drain = [](const FreezeOptions&) { return true; };
    return hooks;
}

}  // namespace

class AdmissionGateTest: public ::testing::Test {
protected:
    FreezeLifecycleController controller_;
    AdmissionGate             gate_{&controller_, "test_instance_0"};
};

TEST_F(AdmissionGateTest, RunningAdmits) {
    ASSERT_EQ(controller_.state(), FreezeState::RUNNING);
    EXPECT_TRUE(gate_.check().ok());
    const auto detail = gate_.checkDetail();
    EXPECT_TRUE(detail.admitted);
    EXPECT_EQ(detail.error_code, 0);
    EXPECT_EQ(detail.state, "RUNNING");
    EXPECT_EQ(detail.instance_id, "test_instance_0");
}

TEST_F(AdmissionGateTest, NullControllerAdmits) {
    AdmissionGate null_gate(nullptr, "no_controller");
    EXPECT_TRUE(null_gate.check().ok());
    EXPECT_TRUE(null_gate.checkDetail().admitted);
}

TEST_F(AdmissionGateTest, DrainingRejects) {
    FreezeHooks hooks;
    // Drain "timeout": controller stays in DRAINING per design.
    hooks.drain = [](const FreezeOptions&) { return false; };
    controller_.setHooks(hooks);
    EXPECT_FALSE(controller_.freeze(FreezeOptions{}).ok);
    ASSERT_EQ(controller_.state(), FreezeState::DRAINING);

    const auto status = gate_.check();
    EXPECT_EQ(status.error_code(), grpc::StatusCode::UNAVAILABLE);
    const auto detail = gate_.checkDetail();
    EXPECT_FALSE(detail.admitted);
    EXPECT_EQ(detail.state, "DRAINING");
}

TEST_F(AdmissionGateTest, FreezingRejects) {
    FreezeHooks hooks = successHooks();
    // Observe gate behavior while the controller is mid-FREEZING.
    hooks.pauseKvMemory = [this](const FreezeOptions&) {
        EXPECT_EQ(controller_.state(), FreezeState::FREEZING);
        const auto status = gate_.check();
        EXPECT_EQ(status.error_code(), grpc::StatusCode::UNAVAILABLE);
        EXPECT_EQ(gate_.checkDetail().state, "FREEZING");
        return true;
    };
    controller_.setHooks(hooks);
    EXPECT_TRUE(controller_.freeze(FreezeOptions{}).ok);
}

TEST_F(AdmissionGateTest, FrozenRejects) {
    controller_.setHooks(successHooks());
    ASSERT_TRUE(controller_.freeze(FreezeOptions{}).ok);
    ASSERT_EQ(controller_.state(), FreezeState::FROZEN);

    const auto status = gate_.check();
    EXPECT_EQ(status.error_code(), grpc::StatusCode::UNAVAILABLE);
    const auto detail = gate_.checkDetail();
    EXPECT_FALSE(detail.admitted);
    EXPECT_EQ(detail.state, "FROZEN");
    EXPECT_EQ(detail.error_code, kEngineUnavailable);
}

TEST_F(AdmissionGateTest, ResumingRejects) {
    FreezeHooks hooks = successHooks();
    // Observe gate behavior while the controller is mid-RESUMING.
    hooks.resumeKvMemory = [this]() {
        EXPECT_EQ(controller_.state(), FreezeState::RESUMING);
        const auto status = gate_.check();
        EXPECT_EQ(status.error_code(), grpc::StatusCode::UNAVAILABLE);
        EXPECT_EQ(gate_.checkDetail().state, "RESUMING");
        return true;
    };
    controller_.setHooks(hooks);
    ASSERT_TRUE(controller_.freeze(FreezeOptions{}).ok);
    EXPECT_TRUE(controller_.resume().ok);
}

TEST_F(AdmissionGateTest, ErrorRejects) {
    FreezeHooks hooks   = successHooks();
    hooks.pauseKvMemory = [](const FreezeOptions&) { return false; };
    controller_.setHooks(hooks);
    EXPECT_FALSE(controller_.freeze(FreezeOptions{}).ok);
    ASSERT_EQ(controller_.state(), FreezeState::ERROR);

    const auto status = gate_.check();
    EXPECT_EQ(status.error_code(), grpc::StatusCode::UNAVAILABLE);
    const auto detail = gate_.checkDetail();
    EXPECT_FALSE(detail.admitted);
    EXPECT_EQ(detail.state, "ERROR");
}

TEST_F(AdmissionGateTest, ErrorBodyFieldsComplete) {
    controller_.setHooks(successHooks());
    ASSERT_TRUE(controller_.freeze(FreezeOptions{}).ok);
    ASSERT_EQ(controller_.state(), FreezeState::FROZEN);

    // Structured detail carries the full M4 error body.
    const auto detail = gate_.checkDetail();
    EXPECT_EQ(detail.error_code, kEngineUnavailable);
    EXPECT_EQ(detail.error_code, static_cast<int64_t>(ErrorCode::ENGINE_UNAVAILABLE));
    EXPECT_EQ(detail.error_code_str, "ENGINE_UNAVAILABLE");
    EXPECT_EQ(detail.instance_id, "test_instance_0");
    EXPECT_EQ(detail.freeze_epoch, controller_.freezeEpoch());
    EXPECT_GE(detail.freeze_epoch, 1);
    EXPECT_EQ(detail.state, "FROZEN");
    EXPECT_FALSE(detail.message.empty());

    // grpc::Status error_details round-trips through ErrorDetailsPB.
    const auto status = gate_.check();
    EXPECT_EQ(status.error_code(), grpc::StatusCode::UNAVAILABLE);
    EXPECT_EQ(status.error_message(), detail.message);
    ErrorDetailsPB details;
    ASSERT_TRUE(details.ParseFromString(status.error_details()));
    EXPECT_EQ(details.error_code(), kEngineUnavailable);
    EXPECT_EQ(details.error_code_str(), "ENGINE_UNAVAILABLE");
    EXPECT_EQ(details.error_message(), detail.message);
    EXPECT_EQ(details.instance_id(), "test_instance_0");
    EXPECT_EQ(details.freeze_epoch(), detail.freeze_epoch);
    EXPECT_EQ(details.state(), "FROZEN");

    // JSON body for the HTTP layer contains the same fields.
    const auto json = AdmissionGate::toJson(detail);
    EXPECT_NE(json.find("\"error_code\":8600"), std::string::npos);
    EXPECT_NE(json.find("\"error_code_str\":\"ENGINE_UNAVAILABLE\""), std::string::npos);
    EXPECT_NE(json.find("\"instance_id\":\"test_instance_0\""), std::string::npos);
    EXPECT_NE(json.find("\"freeze_epoch\":" + std::to_string(detail.freeze_epoch)), std::string::npos);
    EXPECT_NE(json.find("\"state\":\"FROZEN\""), std::string::npos);
}

TEST_F(AdmissionGateTest, AdmitsAgainAfterResume) {
    controller_.setHooks(successHooks());
    ASSERT_TRUE(controller_.freeze(FreezeOptions{}).ok);
    EXPECT_EQ(gate_.check().error_code(), grpc::StatusCode::UNAVAILABLE);

    ASSERT_TRUE(controller_.resume().ok);
    ASSERT_EQ(controller_.state(), FreezeState::RUNNING);
    EXPECT_TRUE(gate_.check().ok());
    const auto detail = gate_.checkDetail();
    EXPECT_TRUE(detail.admitted);
    EXPECT_EQ(detail.state, "RUNNING");
    // Epoch from the completed freeze cycle is still visible.
    EXPECT_GE(detail.freeze_epoch, 1);
}

}  // namespace rtp_llm
