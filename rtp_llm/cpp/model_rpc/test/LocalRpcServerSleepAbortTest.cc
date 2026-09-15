#include <chrono>
#include <future>
#include <memory>
#include <thread>

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "rtp_llm/cpp/engine_base/sleep/AdmissionGate.h"
#include "rtp_llm/cpp/engine_base/sleep/SleepLifecycleController.h"
#include "rtp_llm/cpp/model_rpc/LocalRpcServer.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {
namespace {

ModelConfig testModelConfig() {
    ModelConfig config;
    config.max_seq_len                  = 8;
    config.vocab_size                   = 128;
    config.attn_config.tokens_per_block = 8;
    return config;
}

std::shared_ptr<GenerateInput> makeInput(int64_t request_id, bool streaming) {
    auto input                           = std::make_shared<GenerateInput>();
    input->request_id                    = request_id;
    input->generate_config               = std::make_shared<GenerateConfig>();
    input->generate_config->is_streaming = streaming;
    input->input_ids                     = torch::tensor(std::vector<int32_t>{1}, torch::kInt32);
    input->begin_time_us                 = currentTimeUs();
    return input;
}

std::shared_ptr<NormalGenerateStream> makeStream(int64_t request_id, bool streaming) {
    return std::make_shared<NormalGenerateStream>(
        makeInput(request_id, streaming), testModelConfig(), RuntimeConfig{}, ResourceContext{}, nullptr);
}

// A controller stuck in DRAINING rejects admission: sleep() with a failing drain
// hook stays in DRAINING per SleepLifecycleController design.
std::shared_ptr<SleepLifecycleController> drainingController() {
    auto       controller = std::make_shared<SleepLifecycleController>(true);
    SleepHooks hooks;
    hooks.drain = [](const SleepOptions&) { return false; };
    controller->setHooks(hooks);
    controller->sleep(SleepOptions{});
    return controller;
}

}  // namespace

TEST(LocalRpcServerSleepAbortTest, DirectSleepRpcRejectsNonEmptyTags) {
    auto           controller = std::make_shared<SleepLifecycleController>(true);
    LocalRpcServer server;
    server.admission_gate_ = std::make_shared<AdmissionGate>(controller.get(), "test_instance");

    grpc::ServerContext context;
    SleepRequestPB      request;
    EmptyPB             response;
    request.set_level(1);
    request.set_mode("wait");
    request.add_tags("weights");

    const auto status = server.SleepServing(&context, &request, &response);

    EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(controller->state(), SleepState::RUNNING);
    EXPECT_EQ(controller->sleepEpoch(), 0);
}

TEST(LocalRpcServerSleepAbortTest, HealthReportsUnavailableWhileSleepingAndOkAfterWake) {
    SleepLifecycleController controller(true);
    SleepHooks               hooks;
    hooks.drain = [](const SleepOptions&) { return true; };
    controller.setHooks(hooks);
    LocalRpcServer server;
    server.admission_gate_ = std::make_shared<AdmissionGate>(&controller, "test_instance");
    grpc::ServerContext context;
    EmptyPB             request;

    CheckHealthResponsePB running_response;
    EXPECT_TRUE(server.CheckHealth(&context, &request, &running_response).ok());
    EXPECT_EQ(running_response.health(), "OK");

    ASSERT_TRUE(controller.sleep(SleepOptions{}).ok);
    CheckHealthResponsePB sleeping_response;
    const auto            status = server.CheckHealth(&context, &request, &sleeping_response);
    EXPECT_EQ(status.error_code(), grpc::StatusCode::UNAVAILABLE);
    EXPECT_TRUE(sleeping_response.health().empty());
    ErrorDetailsPB details;
    ASSERT_TRUE(details.ParseFromString(status.error_details()));
    EXPECT_EQ(details.state(), "SLEEPING");
    EXPECT_EQ(details.sleep_epoch(), controller.sleepEpoch());

    ASSERT_TRUE(controller.wakeUp().ok);
    CheckHealthResponsePB awake_response;
    EXPECT_TRUE(server.CheckHealth(&context, &request, &awake_response).ok());
    EXPECT_EQ(awake_response.health(), "OK");
}

TEST(LocalRpcServerSleepAbortTest, AbortRegistryCancelsOnlyNonStreamingStreams) {
    SleepLifecycleController controller(true);
    LocalRpcServer server;
    server.admission_gate_ = std::make_shared<AdmissionGate>(&controller, "test_instance");

    auto streaming     = makeStream(1, true);
    auto non_streaming = makeStream(2, false);

    auto streaming_guard     = server.registerAbortableStreamForScope(streaming);
    auto non_streaming_guard = server.registerAbortableStreamForScope(non_streaming);

    EXPECT_EQ(streaming_guard, nullptr);
    ASSERT_NE(non_streaming_guard, nullptr);

    EXPECT_EQ(server.cancelAbortableStreams(), 1u);
    EXPECT_FALSE(streaming->hasError());
    ASSERT_TRUE(non_streaming->hasError());
    EXPECT_EQ(non_streaming->statusInfo().code(), ErrorCode::CANCELLED);

    non_streaming_guard.reset();
    EXPECT_EQ(server.cancelAbortableStreams(), 0u);
}

TEST(LocalRpcServerSleepAbortTest, DisabledSleepDoesNotRegisterAbortableStreams) {
    LocalRpcServer server;
    auto stream = makeStream(4, false);
    // No sleep admission gate is installed when the startup switch is OFF.
    EXPECT_EQ(server.registerAbortableStreamForScope(stream), nullptr);
    EXPECT_TRUE(server.abortable_streams_.empty());
    EXPECT_FALSE(stream->hasError());
}

TEST(LocalRpcServerSleepAbortTest, LegacyControlsCannotBypassSleepAdmission) {
    SleepLifecycleController controller(true);
    SleepHooks               hooks;
    hooks.drain = [](const SleepOptions&) { return true; };
    controller.setHooks(hooks);
    LocalRpcServer server;
    server.admission_gate_ = std::make_shared<AdmissionGate>(&controller, "test_instance");

    ASSERT_TRUE(controller.sleep(SleepOptions{}).ok);
    for (auto rpc : {&LocalRpcServer::SetPause, &LocalRpcServer::SetRestart}) {
        grpc::ServerContext context;
        EmptyPB             request;
        EmptyPB             response;
        auto                status = (server.*rpc)(&context, &request, &response);
        EXPECT_EQ(status.error_code(), grpc::StatusCode::UNAVAILABLE);
        ErrorDetailsPB details;
        ASSERT_TRUE(details.ParseFromString(status.error_details()));
        EXPECT_EQ(details.state(), "SLEEPING");
        EXPECT_EQ(controller.state(), SleepState::SLEEPING);
    }

    ASSERT_TRUE(controller.wakeUp().ok);
    for (auto rpc : {&LocalRpcServer::SetPause, &LocalRpcServer::SetRestart}) {
        grpc::ServerContext context;
        EmptyPB             request;
        EmptyPB             response;
        // Admission is open again. A missing engine fails explicitly rather
        // than dereferencing it (the sleeping path must never reach it).
        auto status = (server.*rpc)(&context, &request, &response);
        EXPECT_EQ(status.error_code(), grpc::StatusCode::FAILED_PRECONDITION);
        EXPECT_EQ(status.error_message(), "engine is not initialized");
    }
}

TEST(LocalRpcServerSleepAbortTest, LegacyControlsCannotBypassDrainAdmission) {
    auto           controller = drainingController();
    LocalRpcServer server;
    server.admission_gate_ = std::make_shared<AdmissionGate>(controller.get(), "test_instance");
    for (auto rpc : {&LocalRpcServer::SetPause, &LocalRpcServer::SetRestart}) {
        grpc::ServerContext context;
        EmptyPB             request;
        EmptyPB             response;
        auto                status = (server.*rpc)(&context, &request, &response);
        EXPECT_EQ(status.error_code(), grpc::StatusCode::UNAVAILABLE);
        EXPECT_EQ(controller->state(), SleepState::DRAINING);
    }
}

TEST(LocalRpcServerSleepAbortTest, NormalGenerateStreamReportErrorWakesOutputWaiter) {
    auto stream = makeStream(3, false);

    auto output = std::async(std::launch::async, [stream]() { return stream->nextOutput(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    stream->reportError(ErrorCode::CANCELLED, "request cancelled by sleep abort");

    ASSERT_EQ(output.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    const auto result = output.get();
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), ErrorCode::CANCELLED);
}

// --- P1: GPU/KV-touching RPCs must be gated by admission so they cannot start
// once sleep has closed the gate (else they race weight pause / KV release). ---

TEST(LocalRpcServerAdmissionTest, ExecuteFunctionRejectedWhenNotRunning) {
    auto controller = drainingController();
    ASSERT_EQ(controller->state(), SleepState::DRAINING);

    LocalRpcServer server;
    server.admission_gate_ = std::make_shared<AdmissionGate>(controller.get(), "test_instance");

    grpc::ServerContext  context;
    ::FunctionRequestPB  request;
    ::FunctionResponsePB response;
    const auto           status = server.ExecuteFunction(&context, &request, &response);

    EXPECT_EQ(status.error_code(), grpc::StatusCode::UNAVAILABLE);
    EXPECT_EQ(controller->activeAdmissionCount(), 0);  // rejected -> no lease held
}

TEST(LocalRpcServerAdmissionTest, ExecuteFunctionAdmittedWhenRunning) {
    SleepLifecycleController controller(true);
    ASSERT_EQ(controller.state(), SleepState::RUNNING);

    LocalRpcServer server;
    server.admission_gate_ = std::make_shared<AdmissionGate>(&controller, "test_instance");

    grpc::ServerContext  context;
    ::FunctionRequestPB  request;
    ::FunctionResponsePB response;
    // Admission passes; with no engine wired the RPC then fails downstream (not
    // UNAVAILABLE). The point is that it was admitted and the lease is released
    // once the handler returns.
    const auto status = server.ExecuteFunction(&context, &request, &response);

    EXPECT_NE(status.error_code(), grpc::StatusCode::UNAVAILABLE);
    EXPECT_EQ(controller.activeAdmissionCount(), 0);
}

TEST(LocalRpcServerAdmissionTest, UpdateWeightsRejectedWhenNotRunning) {
    auto controller = drainingController();
    ASSERT_EQ(controller->state(), SleepState::DRAINING);

    LocalRpcServer server;
    server.admission_gate_ = std::make_shared<AdmissionGate>(controller.get(), "test_instance");

    grpc::ServerContext      context;
    ::UpdateWeightsRequestPB request;
    request.set_name("w");
    request.set_desc("d");
    request.set_method("m");
    ::EmptyPB  response;
    const auto status = server.UpdateWeights(&context, &request, &response);

    EXPECT_EQ(status.error_code(), grpc::StatusCode::UNAVAILABLE);
    EXPECT_EQ(controller->activeAdmissionCount(), 0);
}

}  // namespace rtp_llm
