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

TEST(LocalRpcServerSleepAbortTest, AbortRegistryCancelsOnlyNonStreamingStreams) {
    LocalRpcServer server;

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

TEST(LocalRpcServerWakeHealthTest, RejectsKvControllerWithoutAttachedBacking) {
    auto controller = std::make_shared<KVCachePhysicalMemoryController>(nullptr);
    EXPECT_FALSE(LocalRpcServer::validateKvMemoryControllerForWake(controller));

    ASSERT_NE(controller->allocateOrAttach(reinterpret_cast<void*>(0x1000), 4096), nullptr);
    EXPECT_TRUE(LocalRpcServer::validateKvMemoryControllerForWake(controller));
}

TEST(LocalRpcServerWakeHealthTest, AllowsModelsWithoutKvController) {
    EXPECT_TRUE(LocalRpcServer::validateKvMemoryControllerForWake(nullptr));
}

}  // namespace rtp_llm
