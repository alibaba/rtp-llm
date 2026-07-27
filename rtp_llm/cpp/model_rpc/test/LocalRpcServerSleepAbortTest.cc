#include <chrono>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

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
    hooks.drain = [](const SleepOptions&, const DrainCancellationPredicate&) { return false; };
    controller->setHooks(hooks);
    controller->sleep(SleepOptions{});
    return controller;
}

SleepOptions levelThreeSleepOptions() {
    SleepOptions options;
    options.level = 3;
    return options;
}

}  // namespace

// LocalRpcServer currently exposes CUDA graph invalidation, Mega symmetric-memory
// teardown, DeepEP teardown, and ProcessGroup teardown as one controller hook.
// These tests cover ordering and fail-closed behavior at that hook boundary. The
// ordering inside LocalRpcServer's Python calls still requires an integration test
// because there is no per-resource injection point.

TEST(LocalRpcServerLevelThreeSleepTest, ControllerBoundaryOrdersTeardownBeforeCheckpointEligibleRelease) {
    SleepLifecycleController controller(true);
    controller.setConfiguredLevel(3);

    std::vector<std::string> calls;
    SleepHooks               hooks;
    hooks.synchronizeAndDeregisterMr = [&calls](const SleepOptions&) {
        calls.push_back("deregister_mr");
        return true;
    };
    hooks.teardownCollectives = [&calls](const SleepOptions&) {
        // This is the atomic LocalRpcServer hook contract. Its implementation
        // performs these four operations in this order.
        calls.push_back("invalidate_cuda_graphs");
        calls.push_back("release_mega_symm");
        calls.push_back("destroy_deepep");
        calls.push_back("teardown_distributed");
        return true;
    };
    hooks.releaseKvMemoryBacking = [&calls](const SleepOptions&) {
        calls.push_back("release_kv");
        return true;
    };
    hooks.releaseRestorableGpuMemory = [&calls](const SleepOptions&) {
        calls.push_back("release_restorable_gpu_memory");
        return true;
    };
    controller.setHooks(hooks);

    const auto result = controller.sleep(levelThreeSleepOptions());

    ASSERT_TRUE(result.ok) << result.message;
    EXPECT_EQ(controller.state(), SleepState::SLEEPING);
    EXPECT_EQ(calls,
              (std::vector<std::string>{"deregister_mr",
                                        "invalidate_cuda_graphs",
                                        "release_mega_symm",
                                        "destroy_deepep",
                                        "teardown_distributed",
                                        "release_kv",
                                        "release_restorable_gpu_memory"}));
}

TEST(LocalRpcServerLevelThreeSleepTest, TeardownExceptionsFailClosedBeforeCheckpointEligibleRelease) {
    for (const std::string& failed_resource : {"symmetric memory", "communicator"}) {
        SCOPED_TRACE(failed_resource);
        SleepLifecycleController controller(true);
        controller.setConfiguredLevel(3);

        bool       release_kv_called         = false;
        bool       release_restorable_called = false;
        SleepHooks hooks;
        hooks.teardownCollectives = [&failed_resource](const SleepOptions&) -> bool {
            throw std::runtime_error(failed_resource + " teardown failed");
        };
        hooks.releaseKvMemoryBacking = [&release_kv_called](const SleepOptions&) {
            release_kv_called = true;
            return true;
        };
        hooks.releaseRestorableGpuMemory = [&release_restorable_called](const SleepOptions&) {
            release_restorable_called = true;
            return true;
        };
        controller.setHooks(hooks);

        const auto result = controller.sleep(levelThreeSleepOptions());

        EXPECT_FALSE(result.ok);
        EXPECT_EQ(result.code, SleepResult::Code::FAILED_PRECONDITION);
        EXPECT_EQ(controller.state(), SleepState::ERROR);
        EXPECT_FALSE(controller.admit());
        EXPECT_FALSE(release_kv_called);
        EXPECT_FALSE(release_restorable_called);
        EXPECT_NE(controller.status().last_error.find("teardownCollectives"), std::string::npos);

        // ERROR is observable and stable rather than a transient half-state.
        // Recovery is an explicit process restart; wake must not reopen this instance.
        const auto wake_result = controller.wakeUp();
        EXPECT_FALSE(wake_result.ok);
        EXPECT_EQ(controller.state(), SleepState::ERROR);
        EXPECT_FALSE(controller.admit());
    }
}

TEST(LocalRpcServerLevelThreeWakeTest, RestoreEagerlyRebuildsMegaBeforeGraphsAndRunning) {
    SleepLifecycleController controller(true);
    controller.setConfiguredLevel(3);

    std::vector<std::string> calls;
    SleepHooks               hooks;
    hooks.restoreRestorableGpuMemory = [&calls]() {
        calls.push_back("restore_restorable_gpu_memory");
        return true;
    };
    hooks.restoreKvMemoryBackingAndResetMetadata = [&calls]() {
        calls.push_back("restore_kv");
        return true;
    };
    hooks.rebuildCollectives = [&calls]() {
        // LocalRpcServer performs these operations in-order inside this hook.
        calls.push_back("rebuild_distributed");
        calls.push_back("rebuild_deepep");
        calls.push_back("rebuild_mega_symm");
        return true;
    };
    hooks.registerMr = [&calls]() {
        calls.push_back("register_mr");
        return true;
    };
    hooks.recaptureCollectiveGraphs = [&calls]() {
        calls.push_back("recapture_cuda_graphs");
        return true;
    };
    hooks.restartEngine = [&calls]() {
        calls.push_back("restart_engine");
        return true;
    };
    controller.setHooks(hooks);

    ASSERT_TRUE(controller.sleep(levelThreeSleepOptions()).ok);
    const auto result = controller.wakeUp();

    ASSERT_TRUE(result.ok) << result.message;
    EXPECT_EQ(controller.state(), SleepState::RUNNING);
    EXPECT_TRUE(controller.admit());
    EXPECT_EQ(calls,
              (std::vector<std::string>{"restore_restorable_gpu_memory",
                                        "restore_kv",
                                        "rebuild_distributed",
                                        "rebuild_deepep",
                                        "rebuild_mega_symm",
                                        "register_mr",
                                        "recapture_cuda_graphs",
                                        "restart_engine"}));
}

TEST(LocalRpcServerLevelThreeWakeTest, RebuildFailuresNeverMarkControllerRunning) {
    for (const std::string& failed_stage : {"rebuild_collectives", "recapture_graphs"}) {
        SCOPED_TRACE(failed_stage);
        SleepLifecycleController controller(true);
        controller.setConfiguredLevel(3);

        bool       graph_called   = false;
        bool       restart_called = false;
        SleepHooks hooks;
        hooks.rebuildCollectives = [&failed_stage]() -> bool {
            if (failed_stage == "rebuild_collectives") {
                throw std::runtime_error("distributed or DeepEP rebuild failed");
            }
            return true;
        };
        hooks.recaptureCollectiveGraphs = [&failed_stage, &graph_called]() -> bool {
            graph_called = true;
            if (failed_stage == "recapture_graphs") {
                throw std::runtime_error("CUDA graph recapture failed");
            }
            return true;
        };
        hooks.restartEngine = [&restart_called]() {
            restart_called = true;
            return true;
        };
        controller.setHooks(hooks);

        ASSERT_TRUE(controller.sleep(levelThreeSleepOptions()).ok);
        const auto result = controller.wakeUp();

        EXPECT_FALSE(result.ok);
        EXPECT_EQ(result.code, SleepResult::Code::FAILED_PRECONDITION);
        EXPECT_EQ(controller.state(), SleepState::ERROR);
        EXPECT_FALSE(controller.admit());
        EXPECT_EQ(graph_called, failed_stage == "recapture_graphs");
        EXPECT_FALSE(restart_called);
        const std::string expected_error =
            failed_stage == "rebuild_collectives" ? "rebuildCollectives" : "recaptureCollectiveGraphs";
        EXPECT_NE(controller.status().last_error.find(expected_error), std::string::npos);
    }
}

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

// The multicast keeper holder identity pinned by the Python collective layer must
// round-trip through the process-global setter into SleepStatusResponsePB so the
// durable checkpoint manifest can persist and later verify it. Unset -> empty
// (non-keeper deployments), set -> "hi:lo", cleared -> empty again.
TEST(LocalRpcServerHolderInstanceTest, MulticastHolderInstanceRoundTripsIntoSleepStatus) {
    LocalRpcServer::clearMulticastHolderInstance();
    EXPECT_EQ(LocalRpcServer::multicastHolderInstanceString(), "");

    {
        SleepStatusResponsePB response;
        response.set_holder_instance(LocalRpcServer::multicastHolderInstanceString());
        EXPECT_EQ(response.holder_instance(), "");
    }

    LocalRpcServer::setMulticastHolderInstance(0x1122334455667788ULL, 0x99aabbccddeeff00ULL);
    const std::string expected = std::to_string(0x1122334455667788ULL) + ":" + std::to_string(0x99aabbccddeeff00ULL);
    EXPECT_EQ(LocalRpcServer::multicastHolderInstanceString(), expected);

    {
        SleepStatusResponsePB response;
        response.set_holder_instance(LocalRpcServer::multicastHolderInstanceString());
        EXPECT_EQ(response.holder_instance(), expected);
    }

    LocalRpcServer::clearMulticastHolderInstance();
    EXPECT_EQ(LocalRpcServer::multicastHolderInstanceString(), "");
    {
        SleepStatusResponsePB response;
        response.set_holder_instance(LocalRpcServer::multicastHolderInstanceString());
        EXPECT_EQ(response.holder_instance(), "");
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
