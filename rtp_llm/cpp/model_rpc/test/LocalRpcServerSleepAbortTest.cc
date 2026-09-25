#include "rtp_llm/cpp/engine_base/sleep/test/BoundSleepLifecycleController.h"
#include <chrono>
#include <future>
#include <memory>
#include <optional>
#include <thread>

#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include <torch/torch.h>

#include "rtp_llm/cpp/cache/connector/KVCacheConnectorCoordinator.h"
#include "rtp_llm/cpp/engine_base/sleep/AdmissionGate.h"
#include "rtp_llm/cpp/engine_base/sleep/SleepLifecycleController.h"
#include "rtp_llm/cpp/model_rpc/LocalRpcServer.h"
#include "rtp_llm/cpp/model_rpc/DecodeRpcServer.h"
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
// hook stays in DRAINING per BoundSleepLifecycleController design.
std::shared_ptr<BoundSleepLifecycleController> drainingController() {
    auto       controller = std::make_shared<BoundSleepLifecycleController>(true);
    SleepHooks hooks;
    hooks.drain = [](const SleepOptions&) { return false; };
    controller->setHooks(hooks);
    controller->sleep(SleepOptions{});
    return controller;
}

}  // namespace

namespace {
class HookTestScheduler: public SchedulerBase {
public:
    absl::Status enqueue(const GenerateStreamPtr&) override {
        return absl::OkStatus();
    }
    std::pair<std::vector<bool>, std::vector<GenerateStreamPtr>>
    enqueueGroup(const std::vector<GenerateStreamPtr>&) override {
        return {{}, {}};
    }
    absl::StatusOr<std::list<GenerateStreamPtr>> schedule() override {
        return std::list<GenerateStreamPtr>{};
    }
    absl::Status stop() override {
        return absl::OkStatus();
    }
    bool empty() override {
        return true;
    }
    int64_t lastScheduleTime() override {
        return 0;
    }
    int64_t onflightStreams() override {
        return 0;
    }
};

class HookTestEngine: public EngineBase {
public:
    explicit HookTestEngine(int& destructions): EngineBase(EngineInitParams{}), destructions_(destructions) {
        scheduler_ = std::make_unique<HookTestScheduler>();
        sleepController().bindAdmission(scheduler_->admission());
    }
    ~HookTestEngine() override {
        ++destructions_;
    }
    GenerateStreamPtr enqueue(const std::shared_ptr<GenerateInput>&) override {
        return nullptr;
    }
    void         enqueue(GenerateStreamPtr&) override {}
    absl::Status stop() override {
        return absl::OkStatus();
    }
    absl::StatusOr<GenerateStreamPtr> preRun(const std::shared_ptr<GenerateInput>&, preRunMode) override {
        return absl::UnimplementedError("unused");
    }
    KVCacheInfo getCacheStatusInfo(int64_t, bool) override {
        return {};
    }

private:
    int& destructions_;
};

struct HookOwnerTestState {
    std::atomic<int>   calls{0};
    std::atomic<int>   destructions{0};
    std::atomic<bool>  destroyed_with_gil{false};
    bool               block{false};
    std::promise<void> entered;
    std::promise<void> release;
};

class HookOwnerTestServer: public LocalRpcServer {
public:
    explicit HookOwnerTestServer(std::shared_ptr<HookOwnerTestState> state): state_(std::move(state)) {}
    ~HookOwnerTestServer() override {
        state_->destroyed_with_gil = !Py_IsInitialized() || PyGILState_Check();
        ++state_->destructions;
    }
    size_t activeCacheTransferCount() override {
        // Keep the synchronization storage independent of the server so the RED
        // run can observe premature destruction without dereferencing freed data.
        auto state = state_;
        ++state->calls;
        if (state->block) {
            state->entered.set_value();
            state->release.get_future().wait();
        }
        return 0;
    }

private:
    std::shared_ptr<HookOwnerTestState> state_;
};
}  // namespace

TEST(LocalRpcServerSleepAbortTest, CopiedServiceCallbacksRejectExpiredOwner) {
    for (bool sleep_enabled : {false, true}) {
        int  destructions = 0;
        auto engine       = std::make_shared<HookTestEngine>(destructions);
        engine->sleepController().setEnabled(sleep_enabled);
        auto state = std::make_shared<HookOwnerTestState>();
        // Retain storage after shared ownership expires: the old implementation
        // can fail its assertions without deliberately executing a heap UAF.
        std::unique_ptr<HookOwnerTestServer> retired;
        auto server     = std::shared_ptr<HookOwnerTestServer>(new HookOwnerTestServer(state),
                                                           [&](auto* value) { retired.reset(value); });
        server->engine_ = engine;
        server->installSleepHooks();
        auto&                         drain      = engine->getScheduler().drainManager();
        auto                          counter    = drain.counters_.at("rpc_cache_transfer").fn;
        auto                          cancel     = drain.cancel_callback_;
        auto                          diagnostic = engine->sleepController().hooks_.hookFailureDetail;
        std::weak_ptr<LocalRpcServer> weak       = server;
        server.reset();
        ASSERT_TRUE(weak.expired());
        EXPECT_THROW(counter(), std::runtime_error);
        EXPECT_THROW(cancel(), std::runtime_error);
        EXPECT_THROW(diagnostic("unrelated"), std::runtime_error);
        EXPECT_EQ(state->calls.load(), 0);
        EXPECT_FALSE(drain.drained());
        EXPECT_EQ(drain.activeCacheTransferCount(), 1);
        retired.reset();
    }
}

TEST(LocalRpcServerSleepAbortTest, InFlightCounterKeepsDerivedOwnerAliveAndReleasesWithGil) {
    // Own the interpreter only in the standalone native test runtime. Declare it
    // first so every Python-owning fixture and joined callback dies before it.
    std::optional<py::scoped_interpreter> interpreter;
    if (!Py_IsInitialized()) {
        interpreter.emplace();
    }
    ASSERT_TRUE(Py_IsInitialized());
    py::gil_scoped_acquire hold_gil;
    int                    destructions = 0;
    auto                   engine       = std::make_shared<HookTestEngine>(destructions);
    auto                   state        = std::make_shared<HookOwnerTestState>();
    state->block                        = true;
    auto server                         = std::make_shared<HookOwnerTestServer>(state);
    server->engine_                     = engine;
    server->installSleepHooks();
    auto                          counter = engine->getScheduler().drainManager().counters_.at("rpc_cache_transfer").fn;
    std::weak_ptr<LocalRpcServer> weak    = server;
    auto                          running = std::async(std::launch::async, [counter] { return counter(); });
    state->entered.get_future().wait();
    server.reset();
    EXPECT_FALSE(weak.expired());
    EXPECT_EQ(state->destructions.load(), 0);
    state->release.set_value();
    {
        std::optional<py::gil_scoped_release> release_gil;
        if (Py_IsInitialized() && PyGILState_Check()) {
            release_gil.emplace();
        }
        EXPECT_EQ(running.get(), 0);
    }
    EXPECT_TRUE(weak.expired());
    EXPECT_EQ(state->destructions.load(), 1);
    EXPECT_TRUE(state->destroyed_with_gil.load());
}

TEST(LocalRpcServerSleepAbortTest, AbortRegistrationTokenDoesNotTouchExpiredOwner) {
    auto                          state  = std::make_shared<HookOwnerTestState>();
    auto                          server = std::make_shared<HookOwnerTestServer>(state);
    BoundSleepLifecycleController controller(true);
    server->admission_gate_ = std::make_shared<AdmissionGate>(&controller, "test_instance");
    auto stream             = makeStream(987, false);
    auto token              = server->registerAbortableStreamForScope(stream);
    ASSERT_NE(token, nullptr);
    ASSERT_EQ(server->abortable_streams_->streams.size(), 1);
    std::weak_ptr<LocalRpcServer>                          weak_owner    = server;
    std::weak_ptr<LocalRpcServer::AbortableStreamRegistry> weak_registry = server->abortable_streams_;
    server.reset();
    ASSERT_TRUE(weak_owner.expired());
    ASSERT_TRUE(weak_registry.expired());
    token.reset();
    EXPECT_EQ(state->destructions.load(), 1);
}

TEST(LocalRpcServerSleepAbortTest, AbortRegistrationTokenCleanupDoesNotRequireGil) {
    std::optional<py::scoped_interpreter> interpreter;
    if (!Py_IsInitialized()) {
        interpreter.emplace();
    }
    ASSERT_TRUE(Py_IsInitialized());
    py::gil_scoped_acquire        hold_gil;
    BoundSleepLifecycleController controller(true);
    auto                          server = std::make_shared<LocalRpcServer>();
    server->admission_gate_              = std::make_shared<AdmissionGate>(&controller, "test_instance");
    auto stream                          = makeStream(988, false);
    auto token                           = server->registerAbortableStreamForScope(stream);
    ASSERT_NE(token, nullptr);
    auto cleanup = std::async(std::launch::async, [token = std::move(token)]() mutable { token.reset(); });
    // Keep the GIL on this thread: unregistering a pure C++ registry must not
    // wait on Python. Release it only after recording the bounded RED result.
    EXPECT_EQ(cleanup.wait_for(std::chrono::milliseconds(200)), std::future_status::ready);
    {
        py::gil_scoped_release release_gil;
        cleanup.get();
    }
    EXPECT_EQ(server->cancelAbortableStreams(), 0u);
}

TEST(LocalRpcServerSleepAbortTest, ProductionHooksDoNotKeepTheirEngineOwnerAlive) {
    for (const bool sleep_enabled : {false, true}) {
        int   destructions  = 0;
        auto  service_owner = std::make_shared<LocalRpcServer>();
        auto& server        = *service_owner;
        auto  engine        = std::make_shared<HookTestEngine>(destructions);
        engine->sleepController().setEnabled(sleep_enabled);
        std::weak_ptr<EngineBase> weak_engine = engine;
        server.engine_                        = engine;
        server.installSleepHooks();
        engine.reset();
        server.engine_.reset();
        EXPECT_TRUE(weak_engine.expired()) << "sleep_enabled=" << sleep_enabled;
        EXPECT_EQ(destructions, 1);
        // Clean up the deliberately reproduced legacy cycle on a red run.
        if (auto retained = weak_engine.lock()) {
            retained->sleepController().setHooks({});
        }
    }
}

TEST(LocalRpcServerSleepAbortTest, DirectSleepRpcRejectsNonEmptyTags) {
    auto           controller = std::make_shared<BoundSleepLifecycleController>(true);
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
    BoundSleepLifecycleController controller(true);
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
    BoundSleepLifecycleController controller(true);
    auto                          service_owner = std::make_shared<LocalRpcServer>();
    auto&                         server        = *service_owner;
    server.admission_gate_                      = std::make_shared<AdmissionGate>(&controller, "test_instance");

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
    auto           stream = makeStream(4, false);
    // No sleep admission gate is installed when the startup switch is OFF.
    EXPECT_EQ(server.registerAbortableStreamForScope(stream), nullptr);
    EXPECT_TRUE(server.abortable_streams_->streams.empty());
    EXPECT_FALSE(stream->hasError());
}

TEST(LocalRpcServerSleepAbortTest, LegacyControlsCannotBypassSleepAdmission) {
    BoundSleepLifecycleController controller(true);
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

TEST(LocalRpcServerAdmissionTest, MemoryCopyRejectedBeforeAccessingCache) {
    for (bool draining : {false, true}) {
        auto       controller = std::make_shared<SleepLifecycleController>(true);
        SleepHooks hooks;
        hooks.drain = [draining](const SleepOptions&) { return !draining; };
        controller->setHooks(hooks);
        controller->sleep(SleepOptions{});
        ASSERT_EQ(controller->state(), draining ? SleepState::DRAINING : SleepState::SLEEPING);
        LocalRpcServer server;
        server.admission_gate_ = std::make_shared<AdmissionGate>(controller.get(), "test_instance");
        // No engine is installed: admitted execution would fail before copying.
        for (auto direction : {MemoryOperationRequestPB::H2D, MemoryOperationRequestPB::D2H}) {
            SCOPED_TRACE(::testing::Message() << "draining=" << draining << " direction=" << direction);
            grpc::ServerContext context;
            FunctionRequestPB   request;
            FunctionResponsePB  response;
            auto*               copy = request.mutable_mem_request();
            copy->set_copy_direction(direction);
            copy->add_copy_items()->set_mem_block(0);
            const auto status = server.ExecuteFunction(&context, &request, &response);
            EXPECT_EQ(status.error_code(), grpc::StatusCode::UNAVAILABLE);
            EXPECT_FALSE(response.has_mem_response());
            EXPECT_EQ(controller->activeAdmissionCount(), 0);
        }
    }
}

TEST(LocalRpcServerAdmissionTest, ExecuteFunctionAdmittedWhenRunning) {
    BoundSleepLifecycleController controller(true);
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

TEST(LocalRpcServerAdmissionTest, MemoryCopyContinuationPassesDrainingGate) {
    auto           controller = drainingController();
    LocalRpcServer server;
    server.admission_gate_ = std::make_shared<AdmissionGate>(controller.get(), "memory-peer");
    for (auto direction : {MemoryOperationRequestPB::H2D, MemoryOperationRequestPB::D2H}) {
        grpc::ServerContext context;
        FunctionRequestPB   request;
        request.mutable_mem_request()->set_copy_direction(direction);
        FunctionResponsePB response;
        const auto         status = server.ExecuteFunction(&context, &request, &response);
        // The real handler must pass admission and reach the deliberately
        // unwired engine, not reject this internal operation as a new root.
        EXPECT_EQ(status.error_code(), grpc::StatusCode::INTERNAL);
        EXPECT_EQ(status.error_message(), "engine is null");
        EXPECT_EQ(controller->activeAdmissionCount(), 0);
    }
}

TEST(LocalRpcServerAdmissionTest, ConnectorDispatchOwnsConservativeContinuationPolicy) {
    struct PolicyCase {
        FunctionRequestPB::RequestCase request_case;
        bool                           continuation;
    };
    for (const auto& test : {PolicyCase{FunctionRequestPB::REQUEST_NOT_SET, false},
                             PolicyCase{FunctionRequestPB::kMemRequest, true},
                             PolicyCase{FunctionRequestPB::kRemoteRequest, true},
                             PolicyCase{FunctionRequestPB::kP2PRequest, false}}) {
        FunctionRequestPB request;
        switch (test.request_case) {
            case FunctionRequestPB::kMemRequest:
                request.mutable_mem_request();
                break;
            case FunctionRequestPB::kRemoteRequest:
                request.mutable_remote_request();
                break;
            case FunctionRequestPB::kP2PRequest:
                request.mutable_p2p_request();
                break;
            default:
                break;
        }
        EXPECT_EQ(KVCacheConnectorCoordinator::isCacheTransferContinuation(request), test.continuation)
            << request.DebugString();
    }
    // A field from a newer sender must not turn into a continuation merely
    // because the local proto cannot recognize its oneof variant yet.
    FunctionRequestPB unknown;
    ASSERT_TRUE(unknown.ParseFromString(std::string("\x22\x00", 2)));  // unknown message field 4
    EXPECT_EQ(unknown.request_case(), FunctionRequestPB::REQUEST_NOT_SET);
    EXPECT_FALSE(KVCacheConnectorCoordinator::isCacheTransferContinuation(unknown));
}

TEST(LocalRpcServerAdmissionTest, RemoteCacheContinuationPassesDrainingGate) {
    auto           controller = drainingController();
    LocalRpcServer server;
    server.admission_gate_ = std::make_shared<AdmissionGate>(controller.get(), "remote-cache-peer");
    grpc::ServerContext context;
    FunctionRequestPB   request;
    request.mutable_remote_request();
    FunctionResponsePB response;
    const auto         status = server.ExecuteFunction(&context, &request, &response);
    EXPECT_EQ(status.error_code(), grpc::StatusCode::INTERNAL);
    EXPECT_EQ(status.error_message(), "engine is null");
    EXPECT_EQ(controller->activeAdmissionCount(), 0);
}

TEST(LocalRpcServerAdmissionTest, KvFunctionsCannotCrossClosedFreezeGate) {
    BoundSleepLifecycleController controller(true);
    SleepOptions             options;
    options.prepare_only = true;
    ASSERT_TRUE(controller.sleep(options).ok);
    ASSERT_EQ(controller.state(), SleepState::DRAINING);
    LocalRpcServer server;
    server.admission_gate_ = std::make_shared<AdmissionGate>(&controller, "frozen-peer");
    for (bool remote : {false, true}) {
        grpc::ServerContext context;
        FunctionRequestPB   request;
        if (remote) {
            request.mutable_remote_request();
        } else {
            request.mutable_mem_request();
        }
        FunctionResponsePB response;
        EXPECT_EQ(server.ExecuteFunction(&context, &request, &response).error_code(), grpc::StatusCode::UNAVAILABLE);
    }
    EXPECT_EQ(controller.activeAdmissionCount(), 0);
}

TEST(LocalRpcServerAdmissionTest, P2pFunctionKeepsRootAdmissionDuringDrain) {
    auto           controller = drainingController();
    LocalRpcServer server;
    server.admission_gate_ = std::make_shared<AdmissionGate>(controller.get(), "p2p-peer");
    grpc::ServerContext context;
    FunctionRequestPB   request;
    request.mutable_p2p_request();
    FunctionResponsePB response;
    EXPECT_EQ(server.ExecuteFunction(&context, &request, &response).error_code(), grpc::StatusCode::UNAVAILABLE);
}

TEST(LocalRpcServerAdmissionTest, KvFunctionsRejectSleepingWakingAndErrorAndReopenAfterWake) {
    BoundSleepLifecycleController controller(true);
    LocalRpcServer           server;
    server.admission_gate_ = std::make_shared<AdmissionGate>(&controller, "cache-peer");
    auto check_status      = [&](grpc::StatusCode expected) {
        for (bool remote : {false, true}) {
            grpc::ServerContext context;
            FunctionRequestPB   request;
            if (remote) {
                request.mutable_remote_request();
            } else {
                request.mutable_mem_request();
            }
            FunctionResponsePB response;
            EXPECT_EQ(server.ExecuteFunction(&context, &request, &response).error_code(), expected);
            EXPECT_EQ(controller.activeAdmissionCount(), 0);
        }
    };
    ASSERT_TRUE(controller.sleep(SleepOptions{}).ok);
    check_status(grpc::StatusCode::UNAVAILABLE);
    WakeUpOptions prepare;
    prepare.prepare_only = true;
    ASSERT_TRUE(controller.wakeUp(prepare).ok);
    ASSERT_EQ(controller.state(), SleepState::WAKING_UP);
    check_status(grpc::StatusCode::UNAVAILABLE);
    WakeUpOptions commit;
    commit.commit_only = true;
    ASSERT_TRUE(controller.wakeUp(commit).ok);
    check_status(grpc::StatusCode::INTERNAL);  // admitted, then missing test engine
    SleepHooks hooks;
    hooks.releaseKvMemoryBacking = [](const SleepOptions&) { return false; };
    controller.setHooks(hooks);
    ASSERT_FALSE(controller.sleep(SleepOptions{}).ok);
    ASSERT_EQ(controller.state(), SleepState::ERROR);
    check_status(grpc::StatusCode::UNAVAILABLE);
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

TEST(LocalRpcServerSleepAbortTest, WrongDpRemoteLoadIsNoopBeforeCheckingDrainingOrSleepingAdmission) {
    BoundSleepLifecycleController controller(true);
    SleepHooks               hooks;
    bool                     allow_drain = false;
    hooks.drain                          = [&](const SleepOptions&) { return allow_drain; };
    controller.setHooks(hooks);
    DecodeRpcServer server;
    server.maga_init_params_.parallelism_config.dp_rank = 3;
    server.admission_gate_                              = std::make_shared<AdmissionGate>(&controller, "decode-peer");
    // No engine/cache store is installed: the wrong-DP no-op must not touch it.
    grpc::ServerContext    context;
    BroadcastLoadRequestPB request;
    request.set_dp_rank(2);
    BroadcastLoadResponsePB response;
    ASSERT_FALSE(controller.sleep(SleepOptions{}).ok);
    ASSERT_EQ(controller.state(), SleepState::DRAINING);
    EXPECT_TRUE(server.RemoteLoad(&context, &request, &response).ok());
    EXPECT_EQ(controller.activeAdmissionCount(), 0);
    allow_drain = true;
    ASSERT_TRUE(controller.sleep(SleepOptions{}).ok);
    EXPECT_TRUE(server.RemoteLoad(&context, &request, &response).ok());
    EXPECT_EQ(controller.activeAdmissionCount(), 0);
    // A real same-DP load must still be rejected while resources are absent.
    request.set_dp_rank(3);
    const auto status = server.RemoteLoad(&context, &request, &response);
    EXPECT_EQ(status.error_code(), grpc::StatusCode::UNAVAILABLE);
    ErrorDetailsPB details;
    ASSERT_TRUE(details.ParseFromString(status.error_details()));
    EXPECT_EQ(details.state(), "SLEEPING");
    EXPECT_EQ(details.error_code(), static_cast<int64_t>(ErrorCode::ENGINE_UNAVAILABLE));
}

}  // namespace rtp_llm
