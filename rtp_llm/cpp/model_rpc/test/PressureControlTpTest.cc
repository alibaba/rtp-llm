#include "rtp_llm/cpp/normal_engine/test/MockEngine.h"
#include "rtp_llm/cpp/model_rpc/LocalRpcServer.h"
#include "rtp_llm/cpp/model_rpc/LocalRpcServiceImpl.h"
#include "rtp_llm/cpp/distribute/CpuTpBroadcaster.h"
#include <cerrno>
#include <filesystem>
#include <fstream>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

namespace rtp_llm {
namespace {

bool waitFile(const std::string& path) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
    while (std::chrono::steady_clock::now() < deadline) {
        if (std::filesystem::exists(path))
            return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return false;
}

void writeFile(const std::string& path, const std::string& value) {
    std::ofstream output(path + ".tmp");
    output << value;
    output.close();
    std::filesystem::rename(path + ".tmp", path);
}

class PressureTpServer: public LocalRpcServer {
public:
    PressureTpServer(const std::shared_ptr<NormalEngine>& engine, int rank, const std::string& peer = "") {
        engine_                                            = engine;
        maga_init_params_.parallelism_config.world_rank    = rank;
        maga_init_params_.parallelism_config.tp_rank       = rank;
        maga_init_params_.parallelism_config.tp_size       = 2;
        maga_init_params_.runtime_config.worker_grpc_addrs = {"", peer};
    }
};

class DelayedPressureService: public LocalRpcServiceImpl {
public:
    explicit DelayedPressureService(std::shared_ptr<LocalRpcServer> server) {
        local_server_ = std::move(server);
    }
    grpc::Status ControlPressureDiagnosticsInternal(grpc::ServerContext*                   context,
                                                    const PressureSessionControlRequestPB* request,
                                                    PressureSessionControlResponsePB*      response) override {
        if (request->action() == PressureSessionControlRequestPB::DISABLE && request->session_id() == "all") {
            return {grpc::StatusCode::UNAVAILABLE, "injected peer failure before control application"};
        }
        // The root's initial wake/ACK happens before this peer even queues its control.
        std::this_thread::sleep_for(std::chrono::milliseconds(request->phase() == "expired" ? 1500 : 200));
        return LocalRpcServiceImpl::ControlPressureDiagnosticsInternal(context, request, response);
    }
    grpc::Status StopProfileInternal(grpc::ServerContext*                context,
                                     const StopProfileInternalRequestPB* request,
                                     StopProfileResponsePB*              response) override {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        return LocalRpcServiceImpl::StopProfileInternal(context, request, response);
    }
};

class PressureControlTpChild: public DeviceTestBase {};

TEST_F(PressureControlTpChild, DISABLED_RunRank) {
    ASSERT_NE(std::getenv("RTP_PRESSURE_TP_RANK"), nullptr);
    ASSERT_NE(std::getenv("RTP_PRESSURE_TP_BASE"), nullptr);
    ::alarm(60);
    const int         rank = std::atoi(std::getenv("RTP_PRESSURE_TP_RANK"));
    const std::string base = std::getenv("RTP_PRESSURE_TP_BASE");
    CpuTpBroadcaster::instance().initialize(rank, 2, base);
    ModelConfig   model;
    RuntimeConfig runtime;
    KVCacheConfig cache;
    auto          params               = createEngineInitParams(CustomConfig{}, model, runtime, cache);
    params.runtime_config.warm_up      = false;
    NormalExecutor::test_model_factory = [&](const GptModelInitParams&) {
        return std::make_unique<MockModel>(model.vocab_size);
    };
    // Initialize cache/model locally, then exercise real idle TP execution using
    // UDS. This avoids requiring a CUDA process group for cache-capacity setup.
    auto engine                           = std::make_shared<NormalEngine>(params, nullptr, false);
    engine->parallelism_config.tp_size    = 2;
    engine->parallelism_config.tp_rank    = rank;
    auto* executor                        = static_cast<NormalExecutor*>(engine->executor_.get());
    executor->parallelism_config_.tp_size = 2;
    executor->parallelism_config_.tp_rank = rank;
    executor->tp_rank_                    = rank;
    engine->step_profiler_.configure(true, "tp_stop_window", 0, 20);
    engine->running_ = true;
    // Start a real collector on the engine worker before entering idle TP sync.
    // Cancellation must not require any further execution steps.
    engine->loop_thread_ = autil::Thread::createThread(
        [&] {
            engine->step_profiler_.startStep();
            engine->step_profiler_.finishStep();
            engine->loop();
        },
        "profile_tp_test");
    NormalExecutor::test_model_factory = nullptr;

    if (rank == 1) {
        auto                   server = std::make_shared<PressureTpServer>(engine, rank);
        DelayedPressureService service(server);
        grpc::ServerBuilder    builder;
        int                    port = 0;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
        builder.RegisterService(&service);
        auto rpc_server = builder.BuildAndStart();
        ASSERT_TRUE(rpc_server);
        writeFile(base + ".port", std::to_string(port));
        ASSERT_TRUE(waitFile(base + ".stop"));
        {
            std::lock_guard<std::mutex> lock(engine->pressure_recorder_->mutex_);
            EXPECT_EQ(engine->pressure_recorder_->session_id_, "all");
            EXPECT_EQ(engine->pressure_recorder_->phase_, "measure");
        }
        auto page = engine->getPressureSteps("all", 0, 10);
        EXPECT_EQ(page.records.size(), 1u) << "root's later session must never be labelled as this peer's old session";
        ASSERT_FALSE(page.records.empty());
        EXPECT_EQ(page.records.front().step_id, 1);
        EXPECT_EQ(page.records.front().phase, "measure");
        EXPECT_GT(page.records.front().tokens, 0);
        // Stop both loops with one final root idle broadcast, then join normally.
        engine->running_ = false;
        writeFile(base + ".stopping", "1");
        ASSERT_TRUE(waitFile(base + ".root-stopped"));
        EXPECT_TRUE(engine->stop().ok());
        rpc_server->Shutdown();
    } else {
        ASSERT_TRUE(waitFile(base + ".port"));
        int port = 0;
        std::ifstream(base + ".port") >> port;
        PressureTpServer server(engine, rank, "127.0.0.1:" + std::to_string(port));
        auto             stop_profile = [&](bool peer_only) {
            StopProfileRequestPB request;
            request.set_trace_name("tp_stop_window");
            request.set_enable_all_rank(true);
            if (peer_only)
                request.add_target_ranks(1);
            StopProfileResponsePB response;
            grpc::ServerContext   context;
            ASSERT_TRUE(server.StopProfile(&context, &request, &response).ok());
            ASSERT_EQ(response.ranks_size(), peer_only ? 1 : 2);
            for (const auto& rank : response.ranks()) {
                EXPECT_TRUE(rank.accepted());
                EXPECT_TRUE(rank.error().empty()) << rank.error();
                EXPECT_TRUE(rank.profile().collection_complete());
                EXPECT_TRUE(rank.profile().collection_error().empty());
            }
        };
        stop_profile(true);
        EXPECT_EQ(engine->getProfileStatus().state, "active");
        stop_profile(false);
        EXPECT_EQ(engine->step_profiler_.profiled_steps_, 1);
        auto control = [&](const std::string&                      session,
                           PressureSessionControlRequestPB::Action action,
                           bool                                    peer_only,
                           const std::string&                      phase = "") {
            PressureSessionControlRequestPB request;
            request.set_session_id(session);
            request.set_action(action);
            request.set_phase(phase);
            if (peer_only)
                request.add_target_ranks(1);
            PressureSessionControlResponsePB response;
            grpc::ServerContext              context;
            const auto                       begin  = std::chrono::steady_clock::now();
            const auto                       status = server.ControlPressureDiagnostics(&context, &request, &response);
            if (peer_only) {
                EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
                EXPECT_EQ(response.ranks_size(), 0);
                return;
            }
            EXPECT_TRUE(status.ok());
            EXPECT_LT(std::chrono::steady_clock::now() - begin, std::chrono::seconds(2));
            EXPECT_EQ(response.ranks_size(), peer_only ? 1 : 2);
            for (const auto& ack : response.ranks()) {
                if (ack.world_rank() == 1
                    && (action == PressureSessionControlRequestPB::DISABLE || session == "next"
                        || phase == "expired")) {
                    EXPECT_FALSE(ack.error().empty());
                    EXPECT_EQ(ack.applied_monotonic_ns(), 0);
                } else {
                    EXPECT_TRUE(ack.error().empty()) << ack.error();
                    EXPECT_GT(ack.applied_monotonic_ns(), 0);
                }
            }
        };
        control("peer-only", PressureSessionControlRequestPB::ENABLE, true);
        control("all", PressureSessionControlRequestPB::ENABLE, false);
        control("all", PressureSessionControlRequestPB::MARK_PHASE, true, "peer-phase");
        control("all", PressureSessionControlRequestPB::DISABLE, true);
        control("all", PressureSessionControlRequestPB::MARK_PHASE, false, "measure");
        auto query                             = std::make_shared<GenerateInput>();
        query->input_ids                       = torch::tensor({1, 2, 3}, torch::kInt32);
        query->generate_config                 = std::make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 1;
        auto stream                            = engine->enqueue(query);
        EXPECT_TRUE(stream->nextOutput(5000).ok());
        auto page = engine->getPressureSteps("all", 0, 10);
        EXPECT_FALSE(page.records.empty());
        if (!page.records.empty()) {
            EXPECT_EQ(page.records.front().step_id, 1);
            EXPECT_EQ(page.records.front().phase, "measure");
            EXPECT_GT(page.records.front().tokens, 0);
        }
        auto run_request = [&](std::vector<int32_t> tokens) {
            auto input                             = std::make_shared<GenerateInput>();
            input->input_ids                       = torch::tensor(tokens, torch::kInt32);
            input->generate_config                 = std::make_shared<GenerateConfig>();
            input->generate_config->max_new_tokens = 1;
            auto stream                            = engine->enqueue(input);
            EXPECT_TRUE(stream->nextOutput(5000).ok());
        };
        control("all", PressureSessionControlRequestPB::MARK_PHASE, false, "expired");
        // Allow the delayed peer handler to see its expired RPC before checking its retained phase.
        std::this_thread::sleep_for(std::chrono::milliseconds(700));
        run_request({5, 6});
        // The root disables A, but the peer never receives that request. The
        // next full-group ENABLE succeeds locally and is rejected by the peer.
        control("all", PressureSessionControlRequestPB::DISABLE, false);
        // The peer still has A enabled while the root has no active recorder.
        run_request({4, 5});
        control("next", PressureSessionControlRequestPB::ENABLE, false);
        run_request({3, 4});
        auto next_page = engine->getPressureSteps("next", 0, 10);
        EXPECT_EQ(next_page.records.size(), 1u);
        writeFile(base + ".stop", "1");
        ASSERT_TRUE(waitFile(base + ".stopping"));
        EXPECT_TRUE(engine->stop().ok());
        writeFile(base + ".root-stopped", "1");
    }
    engine.reset();
    CpuTpBroadcaster::instance().reset();
    ::alarm(0);
}

TEST(PressureControlTpTest, IdleDelayedGroupAppliesPhaseAndRecordsOneRealForward) {
    const char*       tmpdir  = std::getenv("TMPDIR");
    std::string       pattern = std::string(tmpdir ? tmpdir : "/tmp") + "/pressure_tp.XXXXXX";
    std::vector<char> path(pattern.begin(), pattern.end());
    path.push_back('\0');
    ASSERT_NE(::mkdtemp(path.data()), nullptr);
    const std::string  base = std::string(path.data()) + "/b";
    std::vector<pid_t> children;
    for (int rank = 0; rank < 2; ++rank) {
        std::vector<std::string> environment;
        for (char** entry = environ; *entry; ++entry) {
            const std::string value(*entry);
            if (value.rfind("GTEST_", 0) && value.rfind("RTP_PRESSURE_TP_", 0)
                && value.rfind("TEST_PREMATURE_EXIT_FILE=", 0)) {
                environment.push_back(value);
            }
        }
        environment.push_back("RTP_PRESSURE_TP_RANK=" + std::to_string(rank));
        environment.push_back("RTP_PRESSURE_TP_BASE=" + base);
        std::vector<char*> env;
        for (auto& value : environment)
            env.push_back(value.data());
        env.push_back(nullptr);
        std::vector<std::string> arguments = {"/proc/self/exe",
                                              "--gtest_filter=PressureControlTpChild.DISABLED_RunRank",
                                              "--gtest_also_run_disabled_tests",
                                              "--gtest_output="};
        std::vector<char*>       args;
        for (auto& value : arguments)
            args.push_back(value.data());
        args.push_back(nullptr);
        pid_t      pid    = -1;
        const auto result = ::posix_spawn(&pid, "/proc/self/exe", nullptr, nullptr, args.data(), env.data());
        EXPECT_EQ(result, 0);
        if (!result)
            children.push_back(pid);
    }
    for (auto pid : children) {
        int   status = 0;
        pid_t waited;
        do {
            waited = ::waitpid(pid, &status, 0);
        } while (waited < 0 && errno == EINTR);
        EXPECT_EQ(waited, pid);
        EXPECT_TRUE(WIFEXITED(status)) << status;
        if (WIFEXITED(status)) {
            EXPECT_EQ(WEXITSTATUS(status), 0);
        }
    }
    std::filesystem::remove_all(path.data());
}
}  // namespace
}  // namespace rtp_llm
