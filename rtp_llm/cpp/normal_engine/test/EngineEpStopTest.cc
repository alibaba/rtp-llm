#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#include <pybind11/embed.h>
#include <torch/csrc/distributed/c10d/FileStore.hpp>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include "rtp_llm/cpp/distribute/CpuTpBroadcaster.h"
#include "rtp_llm/cpp/normal_engine/test/MockEngine.h"

extern char** environ;

namespace rtp_llm {
void registerExecCtxOps(pybind11::module& m);

namespace {

using ProcessGroup = c10::intrusive_ptr<c10d::ProcessGroupNCCL>;

ProcessGroup makeGroup(const c10::intrusive_ptr<c10d::Store>& store, int rank, int size) {
    auto options     = c10d::ProcessGroupNCCL::Options::create();
    options->timeout = std::chrono::seconds(20);
    return c10::make_intrusive<c10d::ProcessGroupNCCL>(store, rank, size, options);
}

class EpCollectiveExecutor: public Executor {
public:
    EpCollectiveExecutor(const ParallelismConfig& config, ProcessGroup group):
        config_(config), group_(std::move(group)) {}

    absl::Status process(const std::list<GenerateStreamPtr>&, int64_t) override {
        GptModelInputs inputs;
        tpSyncModelInputs(inputs, config_);
        execution_stopped_ = inputs.stop_execution;
        if (execution_stopped_) {
            return absl::OkStatus();
        }
        if (!group_) {
            ++iterations;
            return absl::OkStatus();
        }
        // Real EP traffic across both TP groups. A group exiting one iteration
        // early strands the other group here, even if TP stop frames work.
        auto send = torch::full({4}, config_.world_rank + 1, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        auto recv = torch::empty_like(send);
        std::vector<int64_t> splits;
        group_->alltoall_base(recv, send, splits, splits)->wait();
        auto expected = torch::arange(1, 5, torch::kFloat32);
        EXPECT_TRUE(torch::equal(recv.cpu(), expected));
        ++iterations;
        return absl::OkStatus();
    }

    int iterations = 0;

private:
    ParallelismConfig config_;
    ProcessGroup      group_;
};

class EngineEpStopChild: public DeviceTestBase {
public:
    void initTestDevices() override {
        const char* value = std::getenv("RTP_ENGINE_EP_TEST_RANK");
        ASSERT_NE(value, nullptr);
        ::alarm(90);
        initRuntime(std::atoi(value), false, false, MlaOpsType::AUTO);
    }
};

TEST_F(EngineEpStopChild, DISABLED_RunRank) {
    const int   rank     = std::atoi(std::getenv("RTP_ENGINE_EP_TEST_RANK"));
    const int   tp_size  = std::atoi(std::getenv("RTP_ENGINE_EP_TEST_TP_SIZE"));
    const char* base_env = std::getenv("RTP_ENGINE_EP_TEST_BASE");
    ASSERT_NE(base_env, nullptr);
    const std::string      base(base_env);
    py::scoped_interpreter interpreter;
    // Match the existing native engine fixture's embedded Torch setup.
    py::exec(R"(
import os, sys
sys.path = [p for p in sys.path if not os.path.isdir(os.path.join(p, "nvidia"))]
sys.modules['_hashlib'] = None
import torch
)");
    auto module = py::module_::import("types").attr("ModuleType")("engine_ep_stop").cast<py::module_>();
    registerExecCtxOps(module);
    auto store = c10::make_intrusive<c10d::FileStore>(base + "/store", 4);
    auto ep    = makeGroup(c10::make_intrusive<c10d::PrefixStore>("ep", store), rank, 4);
    py::exec(R"(
import torch.distributed as dist
dist.init_process_group(
    backend="gloo",
    init_method="file://" + os.environ["RTP_ENGINE_EP_TEST_BASE"] + "/control_store",
    rank=int(os.environ["RTP_ENGINE_EP_TEST_RANK"]),
    world_size=4,
)
leaders = dist.new_group(
    ranks=list(range(0, 4, int(os.environ["RTP_ENGINE_EP_TEST_TP_SIZE"]))),
    backend="gloo",
)
)");
    auto       dist           = py::module_::import("torch.distributed");
    py::object leaders        = py::globals()["leaders"];
    bool       expert_traffic = true;
    int        stop_votes     = 0;
    module.attr("register_comm_ops")(
        py::cpp_function([](py::args) { throw std::runtime_error("unexpected broadcast"); }),
        py::cpp_function([&](torch::Tensor value, int op, int mode, py::object) {
            EXPECT_EQ(op, static_cast<int>(ReduceOp::Max));
            EXPECT_EQ(mode, static_cast<int>(ParallelMode::DP));
            EXPECT_EQ(rank % tp_size, 0);
            EXPECT_TRUE(value.is_cpu()) << "stop votes must not synchronize the GPU";
            EXPECT_TRUE(expert_traffic) << "dense replicas must not vote across DP";
            ++stop_votes;
            dist.attr("all_reduce")(
                value, py::arg("op") = dist.attr("ReduceOp").attr("MAX"), py::arg("group") = leaders);
            return value;
        }),
        py::cpp_function([](py::args) { throw std::runtime_error("unexpected allgather"); }));
    auto& broadcaster = CpuTpBroadcaster::instance();
    if (tp_size > 1) {
        broadcaster.initialize(rank % tp_size, tp_size, base + "/tp" + std::to_string(rank / tp_size));
    }
    for (int trial = 0; trial < 18; ++trial) {
        auto engine = createMockEngine(CustomConfig{});
        ASSERT_TRUE(engine->stop().ok());
        const int model_kind = trial % 3;
        expert_traffic       = model_kind != 0;
        // The mock engine starts as a single-rank model. Match the production
        // SM120 DP/EP initialization when installing this multi-rank topology.
        engine->sm120_ep_stop_sync_      = true;
        stop_votes                       = 0;
        engine->model_config_.expert_num = model_kind == 1 ? 4 : 0;
        if (model_kind == 2) {
            auto draft                  = std::make_unique<ProposeModelEngineInitParams>(SP_TYPE_VANILLA, 1);
            draft->vanilla_model_params = std::make_unique<EngineInitParams>();
            draft->vanilla_model_params->model_config_.expert_num = 4;
            engine->propose_params_                               = std::move(draft);
        }
        auto& config            = engine->parallelism_config;
        config.tp_size          = tp_size;
        config.tp_rank          = rank % tp_size;
        config.dp_size          = 4 / tp_size;
        config.dp_rank          = rank / tp_size;
        config.ep_size          = 4;
        config.ep_rank          = rank;
        config.world_size       = 4;
        config.world_rank       = rank;
        auto  executor          = std::make_unique<EpCollectiveExecutor>(config, expert_traffic ? ep : nullptr);
        auto* observed          = executor.get();
        engine->executor_       = std::move(executor);
        engine->running_        = true;
        engine->stop_requested_ = false;
        const int first_leader  = (trial % (4 / tp_size)) * tp_size;
        const int stop_step     = 3 + trial % 3;
        for (int step = 0; step <= stop_step + 2 && engine->running_; ++step) {
            if (step >= stop_step && rank == first_leader) {
                engine->stop_requested_ = true;
            }
            // Stagger requests: the other leader would stop two iterations
            // later, while its followers can already have local stop set.
            if (step >= stop_step + 2 || (step >= stop_step && rank % tp_size != 0)) {
                engine->stop_requested_ = true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds((rank + trial) % 4));
            ASSERT_TRUE(engine->step().ok());
        }
        const int expected_steps = stop_step + (!expert_traffic && rank / tp_size != first_leader / tp_size ? 2 : 0);
        EXPECT_EQ(observed->iterations, expected_steps);
        EXPECT_EQ(stop_votes, expert_traffic && rank % tp_size == 0 ? stop_step + 1 : 0);
        if (tp_size > 1) {
            EXPECT_TRUE(observed->executionStopped());
        }
        EXPECT_TRUE(engine->stop_requested_.load());
        EXPECT_FALSE(engine->running_.load());
        ASSERT_TRUE(engine->stop().ok());
        auto                       fence = torch::zeros({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
        std::vector<torch::Tensor> tensors{fence};
        ep->allreduce(tensors)->wait();
        fence.cpu();
    }
    broadcaster.reset();
    module.attr("clear_comm_ops")();
    dist.attr("destroy_process_group")();
    ep.reset();
    ::alarm(0);
}

class EngineEpStopTest: public ::testing::TestWithParam<int> {};

TEST_P(EngineEpStopTest, StaggeredTpLeadersStopAllEpRanks) {
    ASSERT_GE(torch::cuda::device_count(), 4);
    std::string       pattern = "/tmp/engine_ep.XXXXXX";
    std::vector<char> path(pattern.begin(), pattern.end());
    path.push_back('\0');
    ASSERT_NE(::mkdtemp(path.data()), nullptr);
    const std::string  base(path.data());
    std::vector<pid_t> children;
    for (int rank = 0; rank < 4; ++rank) {
        std::vector<std::string> values;
        for (char** entry = environ; *entry != nullptr; ++entry) {
            std::string value(*entry);
            if (value.rfind("GTEST_", 0) != 0 && value.rfind("RTP_ENGINE_EP_TEST_", 0) != 0) {
                values.push_back(value);
            }
        }
        values.push_back("RTP_ENGINE_EP_TEST_RANK=" + std::to_string(rank));
        values.push_back("RTP_ENGINE_EP_TEST_BASE=" + base);
        values.push_back("RTP_ENGINE_EP_TEST_TP_SIZE=" + std::to_string(GetParam()));
        std::vector<char*> env;
        for (auto& value : values) {
            env.push_back(value.data());
        }
        env.push_back(nullptr);
        std::vector<std::string> args{"/proc/self/exe",
                                      "--gtest_filter=EngineEpStopChild.DISABLED_RunRank",
                                      "--gtest_also_run_disabled_tests",
                                      "--gtest_output="};
        std::vector<char*>       argv;
        for (auto& value : args) {
            argv.push_back(value.data());
        }
        argv.push_back(nullptr);
        pid_t     pid    = -1;
        const int result = ::posix_spawn(&pid, "/proc/self/exe", nullptr, nullptr, argv.data(), env.data());
        EXPECT_EQ(result, 0);
        if (result == 0) {
            children.push_back(pid);
        }
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(100);
    for (pid_t pid : children) {
        int   status = 0;
        pid_t waited;
        do {
            waited = ::waitpid(pid, &status, WNOHANG);
            if (waited == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        } while ((waited == 0 || (waited < 0 && errno == EINTR)) && std::chrono::steady_clock::now() < deadline);
        if (waited == 0) {
            // Only terminate this test's own child; always reap failures.
            ::kill(pid, SIGKILL);
            do {
                waited = ::waitpid(pid, &status, 0);
            } while (waited < 0 && errno == EINTR);
        }
        EXPECT_EQ(waited, pid);
        EXPECT_TRUE(WIFEXITED(status)) << "EP stop child failed: " << status;
        if (WIFEXITED(status)) {
            EXPECT_EQ(WEXITSTATUS(status), 0);
        }
    }
    ::unlink((base + "/tp0_0.sock").c_str());
    ::unlink((base + "/tp1_0.sock").c_str());
    ::unlink((base + "/store").c_str());
    ::unlink((base + "/control_store").c_str());
    ::rmdir(base.c_str());
}

INSTANTIATE_TEST_SUITE_P(TensorParallelSizes, EngineEpStopTest, ::testing::Values(1, 2));

}  // namespace
}  // namespace rtp_llm
