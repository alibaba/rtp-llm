#include <array>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <limits.h>
#include <stdexcept>
#include <string>
#include <thread>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <gtest/gtest.h>
#include <pybind11/embed.h>

#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace py = pybind11;

namespace rtp_llm {

void registerExecCtxOps(pybind11::module& module);

namespace {

constexpr int         kWorldSize          = 2;
constexpr auto        kWatchdogTimeout    = std::chrono::seconds(45);
constexpr auto        kTerminationGrace   = std::chrono::milliseconds(500);
constexpr auto        kWatchdogPollPeriod = std::chrono::milliseconds(20);
constexpr const char* kRankEnvironment    = "RTP_LLM_MODEL_INPUT_SYNC_RANK";
constexpr const char* kPortEnvironment    = "RTP_LLM_MODEL_INPUT_SYNC_PORT";
constexpr const char* kWorkerFilter       = "ModelInputMetadataSyncWorker.runRank";

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

uint16_t reserveLoopbackPort() {
    const int socket_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_fd < 0) {
        throw std::runtime_error("failed to create model-input sync test socket");
    }

    sockaddr_in address{};
    address.sin_family      = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port        = 0;
    if (bind(socket_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
        close(socket_fd);
        throw std::runtime_error("failed to reserve model-input sync test port");
    }

    socklen_t address_length = sizeof(address);
    if (getsockname(socket_fd, reinterpret_cast<sockaddr*>(&address), &address_length) != 0) {
        close(socket_fd);
        throw std::runtime_error("failed to read model-input sync test port");
    }
    const uint16_t port = ntohs(address.sin_port);
    close(socket_fd);
    return port;
}

class PythonCollectiveEnvironment {
public:
    PythonCollectiveEnvironment(int rank, uint16_t port) {
        globals_["rank"]       = rank;
        globals_["world_size"] = kWorldSize;
        globals_["port"]       = port;
        py::exec(R"PY(
import datetime
import torch
import torch.distributed as dist

if torch.cuda.device_count() < world_size:
    raise RuntimeError(
        f"model-input sync regression requires {world_size} devices, "
        f"found {torch.cuda.device_count()}"
    )

torch.cuda.set_device(rank)
dist.init_process_group(
    backend="nccl",
    init_method=f"tcp://127.0.0.1:{port}",
    rank=rank,
    world_size=world_size,
    timeout=datetime.timedelta(seconds=30),
)

broadcast_calls = 0
first_broadcast_is_single_int64_header = False

def test_broadcast(tensors, root, mode):
    global broadcast_calls
    global first_broadcast_is_single_int64_header
    if broadcast_calls == 0:
        first_broadcast_is_single_int64_header = (
            len(tensors) == 1
            and tensors[0].dtype == torch.int64
            and not tensors[0].is_cuda
        )
    broadcast_calls += 1
    for tensor in tensors:
        if tensor.is_cuda:
            communication_tensor = tensor
            copy_back = False
        else:
            communication_tensor = tensor.to(torch.device("cuda", rank))
            copy_back = True
        dist.broadcast(communication_tensor, src=root)
        if copy_back:
            tensor.copy_(communication_tensor.cpu())

def unexpected_allreduce(tensor, op, mode, dest):
    raise RuntimeError("unexpected all-reduce in model-input sync regression")

def unexpected_allgather(recv_buffers, mode, send_buffers, inplace):
    raise RuntimeError("unexpected all-gather in model-input sync regression")
)PY",
                 globals_);

        auto types = py::module_::import("types");
        module_    = types.attr("ModuleType")("_model_input_sync_test_ops").cast<py::module_>();
        registerExecCtxOps(module_);
        module_.attr("register_comm_ops")(globals_["test_broadcast"],
                                          globals_["unexpected_allreduce"],
                                          globals_["unexpected_allgather"]);
    }

    ~PythonCollectiveEnvironment() {
        try {
            module_.attr("clear_comm_ops")();
            py::exec(R"PY(
if dist.is_initialized():
    dist.destroy_process_group()
)PY",
                     globals_);
        } catch (const std::exception& e) {
            std::cerr << "model-input collective cleanup failed: " << e.what() << std::endl;
        }
    }

    int broadcastCallCount() const {
        return globals_["broadcast_calls"].cast<int>();
    }

    bool firstBroadcastIsSingleInt64Header() const {
        return globals_["first_broadcast_is_single_int64_header"].cast<bool>();
    }

    void sentinelBroadcast(int rank) {
        const int calls_before = broadcastCallCount();
        auto      sentinel     = torch::tensor({rank == 0 ? 0x51A7 : 0}, torch::kInt64);
        execBroadcast({{sentinel}, 0});
        require(sentinel.item<int64_t>() == 0x51A7, "sentinel broadcast returned the wrong value");
        require(broadcastCallCount() == calls_before + 1, "sentinel used an unexpected collective sequence");
    }

private:
    py::dict    globals_;
    py::module_ module_;
};

ParallelismConfig makeParallelismConfig(int rank) {
    ParallelismConfig config;
    config.tp_size          = kWorldSize;
    config.tp_rank          = rank;
    config.world_size       = kWorldSize;
    config.world_rank       = rank;
    config.local_world_size = kWorldSize;
    config.local_rank       = rank;
    config.dp_size          = 1;
    config.dp_rank          = 0;
    return config;
}

void runMetadataSync(int rank, const PythonCollectiveEnvironment& collective_environment) {
    constexpr size_t kKvBlockStrideBytes = (size_t{1} << 33) + 123;

    GptModelInputs inputs;
    inputs.kv_block_stride_bytes      = 0;
    inputs.kv_scale_stride_bytes      = 0;
    inputs.seq_size_per_block         = 0;
    inputs.kernel_seq_size_per_block  = 0;
    if (rank == 0) {
        inputs.combo_tokens             = torch::tensor({11, 12}, torch::kInt32);
        inputs.input_lengths            = torch::tensor({2}, torch::kInt32);
        inputs.sequence_lengths         = torch::empty({0}, torch::kInt32);
        inputs.prefix_lengths           = torch::tensor({0}, torch::kInt32);
        inputs.kv_cache_kernel_block_id = torch::tensor({3, 4}, torch::kInt32).reshape({1, 1, 2});
        inputs.kv_cache_block_id        = torch::tensor({5, 6}, torch::kInt32).reshape({1, 1, 2});
        inputs.kv_cache_layer_to_group  = torch::tensor({0}, torch::kInt32);
        inputs.kv_cache_group_types     = torch::tensor({1}, torch::kInt32);
        inputs.kv_cache_update_mapping  = torch::empty({0, 2}, torch::kInt32);
        inputs.request_id               = torch::tensor({701}, torch::kInt64);
        inputs.request_pd_separation    = torch::tensor({true}, torch::kBool);
        inputs.cache_keys               = torch::tensor({101, 102}, torch::kInt64).reshape({1, 2});
        inputs.lm_output_indexes        = torch::tensor({1}, torch::kInt32);
        inputs.lm_output_lengths        = torch::tensor({1}, torch::kInt32);
        inputs.kv_block_stride_bytes    = kKvBlockStrideBytes;
        inputs.kv_scale_stride_bytes    = 67584;
        inputs.seq_size_per_block       = 512;
        inputs.kernel_seq_size_per_block = 64;
        inputs.pd_separation             = true;
        inputs.decode_entrance           = true;
        inputs.need_all_logits           = true;
        inputs.need_moe_gating           = true;
        inputs.warmup                    = true;
        inputs.is_target_verify          = true;
    }

    const auto config       = makeParallelismConfig(rank);
    const int  calls_before = collective_environment.broadcastCallCount();
    tpSyncModelInputs(inputs, config);

    require(collective_environment.broadcastCallCount() == calls_before + 2,
            "model-input sync changed the header or packed collective count");
    require(collective_environment.firstBroadcastIsSingleInt64Header(),
            "model-input metadata must use one int64 header tensor");
    require(inputs.pd_separation, "P/D separation flag was not synchronized");
    require(inputs.decode_entrance, "Decode entrance flag was not synchronized");
    require(inputs.need_all_logits, "all-logits flag was not synchronized");
    require(inputs.need_moe_gating, "MoE gating flag was not synchronized");
    require(inputs.warmup, "warm-up flag was not synchronized");
    require(inputs.is_target_verify, "target-verify flag was not synchronized");
    require(inputs.kv_block_stride_bytes == kKvBlockStrideBytes, "KV block stride was not synchronized");
    require(inputs.kv_scale_stride_bytes == 67584, "KV scale stride was not synchronized");
    require(inputs.seq_size_per_block == 512, "tokens per block was not synchronized");
    require(inputs.kernel_seq_size_per_block == 64, "kernel tokens per block was not synchronized");
    require(inputs.request_pd_separation.defined()
                && inputs.request_pd_separation.sizes() == torch::IntArrayRef({1})
                && torch::equal(inputs.request_pd_separation, torch::tensor({true}, torch::kBool)),
            "per-request P/D separation metadata was not synchronized");
    require(inputs.cache_keys.defined() && inputs.cache_keys.sizes() == torch::IntArrayRef({1, 2}),
            "cache keys were not allocated on every rank");
    require(torch::equal(inputs.cache_keys, torch::tensor({101, 102}, torch::kInt64).reshape({1, 2})),
            "cache keys were not synchronized");

    inputs.pd_separation = false;
    if (rank == 0) {
        inputs.request_pd_separation = torch::tensor({false}, torch::kBool);
    } else {
        inputs.request_pd_separation = torch::Tensor();
        inputs.cache_keys            = torch::Tensor();
    }
    const int non_pd_calls_before = collective_environment.broadcastCallCount();
    tpSyncModelInputs(inputs, config);
    require(collective_environment.broadcastCallCount() == non_pd_calls_before + 2,
            "non-P/D model-input sync changed the collective count");
    require(!inputs.pd_separation, "non-P/D separation flag was not synchronized");
    require(inputs.request_pd_separation.defined()
                && torch::equal(inputs.request_pd_separation, torch::tensor({false}, torch::kBool)),
            "non-P/D per-request metadata was not synchronized");
    if (rank != 0) {
        require(!inputs.cache_keys.defined(), "non-P/D sync allocated cache keys on a non-root rank");
    }
}

int runRank(int rank, uint16_t port) {
    try {
        py::scoped_interpreter interpreter;
        PythonCollectiveEnvironment collective_environment(rank, port);
        initRuntime(rank, false, false, MlaOpsType::AUTO);
        runMetadataSync(rank, collective_environment);
        collective_environment.sentinelBroadcast(rank);
        return EXIT_SUCCESS;
    } catch (const py::error_already_set& e) {
        std::cerr << "model-input sync rank " << rank << " Python failure: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "model-input sync rank " << rank << " failure: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "model-input sync rank " << rank << " unknown failure" << std::endl;
    }
    return EXIT_FAILURE;
}

std::string selfExecutable() {
    std::array<char, PATH_MAX> executable_path{};
    const ssize_t size = readlink("/proc/self/exe", executable_path.data(), executable_path.size() - 1);
    if (size <= 0 || static_cast<size_t>(size) >= executable_path.size()) {
        throw std::runtime_error("failed to resolve model-input sync test executable");
    }
    executable_path[static_cast<size_t>(size)] = '\0';
    return executable_path.data();
}

[[noreturn]] void execRank(const std::string& executable, int rank, uint16_t port) {
    const std::string rank_string = std::to_string(rank);
    const std::string port_string = std::to_string(port);
    if (setenv(kRankEnvironment, rank_string.c_str(), 1) != 0
        || setenv(kPortEnvironment, port_string.c_str(), 1) != 0) {
        std::_Exit(126);
    }
    unsetenv("TEST_PREMATURE_EXIT_FILE");

    const std::string filter_argument = std::string("--gtest_filter=") + kWorkerFilter;
    execl(executable.c_str(),
          executable.c_str(),
          filter_argument.c_str(),
          "--gtest_color=no",
          static_cast<char*>(nullptr));
    std::_Exit(127);
}

void terminateAndReap(std::array<pid_t, kWorldSize>& children) {
    for (const pid_t child : children) {
        if (child > 0) {
            kill(child, SIGTERM);
        }
    }

    const auto grace_deadline = std::chrono::steady_clock::now() + kTerminationGrace;
    while (std::chrono::steady_clock::now() < grace_deadline) {
        bool any_running = false;
        for (auto& child : children) {
            if (child <= 0) {
                continue;
            }
            int         status = 0;
            const pid_t result = waitpid(child, &status, WNOHANG);
            if (result == child) {
                child = -1;
            } else if (result == 0) {
                any_running = true;
            }
        }
        if (!any_running) {
            return;
        }
        std::this_thread::sleep_for(kWatchdogPollPeriod);
    }

    for (auto& child : children) {
        if (child > 0) {
            kill(child, SIGKILL);
            waitpid(child, nullptr, 0);
            child = -1;
        }
    }
}

void runWithWatchdog() {
    const uint16_t                port       = reserveLoopbackPort();
    const std::string             executable = selfExecutable();
    std::array<pid_t, kWorldSize> children{-1, -1};
    for (int rank = 0; rank < kWorldSize; ++rank) {
        const pid_t child = fork();
        if (child == 0) {
            execRank(executable, rank, port);
        }
        if (child < 0) {
            terminateAndReap(children);
            FAIL() << "failed to fork model-input sync rank " << rank;
        }
        children[rank] = child;
    }

    std::array<int, kWorldSize>  statuses{};
    std::array<bool, kWorldSize> finished{};
    size_t                       finished_count = 0;
    const auto                   deadline       = std::chrono::steady_clock::now() + kWatchdogTimeout;
    while (finished_count < kWorldSize && std::chrono::steady_clock::now() < deadline) {
        for (int rank = 0; rank < kWorldSize; ++rank) {
            if (finished[rank]) {
                continue;
            }
            const pid_t result = waitpid(children[rank], &statuses[rank], WNOHANG);
            if (result == children[rank]) {
                finished[rank] = true;
                children[rank] = -1;
                ++finished_count;
            } else if (result < 0 && errno != EINTR) {
                terminateAndReap(children);
                FAIL() << "waitpid failed for model-input sync rank " << rank;
            }
        }
        if (finished_count < kWorldSize) {
            std::this_thread::sleep_for(kWatchdogPollPeriod);
        }
    }

    if (finished_count != kWorldSize) {
        terminateAndReap(children);
        FAIL() << "model-input sync regression exceeded " << kWatchdogTimeout.count() << " second watchdog";
    }

    for (int rank = 0; rank < kWorldSize; ++rank) {
        ASSERT_TRUE(WIFEXITED(statuses[rank])) << "rank " << rank << " terminated by signal " << WTERMSIG(statuses[rank]);
        EXPECT_EQ(WEXITSTATUS(statuses[rank]), EXIT_SUCCESS) << "rank " << rank << " failed";
    }
}

}  // namespace

class ModelInputMetadataSyncTest: public ::testing::Test {};
class ModelInputMetadataSyncWorker: public ::testing::Test {};

TEST_F(ModelInputMetadataSyncWorker, runRank) {
    const char* rank_string = std::getenv(kRankEnvironment);
    const char* port_string = std::getenv(kPortEnvironment);
    ASSERT_NE(rank_string, nullptr);
    ASSERT_NE(port_string, nullptr);

    const int rank = std::stoi(rank_string);
    const int port = std::stoi(port_string);
    ASSERT_GE(rank, 0);
    ASSERT_LT(rank, kWorldSize);
    ASSERT_GT(port, 0);
    ASSERT_LE(port, std::numeric_limits<uint16_t>::max());
    EXPECT_EQ(runRank(rank, static_cast<uint16_t>(port)), EXIT_SUCCESS);
}

TEST_F(ModelInputMetadataSyncTest, metadataAndCacheKeysReachEveryRank) {
    runWithWatchdog();
}

}  // namespace rtp_llm
