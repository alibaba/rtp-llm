#include <array>
#include <chrono>
#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <limits.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <pybind11/embed.h>

// Reuse the executor test doubles and input builders without exposing them as
// production APIs. The target below filters out the tests registered here.
#include "rtp_llm/cpp/normal_engine/speculative/test/MtpExecutorTest.cc"

#undef private

namespace py = pybind11;

namespace rtp_llm {

void registerExecCtxOps(pybind11::module& module);

namespace {

constexpr int         kWorldSize          = 2;
constexpr auto        kWatchdogTimeout    = std::chrono::seconds(45);
constexpr auto        kTerminationGrace   = std::chrono::milliseconds(500);
constexpr auto        kWatchdogPollPeriod = std::chrono::milliseconds(20);
constexpr const char* kTestFilter         = "MtpExecutorDistributedAbortTest";
constexpr const char* kWorkerFilter       = "MtpExecutorDistributedAbortWorker.runRank";
constexpr const char* kRankEnvironment    = "RTP_LLM_DISTRIBUTED_ABORT_RANK";
constexpr const char* kPortEnvironment    = "RTP_LLM_DISTRIBUTED_ABORT_PORT";
constexpr const char* kPathEnvironment    = "RTP_LLM_DISTRIBUTED_ABORT_PATH";

enum class AbortPath {
    PREFILL,
    DECODE,
    DECODE_LOCAL_CPU_HIDDEN,
    DECODE_INTERMEDIATE_DRAFT_TOKEN,
};

class MtpExecutorTestHarness final: public MtpExecutorTest {
public:
    void TestBody() override {}
};

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

uint16_t reserveLoopbackPort() {
    const int socket_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_fd < 0) {
        throw std::runtime_error("failed to create watchdog test socket");
    }

    sockaddr_in address{};
    address.sin_family      = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port        = 0;
    if (bind(socket_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
        close(socket_fd);
        throw std::runtime_error("failed to reserve watchdog test port");
    }

    socklen_t address_length = sizeof(address);
    if (getsockname(socket_fd, reinterpret_cast<sockaddr*>(&address), &address_length) != 0) {
        close(socket_fd);
        throw std::runtime_error("failed to read watchdog test port");
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
        f"distributed abort regression requires {world_size} devices, "
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

def test_broadcast(tensors, root, mode):
    global broadcast_calls
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
    raise RuntimeError("unexpected all-reduce in distributed abort regression")

def unexpected_allgather(recv_buffers, mode, send_buffers, inplace):
    raise RuntimeError("unexpected all-gather in distributed abort regression")
)PY",
                 globals_);

        auto types  = py::module_::import("types");
        module_ = types.attr("ModuleType")("_distributed_abort_test_ops").cast<py::module_>();
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
            std::cerr << "collective cleanup failed: " << e.what() << std::endl;
        }
    }

    int broadcastCallCount() const {
        return globals_["broadcast_calls"].cast<int>();
    }

    void requireExecutorBroadcast() const {
        require(broadcastCallCount() > 0, "executor did not invoke the Python broadcast callback");
    }

    void sentinelBroadcast(int rank) {
        const int calls_before = broadcastCallCount();
        auto sentinel = torch::tensor({rank == 0 ? 0x51A7 : 0}, torch::kInt64);
        execBroadcast({{sentinel}, 0});
        require(sentinel.item<int64_t>() == 0x51A7, "sentinel broadcast returned the wrong value");
        require(broadcastCallCount() == calls_before + 1, "sentinel did not use the Python broadcast callback");
    }

private:
    py::dict    globals_;
    py::module_ module_;
};

void configureDistributedExecutor(MtpExecutor& executor, int rank, RoleType role) {
    executor.tp_rank_                               = rank;
    executor.role_type_                             = role;
    executor.parallelism_config_.tp_size            = kWorldSize;
    executor.parallelism_config_.tp_rank            = rank;
    executor.parallelism_config_.world_size         = kWorldSize;
    executor.parallelism_config_.world_rank         = rank;
    executor.parallelism_config_.local_world_size   = kWorldSize;
    executor.parallelism_config_.local_rank         = rank;
    executor.parallelism_config_.dp_size            = 1;
    executor.parallelism_config_.dp_rank            = 0;
    executor.parallelism_config_.ep_size            = 1;
    executor.parallelism_config_.ep_rank            = 0;
}

void runPrefillAbort(int rank, MtpExecutorTestHarness& harness) {
    MtpExecutorTestConfig test_config;
    test_config.gen_num_per_cycle = 4;
    auto components               = harness.createMtpExecutorComponents(test_config);
    configureDistributedExecutor(*components.executor, rank, RoleType::PREFILL);

    std::list<GenerateStreamPtr> streams;
    GenerateStreamPtr           stream;
    if (rank == 0) {
        stream = harness.createContextStream(
            components.model_config, components.runtime_config, components.resource_context, {0, 1, 2, 3});
        streams.push_back(stream);
    }

    GptModelInputs target_input;
    target_input.combo_tokens      = torch::tensor({0, 1, 2, 3}, torch::kInt32);
    target_input.input_lengths     = torch::tensor({4}, torch::kInt32);
    target_input.prefix_lengths    = torch::tensor({0}, torch::kInt32);
    target_input.lm_output_indexes = torch::tensor({3}, torch::kInt32);

    GptModelOutputs target_output;
    target_output.logits            = torch::tensor({0.1f, 0.2f, 0.3f, 0.4f}).reshape({1, 4});
    target_output.all_hidden_states = torch::zeros({4, 2}, torch::kFloat32);
    components.fake_target_model->setInputs({target_input});
    components.fake_target_model->setOutputs({target_output});

    if (rank == 0) {
        components.fake_sampler->setInputs({SamplerInputs{target_output.logits}});
        SamplerOutput sampler_output;
        sampler_output.token_ids = torch::tensor({1}, torch::kInt32).reshape({1, 1});
        sampler_output.success   = torch::tensor({false}, torch::kBool);
        components.fake_sampler->setOutputs({sampler_output});
    }

    harness.setupFakeModels(components.executor.get(),
                            std::move(components.fake_target_model),
                            std::move(components.fake_draft_model),
                            std::move(components.fake_fast_topk_sampler),
                            std::move(components.fake_speculative_sampler),
                            std::move(components.fake_sampler));

    const auto status = components.executor->process(streams);
    require(status.ok(), "prefill abort process returned an error: " + status.ToString());
    if (rank == 0) {
        require(stream->hasError(), "prefill sampler failure was not reported to the stream");
        require(stream->iterCount() == 0, "prefill sampler abort did not restore the stream iteration count");
    }
    require(components.executor->process({}).ok(), "empty prefill process failed after sampler abort");
}

void runDecodeAbort(int rank, MtpExecutorTestHarness& harness) {
    MtpExecutorTestConfig test_config;
    test_config.gen_num_per_cycle   = 1;
    test_config.vocab_size_override = 4;
    auto components                 = harness.createMtpExecutorComponents(test_config);
    configureDistributedExecutor(*components.executor, rank, RoleType::DECODE);

    std::list<GenerateStreamPtr> streams;
    GenerateStreamPtr           stream;
    if (rank == 0) {
        StreamSpecUpdateInfo spec_update_info{torch::tensor({{2}}, torch::kInt32),
                                               1,
                                               3,
                                               torch::tensor({{0.03f, 0.04f}}),
                                               torch::tensor({{0.0f, 0.0f, 1.0f, 0.0f}})};
        stream = harness.createDecodeStream(components.model_config,
                                            components.runtime_config,
                                            components.resource_context,
                                            {0, 1},
                                            spec_update_info);
        streams.push_back(stream);
    }

    GptModelInputs target_input;
    target_input.combo_tokens      = torch::tensor({2, 3}, torch::kInt32);
    target_input.input_lengths     = torch::tensor({2}, torch::kInt32);
    target_input.prefix_lengths    = torch::tensor({2}, torch::kInt32);
    target_input.lm_output_indexes = torch::tensor({0, 1}, torch::kInt32);

    GptModelOutputs target_output;
    target_output.logits            = torch::zeros({2, 4}, torch::kFloat32);
    target_output.all_hidden_states = torch::zeros({2, 2}, torch::kFloat32);
    components.fake_target_model->setInputs({target_input});
    components.fake_target_model->setOutputs({target_output});

    if (rank == 0) {
        components.fake_sampler->setInputs({SamplerInputs{target_output.logits}});
        SamplerOutput sampler_output;
        sampler_output.token_ids = torch::zeros({1, 2}, torch::kInt32);
        sampler_output.success   = torch::tensor({true, false}, torch::kBool);
        components.fake_sampler->setOutputs({sampler_output});
    }

    harness.setupFakeModels(components.executor.get(),
                            std::move(components.fake_target_model),
                            std::move(components.fake_draft_model),
                            std::move(components.fake_fast_topk_sampler),
                            std::move(components.fake_speculative_sampler),
                            std::move(components.fake_sampler));

    const auto status = components.executor->process(streams);
    require(status.ok(), "decode abort process returned an error: " + status.ToString());
    if (rank == 0) {
        require(stream->hasError(), "decode sampler failure was not reported to the stream");
        require(stream->iterCount() == 0, "decode sampler abort did not restore the stream iteration count");
    }
    require(components.executor->process({}).ok(), "empty decode process failed after sampler abort");
}

void runDecodeLocalCpuHiddenAbort(int rank, MtpExecutorTestHarness& harness) {
    MtpExecutorTestConfig test_config;
    test_config.gen_num_per_cycle   = 4;
    test_config.vocab_size_override = 4;
    auto components                 = harness.createMtpExecutorComponents(test_config);
    configureDistributedExecutor(*components.executor, rank, RoleType::DECODE);

    std::list<GenerateStreamPtr> streams;
    GenerateStreamPtr           stream;
    if (rank == 0) {
        StreamSpecUpdateInfo spec_update_info{torch::tensor({{2}}, torch::kInt32),
                                               1,
                                               3,
                                               torch::tensor({{0.03f, 0.04f}}, torch::kFloat32),
                                               torch::tensor({{0.0f, 0.0f, 1.0f, 0.0f}}, torch::kFloat32)};
        stream = harness.createDecodeStream(components.model_config,
                                            components.runtime_config,
                                            components.resource_context,
                                            {0, 1},
                                            spec_update_info);
        auto output           = stream->getSPOutputBuffer();
        output->tensors_holder.clear();
        output->all_probs     = torch::tensor({{0.0f, 0.0f, 1.0f, 0.0f}}, torch::kFloat32).to(torch::kCUDA);
        output->hidden_states = torch::zeros(
            {1, static_cast<int64_t>(components.executor->hidden_size_)},
            torch::TensorOptions().dtype(dataTypeToTorchType(components.executor->data_type_)));
        const auto latest_tokens = stream->getLatestTokens(1);
        require(latest_tokens.size() == 1 && output->tokens.data_ptr<int32_t>()[0] == latest_tokens[0],
                "test proposal token must match the stream tail");
        require(output->all_probs.is_cuda() && output->all_probs.scalar_type() == torch::kFloat32
                    && output->all_probs.is_contiguous() && output->all_probs.size(0) == 1
                    && static_cast<size_t>(output->all_probs.size(1)) == components.executor->propose_vocab_size_,
                "test probabilities must satisfy the local input contract");
        require(output->hidden_states.scalar_type() == dataTypeToTorchType(components.executor->data_type_)
                    && output->hidden_states.is_contiguous() && output->hidden_states.size(0) == 1
                    && static_cast<size_t>(output->hidden_states.size(1)) == components.executor->hidden_size_,
                "test hidden states must satisfy dtype and shape constraints");
        require(output->hidden_states.is_cpu(), "test hidden states must start on CPU");
        streams.push_back(stream);
    }

    auto* target_model = components.fake_target_model.get();
    auto* draft_model  = components.fake_draft_model.get();
    harness.setupFakeModels(components.executor.get(),
                            std::move(components.fake_target_model),
                            std::move(components.fake_draft_model),
                            std::move(components.fake_fast_topk_sampler),
                            std::move(components.fake_speculative_sampler),
                            std::move(components.fake_sampler));

    const auto status = components.executor->process(streams);
    require(status.ok(), "local CPU hidden input abort returned an error: " + status.ToString());
    if (rank == 0) {
        require(stream->hasError(), "local CPU hidden input error was not reported");
        require(stream->statusInfo().code() == ErrorCode::INVALID_PARAMS,
                "local CPU hidden input returned the wrong error code");
        require(stream->iterCount() == 0, "decode input abort changed the stream iteration count");
    }
    require(target_model->forwardCallCount() == 0, "target model ran after local CPU hidden input error");
    require(draft_model->forwardCallCount() == 0, "draft model ran after local CPU hidden input error");
    require(components.executor->process({}).ok(), "empty decode process failed after local CPU hidden input abort");
}

void runDecodeIntermediateDraftTokenAbort(int rank, MtpExecutorTestHarness& harness) {
    MtpExecutorTestConfig test_config;
    test_config.gen_num_per_cycle                  = 3;
    test_config.vocab_size_override                = 4;
    test_config.input_vocab_size_override          = 4;
    test_config.proposal_input_vocab_size_override = 3;
    auto components                                = harness.createMtpExecutorComponents(test_config);
    configureDistributedExecutor(*components.executor, rank, RoleType::DECODE);

    std::list<GenerateStreamPtr> streams;
    GenerateStreamPtr           stream;
    if (rank == 0) {
        StreamSpecUpdateInfo spec_update_info{torch::tensor({{1}}, torch::kInt32),
                                               1,
                                               2,
                                               torch::tensor({{0.03f, 0.04f}}, torch::kFloat32),
                                               torch::tensor({{0.0f, 0.0f, 1.0f, 0.0f}}, torch::kFloat32)};
        stream = harness.createDecodeStream(components.model_config,
                                            components.runtime_config,
                                            components.resource_context,
                                            {0, 1},
                                            spec_update_info);
        streams.push_back(stream);
    }

    GptModelInputs draft_input;
    draft_input.combo_tokens       = torch::tensor({2}, torch::kInt32);
    draft_input.input_lengths      = torch::tensor({2}, torch::kInt32);
    draft_input.sequence_lengths   = torch::tensor({2}, torch::kInt32);
    draft_input.lm_output_indexes  = torch::tensor({0}, torch::kInt32);
    draft_input.last_hidden_states = torch::tensor({0.03f, 0.04f}).reshape({1, 2});
    auto draft_output              = harness.createRandomGptModelOutputs(1, 4, 2);
    components.fake_draft_model->setInputs({draft_input});
    components.fake_draft_model->setOutputs({draft_output});

    speculative::FastTopKSamplerOutput draft_sampler_output;
    draft_sampler_output.token_ids = torch::tensor({3}, torch::kInt32).reshape({1, 1});
    draft_sampler_output.all_probs = torch::zeros({1, 4}, torch::kFloat32);
    components.fake_fast_topk_sampler->setInputs({draft_output.logits});
    components.fake_fast_topk_sampler->setOutputs({draft_sampler_output});

    auto* target_model = components.fake_target_model.get();
    auto* draft_model  = components.fake_draft_model.get();
    harness.setupFakeModels(components.executor.get(),
                            std::move(components.fake_target_model),
                            std::move(components.fake_draft_model),
                            std::move(components.fake_fast_topk_sampler),
                            std::move(components.fake_speculative_sampler),
                            std::move(components.fake_sampler));

    const auto status = components.executor->process(streams);
    require(status.ok(), "intermediate draft token abort returned an error: " + status.ToString());
    if (rank == 0) {
        require(stream->hasError(), "intermediate draft token error was not reported");
        require(stream->statusInfo().code() == ErrorCode::OUT_OF_VOCAB_RANGE,
                "intermediate draft token returned the wrong error code");
        require(stream->iterCount() == 0, "intermediate draft token abort changed the stream iteration count");
    }
    require(target_model->forwardCallCount() == 0, "target model ran after intermediate draft token error");
    require(draft_model->forwardCallCount() == 1, "intermediate draft token path ran the wrong number of forwards");
    require(components.executor->process({}).ok(),
            "empty decode process failed after intermediate draft token abort");
}

int runRankWithInterpreter(int rank, uint16_t port, AbortPath path) {
    try {
        PythonCollectiveEnvironment collective_environment(rank, port);
        initRuntime(rank, false, false, MlaOpsType::AUTO);

        MtpExecutorTestHarness harness;
        switch (path) {
            case AbortPath::PREFILL:
                runPrefillAbort(rank, harness);
                break;
            case AbortPath::DECODE:
                runDecodeAbort(rank, harness);
                break;
            case AbortPath::DECODE_LOCAL_CPU_HIDDEN:
                runDecodeLocalCpuHiddenAbort(rank, harness);
                break;
            case AbortPath::DECODE_INTERMEDIATE_DRAFT_TOKEN:
                runDecodeIntermediateDraftTokenAbort(rank, harness);
                break;
        }
        collective_environment.requireExecutorBroadcast();
        collective_environment.sentinelBroadcast(rank);
        if (::testing::Test::HasFailure()) {
            throw std::runtime_error("rank recorded an assertion failure");
        }
        return EXIT_SUCCESS;
    } catch (const py::error_already_set& e) {
        std::cerr << kTestFilter << " rank " << rank << " Python failure: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << kTestFilter << " rank " << rank << " failure: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << kTestFilter << " rank " << rank << " unknown failure" << std::endl;
    }
    return EXIT_FAILURE;
}

int runRank(int rank, uint16_t port, AbortPath path) {
    try {
        py::scoped_interpreter interpreter;
        return runRankWithInterpreter(rank, port, path);
    } catch (const std::exception& e) {
        std::cerr << kTestFilter << " rank " << rank << " interpreter failure: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << kTestFilter << " rank " << rank << " unknown interpreter failure" << std::endl;
    }
    return EXIT_FAILURE;
}

std::string selfExecutable() {
    std::array<char, PATH_MAX> executable_path{};
    const ssize_t size = readlink("/proc/self/exe", executable_path.data(), executable_path.size() - 1);
    if (size <= 0 || static_cast<size_t>(size) >= executable_path.size()) {
        throw std::runtime_error("failed to resolve distributed abort test executable");
    }
    executable_path[static_cast<size_t>(size)] = '\0';
    return executable_path.data();
}

[[noreturn]] void execRank(const std::string& executable, int rank, uint16_t port, AbortPath path) {
    const std::string rank_string = std::to_string(rank);
    const std::string port_string = std::to_string(port);
    const char* path_string = path == AbortPath::PREFILL ? "prefill" :
                              path == AbortPath::DECODE ? "decode" :
                              path == AbortPath::DECODE_LOCAL_CPU_HIDDEN ? "decode_local_cpu_hidden" :
                                                                          "decode_intermediate_draft_token";
    if (setenv(kRankEnvironment, rank_string.c_str(), 1) != 0
        || setenv(kPortEnvironment, port_string.c_str(), 1) != 0
        || setenv(kPathEnvironment, path_string, 1) != 0) {
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

void runWithWatchdog(AbortPath path) {
    const uint16_t                port       = reserveLoopbackPort();
    const std::string             executable = selfExecutable();
    std::array<pid_t, kWorldSize> children{-1, -1};
    for (int rank = 0; rank < kWorldSize; ++rank) {
        const pid_t child = fork();
        if (child == 0) {
            execRank(executable, rank, port, path);
        }
        if (child < 0) {
            terminateAndReap(children);
            FAIL() << "failed to fork distributed abort test rank " << rank;
        }
        children[rank] = child;
    }

    std::array<int, kWorldSize> statuses{};
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
                FAIL() << "waitpid failed for distributed abort test rank " << rank;
            }
        }
        if (finished_count < kWorldSize) {
            std::this_thread::sleep_for(kWatchdogPollPeriod);
        }
    }

    if (finished_count != kWorldSize) {
        terminateAndReap(children);
        FAIL() << "distributed abort regression exceeded " << kWatchdogTimeout.count() << " second watchdog";
    }

    for (int rank = 0; rank < kWorldSize; ++rank) {
        ASSERT_TRUE(WIFEXITED(statuses[rank]))
            << "rank " << rank << " terminated by signal " << WTERMSIG(statuses[rank]);
        EXPECT_EQ(WEXITSTATUS(statuses[rank]), EXIT_SUCCESS) << "rank " << rank << " failed";
    }
}

}  // namespace

class MtpExecutorDistributedAbortTest: public EngineBaseTest {};

class MtpExecutorDistributedAbortWorker: public ::testing::Test {};

TEST_F(MtpExecutorDistributedAbortWorker, runRank) {
    const char* rank_string = std::getenv(kRankEnvironment);
    const char* port_string = std::getenv(kPortEnvironment);
    const char* path_string = std::getenv(kPathEnvironment);
    ASSERT_NE(rank_string, nullptr);
    ASSERT_NE(port_string, nullptr);
    ASSERT_NE(path_string, nullptr);

    const int rank = std::stoi(rank_string);
    const int port = std::stoi(port_string);
    ASSERT_GE(rank, 0);
    ASSERT_LT(rank, kWorldSize);
    ASSERT_GT(port, 0);
    ASSERT_LE(port, std::numeric_limits<uint16_t>::max());

    AbortPath path;
    if (std::string(path_string) == "prefill") {
        path = AbortPath::PREFILL;
    } else if (std::string(path_string) == "decode") {
        path = AbortPath::DECODE;
    } else if (std::string(path_string) == "decode_local_cpu_hidden") {
        path = AbortPath::DECODE_LOCAL_CPU_HIDDEN;
    } else if (std::string(path_string) == "decode_intermediate_draft_token") {
        path = AbortPath::DECODE_INTERMEDIATE_DRAFT_TOKEN;
    } else {
        FAIL() << "invalid distributed abort worker path";
    }
    EXPECT_EQ(runRank(rank, static_cast<uint16_t>(port), path), EXIT_SUCCESS);
}

TEST_F(MtpExecutorDistributedAbortTest, prefillSamplerFailureKeepsRanksAligned) {
    runWithWatchdog(AbortPath::PREFILL);
}

TEST_F(MtpExecutorDistributedAbortTest, decodeSamplerFailureKeepsRanksAligned) {
    runWithWatchdog(AbortPath::DECODE);
}

TEST_F(MtpExecutorDistributedAbortTest, decodeLocalCpuHiddenFailureKeepsRanksAligned) {
    runWithWatchdog(AbortPath::DECODE_LOCAL_CPU_HIDDEN);
}

TEST_F(MtpExecutorDistributedAbortTest, decodeIntermediateDraftTokenFailureKeepsRanksAligned) {
    runWithWatchdog(AbortPath::DECODE_INTERMEDIATE_DRAFT_TOKEN);
}

}  // namespace rtp_llm
