#include "rtp_llm/cpp/distribute/CpuBroadcaster.h"
#include "rtp_llm/cpp/models/ModelTypes.h"

#include "gtest/gtest.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <errno.h>
#include <sys/wait.h>
#include <unistd.h>

namespace c10 {
namespace detail {

// Test-only compatibility for local CPU torch builds that expose old-ABI c10
// symbols while this repository is compiled with the default C++11 ABI.
__attribute__((weak)) void
torchCheckFail(const char* file, const char* func, unsigned int line, const std::string& msg) {
    throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + " " + func + ": " + msg);
}

__attribute__((weak)) void torchInternalAssertFail(
    const char* file, const char* func, unsigned int line, const char* cond, const std::string& msg) {
    throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + " " + func + " " + cond + ": " + msg);
}

}  // namespace detail
}  // namespace c10

namespace rtp_llm {

namespace {
thread_local int g_test_tp_rank = -1;
}  // namespace

void execBroadcast(const BroadcastParams& params) {
    auto& bcast = CpuBroadcaster::instance();
    for (auto& t : params.buffers) {
        auto host  = torch::empty({static_cast<int64_t>(t.nbytes())}, torch::kUInt8).pin_memory();
        auto bytes = torch::from_blob(
            t.data_ptr(), {static_cast<int64_t>(t.nbytes())}, torch::TensorOptions(torch::kUInt8).device(t.device()));
        if (g_test_tp_rank == params.root) {
            host.copy_(bytes);
        }
        bcast.broadcast(host.data_ptr(), host.nbytes(), params.root);
        if (g_test_tp_rank != params.root) {
            bytes.copy_(host);
        }
    }
}

void execBroadcastCpu(const BroadcastParams& params, bool allow_fallback) {
    (void)allow_fallback;
    auto& bcast = CpuBroadcaster::instance();
    for (auto& t : params.buffers) {
        auto contig = t.contiguous();
        bcast.broadcast(contig.data_ptr(), contig.nbytes(), params.root);
        if (!contig.is_same(t)) {
            t.copy_(contig);
        }
    }
}

namespace {

std::string makeTempBase() {
    std::string       pattern = "/tmp/tp_sync_model_inputs_test.XXXXXX";
    std::vector<char> buf(pattern.begin(), pattern.end());
    buf.push_back('\0');
    char* dir = ::mkdtemp(buf.data());
    if (dir == nullptr) {
        throw std::runtime_error(std::string("mkdtemp failed: ") + std::strerror(errno));
    }
    return std::string(dir) + "/bcast";
}

void cleanupTempBase(const std::string& base) {
    for (int rank = 0; rank <= 2; ++rank) {
        ::unlink((base + "_" + std::to_string(rank) + ".sock").c_str());
    }
    const auto slash = base.rfind('/');
    if (slash != std::string::npos) {
        ::rmdir(base.substr(0, slash).c_str());
    }
}

pid_t spawnChild(const std::function<int()>& fn) {
    pid_t pid = ::fork();
    if (pid == 0) {
        ::alarm(30);
        try {
            int rc = fn();
            std::fflush(stdout);
            std::fflush(stderr);
            ::_exit(rc);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "child exception: %s\n", e.what());
            std::fflush(stderr);
            ::_exit(1);
        } catch (...) {
            std::fprintf(stderr, "child unknown exception\n");
            std::fflush(stderr);
            ::_exit(1);
        }
    }
    return pid;
}

void expectChildrenOk(const std::vector<pid_t>& pids) {
    for (pid_t pid : pids) {
        ASSERT_GT(pid, 0);
        int   status = 0;
        pid_t waited = ::waitpid(pid, &status, 0);
        ASSERT_EQ(waited, pid);
        ASSERT_TRUE(WIFEXITED(status)) << "child " << pid << " terminated by signal";
        EXPECT_EQ(WEXITSTATUS(status), 0) << "child " << pid << " failed";
    }
}

template<typename T>
std::vector<T> tensorToVector(const torch::Tensor& tensor) {
    auto     contiguous = tensor.contiguous();
    const T* data       = contiguous.data_ptr<T>();
    return std::vector<T>(data, data + contiguous.numel());
}

template<typename T>
bool expectVectorEquals(const torch::Tensor& tensor, const std::vector<T>& expected, const char* name, int rank) {
    const auto actual = tensorToVector<T>(tensor);
    if (actual == expected) {
        return true;
    }
    std::fprintf(stderr, "rank %d got bad %s\n", rank, name);
    return false;
}

torch::Tensor int32Tensor(const std::vector<int32_t>& values) {
    return torch::tensor(values, torch::TensorOptions(torch::kInt32));
}

torch::Tensor int64Tensor(const std::vector<int64_t>& values) {
    return torch::tensor(values, torch::TensorOptions(torch::kInt64));
}

torch::Tensor boolTensor(const std::vector<bool>& values) {
    std::vector<uint8_t> raw(values.begin(), values.end());
    return torch::tensor(raw, torch::TensorOptions(torch::kBool));
}

GptModelInputs makeRootInputs() {
    GptModelInputs inputs;
    inputs.combo_tokens          = int32Tensor({11, 12, 13});
    inputs.input_lengths         = int32Tensor({2, 1});
    inputs.sequence_lengths      = int32Tensor({5, 6});
    inputs.prefix_lengths        = int32Tensor({0, 2});
    inputs.request_id            = int64Tensor({101, 202});
    inputs.request_pd_separation = boolTensor({false, true});
    inputs.lm_output_indexes     = int32Tensor({1, 2});
    inputs.lm_output_lengths     = int32Tensor({7, 8});
    inputs.combo_position_ids    = int32Tensor({0, 1, 2});
    inputs.text_tokens_mask      = int32Tensor({1, 1, 0});
    return inputs;
}

bool probeCapability(const std::function<void()>& fn) {
    pid_t pid = ::fork();
    if (pid == 0) {
        ::alarm(10);
        try {
            fn();
            ::_exit(0);
        } catch (...) { ::_exit(1); }
    }
    int status = 0;
    if (pid <= 0 || ::waitpid(pid, &status, 0) != pid) {
        return false;
    }
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

bool pinMemoryAvailable() {
    return probeCapability([] { (void)torch::empty({1}, torch::TensorOptions(torch::kUInt8)).pin_memory(); });
}

bool cudaAvailable() {
    return probeCapability([] {
        auto gpu = torch::empty({1}, torch::TensorOptions(torch::kUInt8).device(torch::kCUDA));
        gpu.fill_(1);
        (void)gpu.cpu();
    });
}

int smokeChild(int rank, const std::string& base, bool include_gpu) {
    g_test_tp_rank = rank;
    auto& bcast    = CpuBroadcaster::instance();
    bcast.reset();
    bcast.initialize(rank, 2, base);

    ParallelismConfig config;
    config.tp_size          = 2;
    config.tp_rank          = rank;
    config.world_size       = 2;
    config.world_rank       = rank;
    config.local_world_size = 2;
    config.local_rank       = rank;

    GptModelInputs inputs;
    if (rank == 0) {
        inputs = makeRootInputs();
        if (include_gpu) {
            inputs.last_hidden_states =
                torch::tensor({1.25f, 2.5f, 3.75f, 4.5f, 5.25f, 6.5f}, torch::TensorOptions(torch::kFloat32))
                    .reshape({3, 2})
                    .to(torch::kCUDA);
        }
    }
    tpSyncModelInputs(inputs, config);

    bool ok = true;
    ok &= expectVectorEquals<int32_t>(inputs.combo_tokens, {11, 12, 13}, "combo_tokens", rank);
    ok &= expectVectorEquals<int32_t>(inputs.input_lengths, {2, 1}, "input_lengths", rank);
    ok &= expectVectorEquals<int32_t>(inputs.sequence_lengths, {5, 6}, "sequence_lengths", rank);
    ok &= expectVectorEquals<int32_t>(inputs.prefix_lengths, {0, 2}, "prefix_lengths", rank);
    ok &= expectVectorEquals<int64_t>(inputs.request_id, {101, 202}, "request_id", rank);
    ok &= expectVectorEquals<bool>(inputs.request_pd_separation, {false, true}, "request_pd_separation", rank);
    ok &= expectVectorEquals<int32_t>(inputs.lm_output_indexes, {1, 2}, "lm_output_indexes", rank);
    ok &= expectVectorEquals<int32_t>(inputs.lm_output_lengths, {7, 8}, "lm_output_lengths", rank);
    ok &= expectVectorEquals<int32_t>(inputs.combo_position_ids, {0, 1, 2}, "combo_position_ids", rank);
    ok &= expectVectorEquals<int32_t>(inputs.text_tokens_mask, {1, 1, 0}, "text_tokens_mask", rank);
    if (include_gpu) {
        ok &= expectVectorEquals<float>(inputs.last_hidden_states.cpu().reshape({6}),
                                        {1.25f, 2.5f, 3.75f, 4.5f, 5.25f, 6.5f},
                                        "last_hidden_states",
                                        rank);
    }

    bcast.reset();
    return ok ? 0 : 1;
}

TEST(TpSyncModelInputsTest, MetadataSmokeTp2) {
    if (!pinMemoryAvailable()) {
        GTEST_SKIP() << "pin_memory is unavailable in this torch build";
    }
    const bool include_gpu = cudaAvailable();

    const std::string  base = makeTempBase();
    std::vector<pid_t> pids;
    pids.push_back(spawnChild([=] { return smokeChild(0, base, include_gpu); }));
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    pids.push_back(spawnChild([=] { return smokeChild(1, base, include_gpu); }));
    expectChildrenOk(pids);
    cleanupTempBase(base);
}

}  // namespace
}  // namespace rtp_llm
