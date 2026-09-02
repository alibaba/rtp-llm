#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <thread>

#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "rtp_llm/cpp/engine_base/TorchProfiler.h"

namespace {

constexpr char kChildModeEnv[] = "RTP_LLM_CUPTI_TRAP_CHILD_MODE";
constexpr auto kChildTimeout   = std::chrono::seconds(30);

__global__ void warmupKernel() {}

__global__ void trapKernel() {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        __trap();
    }
}

std::filesystem::path testRoot() {
    if (const char* outputs = std::getenv("TEST_UNDECLARED_OUTPUTS_DIR")) {
        return std::filesystem::path(outputs) / "cupti_trap_coredump_repro";
    }
    const char* tmp = std::getenv("TEST_TMPDIR");
    return std::filesystem::path(tmp ? tmp : "/tmp") / "cupti_trap_coredump_repro";
}

bool hasNonEmptyCoredump(const std::filesystem::path& dir) {
    if (!std::filesystem::exists(dir)) {
        return false;
    }
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.path().filename().string().find("cuda_coredump_") == 0 && entry.file_size() > 0) {
            return true;
        }
    }
    return false;
}

int runChild(const std::string& mode) {
    const bool teardown = mode == "teardown";
    const auto out_dir  = testRoot() / mode;
    std::filesystem::create_directories(out_dir);

    std::fprintf(stderr, "CHILD mode=%s profiler_start\n", mode.c_str());
    rtp_llm::TorchProfile profiler("torch_profile_", out_dir.string());
    profiler.start();

    warmupKernel<<<1, 1>>>();
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "CHILD mode=%s warmup_failed\n", mode.c_str());
        return 2;
    }

    profiler.stop();
    std::fprintf(stderr, "CHILD mode=%s profiler_stopped\n", mode.c_str());

    if (teardown) {
        // Kineto tears CUPTI down asynchronously and finalizes from a CUDA API
        // exit callback. Repeated harmless API calls give that path a chance to
        // complete before the trap is launched.
        for (int i = 0; i < 100; ++i) {
            int device = -1;
            (void)cudaGetDevice(&device);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    std::fprintf(stderr, "CHILD mode=%s trap_launch\n", mode.c_str());
    trapKernel<<<1, 1>>>();
    const cudaError_t status = cudaDeviceSynchronize();
    std::fprintf(stderr, "CHILD mode=%s sync_returned=%s\n", mode.c_str(), cudaGetErrorString(status));
    return status == cudaSuccess ? 3 : 0;
}

struct ChildResult {
    bool timed_out = false;
    bool has_core  = false;
    int  status    = 0;
};

ChildResult runIsolated(const std::string& mode) {
    const auto out_dir = testRoot() / mode;
    std::filesystem::remove_all(out_dir);
    std::filesystem::create_directories(out_dir);

    const pid_t pid = fork();
    if (pid < 0) {
        ChildResult result;
        result.status = -1;
        return result;
    }
    if (pid == 0) {
        // Configure coredump before exec, so libcuda sees these values before
        // the child creates its first CUDA context.
        setenv(kChildModeEnv, mode.c_str(), 1);
        setenv("CUDA_ENABLE_COREDUMP_ON_EXCEPTION", "1", 1);
        setenv("CUDA_COREDUMP_SHOW_PROGRESS", "1", 1);
        // Match the production service. Keep kernel images and per-thread
        // memory so the control coredump remains useful in cuda-gdb.
        setenv("CUDA_COREDUMP_GENERATION_FLAGS", "skip_global_memory,skip_constbank_memory", 1);
        const std::string core_pattern = (out_dir / "cuda_coredump_%p").string();
        setenv("CUDA_COREDUMP_FILE", core_pattern.c_str(), 1);
        // Both settings materially change exception/timing behavior and would
        // confound whether the CUPTI subscriber itself blocks coredump progress.
        unsetenv("CUDA_DEVICE_WAITS_ON_EXCEPTION");
        unsetenv("CUDA_LAUNCH_BLOCKING");
        unsetenv("CUDA_ENABLE_CPU_COREDUMP_ON_EXCEPTION");
        unsetenv("CUDA_ENABLE_LIGHTWEIGHT_COREDUMP");
        unsetenv("CUDA_ENABLE_USER_TRIGGERED_COREDUMP");
        unsetenv("CUDA_COREDUMP_PIPE");

        if (mode == "teardown") {
            setenv("TEARDOWN_CUPTI", "1", 1);
            // Leave lazy re-init enabled: after teardown the subscriber stays
            // absent until another profiling session explicitly starts.
            unsetenv("DISABLE_CUPTI_LAZY_REINIT");
        } else {
            unsetenv("TEARDOWN_CUPTI");
            unsetenv("DISABLE_CUPTI_LAZY_REINIT");
        }

        execl("/proc/self/exe", "/proc/self/exe", static_cast<char*>(nullptr));
        _exit(127);
    }

    ChildResult result;
    const auto  deadline = std::chrono::steady_clock::now() + kChildTimeout;
    while (std::chrono::steady_clock::now() < deadline) {
        const pid_t waited = waitpid(pid, &result.status, WNOHANG);
        if (waited == pid) {
            result.has_core = hasNonEmptyCoredump(out_dir);
            return result;
        }
        if (waited == -1) {
            result.status   = -1;
            result.has_core = hasNonEmptyCoredump(out_dir);
            return result;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    result.timed_out = true;
    kill(pid, SIGKILL);
    waitpid(pid, &result.status, 0);
    result.has_core = hasNonEmptyCoredump(out_dir);
    return result;
}

void logResult(const char* mode, const ChildResult& result) {
    std::fprintf(stderr,
                 "PARENT mode=%s timed_out=%d has_core=%d wait_status=0x%x\\n",
                 mode,
                 result.timed_out,
                 result.has_core,
                 result.status);
}

TEST(CuptiTrapCoredumpReproTest, PersistentSubscriberBlocksTrapCoredump) {
    const ChildResult teardown = runIsolated("teardown");
    logResult("teardown", teardown);
    ASSERT_NE(teardown.status, -1) << "failed to start or wait for teardown child";
    ASSERT_FALSE(teardown.timed_out) << "CUPTI teardown control hung";
    ASSERT_TRUE(teardown.has_core) << "CUPTI teardown control did not generate a CUDA coredump";

    const ChildResult persistent = runIsolated("persistent");
    logResult("persistent", persistent);
    ASSERT_NE(persistent.status, -1) << "failed to start or wait for persistent child";
    EXPECT_TRUE(persistent.timed_out) << "persistent CUPTI subscriber did not reproduce the handler hang";
    EXPECT_FALSE(persistent.has_core) << "persistent CUPTI subscriber unexpectedly completed a CUDA coredump";
}

}  // namespace

int main(int argc, char** argv) {
    if (const char* mode = std::getenv(kChildModeEnv)) {
        return runChild(mode);
    }
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
