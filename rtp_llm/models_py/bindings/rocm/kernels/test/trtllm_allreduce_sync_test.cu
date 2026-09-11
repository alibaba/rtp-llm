#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>

#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "rtp_llm/models_py/bindings/rocm/kernels/trtllm_allreduce_sync.cuh"

namespace {

constexpr int kWorldSize       = 2;
constexpr int kBlocks          = 4;
constexpr int kThreadsPerBlock = 64;
constexpr int kReplays         = 2;
constexpr int kChildTimeoutSec = 20;
constexpr int kSkipExitCode    = 77;

class HipError: public std::runtime_error {
public:
    explicit HipError(const std::string& message): std::runtime_error(message) {}
};

void hipCheck(hipError_t status, const char* expression, const char* file, int line) {
    if (status != hipSuccess) {
        throw HipError(std::string(file) + ":" + std::to_string(line) + " " + expression + ": "
                       + hipGetErrorString(status));
    }
}

#define HIP_CHECK(expression) hipCheck((expression), #expression, __FILE__, __LINE__)

enum class ProbePath {
    kOneStage,
    kTwoStage,
};

__global__ void singleSyncProbe(rtp_llm::CommDeviceMeta<kWorldSize> meta, rtp_llm::SyncEpoch* observed) {
    rtp_llm::SyncComm<kWorldSize> comm(meta);
    comm.sync();
    if (threadIdx.x == 0) {
        observed[blockIdx.x] = comm.flag;
    }
}

template<ProbePath Path>
__global__ void syncProbe(rtp_llm::CommDeviceMeta<kWorldSize> meta, rtp_llm::SyncEpoch* observed) {
    rtp_llm::SyncComm<kWorldSize> comm(meta);
    if constexpr (Path == ProbePath::kOneStage) {
        // Both 1-stage production kernels use this entry/exit sequence.
        comm.sync();
        comm.sync();
    } else {
        // Both 2-stage production kernels use this sequence: relaxed/non-final,
        // acquire-release/final, then the workspace-protection exit barrier.
        comm.template sync<true, false>();
        comm.template sync<false, true>();
        comm.sync();
    }
    if (threadIdx.x == 0) {
        observed[blockIdx.x] = comm.flag;
    }
}

struct DeviceBuffers {
    rtp_llm::SyncEpoch* clock      = nullptr;
    rtp_llm::SyncEpoch* flags      = nullptr;
    rtp_llm::SyncEpoch* observed   = nullptr;
    hipStream_t         stream     = nullptr;
    hipGraph_t          graph      = nullptr;
    hipGraphExec_t      graph_exec = nullptr;
};

class ScenarioSkip: public std::runtime_error {
public:
    explicit ScenarioSkip(const std::string& message): std::runtime_error(message) {}
};

void destroyBuffers(std::array<DeviceBuffers, kWorldSize>& buffers) {
    for (int rank = 0; rank < kWorldSize; ++rank) {
        hipSetDevice(rank);
        if (buffers[rank].graph_exec != nullptr) {
            hipGraphExecDestroy(buffers[rank].graph_exec);
        }
        if (buffers[rank].graph != nullptr) {
            hipGraphDestroy(buffers[rank].graph);
        }
        if (buffers[rank].stream != nullptr) {
            hipStreamDestroy(buffers[rank].stream);
        }
        if (buffers[rank].observed != nullptr) {
            hipFree(buffers[rank].observed);
        }
        if (buffers[rank].flags != nullptr) {
            hipFree(buffers[rank].flags);
        }
        if (buffers[rank].clock != nullptr) {
            hipFree(buffers[rank].clock);
        }
    }
}

void enablePeerAccess() {
    for (int device = 0; device < kWorldSize; ++device) {
        const int peer = 1 - device;
        int       can_access_peer{};
        HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, device, peer));
        if (!can_access_peer) {
            throw ScenarioSkip("the selected GPUs do not support peer access");
        }
        HIP_CHECK(hipSetDevice(device));
        const hipError_t status = hipDeviceEnablePeerAccess(peer, 0);
        if (status != hipSuccess && status != hipErrorPeerAccessAlreadyEnabled) {
            hipCheck(status, "hipDeviceEnablePeerAccess", __FILE__, __LINE__);
        }
    }
}

template<ProbePath Path>
void launchProbe(const rtp_llm::CommDeviceMeta<kWorldSize>& meta, DeviceBuffers& buffers) {
    syncProbe<Path><<<kBlocks, kThreadsPerBlock, 0, buffers.stream>>>(meta, buffers.observed);
    HIP_CHECK(hipGetLastError());
}

void launchSingleProbe(const rtp_llm::CommDeviceMeta<kWorldSize>& meta, DeviceBuffers& buffers) {
    singleSyncProbe<<<1, kThreadsPerBlock, 0, buffers.stream>>>(meta, buffers.observed);
    HIP_CHECK(hipGetLastError());
}

void allocateBuffers(std::array<DeviceBuffers, kWorldSize>& buffers, rtp_llm::SyncEpoch seed) {
    int device_count{};
    HIP_CHECK(hipGetDeviceCount(&device_count));
    if (device_count < kWorldSize) {
        throw ScenarioSkip("requires two visible ROCm devices");
    }
    enablePeerAccess();
    const std::array<rtp_llm::SyncEpoch, kBlocks>              clocks{seed, seed, seed, seed};
    const std::array<rtp_llm::SyncEpoch, kBlocks * kWorldSize> flags{
        seed,
        seed,
        seed,
        seed,
        seed,
        seed,
        seed,
        seed,
    };
    for (int rank = 0; rank < kWorldSize; ++rank) {
        HIP_CHECK(hipSetDevice(rank));
        HIP_CHECK(hipMalloc(&buffers[rank].clock, sizeof(clocks)));
        HIP_CHECK(hipMalloc(&buffers[rank].flags, sizeof(flags)));
        HIP_CHECK(hipMalloc(&buffers[rank].observed, sizeof(clocks)));
        HIP_CHECK(hipMemcpy(buffers[rank].clock, clocks.data(), sizeof(clocks), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(buffers[rank].flags, flags.data(), sizeof(flags), hipMemcpyHostToDevice));
        HIP_CHECK(hipStreamCreate(&buffers[rank].stream));
    }
}

std::array<rtp_llm::CommDeviceMeta<kWorldSize>, kWorldSize>
makeMetas(const std::array<DeviceBuffers, kWorldSize>& buffers) {
    std::array<rtp_llm::CommDeviceMeta<kWorldSize>, kWorldSize> metas{};
    for (int rank = 0; rank < kWorldSize; ++rank) {
        metas[rank].barrier_flag_ptrs[0] = buffers[0].flags;
        metas[rank].barrier_flag_ptrs[1] = buffers[1].flags;
        metas[rank].sync_clock           = buffers[rank].clock;
        metas[rank].rank                 = rank;
        metas[rank].nranks               = kWorldSize;
    }
    return metas;
}

template<ProbePath Path>
void runScenario(rtp_llm::SyncEpoch seed, bool graph_capture, int rank_one_delay_ms) {
    std::array<DeviceBuffers, kWorldSize> buffers{};
    try {
        allocateBuffers(buffers, seed);
        const auto metas = makeMetas(buffers);

        // Enqueue both ranks before synchronizing either stream. Rank 1 is delayed
        // to permit scheduling skew without concurrent host-side graph capture.
        if (graph_capture) {
            for (int rank = 0; rank < kWorldSize; ++rank) {
                HIP_CHECK(hipSetDevice(rank));
                HIP_CHECK(hipStreamBeginCapture(buffers[rank].stream, hipStreamCaptureModeGlobal));
                launchProbe<Path>(metas[rank], buffers[rank]);
                HIP_CHECK(hipStreamEndCapture(buffers[rank].stream, &buffers[rank].graph));
                HIP_CHECK(hipGraphInstantiate(&buffers[rank].graph_exec, buffers[rank].graph, nullptr, nullptr, 0));
            }
            for (int replay = 0; replay < kReplays; ++replay) {
                for (int rank = 0; rank < kWorldSize; ++rank) {
                    HIP_CHECK(hipSetDevice(rank));
                    if (rank == 1 && rank_one_delay_ms > 0) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(rank_one_delay_ms));
                    }
                    HIP_CHECK(hipGraphLaunch(buffers[rank].graph_exec, buffers[rank].stream));
                }
            }
        } else {
            for (int replay = 0; replay < kReplays; ++replay) {
                for (int rank = 0; rank < kWorldSize; ++rank) {
                    HIP_CHECK(hipSetDevice(rank));
                    if (rank == 1 && rank_one_delay_ms > 0) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(rank_one_delay_ms));
                    }
                    launchProbe<Path>(metas[rank], buffers[rank]);
                }
            }
        }
        for (int rank = 0; rank < kWorldSize; ++rank) {
            HIP_CHECK(hipSetDevice(rank));
            HIP_CHECK(hipStreamSynchronize(buffers[rank].stream));
        }

        constexpr int            kSyncCallsPerReplay = Path == ProbePath::kOneStage ? 2 : 3;
        const rtp_llm::SyncEpoch expected            = seed + kReplays * kSyncCallsPerReplay;
        for (int rank = 0; rank < kWorldSize; ++rank) {
            std::array<rtp_llm::SyncEpoch, kBlocks>              received_clocks{};
            std::array<rtp_llm::SyncEpoch, kBlocks>              received_observed{};
            std::array<rtp_llm::SyncEpoch, kBlocks * kWorldSize> received_flags{};
            HIP_CHECK(hipSetDevice(rank));
            HIP_CHECK(
                hipMemcpy(received_clocks.data(), buffers[rank].clock, sizeof(received_clocks), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(
                received_observed.data(), buffers[rank].observed, sizeof(received_observed), hipMemcpyDeviceToHost));
            HIP_CHECK(
                hipMemcpy(received_flags.data(), buffers[rank].flags, sizeof(received_flags), hipMemcpyDeviceToHost));
            for (rtp_llm::SyncEpoch value : received_clocks) {
                if (value != expected) {
                    throw std::runtime_error("sync clock did not advance through the expected modulo-2^32 epoch");
                }
            }
            for (rtp_llm::SyncEpoch value : received_observed) {
                if (value != expected) {
                    throw std::runtime_error("probe observed an unexpected final epoch");
                }
            }
            for (rtp_llm::SyncEpoch value : received_flags) {
                if (value != expected) {
                    throw std::runtime_error("barrier flag did not advance through the expected modulo-2^32 epoch");
                }
            }
        }
    } catch (...) {
        destroyBuffers(buffers);
        throw;
    }
    destroyBuffers(buffers);
}

void runStaleEpochGate() {
    constexpr rtp_llm::SyncEpoch          kSeed = 0xffffffffu;
    std::array<DeviceBuffers, kWorldSize> buffers{};
    hipEvent_t                            rank_zero_done = nullptr;
    try {
        allocateBuffers(buffers, kSeed);
        const auto metas = makeMetas(buffers);

        HIP_CHECK(hipSetDevice(0));
        HIP_CHECK(hipEventCreate(&rank_zero_done));
        launchSingleProbe(metas[0], buffers[0]);
        HIP_CHECK(hipEventRecord(rank_zero_done, buffers[0].stream));

        // rank 0 stores the expected epoch (0) into rank 1's local slot before
        // polling rank 0's still-stale slot. This copy does not synchronize rank
        // 0's stream, so it distinguishes a stale-aware wait from unsigned '<'.
        rtp_llm::SyncEpoch published        = kSeed;
        const auto         publish_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (published != rtp_llm::SyncEpoch{0} && std::chrono::steady_clock::now() < publish_deadline) {
            HIP_CHECK(hipSetDevice(1));
            HIP_CHECK(hipMemcpy(&published, buffers[1].flags, sizeof(published), hipMemcpyDeviceToHost));
            if (published != rtp_llm::SyncEpoch{0}) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }

        bool        rank_zero_was_blocked = false;
        std::string gate_error;
        if (published != rtp_llm::SyncEpoch{0}) {
            gate_error = "rank 0 did not publish epoch zero before the stale-gate deadline";
        } else {
            HIP_CHECK(hipSetDevice(0));
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            const hipError_t event_status = hipEventQuery(rank_zero_done);
            rank_zero_was_blocked         = event_status == hipErrorNotReady;
            if (!rank_zero_was_blocked) {
                gate_error = event_status == hipSuccess ?
                                 "rank 0 completed while rank 1 still exposed the stale UINT_MAX flag" :
                                 std::string("hipEventQuery failed: ") + hipGetErrorString(event_status);
            }
        }

        // Always release rank 0 before reporting the stale-gate result. Otherwise
        // cleanup could block behind the deliberately stalled peer kernel.
        HIP_CHECK(hipSetDevice(1));
        launchSingleProbe(metas[1], buffers[1]);
        HIP_CHECK(hipSetDevice(0));
        HIP_CHECK(hipStreamSynchronize(buffers[0].stream));
        HIP_CHECK(hipSetDevice(1));
        HIP_CHECK(hipStreamSynchronize(buffers[1].stream));

        rtp_llm::SyncEpoch rank_zero_clock{};
        rtp_llm::SyncEpoch rank_one_clock{};
        rtp_llm::SyncEpoch rank_zero_flag{};
        rtp_llm::SyncEpoch rank_one_flag{};
        HIP_CHECK(hipSetDevice(0));
        HIP_CHECK(hipMemcpy(&rank_zero_clock, buffers[0].clock, sizeof(rank_zero_clock), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(&rank_one_flag, buffers[0].flags + 1, sizeof(rank_one_flag), hipMemcpyDeviceToHost));
        HIP_CHECK(hipSetDevice(1));
        HIP_CHECK(hipMemcpy(&rank_one_clock, buffers[1].clock, sizeof(rank_one_clock), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(&rank_zero_flag, buffers[1].flags, sizeof(rank_zero_flag), hipMemcpyDeviceToHost));

        if (!rank_zero_was_blocked) {
            throw std::runtime_error(gate_error);
        }
        if (rank_zero_clock != rtp_llm::SyncEpoch{0} || rank_one_clock != rtp_llm::SyncEpoch{0}
            || rank_zero_flag != rtp_llm::SyncEpoch{0} || rank_one_flag != rtp_llm::SyncEpoch{0}) {
            throw std::runtime_error("the two ranks did not complete epoch zero after the peer was released");
        }
    } catch (...) {
        if (rank_zero_done != nullptr) {
            hipEventDestroy(rank_zero_done);
        }
        destroyBuffers(buffers);
        throw;
    }
    HIP_CHECK(hipEventDestroy(rank_zero_done));
    destroyBuffers(buffers);
}

template<typename Scenario>
int runInChild(Scenario&& scenario) {
    const pid_t child = fork();
    if (child < 0) {
        return EXIT_FAILURE;
    }
    if (child == 0) {
        try {
            scenario();
            std::_Exit(EXIT_SUCCESS);
        } catch (const ScenarioSkip& error) {
            std::fprintf(stderr, "SKIP: %s\n", error.what());
            std::_Exit(kSkipExitCode);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "FAIL: %s\n", error.what());
            std::_Exit(EXIT_FAILURE);
        }
    }

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(kChildTimeoutSec);
    int        status{};
    while (std::chrono::steady_clock::now() < deadline) {
        const pid_t result = waitpid(child, &status, WNOHANG);
        if (result == child) {
            if (WIFEXITED(status)) {
                return WEXITSTATUS(status);
            }
            return EXIT_FAILURE;
        }
        if (result < 0) {
            return EXIT_FAILURE;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    std::fprintf(stderr, "FAIL: GPU sync child exceeded %d seconds; terminating it\n", kChildTimeoutSec);
    kill(child, SIGKILL);
    waitpid(child, &status, 0);
    return EXIT_FAILURE;
}

class TrtllmAllReduceSyncTest: public ::testing::TestWithParam<std::tuple<ProbePath, bool, rtp_llm::SyncEpoch>> {};

TEST(SyncEpochReachedTest, AcceptsEqualAndBoundedAheadEpochs) {
    using rtp_llm::SyncEpoch;
    EXPECT_TRUE(rtp_llm::details::syncEpochReached(SyncEpoch{0}, SyncEpoch{0}));
    EXPECT_TRUE(rtp_llm::details::syncEpochReached(SyncEpoch{0}, SyncEpoch{0xffffffffu}));
    EXPECT_TRUE(rtp_llm::details::syncEpochReached(SyncEpoch{0x80000000u}, SyncEpoch{0x7fffffffu}));
    EXPECT_FALSE(rtp_llm::details::syncEpochReached(SyncEpoch{0xffffffffu}, SyncEpoch{0}));
    EXPECT_FALSE(rtp_llm::details::syncEpochReached(SyncEpoch{0x7fffffffu}, SyncEpoch{0x80000000u}));
}

TEST(TrtllmAllReduceSyncTest, RejectsStaleUintMaxUntilPeerPublishesEpochZero) {
    const int exit_code = runInChild([] { runStaleEpochGate(); });
    if (exit_code == kSkipExitCode) {
        GTEST_SKIP() << "two ROCm devices with P2P access are required";
    }
    EXPECT_EQ(exit_code, EXIT_SUCCESS);
}

TEST_P(TrtllmAllReduceSyncTest, AdvancesEqualClocksAndFlagsAcrossCounterBoundaries) {
    const auto [path, graph_capture, seed] = GetParam();
    const int exit_code                    = path == ProbePath::kOneStage ?
                                                 runInChild([&] { runScenario<ProbePath::kOneStage>(seed, graph_capture, 10); }) :
                                                 runInChild([&] { runScenario<ProbePath::kTwoStage>(seed, graph_capture, 10); });
    if (exit_code == kSkipExitCode) {
        GTEST_SKIP() << "two ROCm devices with P2P access are required";
    }
    EXPECT_EQ(exit_code, EXIT_SUCCESS);
}

std::string parameterName(const ::testing::TestParamInfo<TrtllmAllReduceSyncTest::ParamType>& info) {
    const auto [path, graph_capture, seed] = info.param;
    const char* path_name                  = path == ProbePath::kOneStage ? "OneStage" : "TwoStage";
    const char* graph_name                 = graph_capture ? "Graph" : "Eager";
    const char* epoch_name = seed == rtp_llm::SyncEpoch{0x7ffffffd} ? "IntMaxMinusTwo" : "UintMaxMinusTwo";
    return std::string(path_name) + graph_name + epoch_name;
}

INSTANTIATE_TEST_SUITE_P(AllReduceKernelSyncSequences,
                         TrtllmAllReduceSyncTest,
                         ::testing::Combine(::testing::Values(ProbePath::kOneStage, ProbePath::kTwoStage),
                                            ::testing::Bool(),
                                            ::testing::Values(rtp_llm::SyncEpoch{0x7ffffffd},
                                                              rtp_llm::SyncEpoch{0xfffffffdu})),
                         parameterName);

}  // namespace
