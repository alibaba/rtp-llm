#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

namespace rtp_llm {

// A node-local, shutdown-only rendezvous. Running steps use only a local
// counter; they add no collective, file lock or device synchronization.
class EngineLoopCoordinator {
public:
    EngineLoopCoordinator(std::string               path,
                          int                       rank,
                          int                       size,
                          std::chrono::milliseconds timeout = std::chrono::seconds(120));
    ~EngineLoopCoordinator();
    EngineLoopCoordinator(const EngineLoopCoordinator&)            = delete;
    EngineLoopCoordinator& operator=(const EngineLoopCoordinator&) = delete;

    static std::unique_ptr<EngineLoopCoordinator> createFromEnvironment(int rank, int size, int local_size);
    static bool supportsLocalGroup(int world_size, int local_size, int tp_size, int ep_size);

    void     requestStop();
    void     waitForStopReady(std::chrono::milliseconds drain_timeout);
    bool     beforeStep(const std::atomic<bool>& running);
    void     afterStep();
    void     waitForGpuDrain();
    uint64_t completedSteps() const {
        return completed_;
    }

private:
    struct SharedState;
    class Guard;
    void checkDeadline(const std::chrono::steady_clock::time_point& deadline, const char* phase) const;

    std::string               path_;
    int                       rank_;
    int                       size_;
    int                       fd_    = -1;
    SharedState*              state_ = nullptr;
    std::mutex                mutex_;
    std::chrono::milliseconds timeout_;
    std::atomic<uint64_t>     started_{0};
    uint64_t                  completed_   = 0;
    bool                      stopping_    = false;
    bool                      gpu_drained_ = false;
};

}  // namespace rtp_llm
