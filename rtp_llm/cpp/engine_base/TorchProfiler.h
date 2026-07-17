#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include "torch/csrc/autograd/profiler_kineto.h"

namespace rtp_llm {

namespace tpi = torch::profiler::impl;

struct ProfilingDebugLoggingConfig;

// Low-level profiler wrapper around Kineto.
// IMPORTANT: start() and stop() MUST be called on the same thread (Kineto thread-affinity).
class TorchProfile {
public:
    TorchProfile(const std::string& prefix, std::string output_dir = "");
    ~TorchProfile();
    void start();

    // Stops profiling and returns the result + filename for async saving.
    // Returns {nullptr, ""} if already stopped.
    std::pair<std::unique_ptr<torch::autograd::profiler::ProfilerResult>, std::string> stopAndCollect();

    // Legacy synchronous stop (calls stopAndCollect + save inline).
    void stop();

    TorchProfile(const TorchProfile&)            = delete;
    TorchProfile& operator=(const TorchProfile&) = delete;

private:
    std::string                 prefix_;
    std::string                 output_dir_;
    static std::atomic<size_t>  count_;
    tpi::ProfilerConfig         config_ = tpi::ProfilerConfig(tpi::ProfilerState::KINETO, /*report_input_shapes=*/true);
#if USING_XPU
    std::set<tpi::ActivityType> activities_{tpi::ActivityType::CPU, tpi::ActivityType::XPU};
#else
    std::set<tpi::ActivityType> activities_{tpi::ActivityType::CPU, tpi::ActivityType::CUDA};
#endif
    bool                        stopped_ = true;
};

// Background thread that serializes profiler results to disk without blocking the engine loop.
class ProfilerSaveWorker {
public:
    ProfilerSaveWorker();
    ~ProfilerSaveWorker();

    // Enqueue a save task. Non-blocking, returns immediately.
    void enqueue(std::unique_ptr<torch::autograd::profiler::ProfilerResult> result, std::string file_name);

    ProfilerSaveWorker(const ProfilerSaveWorker&)            = delete;
    ProfilerSaveWorker& operator=(const ProfilerSaveWorker&) = delete;

private:
    void run();

    struct SaveTask {
        std::unique_ptr<torch::autograd::profiler::ProfilerResult> result;
        std::string                                                file_name;
    };

    std::mutex              mu_;
    std::condition_variable cv_;
    std::queue<SaveTask>    tasks_;
    bool                    stop_ = false;
    std::thread             thread_;
};

// Step-window profiler controlled via API (sglang-style).
// Thread-safe: configure() is called from gRPC thread, stepScope() from engine loop thread.
// The actual TorchProfile start/stop happens only inside engine step callbacks, satisfying Kineto thread-affinity.
// Trace export is done asynchronously on a background thread to avoid blocking inference.
class StepWindowProfiler {
public:
    // RAII guard that brackets a single engine step. Construct with stepScope() before
    // process(), destruction after process() advances the profiler state machine.
    // Strictly scope-bound: not movable, not copyable.
    class StepScope {
    public:
        explicit StepScope(StepWindowProfiler& profiler);
        ~StepScope();
        StepScope(const StepScope&)            = delete;
        StepScope& operator=(const StepScope&) = delete;
        StepScope(StepScope&&)                 = delete;
        StepScope& operator=(StepScope&&)      = delete;

    private:
        StepWindowProfiler& profiler_;
    };

    explicit StepWindowProfiler(const std::string& default_output_dir = "", int world_rank = 0);
    ~StepWindowProfiler();

    // Configure a profiling session. Safe to call from any thread (sets atomic state).
    // The actual profiler start happens inside the next stepScope() (on the engine loop
    // thread), satisfying Kineto thread-affinity. If a session is already active, this is
    // a no-op (first-come-first-served).
    void configure(bool enable, const std::string& trace_name, int start_step, int num_steps);

    // Convenience overload: enable service-wide timeline profiling from config.
    // No-op if cfg.gen_timeline_sync is false.
    void configureFromConfig(const ProfilingDebugLoggingConfig& cfg);

    // Returns an RAII scope that wraps a single engine step.
    StepScope stepScope() {
        return StepScope(*this);
    }

    bool enabled() const {
        return enabled_.load(std::memory_order_relaxed);
    }

private:
    friend class StepScope;

    void beginStep();
    void endStep();
    void stopProfiler(const char* reason);

    std::string default_output_dir_;

    // Atomic state set by configure(), read by beginStep()/endStep()
    std::atomic<bool> enabled_{false};
    std::atomic<bool> reconfigure_{false};
    std::atomic<bool> has_profiler_{false};
    std::atomic<int>  start_step_{0};
    std::atomic<int>  num_steps_{0};

    // State managed exclusively on the engine loop thread (inside begin/end/shutdown)
    std::mutex                    mu_;
    std::string                   trace_name_;
    std::shared_ptr<TorchProfile> profiler_;
    int64_t                       waited_steps_   = 0;
    int64_t                       profiled_steps_ = 0;
    int                           world_rank_     = 0;
    ProfilerSaveWorker            save_worker_;
};

}  // namespace rtp_llm
