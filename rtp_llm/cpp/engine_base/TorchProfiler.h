#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include "rtp_llm/cpp/engine_base/ProfileStatus.h"
#include "torch/csrc/autograd/profiler_kineto.h"
#include "rtp_llm/cpp/utils/AsyncProfilingScope.h"

namespace rtp_llm {

namespace tpi = torch::profiler::impl;

struct ProfilingDebugLoggingConfig;

namespace detail {

using DisableProfilerFn = std::unique_ptr<torch::autograd::profiler::ProfilerResult> (*)();

// PyTorch profiler finalization can reject malformed backend traces. Profiling
// is diagnostic-only, so a bad trace must not terminate the inference engine.
std::unique_ptr<torch::autograd::profiler::ProfilerResult>
tryDisableProfiler(DisableProfilerFn disable_profiler, const std::function<void()>& drain_device = {});

}  // namespace detail

// Keep custom records outside Kineto's thread-local event tree. Both synchronous
// and background exports use the same append-only writer.
struct CollectedProfile {
    std::unique_ptr<torch::autograd::profiler::ProfilerResult> kineto;
    std::vector<AsyncProfileEvent>                             async_events;

    const auto& events() const {
        return kineto->events();
    }
    void save(const std::string& path);
};

// Low-level profiler wrapper around Kineto.
// IMPORTANT: start() and stop() MUST be called on the same thread (Kineto thread-affinity).
class TorchProfile {
public:
    TorchProfile(const std::string& prefix, std::string output_dir = "");
    ~TorchProfile();
    void start(std::function<void()> before_enable = {});

    // Stops profiling and returns the result + filename for async saving.
    // Returns {nullptr, ""} if already stopped.
    std::pair<std::unique_ptr<CollectedProfile>, std::string> stopAndCollect();

    // Legacy synchronous stop (calls stopAndCollect + save inline).
    void stop();

    TorchProfile(const TorchProfile&)            = delete;
    TorchProfile& operator=(const TorchProfile&) = delete;

private:
    friend struct TorchProfileTestPeer;

    std::string                          prefix_;
    std::string                          output_dir_;
    static std::atomic<size_t>           count_;
    tpi::ProfilerConfig                  config_ = tpi::ProfilerConfig(tpi::ProfilerState::KINETO, /*report_input_shapes=*/true);
    std::set<tpi::ActivityType>          activities_{tpi::ActivityType::CPU, tpi::ActivityType::CUDA};
    std::thread::id                      owner_thread_;
    bool                                 stopped_        = true;
    int                                  profile_device_ = -1;
    std::shared_ptr<AsyncProfileSession> async_profile_;
};

// Background thread that serializes profiler results to disk without blocking the engine loop.
class ProfilerSaveWorker {
public:
    ProfilerSaveWorker();
    ~ProfilerSaveWorker();

    // Enqueue a save task. Non-blocking, returns immediately.
    void enqueue(std::unique_ptr<CollectedProfile>       result,
                 std::string                             file_name,
                 std::function<void(const std::string&)> completed = {});

    ProfilerSaveWorker(const ProfilerSaveWorker&)            = delete;
    ProfilerSaveWorker& operator=(const ProfilerSaveWorker&) = delete;

private:
    void run();

    struct SaveTask {
        std::unique_ptr<CollectedProfile>       result;
        std::string                             file_name;
        std::function<void(const std::string&)> completed;
    };

    std::mutex              mu_;
    std::condition_variable cv_;
    std::queue<SaveTask>    tasks_;
    bool                    stop_ = false;
    std::thread             thread_;
};

// Step-window profiler controlled via API (sglang-style).
// Collection starts in engine step callbacks and stops on that same thread,
// including idle/shutdown serviceControl boundaries.
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
    // Request only; never disables Kineto on the caller's thread.
    bool requestStop(const std::string& trace_name);
    // Owner-thread only. Does not advance the actual execution-step window.
    void          serviceControl();
    ProfileStatus status() const;
    // Drain Kineto on its owner. Preserve an untouched startup-configured window.
    void finishOnWorkerExit();

    // Convenience overload: enable service-wide timeline profiling from config.
    // No-op if cfg.gen_timeline_sync is false.
    void configureFromConfig(const ProfilingDebugLoggingConfig& cfg);

    // Returns an RAII scope that wraps a single engine step.
    StepScope stepScope() {
        return StepScope(*this);
    }

    // Called around an actual model-forward step. Unlike tick(), these hooks
    // do not consume profiling windows on idle TP synchronization steps.
    void startStep();
    void finishStep();

    bool enabled() const {
        return enabled_.load(std::memory_order_relaxed);
    }

private:
    friend class StepScope;
    friend struct TorchProfileTestPeer;

    void arm(const std::string& trace_name, int start_step, int num_steps, bool from_config);
    void startStep(const std::function<void()>& before_prepare);
    void beginStep();
    void endStep();
    void stopProfiler(const char* reason);

    std::string default_output_dir_;

    // Atomic state set by configure(), read by beginStep()/endStep()
    std::atomic<bool> enabled_{false};
    std::atomic<bool> has_profiler_{false};
    std::atomic<int>  start_step_{0};
    std::atomic<int>  num_steps_{0};

    // Session/status state is mutex-protected; Kineto operations remain owner-thread only.
    mutable std::mutex            mu_;
    std::string                   trace_name_;
    std::string                   last_finished_trace_;
    std::string                   last_cancelled_trace_;
    ProfileStatus                 status_;
    bool                          from_config_ = false;
    std::thread::id               owner_thread_;
    std::shared_ptr<TorchProfile> profiler_;
    int64_t                       waited_steps_   = 0;
    int64_t                       profiled_steps_ = 0;
    int                           world_rank_     = 0;
    ProfilerSaveWorker            save_worker_;
};

}  // namespace rtp_llm
