#pragma once

#include <atomic>
#include <mutex>
#include <string>
#include "torch/csrc/autograd/profiler_kineto.h"

namespace rtp_llm {

namespace tpi = torch::profiler::impl;

// Low-level profiler wrapper around Kineto.
// IMPORTANT: start() and stop() MUST be called on the same thread (Kineto thread-affinity).
class TorchProfile {
public:
    TorchProfile(const std::string& prefix, std::string output_dir = "");
    ~TorchProfile();
    void start();
    void stop();

    TorchProfile(const TorchProfile&)            = delete;
    TorchProfile& operator=(const TorchProfile&) = delete;

private:
    std::string                 prefix_;
    std::string                 output_dir_;
    static std::atomic<size_t>  count_;
    tpi::ProfilerConfig         config_ = tpi::ProfilerConfig(tpi::ProfilerState::KINETO, /*report_input_shapes=*/true);
    std::set<tpi::ActivityType> activities_{tpi::ActivityType::CPU, tpi::ActivityType::CUDA};
    bool                        stopped_ = true;
};

// Step-window profiler controlled via API (sglang-style).
// Thread-safe: configure() is called from gRPC thread, tick() from engine loop thread.
// The actual TorchProfile start/stop happens only inside tick(), satisfying Kineto thread-affinity.
class StepWindowProfiler {
public:
    explicit StepWindowProfiler(const std::string& default_output_dir = "", int world_rank = 0);
    ~StepWindowProfiler();

    // Called from gRPC/API thread to configure a profiling session.
    void configure(bool enable, const std::string& trace_name, int start_step, int num_steps);

    // Called once per engine step() on the engine loop thread.
    // Handles start/stop of the underlying TorchProfile based on step counts.
    void tick();

    bool enabled() const {
        return enabled_.load(std::memory_order_relaxed);
    }

private:
    void stopProfiler(const char* reason);

    std::string default_output_dir_;

    // Atomic state set by configure(), read by tick()
    std::atomic<bool> enabled_{false};
    std::atomic<bool> reconfigure_{false};
    std::atomic<bool> has_profiler_{false};
    std::atomic<int>  start_step_{0};
    std::atomic<int>  num_steps_{0};

    // State managed exclusively on the engine loop thread (inside tick/shutdown)
    std::mutex                    mu_;
    std::string                   trace_name_;
    std::shared_ptr<TorchProfile> profiler_;
    int64_t                       waited_steps_   = 0;
    int64_t                       profiled_steps_ = 0;
    int                           world_rank_     = 0;
};

}  // namespace rtp_llm
