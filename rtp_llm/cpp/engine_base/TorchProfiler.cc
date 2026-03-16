#include "rtp_llm/cpp/engine_base/TorchProfiler.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "autil/TimeUtility.h"
#include <string>

namespace rtp_llm {
namespace tap = torch::autograd::profiler;

// ---- TorchProfile ----

std::atomic<size_t> TorchProfile::count_{0};

TorchProfile::TorchProfile(const std::string& prefix, std::string output_dir):
    prefix_(prefix), output_dir_(output_dir.empty() ? "." : std::move(output_dir)) {}

TorchProfile::~TorchProfile() {
    if (!stopped_) {
        stop();
    }
}

void TorchProfile::start() {
    count_ += 1;
    stopped_ = false;
    tap::prepareProfiler(config_, activities_);
    tap::enableProfiler(config_, activities_);
}

void TorchProfile::stop() {
    if (stopped_) {
        return;
    }
    auto        res       = tap::disableProfiler();
    std::string file_name = output_dir_ + "/" + prefix_ + std::to_string(count_) + ".json";
    res->save(file_name);
    stopped_ = true;
}

// ---- StepWindowProfiler ----

StepWindowProfiler::StepWindowProfiler(const std::string& default_output_dir, int world_rank):
    default_output_dir_(default_output_dir.empty() ? "." : default_output_dir), world_rank_(world_rank) {}

void StepWindowProfiler::configure(bool enable, const std::string& trace_name, int start_step, int num_steps) {
    // First-come-first-served: if a profiling session is already active, ignore new requests
    // to prevent concurrent requests from repeatedly restarting the profiler.
    if (enable && enabled_.load(std::memory_order_relaxed)) {
        RTP_LLM_LOG_INFO("timeline profiling already active, ignoring new configure request");
        return;
    }
    {
        std::lock_guard<std::mutex> lock(mu_);
        trace_name_ = trace_name;
    }
    start_step_.store(std::max(0, start_step));
    num_steps_.store(std::max(0, num_steps));
    enabled_.store(enable);
    reconfigure_.store(true);
    RTP_LLM_LOG_INFO("timeline profiling configured: enable=%d start_step=%d num_steps=%d trace=%s",
                     int(enable),
                     std::max(0, start_step),
                     std::max(0, num_steps),
                     trace_name.c_str());
}

void StepWindowProfiler::tick() {
    // Fast path: no profiling active and no profiler to clean up — zero cost
    if (!enabled_.load(std::memory_order_relaxed) && !has_profiler_.load(std::memory_order_relaxed)) {
        return;
    }

    if (!enabled_.load(std::memory_order_relaxed)) {
        stopProfiler("disabled");
        return;
    }

    // Handle reconfigure: stop current profiler so a new one can start with new settings
    if (reconfigure_.exchange(false)) {
        std::lock_guard<std::mutex> lock(mu_);
        if (profiler_) {
            profiler_->stop();
            profiler_.reset();
            has_profiler_.store(false, std::memory_order_relaxed);
            RTP_LLM_LOG_INFO("timeline profiler stopped for reconfigure");
        }
        waited_steps_   = 0;
        profiled_steps_ = 0;
    }

    std::lock_guard<std::mutex> lock(mu_);

    // If profiler not yet started, check if we've waited enough steps
    if (!profiler_) {
        if (waited_steps_ < start_step_.load()) {
            waited_steps_++;
            return;
        }
        // Build trace prefix
        std::string prefix = trace_name_;
        if (prefix.empty()) {
            prefix = "profiler_wr" + std::to_string(world_rank_) + "_ts"
                     + std::to_string(autil::TimeUtility::currentTimeInMicroSeconds());
        }
        if (prefix.back() != '_') {
            prefix += "_";
        }
        profiler_ = std::make_shared<TorchProfile>(prefix, default_output_dir_);
        has_profiler_.store(true, std::memory_order_relaxed);
        profiler_->start();
        profiled_steps_ = 0;
        RTP_LLM_LOG_INFO("timeline profiler started: prefix=%s start_step=%d num_steps=%d",
                         prefix.c_str(),
                         start_step_.load(),
                         num_steps_.load());
        return;
    }

    // Profiler is running, count steps
    profiled_steps_++;
    const int target = num_steps_.load();
    if (target > 0 && profiled_steps_ >= target) {
        enabled_.store(false);
        profiler_->stop();
        profiler_.reset();
        has_profiler_.store(false, std::memory_order_relaxed);
        RTP_LLM_LOG_INFO("timeline profiler stopped: reached %ld/%d steps", profiled_steps_, target);
    }
}

StepWindowProfiler::~StepWindowProfiler() {
    stopProfiler("destructor");
}

void StepWindowProfiler::stopProfiler(const char* reason) {
    std::lock_guard<std::mutex> lock(mu_);
    if (profiler_) {
        profiler_->stop();
        profiler_.reset();
        has_profiler_.store(false, std::memory_order_relaxed);
        RTP_LLM_LOG_INFO("timeline profiler stopped: reason=%s", reason);
    }
}

}  // namespace rtp_llm
