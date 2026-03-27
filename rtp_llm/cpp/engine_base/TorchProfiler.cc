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

std::pair<std::unique_ptr<tap::ProfilerResult>, std::string> TorchProfile::stopAndCollect() {
    if (stopped_) {
        return {nullptr, ""};
    }
    auto        res       = tap::disableProfiler();
    std::string file_name = output_dir_ + "/" + prefix_ + std::to_string(count_) + ".json";
    stopped_              = true;
    return {std::move(res), std::move(file_name)};
}

void TorchProfile::stop() {
    auto [res, file_name] = stopAndCollect();
    if (res) {
        res->save(file_name);
    }
}

// ---- ProfilerSaveWorker ----

ProfilerSaveWorker::ProfilerSaveWorker(): thread_([this] { run(); }) {}

ProfilerSaveWorker::~ProfilerSaveWorker() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        stop_ = true;
    }
    cv_.notify_one();
    thread_.join();
}

void ProfilerSaveWorker::enqueue(std::unique_ptr<tap::ProfilerResult> result, std::string file_name) {
    {
        std::lock_guard<std::mutex> lock(mu_);
        tasks_.push({std::move(result), std::move(file_name)});
    }
    cv_.notify_one();
}

void ProfilerSaveWorker::run() {
    while (true) {
        SaveTask task;
        {
            std::unique_lock<std::mutex> lock(mu_);
            cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
            if (stop_ && tasks_.empty()) {
                return;
            }
            task = std::move(tasks_.front());
            tasks_.pop();
        }
        RTP_LLM_LOG_INFO("saving profiler trace to %s (async)", task.file_name.c_str());
        try {
            task.result->save(task.file_name);
            RTP_LLM_LOG_INFO("profiler trace saved: %s", task.file_name.c_str());
        } catch (const std::exception& e) {
            RTP_LLM_LOG_ERROR("failed to save profiler trace %s: %s", task.file_name.c_str(), e.what());
        }
    }
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
    static constexpr int kDefaultNumSteps = 3;
    start_step_.store(std::max(0, start_step));
    num_steps_.store(num_steps > 0 ? num_steps : kDefaultNumSteps);
    enabled_.store(enable);
    reconfigure_.store(true);
    RTP_LLM_LOG_INFO("timeline profiling configured: enable=%d start_step=%d num_steps=%d trace=%s",
                     int(enable),
                     start_step_.load(),
                     num_steps_.load(),
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
            auto [res, file_name] = profiler_->stopAndCollect();
            if (res) {
                save_worker_.enqueue(std::move(res), std::move(file_name));
            }
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
        // Fall through to count this step — the profiler is now active and the
        // engine will execute process() (including prefill) after this tick().
    }

    // Profiler is running, count steps (including the step that started it)
    profiled_steps_++;
    const int target = num_steps_.load();
    if (target > 0 && profiled_steps_ >= target) {
        enabled_.store(false);
        auto [res, file_name] = profiler_->stopAndCollect();
        if (res) {
            save_worker_.enqueue(std::move(res), std::move(file_name));
        }
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
        auto [res, file_name] = profiler_->stopAndCollect();
        if (res) {
            save_worker_.enqueue(std::move(res), std::move(file_name));
        }
        profiler_.reset();
        has_profiler_.store(false, std::memory_order_relaxed);
        RTP_LLM_LOG_INFO("timeline profiler stopped: reason=%s", reason);
    }
}

}  // namespace rtp_llm
