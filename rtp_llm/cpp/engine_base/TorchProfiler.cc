#include "rtp_llm/cpp/engine_base/TorchProfiler.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "autil/TimeUtility.h"
#include <algorithm>
#include <cctype>
#include <string>
#if USING_ASCEND
#include <pybind11/pybind11.h>
namespace py = pybind11;
#endif

namespace rtp_llm {
namespace tap = torch::autograd::profiler;

namespace {
// Trust-boundary limits for configure(). Every profiling config source (HTTP after
// Python sanitize, gRPC/ARPC without Python involvement, service-wide config) funnels
// through StepWindowProfiler::configure, so enforcing them here covers all callers.
constexpr size_t kMaxTraceNameLen = 64;
constexpr int    kMaxNumSteps     = 1000;

std::string sanitizeTraceName(const std::string& trace_name) {
    std::string out;
    out.reserve(std::min(trace_name.size(), kMaxTraceNameLen));
    for (char c : trace_name) {
        if (out.size() >= kMaxTraceNameLen) {
            break;
        }
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-') {
            out.push_back(c);
        }
    }
    return out;
}
}  // namespace
namespace detail {

std::unique_ptr<tap::ProfilerResult> tryDisableProfiler(DisableProfilerFn disable_profiler) {
    try {
        return disable_profiler();
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("failed to disable torch profiler; dropping trace and continuing inference: %s", e.what());
    } catch (...) {
        RTP_LLM_LOG_WARNING(
            "failed to disable torch profiler with unknown exception; dropping trace and continuing inference");
    }
    return nullptr;
}

}  // namespace detail
#if USING_ASCEND
// Holds the Python torch_npu.profiler.profile object. All members that touch
// Python reference counting (py::object) must be created/destroyed while
// holding the GIL; the surrounding TorchProfile methods acquire it.
struct AscendProfilerImpl {
    py::object profiler;

    AscendProfilerImpl() : profiler(py::none()) {}

    // Build the torch_npu profiler via the Python recipe
    // (rtp_llm.utils.ascend_profiler.AscendTorchNpuProfiler).
    // Must be called under the GIL.
    void create(const std::string& output_dir, const std::string& trace_name) {
        if (output_dir.empty()) {
            throw std::runtime_error(
                "Ascend profiling requires a non-empty output_dir "
                "(set torch_cuda_profiler_dir).");
        }
        py::object cls = py::module_::import("rtp_llm.utils.ascend_profiler")
                             .attr("AscendTorchNpuProfiler");
        profiler = cls(output_dir, trace_name);
    }

    void start() { profiler.attr("start")(); }

    // stop() triggers on_trace_ready -> writes *_ascend_pt/ dump.
    void stop() { profiler.attr("stop")(); }
};
#endif

// ---- TorchProfile ----

std::atomic<size_t> TorchProfile::count_{0};

TorchProfile::TorchProfile(const std::string& prefix, std::string output_dir):
    prefix_(prefix), output_dir_(output_dir.empty() ? "." : std::move(output_dir)) {}

TorchProfile::~TorchProfile() {
    if (!stopped_) {
        stop();
    }
#if USING_ASCEND
    // If a Python profiler object is still held (e.g. start() succeeded but
    // stop() was never called, or failed mid-way), release it under the GIL
    // because py::object destruction decrements the Python refcount.
    if (ascend_impl_) {
        try {
            py::gil_scoped_acquire gil;
            ascend_impl_.reset();
        } catch (const std::exception& e) {
            RTP_LLM_LOG_ERROR("ascend profiler cleanup failed: %s", e.what());
            ascend_impl_.release();
        }
    }
#endif
}

void TorchProfile::start() {
    count_ += 1;
    stopped_ = false;
#if USING_ASCEND
    // Ascend: drive torch_npu.profiler via Python. The engine loop thread is
    // the same thread that runs the Python forward_micro_batch, so acquiring
    // the GIL here is deadlock-free and the profiler is active while the npu
    // ops are issued during the subsequent process()/forward().
    try {
        py::gil_scoped_acquire gil;
        ascend_impl_ = std::make_unique<AscendProfilerImpl>();
        ascend_impl_->create(output_dir_, prefix_);
        ascend_impl_->start();
        RTP_LLM_LOG_INFO("ascend torch_npu profiler started: dir=%s prefix=%s",
                         output_dir_.c_str(),
                         prefix_.c_str());
        return;
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("ascend profiler start failed, no profiling this window: %s", e.what());
        try {
            // best-effort cleanup of any half-built python object
            py::gil_scoped_acquire gil;
            ascend_impl_.reset();
        } catch (...) {
            ascend_impl_.release();
        }
        stopped_ = true;
        return;
    }
#endif
    // CUDA / default Kineto path
    tap::prepareProfiler(config_, activities_);
    tap::enableProfiler(config_, activities_);
}

std::pair<std::unique_ptr<tap::ProfilerResult>, std::string> TorchProfile::stopAndCollect() {
    if (stopped_) {
        return {nullptr, ""};
    }
    // disableProfiler() pops the active profiler state before finalizing the
    // trace. Mark this wrapper stopped first so an exception is never retried
    // from the destructor.
    stopped_ = true;
#if USING_ASCEND
    if (ascend_impl_) {
        try {
            py::gil_scoped_acquire gil;
            ascend_impl_->stop();   // on_trace_ready writes *_ascend_pt/
            ascend_impl_.reset();   // safe under GIL
        } catch (const std::exception& e) {
            RTP_LLM_LOG_ERROR("ascend profiler stop failed: %s", e.what());
            try {
                py::gil_scoped_acquire gil;
                ascend_impl_.reset();
            } catch (...) {
                ascend_impl_.release();
            }
        }
        // torch_npu wrote the trace itself; no ProfilerResult to enqueue to
        // the async save worker (StepWindowProfiler already skips nullptr).
        return {nullptr, ""};
    }
#endif
    auto res = detail::tryDisableProfiler(&tap::disableProfiler);
    if (!res) {
        return {nullptr, ""};
    }
    std::string file_name = output_dir_ + "/" + prefix_ + std::to_string(count_) + ".json";
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

StepWindowProfiler::StepScope::StepScope(StepWindowProfiler& profiler): profiler_(profiler) {
    profiler_.beginStep();
}

StepWindowProfiler::StepScope::~StepScope() {
    try {
        profiler_.endStep();
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("timeline profiler endStep failed: %s", e.what());
    } catch (...) {
        RTP_LLM_LOG_ERROR("timeline profiler endStep failed with unknown exception");
    }
}

void StepWindowProfiler::configure(bool enable, const std::string& trace_name, int start_step, int num_steps) {
    // First-come-first-served: if a profiling session is already active, ignore new requests
    // to prevent concurrent requests from repeatedly restarting the profiler.
    if (enable && enabled_.load(std::memory_order_relaxed)) {
        RTP_LLM_LOG_INFO("timeline profiling already active, ignoring new configure request");
        return;
    }
    // Trust-boundary sanitization: trace_name flows into the output file path
    // (TorchProfile::stopAndCollect concatenates output_dir + "/" + prefix + ...),
    // and gRPC/ARPC callers do not pass through the Python HTTP entry's sanitize step.
    std::string safe_trace_name = sanitizeTraceName(trace_name);
    {
        std::lock_guard<std::mutex> lock(mu_);
        trace_name_ = safe_trace_name;
    }
    static constexpr int kDefaultNumSteps = 3;
    start_step_.store(std::max(0, start_step));
    num_steps_.store(num_steps > 0 ? std::min(num_steps, kMaxNumSteps) : kDefaultNumSteps);
    enabled_.store(enable);
    reconfigure_.store(true);
    RTP_LLM_LOG_INFO("timeline profiling configured: enable=%d start_step=%d num_steps=%d trace=%s",
                     int(enable),
                     start_step_.load(),
                     num_steps_.load(),
                     safe_trace_name.c_str());
}

void StepWindowProfiler::configureFromConfig(const ProfilingDebugLoggingConfig& cfg) {
    if (!cfg.gen_timeline_sync) {
        return;
    }
    configure(true, cfg.timeline_trace_name, cfg.timeline_start_step, cfg.timeline_num_steps);
}

void StepWindowProfiler::beginStep() {
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
            prefix = "profiler_ts" + std::to_string(autil::TimeUtility::currentTimeInMicroSeconds());
        }
        if (prefix.back() != '_') {
            prefix += "_";
        }
        prefix += "wr" + std::to_string(world_rank_) + "_";
        profiler_ = std::make_shared<TorchProfile>(prefix, default_output_dir_);
        has_profiler_.store(true, std::memory_order_relaxed);
        profiler_->start();
        profiled_steps_ = 0;
        RTP_LLM_LOG_INFO("timeline profiler started: prefix=%s start_step=%d num_steps=%d",
                         prefix.c_str(),
                         start_step_.load(),
                         num_steps_.load());
    }
}

void StepWindowProfiler::endStep() {
    // Fast path: no profiling active and no profiler to clean up — zero cost
    if (!enabled_.load(std::memory_order_relaxed) && !has_profiler_.load(std::memory_order_relaxed)) {
        return;
    }

    if (!enabled_.load(std::memory_order_relaxed)) {
        stopProfiler("disabled");
        return;
    }

    std::lock_guard<std::mutex> lock(mu_);
    if (!profiler_) {
        return;
    }

    // Profiler is running, count steps
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

void StepWindowProfiler::startStep() {
    // Fast path: no profiling active and no profiler to clean up — zero cost.
    if (!enabled_.load(std::memory_order_relaxed) && !has_profiler_.load(std::memory_order_relaxed)) {
        return;
    }

    if (!enabled_.load(std::memory_order_relaxed)) {
        stopProfiler("disabled");
        return;
    }

    // Handle reconfigure on the engine loop thread, preserving Kineto thread
    // affinity for stop/start.
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
    if (profiler_) {
        return;
    }
    if (waited_steps_ < start_step_.load()) {
        waited_steps_++;
        return;
    }

    std::string prefix = trace_name_;
    if (prefix.empty()) {
        prefix = "profiler_ts" + std::to_string(autil::TimeUtility::currentTimeInMicroSeconds());
    }
    if (prefix.back() != '_') {
        prefix += "_";
    }
    prefix += "wr" + std::to_string(world_rank_) + "_";
    profiler_ = std::make_shared<TorchProfile>(prefix, default_output_dir_);
    has_profiler_.store(true, std::memory_order_relaxed);
    profiler_->start();
    profiled_steps_ = 0;
    RTP_LLM_LOG_INFO("timeline profiler started: prefix=%s start_step=%d num_steps=%d",
                     prefix.c_str(),
                     start_step_.load(),
                     num_steps_.load());
}

void StepWindowProfiler::finishStep() {
    if (!has_profiler_.load(std::memory_order_relaxed)) {
        return;
    }

    std::lock_guard<std::mutex> lock(mu_);
    if (!profiler_) {
        return;
    }

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
