#include "rtp_llm/cpp/utils/ExecutionStageRecorder.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <ostream>
#include <utility>
#include <unistd.h>
#if defined(__linux__)
#include <sys/syscall.h>
#endif

namespace rtp_llm {
namespace {

int64_t monotonicUs() {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

int64_t wallTimeUs() {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch())
        .count();
}

int64_t threadId() {
#if defined(__linux__)
    thread_local const int64_t id = syscall(SYS_gettid);
#else
    thread_local const int64_t id = std::hash<std::thread::id>{}(std::this_thread::get_id());
#endif
    return id;
}

void writeJsonString(std::ostream& out, const std::string& value) {
    out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') {
            out << '\\' << c;
        } else if (c < 0x20) {
            out << "\\u00" << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned int>(c) << std::dec;
        } else {
            out << c;
        }
    }
    out << '"';
}

int64_t envInt(const char* name, int64_t fallback, int64_t low, int64_t high) {
    const char* value = std::getenv(name);
    if (!value || !*value) {
        return fallback;
    }
    char* end   = nullptr;
    errno       = 0;
    auto result = std::strtoll(value, &end, 10);
    return errno == 0 && end != value && *end == '\0' && result >= low && result <= high ? result : fallback;
}

ExecutionStageRecorder::Config configFromEnv() {
    ExecutionStageRecorder::Config config;
    const char*                    enabled = std::getenv("RTP_LLM_GPU_STAGE_TRACE");
    config.enabled                         = enabled && std::string(enabled) == "1";
    if (config.enabled) {
        config.capacity         = envInt("RTP_LLM_GPU_STAGE_TRACE_CAPACITY", 4096, 1, 65536);
        config.stall_timeout_ms = envInt("RTP_LLM_GPU_STAGE_TRACE_STALL_MS", 30000, 1, 86400000);
        if (const char* prefix = std::getenv("RTP_LLM_GPU_STAGE_TRACE_FILE"); prefix && *prefix) {
            config.file_prefix = prefix;
        }
    }
    return config;
}

}  // namespace

ExecutionStageRecorder::ExecutionStageRecorder(Config config): config_(std::move(config)), pid_(getpid()) {
    if (config_.enabled && config_.capacity > 0 && config_.stall_timeout_ms > 0) {
        try {
            monitor_ = std::thread([this] { monitor(); });
        } catch (...) {
            std::fprintf(stderr, "[gpu_stage_trace] cannot start monitor\n");
        }
    }
}

ExecutionStageRecorder::~ExecutionStageRecorder() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    wakeup_.notify_all();
    if (monitor_.joinable()) {
        monitor_.join();
    }
}

void ExecutionStageRecorder::appendLocked(Event event) {
    if (history_.size() == config_.capacity) {
        history_.pop_front();
    }
    history_.push_back(std::move(event));
}

uint64_t ExecutionStageRecorder::begin(const char*        stage,
                                       const std::string& details,
                                       int64_t            request_id,
                                       uint64_t           parent_id) noexcept {
    if (!enabled() || config_.capacity == 0) {
        return 0;
    }
    try {
        Event event{0, parent_id, threadId(), monotonicUs(), wallTimeUs(), request_id, stage, details, "begin"};
        std::lock_guard<std::mutex> lock(mutex_);
        event.id = ++next_id_;
        // Retain the oldest unfinished operations even after the history ring wraps.
        if (active_.size() < config_.capacity) {
            active_.emplace(event.id, event);
        } else {
            ++dropped_active_;
        }
        const auto id = event.id;
        appendLocked(std::move(event));
        return id;
    } catch (...) {
        return 0;
    }
}

void ExecutionStageRecorder::end(uint64_t id, const char* phase) noexcept {
    if (!id) {
        return;
    }
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        auto                        it = active_.find(id);
        if (it == active_.end()) {
            return;
        }
        auto event = std::move(it->second);
        active_.erase(it);
        event.monotonic_us = monotonicUs();
        event.wall_time_us = wallTimeUs();
        event.phase        = phase;
        appendLocked(std::move(event));
    } catch (...) {}
}

ExecutionStageRecorder::Snapshot ExecutionStageRecorder::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    Snapshot                    result;
    result.history.assign(history_.begin(), history_.end());
    for (const auto& item : active_) {
        result.active.push_back(item.second);
    }
    result.dropped_active = dropped_active_;
    return result;
}

void ExecutionStageRecorder::dump(const std::string& reason) noexcept {
    if (!enabled()) {
        return;
    }
    try {
        const auto                  data = snapshot();
        const auto                  now  = monotonicUs();
        const auto                  file = config_.file_prefix + "." + std::to_string(pid_) + ".jsonl";
        std::lock_guard<std::mutex> output_lock(output_mutex_);
        std::ofstream               out(file, std::ios::app);
        if (!out) {
            std::fprintf(stderr, "[gpu_stage_trace] cannot open %s\n", file.c_str());
            return;
        }
        out << "{\"pid\":" << pid_ << ",\"rank\":" << rank_.load(std::memory_order_relaxed)
            << ",\"wall_time_us\":" << wallTimeUs() << ",\"monotonic_us\":" << now << ",\"reason\":";
        writeJsonString(out, reason);
        out << ",\"dropped_active\":" << data.dropped_active;
        auto writeEvents = [&](const char* name, const std::vector<Event>& events) {
            out << ",\"" << name << "\":[";
            bool first = true;
            for (const auto& event : events) {
                if (!first) {
                    out << ',';
                }
                first = false;
                out << "{\"id\":" << event.id << ",\"parent_id\":" << event.parent_id << ",\"tid\":" << event.thread_id
                    << ",\"monotonic_us\":" << event.monotonic_us << ",\"wall_time_us\":" << event.wall_time_us
                    << ",\"request_id\":" << event.request_id << ",\"stage\":";
                writeJsonString(out, event.stage);
                out << ",\"phase\":";
                writeJsonString(out, event.phase);
                out << ",\"details\":";
                writeJsonString(out, event.details);
                out << '}';
            }
            out << ']';
        };
        writeEvents("active", data.active);
        writeEvents("history", data.history);
        out << "}\n";
        out.flush();
        std::fprintf(stderr,
                     "[gpu_stage_trace] pid=%ld rank=%d reason=%s active=%zu file=%s\n",
                     static_cast<long>(pid_),
                     rank_.load(std::memory_order_relaxed),
                     reason.c_str(),
                     data.active.size(),
                     file.c_str());
    } catch (...) {
        std::fprintf(stderr, "[gpu_stage_trace] dump failed\n");
    }
}

void ExecutionStageRecorder::monitor() noexcept {
    try {
        int64_t                      last_dump_us = 0;
        std::unique_lock<std::mutex> lock(mutex_);
        while (!wakeup_.wait_for(lock,
                                 std::chrono::milliseconds(std::min<int64_t>(1000, config_.stall_timeout_ms)),
                                 [this] { return stopping_; })) {
            const auto now        = monotonicUs();
            const auto timeout_us = config_.stall_timeout_ms * 1000;
            bool       stalled    = false;
            for (const auto& item : active_) {
                stalled |= now - item.second.monotonic_us >= timeout_us;
            }
            if (stalled && now - last_dump_us >= timeout_us) {
                last_dump_us = now;
                lock.unlock();
                dump("stage_stalled");
                lock.lock();
            }
        }
    } catch (...) {
        std::fprintf(stderr, "[gpu_stage_trace] monitor failed\n");
    }
}

ExecutionStageRecorder& executionStageRecorder() {
    static ExecutionStageRecorder recorder(configFromEnv());
    return recorder;
}

}  // namespace rtp_llm
