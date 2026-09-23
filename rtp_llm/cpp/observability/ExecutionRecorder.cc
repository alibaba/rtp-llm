#include "rtp_llm/cpp/observability/ExecutionRecorder.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "alog/Logger.h"
#include "autil/legacy/jsonizable.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <unistd.h>

namespace rtp_llm {
namespace {
std::string env(const char* name, const std::string& fallback = "") {
    const auto* value = std::getenv(name);
    return value ? value : fallback;
}
size_t positiveEnv(const char* name, size_t fallback) {
    const auto text = env(name);
    if (text.empty())
        return fallback;
    try {
        size_t     end   = 0;
        const auto value = std::stoull(text, &end);
        return text[0] != '-' && end == text.size() && value > 0 ? value : fallback;
    } catch (...) {
        return fallback;
    }
}
}  // namespace

int64_t ExecutionRecorder::unixNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch())
        .count();
}
std::string ExecutionRecorder::quote(const std::string& value) {
    return toJson(value);
}
std::string ExecutionRecorder::toJson(const JsonValue& value) {
    return autil::legacy::ToJsonString(value, /*isCompact=*/true);
}

ExecutionRecorder& ExecutionRecorder::instance() {
    // A shared explicit session ID is required for cross-rank correlation.
    static ExecutionRecorder recorder(env("RTP_LLM_RECORD_DIR"),
                                      env("RTP_LLM_RECORD_SESSION"),
                                      positiveEnv("RTP_LLM_RECORD_QUEUE_SIZE", 4096),
                                      positiveEnv("RTP_LLM_RECORD_TOTAL_BYTES", 1024ULL * 1024 * 1024));
    return recorder;
}

ExecutionRecorder::ExecutionRecorder(std::string directory,
                                     std::string session,
                                     size_t      capacity,
                                     size_t      max_total_bytes):
    directory_(std::move(directory)),
    session_(std::move(session)),
    capacity_(capacity),
    max_total_bytes_(max_total_bytes) {
    if (directory_.empty() || session_.empty() || !capacity_)
        return;
    owner_   = std::to_string(getpid()) + "-" + std::to_string(unixNs());
    replica_ = env("RTP_LLM_RECORD_REPLICA");
    next_id_ = unixNs();
    directory_ += "/owner-" + owner_;
    state_.store(State::Unconfigured, std::memory_order_release);
}
ExecutionRecorder::~ExecutionRecorder() {
    close();
}
bool ExecutionRecorder::configure(int rank, int dp_rank, Json metadata) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_.load(std::memory_order_relaxed) != State::Unconfigured)
        return false;
    try {
        rank_     = rank;
        dp_rank_  = dp_rank;
        metadata_ = std::move(metadata);
        std::filesystem::create_directories(directory_);
        manifest(false);
        // The worker cannot dequeue until this lock is released. Publish Ready
        // only after both the complete manifest and the worker are available.
        worker_ = std::thread([this] { run(); });
        state_.store(State::Ready, std::memory_order_release);
        return true;
    } catch (...) {
        ++errors_;
        state_.store(State::Failed, std::memory_order_release);
        return false;
    }
}
ExecutionRecorder::Json ExecutionRecorder::identity() const {
    return {{"schema_version", 1},
            {"session_id", session_},
            {"replica_id", replica_.empty() ? "dp" + std::to_string(dp_rank_) : replica_},
            {"dp_rank", dp_rank_},
            {"world_rank", rank_},
            {"owner_id", owner_}};
}
bool ExecutionRecorder::submitBatch(std::string line) noexcept {
    return submitBatch([line = std::move(line)] { return line; });
}
bool ExecutionRecorder::submitBatch(std::function<std::string()> make_line) noexcept {
    if (!enabled())
        return false;
    ++generated_;
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!enabled() || stop_ || queue_.size() >= capacity_) {
            ++dropped_;
            return false;
        }
        queue_.push_back(std::move(make_line));
        ready_.notify_one();
        return true;
    } catch (...) {
        ++dropped_;
        return false;
    }
}
void ExecutionRecorder::manifest(bool closed) {
    std::ofstream out(directory_ + "/manifest.json.tmp");
    out.exceptions(std::ios::failbit | std::ios::badbit);
    auto record = identity();
    record.insert({{"closed", closed},
                   {"complete", false},
                   {"storage_backend", std::string("alog")},
                   {"delivery_policy", std::string("best_effort")},
                   {"batch_log_logger", std::string("batch_schedule")},
                   {"batch_log_location", std::string("alog_config")},
                   {"request_log_format", std::string("batch_embedded_v1")},
                   {"model_role", std::string("target")},
                   {"length_source", std::string("cpu_input_snapshot")},
                   {"generated", generated_.load()},
                   {"written", JsonValue()},
                   {"submitted_to_alog", submitted_.load()},
                   {"dropped", dropped_.load()},
                   {"sink_dropped", JsonValue()},
                   {"errors", errors_.load()},
                   {"scope", std::string("engine")},
                   {"request_sampling", false},
                   {"engine", metadata_},
                   {"max_total_bytes", max_total_bytes_},
                   {"bytes_written", JsonValue()},
                   {"bytes_submitted", total_bytes_}});
    out << toJson(record) << '\n';
    out.close();
    std::filesystem::rename(directory_ + "/manifest.json.tmp", directory_ + "/manifest.json");
}
void ExecutionRecorder::run() {
    for (;;) {
        Task task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            ready_.wait(lock, [this] { return stop_ || !queue_.empty(); });
            if (queue_.empty())
                break;
            task = std::move(queue_.front());
            queue_.pop_front();
        }
        try {
            auto* batch_logger = Logger::getBatchScheduleLogger();
            if (!batch_logger->isLevelEnabled(alog::LOG_LEVEL_INFO)) {
                ++dropped_;
                continue;
            }
            const auto line = task();
            if (line.size() >= max_total_bytes_ || total_bytes_ > max_total_bytes_ - line.size() - 1) {
                ++dropped_;
                ++errors_;
                state_.store(State::Failed, std::memory_order_release);
                manifest(false);
                continue;
            }
            // Sink configuration, asynchronous flush and rotation belong to alog.conf.
            // Submission does not imply durable delivery.
            batch_logger->logBinaryMessage(alog::LOG_LEVEL_INFO, line);
            total_bytes_ += line.size() + 1;
            ++submitted_;
            if (submitted_ % 64 == 0)
                manifest(false);
        } catch (...) {
            ++errors_;
            state_.store(State::Failed, std::memory_order_release);
            try {
                manifest(false);
            } catch (...) {}
        }
    }
}
void ExecutionRecorder::reportSnapshotError(const char* message) noexcept {
    markError();
    const auto now =
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count();
    auto next = next_warning_ns_.load(std::memory_order_relaxed);
    if (now >= next
        && next_warning_ns_.compare_exchange_strong(next, now + 60LL * 1000 * 1000 * 1000, std::memory_order_relaxed)) {
        try {
            RTP_LLM_LOG_WARNING("batch recording snapshot failed: %s (warnings limited to once per minute)", message);
        } catch (...) {
            // Observability failures must not interrupt inference.
        }
    }
}

void ExecutionRecorder::close() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        state_.store(State::Closed, std::memory_order_release);
        stop_ = true;
    }
    ready_.notify_one();
    if (worker_.joinable()) {
        worker_.join();
        try {
            // Flush the batch sink only, never shut down global alog.
            Logger::getBatchScheduleLogger()->flush();
            manifest(true);
        } catch (...) {
            ++errors_;
        }
    }
    // Draining queued work may have reported a failure after close began.
    state_.store(State::Closed, std::memory_order_release);
}

RecordedBatch::RecordedBatch(ExecutionRecorder&                   recorder,
                             ExecutionRecorder::Json              record,
                             std::vector<ExecutionRecorder::Json> rows):
    timings(rows.size()), recorder_(recorder), record_(std::move(record)), rows_(std::move(rows)) {}

void RecordedBatch::markTimingError() noexcept {
    if (!timing_error_) {
        timing_error_ = true;
        recorder_.reportSnapshotError("TTFT snapshot slot count mismatch");
    }
}

RecordedBatch::~RecordedBatch() noexcept {
    try {
        recorder_.submitBatch(
            [record = std::move(record_), rows = std::move(rows_), timings = std::move(timings)]() mutable {
                ExecutionRecorder::JsonArray requests;
                requests.reserve(rows.size());
                for (size_t i = 0; i < rows.size(); ++i) {
                    if (timings[i].ttft_us)
                        rows[i]["ttft_us"] = *timings[i].ttft_us;
                    requests.emplace_back(std::move(rows[i]));
                }
                record["requests"] = std::move(requests);
                return ExecutionRecorder::toJson(record);
            });
    } catch (...) {
        recorder_.markError();
    }
}

}  // namespace rtp_llm
