#include "rtp_llm/cpp/observability/ExecutionRecorder.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "alog/Appender.h"
#include "alog/Layout.h"
#include "alog/Logger.h"
#include "aios/alog/src/cpp/EventBase.h"
#include "autil/legacy/jsonizable.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <unistd.h>

namespace alog {
extern EventBase gEventBase;
}

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

int64_t ExecutionRecorder::monotonicNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
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
                                      0,
                                      0,
                                      positiveEnv("RTP_LLM_RECORD_QUEUE_SIZE", 4096),
                                      positiveEnv("RTP_LLM_RECORD_FILE_BYTES", 0),
                                      positiveEnv("RTP_LLM_RECORD_TOTAL_BYTES", 1024ULL * 1024 * 1024));
    return recorder;
}

ExecutionRecorder::ExecutionRecorder(std::string directory,
                                     std::string session,
                                     int         rank,
                                     int         dp_rank,
                                     size_t      capacity,
                                     size_t      max_file_bytes,
                                     size_t      max_total_bytes):
    directory_(std::move(directory)),
    session_(std::move(session)),
    rank_(rank),
    dp_rank_(dp_rank),
    capacity_(capacity),
    max_file_bytes_(max_file_bytes),
    max_total_bytes_(max_total_bytes) {
    if (directory_.empty() || session_.empty() || !capacity_)
        return;
    owner_   = std::to_string(getpid()) + "-" + std::to_string(unixNs());
    replica_ = env("RTP_LLM_RECORD_REPLICA");
    next_id_ = unixNs();
    directory_ += "/owner-" + owner_;
    try {
        std::filesystem::create_directories(directory_);
        manifest(false);
        enabled_ = true;
        worker_  = std::thread([this] { run(); });
    } catch (...) {
        ++errors_;
        enabled_ = false;
    }
}
ExecutionRecorder::~ExecutionRecorder() {
    close();
}
void ExecutionRecorder::configure(int rank, int dp_rank) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (generated_ != 0)
        return;
    rank_    = rank;
    dp_rank_ = dp_rank;
    if (enabled()) {
        try {
            manifest(false);
        } catch (...) {
            ++errors_;
            enabled_ = false;
        }
    }
}
void ExecutionRecorder::setMetadata(Json metadata) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (generated_ != 0)
        return;
    metadata_ = std::move(metadata);
    if (enabled()) {
        try {
            manifest(false);
        } catch (...) {
            ++errors_;
            enabled_ = false;
        }
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
bool ExecutionRecorder::submit(const std::string& file, std::string line) noexcept {
    return submit(file, [line = std::move(line)] { return line; });
}
bool ExecutionRecorder::submit(const std::string& file, std::function<std::string()> make_line) noexcept {
    if (!enabled())
        return false;
    ++generated_;
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stop_ || queue_.size() >= capacity_) {
            ++dropped_;
            return false;
        }
        queue_.push_back({file, std::move(make_line)});
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
                   {"max_file_bytes", max_file_bytes_},
                   {"max_total_bytes", max_total_bytes_},
                   {"bytes_written", JsonValue()},
                   {"bytes_submitted", total_bytes_}});
    JsonArray files;
    for (const auto& entry : files_) {
        for (size_t part = 0; part <= entry.second.part; ++part) {
            files.emplace_back(entry.first + (part ? "." + std::to_string(part) : ""));
        }
    }
    record["files"] = std::move(files);
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
            const auto line = task.make_line();
            if (line.size() >= max_total_bytes_ || total_bytes_ > max_total_bytes_ - line.size() - 1) {
                ++dropped_;
                ++errors_;
                enabled_ = false;
                manifest(false);
                continue;
            }
            if (task.file == "batches.jsonl") {
                // Only alog.conf owns this sink: no per-owner appender, layout,
                // level override or recorder-managed rotation for batch logs.
                RTP_LLM_BATCH_SCHEDULE_LOG_INFO(line);
                total_bytes_ += line.size() + 1;
                ++submitted_;
                if (submitted_ % 64 == 0)
                    manifest(false);
                continue;
            }
            auto& file = files_[task.file];
            if (max_file_bytes_ && file.logger && file.bytes && file.bytes + line.size() + 1 > max_file_bytes_) {
                file.logger->flush();
                file.logger = nullptr;
                file.bytes  = 0;
                ++file.part;
            }
            if (!file.logger) {
                const auto path     = directory_ + "/" + task.file + (file.part ? "." + std::to_string(file.part) : "");
                auto*      appender = static_cast<alog::FileAppender*>(alog::FileAppender::getAppender(path.c_str()));
                auto*      layout   = new alog::PatternLayout();
                layout->setLogPattern("%%m");
                appender->setLayout(layout);
                appender->setAutoFlush(false);
                appender->setCompress(false);
                // Retain recorder's exact byte-based part naming and budget.
                appender->setMaxSize(0);
                appender->setAsyncFlush(true);
                appender->setFlushThreshold(64 * 1024);
                appender->setFlushIntervalInMS(100);
                // setAsyncFlush alone does not register the appender. Mirror
                // Configurator without resetting process-wide logger settings.
                alog::gEventBase.addFileAppender(appender);
                const auto name = "execution_recorder." + owner_ + "." + task.file + "." + std::to_string(file.part);
                file.logger     = alog::Logger::getLogger(name.c_str());
                file.logger->setInheritFlag(false);
                file.logger->setLevel(alog::LOG_LEVEL_INFO);
                file.logger->setAppender(appender);
            }
            // Unlike printf-style log(), this does not truncate at alog.max_msg_len.
            // A void return only means handed to alog, not durably written.
            file.logger->logPureMessage(alog::LOG_LEVEL_INFO, line.c_str());
            file.bytes += line.size() + 1;
            total_bytes_ += line.size() + 1;
            ++submitted_;
            if (submitted_ % 64 == 0)
                manifest(false);
        } catch (...) {
            ++errors_;
            enabled_ = false;
            try {
                manifest(false);
            } catch (...) {}
        }
    }
}
void ExecutionRecorder::close() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        enabled_ = false;
        stop_    = true;
    }
    ready_.notify_one();
    if (worker_.joinable()) {
        worker_.join();
        try {
            // Flush only this recorder's appenders, never shut down global alog.
            getBatchScheduleLogger()->flush();
            for (const auto& entry : files_)
                if (entry.second.logger)
                    entry.second.logger->flush();
            manifest(true);
        } catch (...) {
            ++errors_;
        }
    }
}

RecordedBatch::RecordedBatch(ExecutionRecorder&                   recorder,
                             ExecutionRecorder::Json              record,
                             std::vector<ExecutionRecorder::Json> rows):
    timings(rows.size()), recorder_(recorder), record_(std::move(record)), rows_(std::move(rows)) {}

RecordedBatch::~RecordedBatch() noexcept {
    try {
        recorder_.submit(
            "batches.jsonl",
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

void RecordedRequest::enqueue() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!timing_.enqueue_time_unix_ns)
        timing_.enqueue_time_unix_ns = ExecutionRecorder::unixNs();
}
void RecordedRequest::scheduled() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (timing_.enqueue_time_unix_ns && !timing_.first_scheduled_time_unix_ns)
        timing_.first_scheduled_time_unix_ns = ExecutionRecorder::unixNs();
}
RecordedRequest::ScheduleTiming RecordedRequest::scheduleTiming() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return timing_;
}
}  // namespace rtp_llm
