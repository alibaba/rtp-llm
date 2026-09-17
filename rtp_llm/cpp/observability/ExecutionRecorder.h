#pragma once
#include "autil/legacy/json.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace alog {
class Logger;
}

namespace rtp_llm {

// CPU-only snapshot worker with a best-effort alog file sink.
class ExecutionRecorder {
public:
    using Json      = autil::legacy::json::JsonMap;
    using JsonArray = autil::legacy::json::JsonArray;
    using JsonValue = autil::legacy::Any;
    static std::string        toJson(const JsonValue& value);
    static ExecutionRecorder& instance();
    static std::string        quote(const std::string& value);
    static int64_t            monotonicNs();
    static int64_t            unixNs();
    ExecutionRecorder(std::string directory,
                      std::string session,
                      int         rank,
                      int         dp_rank,
                      size_t      capacity        = 4096,
                      size_t      max_file_bytes  = 0,  // Zero: one append-only file per record type.
                      size_t      max_total_bytes = 1024ULL * 1024 * 1024);
    ~ExecutionRecorder();
    bool enabled() const {
        return enabled_.load(std::memory_order_relaxed);
    }
    // Configure once during engine construction, before requests enter.
    void    configure(int rank, int dp_rank);
    int64_t nextId() {
        return next_id_.fetch_add(1);
    }
    Json               identity() const;
    const std::string& owner() const {
        return owner_;
    }
    void setMetadata(Json metadata);
    void markError() noexcept {
        ++errors_;
    }
    bool     submit(const std::string& file, std::function<std::string()> make_line) noexcept;
    bool     submit(const std::string& file, std::string line) noexcept;
    uint64_t dropped() const {
        return dropped_.load();
    }
    void close();

private:
    void run();
    void manifest(bool closed);
    struct Task {
        std::string                  file;
        std::function<std::string()> make_line;
    };
    struct File {
        alog::Logger* logger = nullptr;  // Owned by alog, not the recorder.
        size_t        bytes  = 0;
        size_t        part   = 0;
    };
    std::string                 directory_, session_, owner_, replica_;
    Json                        metadata_;
    int                         rank_, dp_rank_;
    size_t                      capacity_, max_file_bytes_, max_total_bytes_, total_bytes_ = 0;
    std::atomic<bool>           enabled_{false};
    std::atomic<int64_t>        next_id_{1};
    std::atomic<uint64_t>       generated_{0}, submitted_{0}, dropped_{0}, errors_{0};
    std::mutex                  mutex_;
    std::condition_variable     ready_;
    std::deque<Task>            queue_;
    std::map<std::string, File> files_;
    bool                        stop_ = false;
    std::thread                 worker_;
};

// Filled under the stream update lock; never read by the JSON writer until frozen.
struct RecordedTokenTiming {
    std::optional<int64_t> ttft_us;
    bool                   first_token_produced = false;
};

// Per-forward ownership, not a global ID lookup. Last owner submits even on
// failure; rows that never reached output update retain null TTFT.
class RecordedBatch {
public:
    RecordedBatch(ExecutionRecorder&                   recorder,
                  ExecutionRecorder::Json              record,
                  std::vector<ExecutionRecorder::Json> rows);
    ~RecordedBatch() noexcept;
    RecordedBatch(const RecordedBatch&)                              = delete;
    RecordedBatch&                   operator=(const RecordedBatch&) = delete;
    std::vector<RecordedTokenTiming> timings;

private:
    ExecutionRecorder&                   recorder_;
    ExecutionRecorder::Json              record_;
    std::vector<ExecutionRecorder::Json> rows_;
};

// Shared request identity/timing only. Never emits a separate request log.
class RecordedRequest {
public:
    struct ScheduleTiming {
        std::optional<int64_t> enqueue_time_unix_ns;
        std::optional<int64_t> first_scheduled_time_unix_ns;
    };
    explicit RecordedRequest(ExecutionRecorder& recorder): id_(recorder.nextId()) {}
    int64_t id() const {
        return id_;
    }
    void           enqueue();
    void           scheduled();
    ScheduleTiming scheduleTiming() const;

private:
    int64_t            id_;
    ScheduleTiming     timing_;
    mutable std::mutex mutex_;
};
}  // namespace rtp_llm
