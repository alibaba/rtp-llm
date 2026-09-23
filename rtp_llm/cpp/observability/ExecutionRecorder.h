#pragma once
#include "autil/legacy/json.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace rtp_llm {

// CPU-only batch snapshot worker with the alog-configured batch_schedule sink.
class ExecutionRecorder {
public:
    using Json      = autil::legacy::json::JsonMap;
    using JsonArray = autil::legacy::json::JsonArray;
    using JsonValue = autil::legacy::Any;
    static std::string        toJson(const JsonValue& value);
    static ExecutionRecorder& instance();
    static std::string        quote(const std::string& value);
    static int64_t            unixNs();
    ExecutionRecorder(std::string directory,
                      std::string session,
                      size_t      capacity        = 4096,
                      size_t      max_total_bytes = 1024ULL * 1024 * 1024);
    ~ExecutionRecorder();
    bool enabled() const {
        return state_.load(std::memory_order_acquire) == State::Ready;
    }
    bool needsConfiguration() const {
        return state_.load(std::memory_order_acquire) == State::Unconfigured;
    }
    // Publish rank and metadata together before accepting batches. Returns false
    // for disabled, already configured, closed or failed recorders; never mutates
    // a published configuration. Initialization failures also increment errors.
    [[nodiscard]] bool configure(int rank, int dp_rank, Json metadata);
    int64_t nextId() {
        return next_id_.fetch_add(1);
    }
    Json               identity() const;
    const std::string& owner() const {
        return owner_;
    }
    void markError() noexcept {
        ++errors_;
    }
    void     reportSnapshotError(const char* message) noexcept;
    bool     submitBatch(std::function<std::string()> make_line) noexcept;
    bool     submitBatch(std::string line) noexcept;
    uint64_t dropped() const {
        return dropped_.load();
    }
    void close();

private:
    enum class State { Disabled, Unconfigured, Ready, Failed, Closed };
    void run();
    void manifest(bool closed);
    using Task = std::function<std::string()>;
    std::string                 directory_, session_, owner_, replica_;
    Json                        metadata_;
    int                         rank_ = 0, dp_rank_ = 0;
    size_t                      capacity_, max_total_bytes_, total_bytes_ = 0;
    std::atomic<State>          state_{State::Disabled};
    std::atomic<int64_t>        next_id_{1};
    std::atomic<int64_t>        next_warning_ns_{0};
    std::atomic<uint64_t>       generated_{0}, submitted_{0}, dropped_{0}, errors_{0};
    std::mutex                  mutex_;
    std::condition_variable     ready_;
    std::deque<Task>            queue_;
    bool                        stop_ = false;
    std::thread                 worker_;
};

// Filled under the stream update lock; never read by the JSON writer until frozen.
struct RecordedTokenTiming {
    std::optional<int64_t> ttft_us;
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
    void                            markTimingError() noexcept;
    std::vector<RecordedTokenTiming> timings;

private:
    bool                                 timing_error_ = false;  // Dispatch thread only.
    ExecutionRecorder&                   recorder_;
    ExecutionRecorder::Json              record_;
    std::vector<ExecutionRecorder::Json> rows_;
};

// Immutable request identity, published by the scheduler before the stream is visible.
class RecordedRequest {
public:
    explicit RecordedRequest(ExecutionRecorder& recorder): id_(recorder.nextId()) {}
    int64_t id() const {
        return id_;
    }

private:
    const int64_t id_;
};
}  // namespace rtp_llm
