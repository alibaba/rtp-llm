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

struct ProfilingDebugLoggingConfig;

// CPU-only batch snapshot worker with the alog-configured batch_schedule sink.
class ExecutionRecorder {
public:
    using Json      = autil::legacy::json::JsonMap;
    using JsonArray = autil::legacy::json::JsonArray;
    using JsonValue = autil::legacy::Any;
    static std::string toJson(const JsonValue& value);
    // The singleton is built on first call. The first call normally happens
    // during single-threaded engine init (NormalExecutor ctor) and may carry
    // the engine's ProfilingDebugLoggingConfig (already arg+env resolved on
    // the Python side); every later call returns the same object no matter
    // what it passes. A null or disabled config yields a Disabled recorder.
    static ExecutionRecorder& instance(const ProfilingDebugLoggingConfig* config = nullptr);
    static std::string        quote(const std::string& value);
    static int64_t            unixNs();
    // Empty directory leaves the recorder Disabled. Direct construction is
    // for tests; production code goes through instance().
    ExecutionRecorder(std::string directory, size_t capacity = 4096);
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
    int64_t            nextId() {
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
    enum class State {
        Disabled,
        Unconfigured,
        Ready,
        Failed,
        Closed
    };
    void run();
    void manifest(bool closed);
    void tryWriteManifest(bool closed) noexcept;
    void reportError(const char* operation, const char* message) noexcept;
    using Task = std::function<std::string()>;
    std::string             directory_, owner_;
    Json                    metadata_;
    int                     rank_ = 0, dp_rank_     = 0;
    size_t                  capacity_, total_bytes_ = 0;
    std::atomic<State>      state_{State::Disabled};
    std::atomic<int64_t>    next_id_{1};
    std::atomic<int64_t>    next_warning_ns_{0};
    std::atomic<uint64_t>   generated_{0}, submitted_{0}, dropped_{0}, errors_{0};
    std::mutex              mutex_;
    std::condition_variable ready_;
    std::deque<Task>        queue_;
    bool                    stop_ = false;
    std::thread             worker_;
};

// Filled under the stream update lock; never read by the JSON writer until frozen.
struct RecordedTokenTiming {
    std::optional<int64_t> ttft_us;
    // Speculative verify rows only (guarded by speculative_propose_step > 0
    // in specUpdate): pre-verify cached length and accepted token count are
    // backfilled at dispatch under the stream lock, same pattern as ttft_us.
    std::optional<int64_t> kv_len;
    std::optional<int64_t> accept_len;
};

// Per-forward ownership, not a global ID lookup. Last owner submits even on
// failure; rows that never reached output update retain null TTFT.
class RecordedBatch {
public:
    RecordedBatch(ExecutionRecorder&                   recorder,
                  ExecutionRecorder::Json              record,
                  std::vector<ExecutionRecorder::Json> rows);
    ~RecordedBatch() noexcept;
    RecordedBatch(const RecordedBatch&)            = delete;
    RecordedBatch& operator=(const RecordedBatch&) = delete;
    void           markTimingError() noexcept;
    // Slice the timing window [batch_idx_in, batch_idx_in + cur_batch_size) for
    // stream->update/specUpdate. Returns nullptr and marks a timing error when
    // the window falls outside the recorded rows.
    RecordedTokenTiming* sliceTimings(int64_t batch_idx_in, size_t cur_batch_size) {
        if (batch_idx_in >= 0 && static_cast<size_t>(batch_idx_in) + cur_batch_size <= timings.size())
            return timings.data() + batch_idx_in;
        markTimingError();
        return nullptr;
    }

    // Record the executor step's own wall duration in microseconds: from the
    // step's entry on the executor thread to the output-dispatch handoff
    // (gather, forward and sampling included; the async bookkeeping tail that
    // overlaps the next step is excluded by construction). Called by the
    // owning executor once compute is done and before the batch record moves
    // into dispatch; early-exit submissions leave it unset (omitted means
    // unknown). Sampled decode streams can use it together with kv_len
    // deltas to derive per-step timing offline.
    void setStepDurationUs(int64_t step_duration_us) {
        record_["step_duration_us"] = step_duration_us;
    }
    std::vector<RecordedTokenTiming> timings;

private:
    // Pooled output dispatch (output_dispatcher_worker_count > 0) may call
    // markTimingError() from several worker threads at once; exchange() makes
    // the first caller the only reporter.
    std::atomic<bool>                    timing_error_{false};
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
