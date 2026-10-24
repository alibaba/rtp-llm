#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace rtp_llm {

// CPU-only breadcrumbs: the monitor must remain usable when CUDA calls cannot return.
class ExecutionStageRecorder {
public:
    struct Config {
        bool        enabled          = false;
        size_t      capacity         = 4096;
        int64_t     stall_timeout_ms = 30000;
        std::string file_prefix      = "logs/gpu_stage_trace";
    };

    struct Event {
        uint64_t    id;
        uint64_t    parent_id;
        int64_t     thread_id;
        int64_t     monotonic_us;
        int64_t     wall_time_us;
        int64_t     request_id;
        std::string stage;
        std::string details;
        std::string phase;
    };

    struct Snapshot {
        std::vector<Event> history;
        std::vector<Event> active;
        uint64_t           dropped_active = 0;
    };

    explicit ExecutionStageRecorder(Config config);
    ~ExecutionStageRecorder();
    ExecutionStageRecorder(const ExecutionStageRecorder&)            = delete;
    ExecutionStageRecorder& operator=(const ExecutionStageRecorder&) = delete;

    bool enabled() const noexcept {
        return config_.enabled;
    }
    void setRank(int rank) noexcept {
        rank_.store(rank, std::memory_order_relaxed);
    }
    uint64_t
         begin(const char* stage, const std::string& details, int64_t request_id = 0, uint64_t parent_id = 0) noexcept;
    void end(uint64_t id, const char* phase = "returned") noexcept;
    // Enrich an already-published breadcrumb without moving its start timestamp.
    void     updateDetails(uint64_t id, const std::string& details) noexcept;
    bool     isActive(uint64_t id) const noexcept;
    Snapshot snapshot() const;
    void     dump(const std::string& reason) noexcept;

private:
    void appendLocked(Event event);
    void monitor() noexcept;

    const Config              config_;
    const int64_t             pid_;
    std::atomic<int>          rank_{-1};
    mutable std::mutex        mutex_;
    std::mutex                output_mutex_;
    std::condition_variable   wakeup_;
    bool                      stopping_       = false;
    uint64_t                  next_id_        = 0;
    uint64_t                  dropped_active_ = 0;
    std::deque<Event>         history_;
    std::map<uint64_t, Event> active_;
    std::thread               monitor_;
};

ExecutionStageRecorder& executionStageRecorder();

// A CPU-only host-operation scope. "returned" means that the host API returned,
// not that a kernel or asynchronous copy completed on the device.
class ExecutionStageScope {
public:
    ExecutionStageScope(const char*        stage,
                        const std::string& details    = {},
                        int64_t            request_id = 0,
                        uint64_t           parent_id  = 0) noexcept;
    ExecutionStageScope(ExecutionStageRecorder& recorder,
                        const char*             stage,
                        const std::string&      details    = {},
                        int64_t                 request_id = 0,
                        uint64_t                parent_id  = 0) noexcept;
    ~ExecutionStageScope();
    ExecutionStageScope(const ExecutionStageScope&)            = delete;
    ExecutionStageScope& operator=(const ExecutionStageScope&) = delete;

    uint64_t id() const noexcept {
        return id_;
    }
    uint64_t parentId() const noexcept {
        return parent_id_;
    }
    void annotate(const std::string& details) noexcept;
    void finish(const char* phase = nullptr) noexcept;

private:
    void start(ExecutionStageRecorder& recorder,
               const char*             stage,
               const std::string&      details,
               int64_t                 request_id,
               uint64_t                parent_id) noexcept;

    ExecutionStageRecorder* recorder_            = nullptr;
    uint64_t                id_                  = 0;
    uint64_t                parent_id_           = 0;
    int64_t                 thread_id_           = 0;
    int                     uncaught_exceptions_ = 0;
};

}  // namespace rtp_llm
