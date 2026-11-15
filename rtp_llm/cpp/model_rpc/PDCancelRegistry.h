#pragma once

#include "rtp_llm/cpp/model_rpc/RpcServerRuntimeMeta.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include <algorithm>
#include <cstddef>
#include "absl/status/statusor.h"
#include <atomic>
#include <cstdint>
#include <memory>
#include <map>
#include <set>
#include <mutex>
#include <string>
#include <utility>
#include <unordered_map>
#include <vector>

namespace rtp_llm {

// Decode publishes its terminal after local physical cleanup, retaining the
// control record separately until Prefill cleanup is confirmed or its retention expires.
// Admission and the cancel latch share a lock, including cancel-before-enqueue.
class PDCancelRegistry {
public:
    struct Entry {
        explicit Entry(TaskIdentity task): identity(task) {}
        TaskIdentity      identity;
        std::string       unique_key;
        std::string       downstream_address;
        int64_t           deadline_ms = 0;
        bool              control_only = false;
        ErrorInfo         cancel_reason;
        std::atomic<bool> canceled{false};
        GenerateStreamPtr stream;
        std::atomic<bool> local_done{false};
        bool              downstream_done = false;
        std::atomic<bool> terminal{false};
        bool                 cleanup_overdue = false;
        int64_t           retain_until_ms = 0;
        std::atomic<int64_t> next_cancel_retry_ms{0};
        std::atomic<int64_t> cancel_retry_delay_ms{50};
    };
    using Handle = std::shared_ptr<Entry>;

    explicit PDCancelRegistry(std::shared_ptr<RpcServerRuntimeMeta> meta): meta_(std::move(meta)) {}

    ErrorInfo           admit(const GenerateInputPB& request,
                              const std::string&     key,
                              const std::string&     downstream_address,
                              int64_t                deadline_ms,
                              Handle&                handle,
                              bool                   control_only = false);
    Handle              find(int64_t request_id);
    void                attach(const Handle& handle, const GenerateStreamPtr& stream);
    void                finishLocal(const Handle& handle);
    absl::StatusOr<CancelStatusPB> cancel(int64_t request_id, const ErrorInfo& reason);
    bool                isCanceled(const std::string& unique_key);
    std::vector<Handle> pending();
    void                finishDownstream(const Handle& handle);
    bool                complete(const Handle& handle);

    void beginRead(const std::string& key);
    void endRead(const std::string& key);
    class ReadGuard {
    public:
        ReadGuard(PDCancelRegistry& registry, std::string key): registry_(registry), key_(std::move(key)) {
            registry_.beginRead(key_);
        }
        ~ReadGuard() {
            registry_.endRead(key_);
        }

    private:
        PDCancelRegistry& registry_;
        std::string       key_;
    };

    // Declare before GenerateContext so local completion is observed only after
    // context cleanup, including Prefill RPC joins and P2P load cancellation.
    class CallGuard {
    public:
        CallGuard(PDCancelRegistry& registry, Handle handle): registry_(registry), handle_(std::move(handle)) {}
        ~CallGuard() {
            registry_.finishLocal(handle_);
        }

    private:
        PDCancelRegistry& registry_;
        Handle            handle_;
    };

private:
    void                                    reapFinishedStreams();
    void                                    trimLocked(int64_t now_ms);
    void                                    scheduleExpirationLocked(const Handle& handle, int64_t expires_at_ms);
    void                                    reportPressureLocked();
    static constexpr int64_t                kRetentionMs = 10 * 60 * 1000;
    static constexpr int64_t                kActiveReadExpiryRetryMs = 1000;
    // Cancel-before-admission records have their own budget. Existing request
    // IDs remain cancellable when either budget is full.
    size_t                                  record_capacity_       = 1u << 20;
    size_t                                  control_capacity_      = 1u << 20;
    size_t                                  control_record_count_  = 0;
    size_t                                  overdue_cancels_       = 0;
    size_t                                  request_rejections_    = 0;
    size_t                                  control_rejections_    = 0;
    int64_t                                 last_pressure_log_ms_  = 0;
    static constexpr size_t                 kMaxExpirationsPerPass = 128;
    std::mutex                              mutex_;
    std::unordered_map<int64_t, Handle>     entries_;
    struct CancelFence {
        int64_t   expires_at_ms;
        ErrorInfo reason;
    };
    std::unordered_map<int64_t, CancelFence> absent_fences_;
    std::multimap<int64_t, int64_t>         fence_expirations_;
    std::set<std::pair<int64_t, int64_t>>   entry_expirations_;
    static constexpr size_t                  kPendingBatchSize = 128;
    std::map<int64_t, Handle>                pending_cancels_;
    std::map<int64_t, Handle>                local_pending_cancels_;
    std::map<int64_t, Handle>                pending_stream_releases_;
    int64_t                                  stream_release_cursor_ = 0;
    int64_t                                  local_pending_cursor_ = 0;
    int64_t                                  retry_pending_cursor_ = 0;
    std::unordered_map<std::string, size_t> canceled_keys_;
    std::unordered_map<std::string, size_t> active_reads_;
    std::shared_ptr<RpcServerRuntimeMeta>   meta_;
};

}  // namespace rtp_llm
