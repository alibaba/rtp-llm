#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <map>
#include <set>
#include <utility>
#include <mutex>
#include <thread>
#include <atomic>
#include <vector>
#include <condition_variable>
#include <functional>
#include "rtp_llm/cpp/cache/connector/Meta.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorMetrics.h"
#include <torch/torch.h>

namespace rtp_llm {

struct P2PConnectorResourceEntry {
    int64_t            request_id;         // 请求 ID
    std::string        unique_key;         // 路由唯一标识（从 Meta::P2PRoutingContext 填充）
    KVCacheResourcePtr kv_cache_resource;  // KV cache 资源引用，用于保持引用计数
    int64_t            deadline_ms;        // Prefill 资源持有截止时间
    int64_t            request_deadline_ms;  // 原始请求截止时间
    int64_t            add_time_us;        // 添加时间

    // Published CPU tensors are owned by this payload and must remain read-only.
    struct SideChannelData {
        bool                 has_first_token  = false;
        int64_t              first_token_id   = 0;
        int32_t              total_reuse_len  = 0;
        int32_t              local_reuse_len  = 0;
        int32_t              remote_reuse_len = 0;
        int32_t              memory_reuse_len = 0;
        int32_t              disk_reuse_len   = 0;
        std::vector<int>     propose_tokens;
        torch::Tensor        propose_probs;
        torch::Tensor        propose_hidden;
        std::vector<int32_t> position_ids;
        std::map<std::string, torch::Tensor> first_token_tensors;
    };
};

// Prefill rank 0 holds request KV resources until StartLoad takes ownership.
// Lightweight terminal state remains for one hour after completion/cancellation/timeout.
class P2PConnectorResourceStore {
public:
    P2PConnectorResourceStore(const kmonitor::MetricsReporterPtr& metrics_reporter,
                              int                                 timeout_check_interval_ms);
    ~P2PConnectorResourceStore();

public:
    bool init();

    // GenerateStream registers the local request deadline; duplicates cannot renew it
    // while the record exists. A new handoff attempt must use a new unique_key.
    int64_t requestDeadline(const std::string& unique_key, int64_t timeout_ms);

    // Wait for GenerateStream registration without creating request state.
    // Returns zero when the load wait expires, is cancelled, or the request is terminal.
    int64_t waitForRequestDeadline(const std::string&    unique_key,
                                   int64_t               load_deadline_ms,
                                   std::function<bool()> is_cancelled = nullptr);

public:
    // addResource from Meta (extracts routing from Meta::p2pRouting())
    // Routing fields (unique_key, deadline_ms, request_id) are read from Meta::P2PRoutingContext
    bool addResource(const std::shared_ptr<Meta>& meta, const KVCacheResourcePtr& kv_cache_resource);

    // Init-time hook used to release per-request side resources (for example
    // computed per-layer buffers) when the stream-store drops the request due
    // to timeout or cancellation before decode consumes it.
    void setOnRequestReleased(std::function<void(int64_t, int64_t)> on_request_released);

    // Mark unique_key as cancelled: if the resource is already in the store, remove it
    // immediately; otherwise record the cancellation so that a future addResource() call
    // for the same key is rejected on arrival, preventing blocks from being pinned until
    // the next checkTimeout() cycle.
    void markCancelled(const std::string& unique_key, int64_t request_deadline_ms);

    // Record a terminal request after its resource has been consumed. Late
    // side-channel notifications and duplicate StartLoad calls are rejected.
    void markTerminal(const std::string& unique_key, int64_t request_deadline_ms);

    std::shared_ptr<P2PConnectorResourceEntry> waitAndStealResource(const std::string&    unique_key,
                                                                    int64_t               deadline_ms,
                                                                    int64_t               request_deadline_ms,
                                                                    std::function<bool()> is_cancelled = nullptr);

    // Publish locally computed first-token / SP data for the Prefill StartLoad handler.
    void publishPrefillPayload(const std::string&                           unique_key,
                                int64_t                                      deadline_ms,
                                P2PConnectorResourceEntry::SideChannelData&& data);

    // Wait on Prefill for local computation to publish the response payload.
    bool waitPrefillPayloadReady(const std::string&    unique_key,
                              int64_t               deadline_ms,
                              std::function<bool()> is_cancelled = nullptr);

    // Move the local payload out so the Prefill handler can serialize it for Decode.
    bool takePrefillPayload(const std::string& unique_key, P2PConnectorResourceEntry::SideChannelData& out_data);
    void clearPrefillPayload(const std::string& unique_key);

private:
    static constexpr int64_t kTombstoneRetentionMs = 60LL * 60 * 1000;

    void checkTimeout(int64_t now_ms);
    void runDeadlineLoop();
    std::optional<int64_t> nextDeadlineMsLocked() const;
    void reportMetrics(bool timeout, bool cancelled, int64_t wait_start_time_us);

    struct RequestState {
        int64_t request_deadline_ms;
        int64_t load_deadline_ms = 0;
        int64_t scheduled_deadline_ms = 0;
        int64_t terminal_expire_at_ms = 0;
        bool request_registered = false;
        bool consumed = false;
        bool terminal = false;
        std::optional<P2PConnectorResourceEntry::SideChannelData> side_channel_data;

        int64_t deadlineMs() const {
            return load_deadline_ms > 0 ? load_deadline_ms : request_deadline_ms;
        }
    };

    void scheduleDeadlineCheckLocked(const std::string& unique_key, RequestState& state);

    mutable std::mutex resource_map_mutex_;
    std::condition_variable resource_cv_;
    std::condition_variable deadline_cv_;
    std::map<std::string, std::shared_ptr<P2PConnectorResourceEntry>> resource_map_;
    // Lifecycle survives resource transfer without retaining the KV resource.
    std::map<std::string, RequestState> request_states_;
    // Exactly one index entry per request. Updating a deadline replaces its old
    // entry, so repeated publications cannot accumulate stale timer records.
    std::set<std::pair<int64_t, std::string>> deadline_index_;

    kmonitor::MetricsReporterPtr metrics_reporter_;

    std::thread deadline_thread_;
    bool        stopping_{false};
    uint64_t    deadline_generation_{0};

    std::function<void(int64_t, int64_t)> on_request_released_;

public:
    // Test hook: peek whether a unique_key is terminal.
    // Used by handleRead to decide between GENERATE_TIMEOUT (expired here)
    // and the generic RESOURCE_FAILED.
    bool isMarkedCancelled(const std::string& unique_key) const;

    // Read-only observation hook for the retained request state of a unique_key. Returns -1 when
    // no state is retained; used by tests instead of reaching into request_states_ directly.
    int64_t storedRequestDeadlineMs(const std::string& unique_key) const;
};

}  // namespace rtp_llm
