#pragma once

#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <map>
#include <mutex>
#include <thread>
#include <atomic>
#include <vector>
#include <condition_variable>
#include <functional>
#include "autil/LoopThread.h"
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
    int64_t            request_deadline_ms;  // 原始请求截止时间，用于终态 tombstone
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
    };
};

// Prefill rank 0 holds request KV resources until StartLoad takes ownership.
// Lightweight request state remains until the original request deadline.
class P2PConnectorResourceStore {
public:
    P2PConnectorResourceStore(const kmonitor::MetricsReporterPtr& metrics_reporter,
                              int                                 timeout_check_interval_ms);
    ~P2PConnectorResourceStore();

public:
    bool init();

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

    // Notify side-channel data ready (called by prefill when first token / SP data is produced)
    void notifySideChannelReady(const std::string&                           unique_key,
                                int64_t                                      deadline_ms,
                                P2PConnectorResourceEntry::SideChannelData&& data);

    // Wait for side-channel data ready (called by P2PConnector when filling response)
    bool waitSideChannelReady(const std::string&    unique_key,
                              int64_t               deadline_ms,
                              std::function<bool()> is_cancelled = nullptr);

    // Try to consume side-channel data from the request state (returns true if data was found and moved)
    bool consumeSideChannelData(const std::string& unique_key, P2PConnectorResourceEntry::SideChannelData& out_data);
    void clearSideChannelData(const std::string& unique_key);

private:
    void checkTimeout();
    void reportMetrics(bool timeout, bool cancelled, int64_t wait_start_time_us);

    struct RequestState {
        int64_t request_deadline_ms;
        int64_t load_deadline_ms = 0;
        bool consumed = false;
        bool terminal = false;
        std::optional<P2PConnectorResourceEntry::SideChannelData> side_channel_data;

        int64_t deadlineMs() const {
            return load_deadline_ms > 0 ? load_deadline_ms : request_deadline_ms;
        }
    };

    mutable std::mutex resource_map_mutex_;
    std::condition_variable resource_cv_;
    std::map<std::string, std::shared_ptr<P2PConnectorResourceEntry>> resource_map_;
    // Lifecycle survives resource transfer without retaining the KV resource.
    std::map<std::string, RequestState> request_states_;

    kmonitor::MetricsReporterPtr metrics_reporter_;

    autil::LoopThreadPtr check_timeout_thread_;
    int                  timeout_check_interval_ms_;

    std::function<void(int64_t, int64_t)> on_request_released_;

public:
    // Test hook: peek whether a unique_key is terminal.
    // Used by handleRead to decide between GENERATE_TIMEOUT (expired here)
    // and the generic RESOURCE_FAILED.
    bool isMarkedCancelled(const std::string& unique_key) const;
};

}  // namespace rtp_llm
