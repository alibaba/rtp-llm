#pragma once

#include <chrono>
#include <memory>
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
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

namespace rtp_llm {

struct P2PConnectorResourceEntry {
    int64_t            request_id;         // 请求 ID
    std::string        unique_key;         // 路由唯一标识（从 Meta::P2PRoutingContext 填充）
    KVCacheResourcePtr kv_cache_resource;  // KV cache 资源引用，用于保持引用计数
    int64_t            deadline_ms;        // 持有阶段为 hold deadline，取走后为 transfer deadline
    int64_t            request_deadline_ms;  // 原始请求截止时间，用于终态 tombstone
    int64_t            add_time_us;        // 添加时间
};

// p2p_connector 拉取过程中对应的资源不会释放。当 prefill 请求完成时, 需要将资源从 store 中移除, 如果超时未完成,
// 需要将资源移除。unique_key 目前由 decode 生成, 后续可能由 master 统一生成保证全局唯一。
class P2PConnectorResourceStore {
public:
    P2PConnectorResourceStore(const kmonitor::MetricsReporterPtr& metrics_reporter,
                              int                                 timeout_check_interval_ms,
                              int64_t                             prefill_resource_hold_ms = 60 * 1000,
                              int64_t                             cancelled_keys_ttl_ms    = 3600 * 1000);
    ~P2PConnectorResourceStore();

public:
    bool init();

public:
    // addResource from Meta (extracts routing from Meta::p2pRouting())
    // Routing fields (unique_key, deadline_ms, request_id) are read from Meta::P2PRoutingContext
    bool addResource(const std::shared_ptr<Meta>& meta, const KVCacheResourcePtr& kv_cache_resource);

    // Init-time hooks run under resource_map_mutex_; they must not re-enter ResourceStore.
    // Lock order is ResourceStore -> result store. A rejected hook makes the request terminal.
    void setOnRequestRegistered(std::function<bool(const std::string&, int64_t, int64_t)> on_request_registered);
    void setOnRequestAcquired(std::function<bool(const std::string&, int64_t)> on_request_acquired);

    // Init-time hook for terminal requests, including already-consumed entries.
    // request_id is -1 when no resource entry remains; unique_key is always available.
    void setOnRequestReleased(std::function<void(const std::string&, int64_t, int64_t)> on_request_released);

    // Mark unique_key as cancelled: if the resource is already in the store, remove it
    // immediately; otherwise record the cancellation so that a future addResource() call
    // for the same key is rejected on arrival, preventing blocks from being pinned until
    // the next checkTimeout() cycle.
    void markCancelled(const std::string& unique_key, int64_t request_deadline_ms = 0);

    // Record a terminal request after its resource has been consumed.
    // Duplicate requests are rejected until the request deadline.
    void markTerminal(const std::string& unique_key, int64_t request_deadline_ms = 0);

    std::shared_ptr<P2PConnectorResourceEntry> waitAndStealResource(const std::string&    unique_key,
                                                                    int64_t               deadline_ms,
                                                                    std::function<bool()> is_cancelled = nullptr);

private:
    void checkTimeout();
    void reportMetrics(bool timeout, bool cancelled, int64_t wait_start_time_us);

    // 持有 resource_map_mutex_（unique_lock）时调用。
    bool                                       waitForResourceOrCancellation(std::unique_lock<std::mutex>&         lock,
                                                                             const std::string&                    unique_key,
                                                                             std::chrono::system_clock::time_point timeout_tp,
                                                                             const std::function<bool()>&          is_cancelled);
    std::shared_ptr<P2PConnectorResourceEntry> stealResourceEntryLocked(const std::string& unique_key);

private:
    mutable std::mutex                                                resource_map_mutex_;
    std::condition_variable                                           resource_cv_;
    std::map<std::string, std::shared_ptr<P2PConnectorResourceEntry>> resource_map_;
    // unique_key → expire_at_ms: terminal keys whose resources were cancelled or expired.
    // Keeping the terminal state until the original request deadline makes a late StartLoad
    // fail immediately instead of waiting for a resource that can no longer arrive.
    std::map<std::string, int64_t> cancelled_keys_;
    kmonitor::MetricsReporterPtr metrics_reporter_;

    autil::LoopThreadPtr check_timeout_thread_;
    int                  timeout_check_interval_ms_;

    // Cap entry->deadline_ms to currentTimeMs() + this value, so prefill stops
    // pinning KV blocks for the full business deadline (commonly ~1h) when
    // decode never sends StartLoad. See P2PConnectorResourceStore.cc::addResource.
    int64_t prefill_resource_hold_ms_;
    // Fallback lifetime used only when a legacy caller does not provide a
    // valid original request deadline.
    int64_t cancelled_keys_ttl_ms_;
    std::function<void(const std::string&, int64_t, int64_t)> on_request_released_;
    std::function<bool(const std::string&, int64_t, int64_t)> on_request_registered_;
    std::function<bool(const std::string&, int64_t)> on_request_acquired_;

public:
    // Test hook: peek whether a unique_key is currently in cancelled_keys_.
    // Used by handleRead to decide between GENERATE_TIMEOUT (expired here)
    // and the generic RESOURCE_FAILED.
    bool isMarkedCancelled(const std::string& unique_key) const;
};

}  // namespace rtp_llm
