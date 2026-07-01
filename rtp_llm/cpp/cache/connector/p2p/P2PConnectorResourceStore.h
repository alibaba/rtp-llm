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
#include "rtp_llm/cpp/cache/connector/KVCacheConnector.h"
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
    int64_t            deadline_ms;        // 过期时间
    int64_t            add_time_us;        // 添加时间

    // Side-channel data (filled by prefill when first token / SP data is produced)
    struct SideChannelData {
        bool                 has_first_token  = false;
        int64_t              first_token_id   = 0;
        int32_t              total_reuse_len  = 0;
        int32_t              local_reuse_len  = 0;
        int32_t              remote_reuse_len = 0;
        int32_t              memory_reuse_len = 0;
        std::vector<int>     propose_tokens;
        TensorPB             propose_probs;
        TensorPB             propose_hidden;
        std::vector<int32_t> position_ids;
    };
    SideChannelData         side_channel_data;
    bool                    side_channel_ready = false;
    mutable std::mutex      side_channel_mutex;
    std::condition_variable side_channel_cv;
};

struct P2PSideChannelStoreEntry {
    P2PConnectorResourceEntry::SideChannelData data;
    int64_t                                    deadline_ms = 0;
    int64_t                                    add_time_us = 0;
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

    // Init-time hook used to release per-request side resources (for example
    // computed per-layer buffers) when the stream-store drops the request due
    // to timeout or cancellation before decode consumes it.
    void setOnRequestReleased(std::function<void(int64_t)> on_request_released);

    // Mark unique_key as cancelled: if the resource is already in the store, remove it
    // immediately; otherwise record the cancellation so that a future addResource() call
    // for the same key is rejected on arrival, preventing blocks from being pinned until
    // the next checkTimeout() cycle.
    void markCancelled(const std::string& unique_key);

    std::shared_ptr<P2PConnectorResourceEntry> waitAndStealResource(const std::string&    unique_key,
                                                                    int64_t               deadline_ms,
                                                                    std::function<bool()> is_cancelled = nullptr);

    // Notify side-channel data ready (called by prefill when first token / SP data is produced)
    void notifySideChannelReady(const std::string&                                unique_key,
                                int64_t                                           deadline_ms,
                                const P2PConnectorResourceEntry::SideChannelData& data);

    // Wait for side-channel data ready (called by P2PConnector when filling response)
    bool waitSideChannelReady(const std::string&    unique_key,
                              int64_t               deadline_ms,
                              std::function<bool()> is_cancelled = nullptr);

    // Try to consume side-channel data from the independent map (returns true if data was found and copied)
    bool consumeSideChannelData(const std::string& unique_key, P2PConnectorResourceEntry::SideChannelData& out_data);
    void clearSideChannelData(const std::string& unique_key);

private:
    void checkTimeout();
    void reportMetrics(bool timeout, bool cancelled, int64_t wait_start_time_us);

    // 持有 resource_map_mutex_（unique_lock）时调用。
    bool                                       waitForResourceOrCancellation(std::unique_lock<std::mutex>&         lock,
                                                                             const std::string&                    unique_key,
                                                                             std::chrono::system_clock::time_point timeout_tp,
                                                                             const std::function<bool()>&          is_cancelled);
    std::shared_ptr<P2PConnectorResourceEntry> stealResourceEntryLocked(const std::string& unique_key);
    void                                       clearSideChannelDataLocked(const std::string& unique_key);

private:
    mutable std::mutex                                                resource_map_mutex_;
    std::condition_variable                                           resource_cv_;
    std::map<std::string, std::shared_ptr<P2PConnectorResourceEntry>> resource_map_;
    // unique_key → cancel_time_ms: keys for which the decode-side gRPC was cancelled before the
    // resource arrived. addResource() rejects these to avoid pinning KV blocks; entries are
    // purged by checkTimeout() after kCancelledKeyTTLMs.
    std::map<std::string, int64_t> cancelled_keys_;
    // Side-channel data stored independently from resource entries.
    // This allows notifySideChannelReady to store data even after the entry has been stolen.
    std::mutex                                      side_channel_map_mutex_;
    std::condition_variable                         side_channel_cv_;
    std::map<std::string, P2PSideChannelStoreEntry> side_channel_data_map_;

    kmonitor::MetricsReporterPtr metrics_reporter_;

    autil::LoopThreadPtr check_timeout_thread_;
    int                  timeout_check_interval_ms_;

    // Cap entry->deadline_ms to currentTimeMs() + this value, so prefill stops
    // pinning KV blocks for the full business deadline (commonly ~1h) when
    // decode never sends StartLoad. See P2PConnectorResourceStore.cc::addResource.
    int64_t prefill_resource_hold_ms_;
    // Lifetime of cancelled_keys_ tombstones; should cover the longest expected
    // skew between addResource and a late-arriving StartLoad so handleRead can
    // distinguish "expired on prefill" from "never seen".
    int64_t cancelled_keys_ttl_ms_;
    std::function<void(int64_t)> on_request_released_;

public:
    // Test hook: peek whether a unique_key is currently in cancelled_keys_.
    // Used by handleRead to decide between GENERATE_TIMEOUT (expired here)
    // and the generic RESOURCE_FAILED.
    bool isMarkedCancelled(const std::string& unique_key) const;
};

}  // namespace rtp_llm
