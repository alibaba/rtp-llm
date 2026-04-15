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
    int64_t                      request_id;         // 请求 ID
    std::string                  unique_key;         // 路由唯一标识（从 Meta::P2PRoutingContext 填充）
    KVCacheResourcePtr           kv_cache_resource;  // KV cache 资源引用，用于保持引用计数
    int64_t                      deadline_ms;        // 过期时间
    int64_t                      add_time_us;        // 添加时间

    // Side-channel data (filled by prefill when first token / SP data is produced)
    struct SideChannelData {
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
    SideChannelData             side_channel_data;
    bool                        side_channel_ready = false;
    mutable std::mutex          side_channel_mutex;
    std::condition_variable     side_channel_cv;
};

// p2p_connector 拉取过程中对应的资源不会释放。当 prefill 请求完成时, 需要将资源从 store 中移除, 如果超时未完成,
// 需要将资源移除。unique_key 目前由 decode 生成, 后续可能由 master 统一生成保证全局唯一。
class P2PConnectorResourceStore {
public:
    P2PConnectorResourceStore(const kmonitor::MetricsReporterPtr& metrics_reporter, int timeout_check_interval_ms);
    ~P2PConnectorResourceStore();

public:
    bool init();

public:
    // addResource from Meta (extracts routing from Meta::p2pRouting())
    // Routing fields (unique_key, deadline_ms, request_id) are read from Meta::P2PRoutingContext
    bool addResource(const std::shared_ptr<Meta>& meta, const KVCacheResourcePtr& kv_cache_resource);

    std::shared_ptr<P2PConnectorResourceEntry> waitAndStealResource(const std::string&    unique_key,
                                                                    int64_t               deadline_ms,
                                                                    std::function<bool()> is_cancelled = nullptr);

    // Notify side-channel data ready (called by prefill when first token / SP data is produced)
    void notifySideChannelReady(const std::string& unique_key, const P2PConnectorResourceEntry::SideChannelData& data);

    // Wait for side-channel data ready (called by P2PConnector when filling response)
    bool waitSideChannelReady(const std::string& unique_key, int64_t deadline_ms, std::function<bool()> is_cancelled = nullptr);

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
    kmonitor::MetricsReporterPtr                                      metrics_reporter_;

    autil::LoopThreadPtr check_timeout_thread_;
    int                  timeout_check_interval_ms_;
};

}  // namespace rtp_llm
