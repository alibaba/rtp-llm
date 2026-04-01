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
#include "rtp_llm/cpp/cache/connector/IGenerateStream.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorMetrics.h"

namespace rtp_llm {

struct P2PConnectorResourceEntry {
    int64_t            request_id;         // 请求 ID
    IGenerateStreamPtr generate_stream;    // GenerateStream 接口，包含 token ids 和 reuse 信息
    KVCacheResourcePtr kv_cache_resource;  // KV cache 资源引用，用于保持引用计数
    int64_t            deadline_ms;        // 过期时间
    int64_t            add_time_us;        // 添加时间
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
    bool addResource(const IGenerateStreamPtr& generate_stream, const KVCacheResourcePtr& kv_cache_resource);

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
    kmonitor::MetricsReporterPtr                                      metrics_reporter_;

    autil::LoopThreadPtr check_timeout_thread_;
    int                  timeout_check_interval_ms_;
};

}  // namespace rtp_llm
