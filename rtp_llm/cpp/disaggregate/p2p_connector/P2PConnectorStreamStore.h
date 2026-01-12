#pragma once

#include <memory>
#include <string>
#include <map>
#include <mutex>
#include <thread>
#include <atomic>
#include <vector>
#include "autil/LoopThread.h"
#include "rtp_llm/cpp/cache/connector/KVCacheConnector.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/disaggregate/p2p_connector/P2PConnectorMetrics.h"

namespace rtp_llm {

/// @brief P2P Connector 资源条目，存储 prefill 完成后的关键数据
struct P2PConnectorResourceEntry {
    int64_t            request_id;         // 请求 ID
    IGenerateStreamPtr generate_stream;    // GenerateStream 接口，包含 token ids 和 reuse 信息
    KVCacheResourcePtr kv_cache_resource;  // KV cache 资源引用，用于保持引用计数
    int64_t            deadline_ms;        // 过期时间
    int64_t            add_time_us;        // 添加时间
};

/// @brief P2P Connector Resource Store，存储 prefill 完成后的资源
// 实现思路: 当请求中有 pd_sep_unique_key 时, 认为这个 stream 会有一个对应的 P2PConnector load 的请求, 用来保证
// p2p_connector 拉取过程中对应的资源不会释放。当 prefill 请求完成时, 需要将资源从 store 中移除, 如果超时未完成,
// 需要将资源移除。unique_key 目前由 decode 生成, 后续可能由 master 统一生成保证全局唯一。
class P2PConnectorStreamStore {
public:
    P2PConnectorStreamStore(const kmonitor::MetricsReporterPtr& metrics_reporter);
    ~P2PConnectorStreamStore();

public:
    bool init();

public:
    /// @brief 添加资源条目
    void addResource(const std::string&        unique_key,
                     int64_t                   request_id,
                     const IGenerateStreamPtr& generate_stream,
                     const KVCacheResourcePtr& kv_cache_resource,
                     int64_t                   deadline_ms);

    /// @brief 获取并移除资源条目
    std::shared_ptr<P2PConnectorResourceEntry> stealResource(const std::string& unique_key);

private:
    void checkTimeout();
    void reportMetrics(bool timeout, int64_t wait_start_time_us);

private:
    mutable std::mutex                                                resource_map_mutex_;
    std::map<std::string, std::shared_ptr<P2PConnectorResourceEntry>> resource_map_;
    kmonitor::MetricsReporterPtr                                      metrics_reporter_;

    autil::LoopThreadPtr check_timeout_thread_;
};

}  // namespace rtp_llm
