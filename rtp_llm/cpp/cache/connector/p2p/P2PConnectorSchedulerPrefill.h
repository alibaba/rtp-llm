#pragma once

#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorConfig.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PBroadcastClient.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorMetrics.h"
#include "rtp_llm/cpp/cache/connector/p2p/plan/KVCacheTransferPlanner.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include <functional>
#include <map>
#include <mutex>
#include <utility>
#include <memory>
#include <string>
#include <vector>

namespace rtp_llm {

class P2PConnectorSchedulerPrefill {
public:
    P2PConnectorSchedulerPrefill(P2PConnectorSchedulerConfig                config,
                                 const kmonitor::MetricsReporterPtr&        metrics_reporter,
                                 const std::shared_ptr<P2PBroadcastClient>& tp_broadcast_client);
    ~P2PConnectorSchedulerPrefill() = default;

public:
    ErrorInfo sendKVCache(const KVCacheResourcePtr&                            resource,
                          const std::string&                                   unique_key,
                          int64_t                                              request_id,
                          const std::vector<std::pair<std::string, uint32_t>>& decode_transfer_servers,
                          int64_t                                              deadline_ms,
                          std::function<bool()>                                is_cancelled = nullptr,
                          bool                                                 no_transfer = false,
                          int64_t                                              request_deadline_ms = 0);

    /// @brief 与 decode 侧用**同一个 planner 函数**算镜像 plan。
    /// decode 的 (tp_size, cp_size) 由本端推导：tp_size = decode_transfer_servers.size()，
    /// cp_size = kv_cache_sharded ? decode_tp_size : 1（部署级同配开关）。
    std::shared_ptr<const PlanResult> planFor(int decode_tp_size);

    /// @brief 把 plan 投影成每个 prefill worker 的 route 列表。
    /// prefill 方向不带 layer_blocks —— 在所有被允许的 CP 形态下 (src_rank, dst_rank, tag)
    /// 唯一确定一条 route，而 worker 自己 writeByLayer 产出的本地投影恰好等于该 route 的键集。
    P2PBroadcastClient::RankRoutes buildPrefillRankRoutes(const TransferPlan& plan, size_t worker_num) const;

private:
    /// 轮询 result->done()；is_cancelled 或当前时间超过 deadline_ms 时发 CANCEL_HANDLE_READ。
    /// deadline_exceeded_out 若为非空，在因超时而发起 cancel 时置 true（与客户端 cancel 区分）。
    std::shared_ptr<P2PBroadcastClient::Result>
    waitForBroadcastCompletion(const std::shared_ptr<P2PBroadcastClient::Result>& result,
                               const std::string&                                 unique_key,
                               int64_t                                            request_id,
                               int64_t                                            deadline_ms,
                               std::function<bool()>                              is_cancelled,
                               bool*                                              deadline_exceeded_out = nullptr);

private:
    mutable std::mutex                                       plan_cache_mutex_;
    std::map<int, std::shared_ptr<const PlanResult>>         plan_cache_;
    const P2PConnectorSchedulerConfig   config_;
    kmonitor::MetricsReporterPtr        metrics_reporter_;
    std::shared_ptr<P2PBroadcastClient> tp_broadcast_client_;
};

}  // namespace rtp_llm
