#pragma once

#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorConfig.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PBroadcastClient.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorMetrics.h"
#include "rtp_llm/cpp/cache/connector/p2p/plan/KVCacheTransferPlanner.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <tuple>
#include <utility>
#include <memory>
#include <string>
#include <vector>

namespace rtp_llm {

class P2PSchedulerPrefillRead {
public:
    P2PSchedulerPrefillRead(P2PConnectorSchedulerConfig                config,
                                 const kmonitor::MetricsReporterPtr&        metrics_reporter,
                                 const std::shared_ptr<P2PBroadcastClient>& tp_broadcast_client);
    ~P2PSchedulerPrefillRead() = default;

public:
    using PrefillRouteKeys = std::map<int, std::vector<int64_t>>;

    struct PreparedPlan {
        std::shared_ptr<const TransferPlan> plan;
        uint64_t                            digest = 0;
        std::set<int>                       active_route_ids;
        PrefillRouteKeys                    prefill_route_keys;
    };

    // A null prepared plan is a result-only handoff.
    ErrorInfo sendKVCache(const std::string&                                   unique_key,
                          int64_t                                              request_id,
                          const std::vector<std::pair<std::string, uint32_t>>& decode_transfer_servers,
                          int64_t                                              deadline_ms,
                          const PreparedPlan*                                  prepared_plan,
                          std::function<bool()>                                is_cancelled             = nullptr,
                          int64_t                                              request_deadline_ms      = 0,
                          int64_t                                              communication_timeout_ms = 0);

    /// @brief 与 decode 侧用**同一个 planner 函数**算镜像 plan。
    /// Decode TP width comes from StartLoad. Decode CP is inferred from the local sharding configuration.
    std::shared_ptr<const PlanResult>
    planFor(int decode_tp_size, const std::string& unique_key, const std::string& decode_pipeline_layout = {});

    /// Retain the validated plan across the StartLoad resource wait.
    ErrorResult<PreparedPlan>
    preparePlan(int                                                           decode_tp_size,
                uint64_t                                                      decode_plan_digest,
                const std::string&                                            unique_key,
                size_t                                                        decode_worker_count,
                std::set<int>                                                 active_route_ids,
                const std::string&                                            decode_pipeline_layout = {},
                const google::protobuf::RepeatedPtrField<PrefillRouteKeysPB>& prefill_route_keys     = {});

    /// @brief 把 plan 投影成每个 prefill worker 的 route 列表。
    /// Prefill routes name the destination worker and, for PP, the local layers to send.
    P2PBroadcastClient::RankRoutes buildPrefillRankRoutes(const TransferPlan&     plan,
                                                          size_t                  worker_num,
                                                          const std::set<int>&    active_route_ids   = {},
                                                          const PrefillRouteKeys& prefill_route_keys = {}) const;

private:
    /// 轮询 result->done()；is_cancelled 或当前时间超过 deadline_ms 时发 CANCEL_HANDLE_READ。
    /// deadline_exceeded_out 若为非空，在因超时而发起 cancel 时置 true（与客户端 cancel 区分）。
    std::shared_ptr<P2PBroadcastClient::Result>
    waitForBroadcastCompletion(const std::shared_ptr<P2PBroadcastClient::Result>& result,
                               const std::string&                                 unique_key,
                               int64_t                                            request_id,
                               int64_t                                            deadline_ms,
                               int64_t                                            request_deadline_ms,
                               std::function<bool()>                              is_cancelled,
                               bool*                                              deadline_exceeded_out = nullptr);

private:
    mutable std::mutex                                       plan_cache_mutex_;
    std::map<std::tuple<int, size_t, std::string>, std::shared_ptr<const PlanResult>> plan_cache_;
    const P2PConnectorSchedulerConfig   config_;
    kmonitor::MetricsReporterPtr        metrics_reporter_;
    std::shared_ptr<P2PBroadcastClient> tp_broadcast_client_;
};

}  // namespace rtp_llm
