#pragma once

#include "rtp_llm/cpp/cache/connector/p2p/P2PBroadcastClient.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorConfig.h"
#include "rtp_llm/cpp/cache/connector/p2p/plan/TransferPlan.h"

#include <map>
#include <mutex>
#include <tuple>

namespace rtp_llm {

class KVCacheResource;

class P2PWritebackPlanner {
public:
    P2PWritebackPlanner(const P2PConnectorSchedulerConfig& config, RoleType role): config_(config), role_(role) {}

    std::shared_ptr<const PlanResult> planFor(int                peer_tp,
                                              int                peer_cp,
                                              const std::string& peer_layout,
                                              int                peer_workers = 0,
                                              const std::string& unique_key   = {});
    ErrorInfo buildRankRoutes(const TransferPlan&             plan,
                              KVCacheResource&                resource,
                              size_t                          start,
                              size_t                          count,
                              int                             cp_size,
                              P2PBroadcastClient::RankRoutes& routes,
                              int64_t*                        bytes) const;

private:
    PlanResult buildPlan(int                peer_tp,
                         int                peer_cp,
                         const std::string& peer_layout,
                         int                peer_workers,
                         const std::string& unique_key,
                         size_t             source_replica_offset) const;
    const P2PConnectorSchedulerConfig&                                                          config_;
    const RoleType                                                                              role_;
    std::mutex                                                                                  mutex_;
    std::map<std::tuple<int, int, size_t, std::string, int>, std::shared_ptr<const PlanResult>> cache_;
};

}  // namespace rtp_llm
