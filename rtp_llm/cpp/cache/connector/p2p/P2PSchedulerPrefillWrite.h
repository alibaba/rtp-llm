#pragma once

#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorAsyncContext.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorConfig.h"
#include "rtp_llm/cpp/cache/connector/p2p/plan/TransferPlan.h"
#include <map>
#include <mutex>

namespace rtp_llm {

class P2PSchedulerPrefillWrite {
public:
    P2PSchedulerPrefillWrite(P2PConnectorSchedulerConfig         config,
                             KVCacheAllocatorPtr                 allocator,
                             const kmonitor::MetricsReporterPtr& metrics_reporter,
                             std::shared_ptr<P2PBroadcastClient> client);
    ~P2PSchedulerPrefillWrite();
    bool   init();
    void stop();
    void   handleWrite(const P2PConnectorStartWriteRequestPB& request,
                       P2PConnectorStartWriteResponsePB&      response,
                       std::function<bool()>                  is_cancelled = nullptr);
    size_t inflightContextCount() const {
        return checker_.inflightContextCount();
    }

private:
    std::shared_ptr<const PlanResult> planFor(int decode_tp_size);
    ErrorInfo buildPrefillRankRoutes(const TransferPlan&              plan,
                                     KVCacheResource&                 resource,
                                     size_t                           start_block,
                                     size_t                           block_count,
                                     P2PBroadcastClient::RankRoutes& routes,
                                     int64_t*                         planned_bytes) const;

    const P2PConnectorSchedulerConfig          config_;
    std::mutex                                  plan_cache_mutex_;
    std::map<int, std::shared_ptr<const PlanResult>> plan_cache_;
    KVCacheAllocatorPtr                        allocator_;
    std::shared_ptr<P2PBroadcastClient>        client_;
    kmonitor::MetricsReporterPtr               metrics_reporter_;
    std::vector<std::string>                   transfer_addrs_;
    P2PConnectorAsyncWriteContextChecker       checker_;
    bool                                       initialized_{false};
};

}  // namespace rtp_llm
