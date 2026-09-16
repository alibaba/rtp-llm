#pragma once

#include "rtp_llm/cpp/cache/connector/Meta.h"
#include "rtp_llm/cpp/cache/connector/p2p/DecodeWriteHelper.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorAsyncContext.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorConfig.h"
#include "rtp_llm/cpp/cache/connector/p2p/plan/TransferPlan.h"
#include "autil/LockFreeThreadPool.h"
#include <map>
#include <mutex>

namespace rtp_llm {

class P2PSchedulerDecodeWrite {
public:
    P2PSchedulerDecodeWrite(P2PConnectorSchedulerConfig         config,
                            std::shared_ptr<P2PBroadcastClient> client,
                            kmonitor::MetricsReporterPtr        metrics_reporter = nullptr);
    ~P2PSchedulerDecodeWrite();
    bool                                           init();
    void stop();
    std::shared_ptr<P2PConnectorAsyncWriteContext> asyncWrite(KVCacheResourcePtr resource,
                                                         std::vector<int>        token_ids,
                                                         int                     input_length,
                                                         size_t                  kv_ready_token_count,
                                                         Meta::P2PRoutingContext routing);
    size_t                                         inflightContextCount() const {
        return checker_.inflightContextCount();
    }

private:
    std::shared_ptr<const PlanResult> planFor(int prefill_tp_size);
    ErrorInfo buildDecodeRankRoutes(const TransferPlan&              plan,
                                    KVCacheResource&                 resource,
                                    size_t                           start_block,
                                    size_t                           block_count,
                                    P2PBroadcastClient::RankRoutes& routes,
                                    int64_t*                         planned_bytes) const;

    void startAsyncWriteCalls(const std::shared_ptr<P2PConnectorAsyncWriteContext>& context,
                              const std::vector<int>&                               token_ids,
                              int                                                   input_length,
                              size_t                                                kv_ready_token_count,
                              const Meta::P2PRoutingContext&                        routing,
                              int64_t                                               deadline_ms);

    const P2PConnectorSchedulerConfig    config_;
    std::mutex                            plan_cache_mutex_;
    std::map<int, std::shared_ptr<const PlanResult>> plan_cache_;
    std::shared_ptr<P2PBroadcastClient>  client_;
    kmonitor::MetricsReporterPtr         metrics_reporter_;
    DecodeWriteHelper                    server_caller_;
    std::shared_ptr<autil::LockFreeThreadPool> async_write_pool_;
    P2PConnectorAsyncWriteContextChecker checker_;
};

}  // namespace rtp_llm
