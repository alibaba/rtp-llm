#pragma once

#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache/connector/p2p/DecodeWriteHelper.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PBroadcastClient.h"
#include "autil/ThreadPool.h"
#include <condition_variable>
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
    bool      init();
    void      stop();
    void      handleWrite(const P2PConnectorStartWriteRequestPB& request,
                          P2PConnectorStartWriteResponsePB&      response,
                          std::function<bool()>                  is_cancelled = nullptr);
    void      cancelWrite(const P2PConnectorCancelWriteRequestPB& request, P2PConnectorStartWriteResponsePB& response);
    ErrorInfo workerRegistered(const P2PConnectorBroadcastTpRequestPB& request);
    size_t    inflightContextCount() const;

private:
    struct RequestState {
        explicit RequestState(int64_t deadline): deadline_ms(deadline) {}
        void      fail(const ErrorInfo& error);
        ErrorInfo errorInfo() const;
        bool      resourceHoldPending() const;

        const int64_t           deadline_ms;
        mutable std::mutex      mutex;
        std::condition_variable cv;
        KVCacheResourcePtr      resource;
        std::vector<bool>       registered_workers;
        uint64_t                plan_digest{0};
        bool                    cancelled{false};
        bool                    receive_stopped{true};
        bool                    stopped{false};
        ErrorInfo               error;
    };

    std::shared_ptr<RequestState> findRequest(const std::string& unique_key) const;
    void                          pruneRequestsLocked(int64_t now_ms);
    int64_t                       holdUntilMs(int64_t deadline_ms) const;

    std::shared_ptr<const PlanResult> planFor(int decode_tp_size);
    ErrorInfo                         buildPrefillRankRoutes(const TransferPlan&             plan,
                                                             KVCacheResource&                resource,
                                                             size_t                          start_block,
                                                             size_t                          block_count,
                                                             P2PBroadcastClient::RankRoutes& routes,
                                                             int64_t*                        planned_bytes) const;

    const P2PConnectorSchedulerConfig                config_;
    std::mutex                                       plan_cache_mutex_;
    std::map<int, std::shared_ptr<const PlanResult>> plan_cache_;
    KVCacheAllocatorPtr                              allocator_;
    std::shared_ptr<P2PBroadcastClient>              client_;
    kmonitor::MetricsReporterPtr                     metrics_reporter_;
    std::vector<std::string>                         transfer_addrs_;
    DecodeWriteHelper                                server_caller_;
    std::shared_ptr<autil::ThreadPool>                   control_pool_;
    mutable std::mutex                                   requests_mutex_;
    std::condition_variable                              requests_cv_;
    std::map<std::string, std::shared_ptr<RequestState>> requests_;
    std::multimap<int64_t, std::string>                  request_expirations_;
    bool                                                 stopping_{false};
    bool                                             initialized_{false};
};

}  // namespace rtp_llm
