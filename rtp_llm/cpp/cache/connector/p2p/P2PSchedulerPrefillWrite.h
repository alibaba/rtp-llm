#pragma once

#include "rtp_llm/cpp/cache/CoordinatorCacheManager.h"
#include "rtp_llm/cpp/cache/connector/p2p/DecodeLoadHelper.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PBroadcastClient.h"
#include "autil/ThreadPool.h"
#include <thread>
#include <atomic>
#include <condition_variable>
#include "rtp_llm/cpp/cache/connector/p2p/P2PWritebackPlanner.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PWriteRpcUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PWritebackResourceUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/plan/TransferPlan.h"
#include <future>
#include <map>
#include <mutex>

namespace rtp_llm {

class P2PSchedulerPrefillWrite {
public:
    P2PSchedulerPrefillWrite(P2PConnectorSchedulerConfig              config,
                             std::shared_ptr<CoordinatorCacheManager> coordinator_manager,
                             const kmonitor::MetricsReporterPtr&      metrics_reporter,
                             std::shared_ptr<P2PBroadcastClient>      client);
    ~P2PSchedulerPrefillWrite();
    bool      init();
    void      stop(int64_t steady_deadline_ms = 0);
    void      handleWrite(const P2PConnectorStartWriteBackRequestPB& request,
                          P2PConnectorStartWriteBackResponsePB&      response,
                          std::function<bool()>                      is_cancelled = nullptr);
    void      cancelWrite(const P2PConnectorCancelWriteBackRequestPB& request,
                          P2PConnectorStartWriteBackResponsePB&       response);
    ErrorInfo workerRegistered(const P2PConnectorBroadcastTpRequestPB& request);
    size_t    inflightContextCount() const;

private:
    struct PrefillWriteRequest {
        explicit PrefillWriteRequest(int64_t steady_deadline_ms): stop_at_ms(steady_deadline_ms) {}
        void      fail(const ErrorInfo& error);
        void      cancel(const ErrorInfo& error);
        ErrorInfo beginCommit();
        ErrorInfo errorInfo() const;
        bool      isStopped() const;
        uint64_t  progressSeq() const;
        void      notifyProgress();
        void      waitForProgress(uint64_t observed_seq);

        const int64_t           stop_at_ms;
        const int64_t           start_time_us{currentTimeUs()};
        int64_t                 resource_start_time_us{0};
        mutable std::mutex      mutex;
        std::condition_variable cv;
        uint64_t                progress_seq{0};
        std::vector<bool>       registered_workers;
        uint64_t                plan_digest{0};
        bool                    cancelled{false};
        bool                    committing{false};
        bool                    stopped{false};
        ErrorInfo               error;

    private:
        void failLocked(const ErrorInfo& failure);
    };

    class ActiveWriteHandler {
    public:
        ActiveWriteHandler(P2PSchedulerPrefillWrite& scheduler, const std::string& unique_key);
        ~ActiveWriteHandler();
        ActiveWriteHandler(const ActiveWriteHandler&) = delete;
        ActiveWriteHandler& operator=(const ActiveWriteHandler&) = delete;

        bool admitted() const {
            return scheduler_ != nullptr;
        }
        bool existingRequestStopped() const {
            return existing_request_stopped_;
        }

    private:
        P2PSchedulerPrefillWrite* scheduler_{nullptr};
        bool existing_request_stopped_{true};
    };

    struct WriteTransfer {
        KVCacheResourcePtr                resource;
        std::shared_ptr<LoadAsyncContext> load_context;
        std::future<std::shared_ptr<P2PBroadcastClient::Result>> receive_submission;
        std::shared_ptr<P2PBroadcastClient::Result>              receive_result;
        std::future<std::shared_ptr<P2PBroadcastClient::Result>> send_submission;
        std::shared_ptr<P2PBroadcastClient::Result>              send_result;
        bool receive_started{false};
        bool receive_stopped{true};

        bool targetStopped() const;
        void observe(PrefillWriteRequest& request, const std::function<void()>& notify);
    };

    static constexpr int64_t kControlInitialIntervalMs = 10;

    struct WriteDrain {
        std::string                          unique_key;
        std::shared_ptr<PrefillWriteRequest> state;
        WriteTransfer                        transfer;
        WriteSchedulerMetricsCollector       metrics;
        int64_t                              deadline_ms;
        int64_t                              start_us;
        int64_t                              hold_start_us;
        int64_t                              drain_start_us;

        std::future<std::shared_ptr<P2PBroadcastClient::TpBroadcastResult>> control_submission{};
        std::shared_ptr<P2PBroadcastClient::TpBroadcastResult>              control_result{};
        bool    cancel_confirmed{false};
        int64_t next_control_ms{0};
        int64_t interval_ms{kControlInitialIntervalMs};
    };

    void runDrainLoop();
    void advanceDrain(WriteDrain& drain);
    void finishWrite(WriteDrain& drain);

    std::shared_ptr<PrefillWriteRequest> findRequest(const std::string& unique_key) const;
    std::shared_ptr<PrefillWriteRequest> findRequestLocked(const std::string& unique_key) const;
    void                                 pruneRequestsLocked(int64_t now_ms);
    int64_t                              holdUntilMs(int64_t deadline_ms) const;
    void                                 checkRetainedResources(bool force = true);

    std::shared_ptr<const PlanResult> planFor(int                decode_tp_size,
                                              const std::string& decode_layout = {},
                                              int                workers       = 0,
                                              const std::string& unique_key    = {});
    std::shared_ptr<P2PBroadcastClient> sourceClient(const std::vector<std::string>& addresses);
    ErrorInfo                           buildPrefillRankRoutes(const TransferPlan&             plan,
                                                               KVCacheResource&                resource,
                                                               size_t                          start_block,
                                                               size_t                          block_count,
                                                               P2PBroadcastClient::RankRoutes& routes,
                                                               int64_t*                        planned_bytes) const;
    ErrorInfo                           selectSendRoutes(const P2PConnectorStartWriteBackRequestPB& request,
                                                         const P2PBroadcastClient::RankRoutes&      receive_routes,
                                                         P2PBroadcastClient::RankRoutes&            send_routes) const;

    const P2PConnectorSchedulerConfig                                           config_;
    P2PWritebackPlanner                                                         planner_{config_, RoleType::PREFILL};
    std::shared_ptr<CoordinatorCacheManager>                                    coordinator_manager_;
    std::shared_ptr<P2PBroadcastClient>                                         client_;
    kmonitor::MetricsReporterPtr                                                metrics_reporter_;
    std::vector<std::string>                                                    transfer_addrs_;
    p2p_internal::PeerClientCache<std::vector<std::string>, P2PBroadcastClient> source_clients_;
    std::shared_ptr<autil::ThreadPool>                                          control_pool_;
    mutable std::mutex                                                          requests_mutex_;
    std::condition_variable                                                     requests_cv_;
    std::map<std::string, std::shared_ptr<PrefillWriteRequest>>                 requests_;
    std::map<std::string, std::shared_ptr<PrefillWriteRequest>>                 completed_requests_;
    std::map<std::string, KVCacheResourcePtr>                                   retained_resources_;
    std::vector<std::shared_ptr<LoadAsyncContext>>                              retained_load_contexts_;
    std::multimap<int64_t, std::string>                                         request_expirations_;
    size_t                                                                      active_write_handlers_{0};
    bool                                                                        stopping_{false};
    bool                                                                        stopped_{false};
    std::atomic<bool>                                                           drain_aborted_{false};
    bool                                                                        initialized_{false};
    WritebackRetainedKv                                                         retained_kv_;
    std::atomic<int64_t>                                                        last_retained_check_ms_{0};
    std::thread                                                                 drain_thread_;
    std::map<std::string, std::shared_ptr<WriteDrain>>                          pending_drains_;
};

}  // namespace rtp_llm
