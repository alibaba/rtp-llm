#pragma once

#include "rtp_llm/cpp/cache/connector/p2p/P2PWriteWorkerUtil.h"
#include <condition_variable>
#include <map>
#include <thread>
#include <unordered_set>

namespace rtp_llm {

class P2PWorkerPrefillWrite {
public:
    P2PWorkerPrefillWrite(P2PConnectorWorkerConfig             config,
                          std::shared_ptr<LayerBlockConverter> converter,
                          transfer::IKVCacheReceiverPtr        receiver);
    ~P2PWorkerPrefillWrite();
    bool      init();
    ErrorInfo handleWrite(const std::string&         unique_key,
                          int64_t                    deadline_ms,
                          const P2PWorkerRoutePlan&  worker_plan,
                          std::function<ErrorInfo()> on_registered = {});
    bool      cancelWrite(const std::string& unique_key, int64_t deadline_ms);
    bool
    queryLeaseStatus(const std::string& unique_key, bool& sealed, int& started_ops, int& finished_ops, bool& stopped);

private:
    using ExpiringEntry = std::pair<std::string, std::weak_ptr<p2p_internal::WriteTaskGroup>>;

    void    cleanup();
    int64_t cleanupLocked(int64_t now_ms);
    void    runCleanupLoop();
    void    stopCleanupThread();
    void    removeRecvTasks(const std::shared_ptr<p2p_internal::WriteTaskGroup>& group);

    P2PConnectorWorkerConfig                                             config_;
    std::shared_ptr<LayerBlockConverter>                                 converter_;
    transfer::IKVCacheReceiverPtr                                        receiver_;
    std::mutex                                                           mutex_;
    std::condition_variable                                              cleanup_cv_;
    bool                                                                 stopping_{false};
    std::map<std::string, std::shared_ptr<p2p_internal::WriteTaskGroup>> groups_;
    std::unordered_set<std::string>                                      active_groups_;
    std::multimap<int64_t, ExpiringEntry>                                expiring_groups_;
    std::thread                                                          cleanup_thread_;
};

}  // namespace rtp_llm
