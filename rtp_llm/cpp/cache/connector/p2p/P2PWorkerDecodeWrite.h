#pragma once

#include "rtp_llm/cpp/cache/connector/p2p/P2PWriteWorkerUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/IKVCacheSender.h"
#include "autil/ThreadPool.h"
#include <condition_variable>
#include <map>
#include <thread>
#include <unordered_set>

namespace rtp_llm {

class P2PWorkerDecodeWrite {
public:
    P2PWorkerDecodeWrite(P2PConnectorWorkerConfig              config,
                         std::shared_ptr<LayerBlockConverter>  converter,
                         transfer::IKVCacheSenderPtr           sender,
                         std::shared_ptr<WritebackStepTracker> step_tracker = nullptr);
    ~P2PWorkerDecodeWrite();
    bool      init(size_t sender_thread_count = 4, size_t sender_queue_size = 10000);
    ErrorInfo write(const std::string&        unique_key,
                    int64_t                   deadline_ms,
                    const P2PWorkerRoutePlan& worker_plan,
                    uint64_t                  kv_step = 0);
    std::shared_ptr<p2p_internal::WriteTaskGroup>
         holdSourceResource(const std::string& unique_key, int64_t deadline_ms, KVCacheResourcePtr source_resource);
    bool cancelWrite(const std::string& unique_key, int64_t deadline_ms);
    bool
    queryLeaseStatus(const std::string& unique_key, bool& sealed, int& started_ops, int& finished_ops, bool& stopped);

private:
    using ExpiringEntry = std::pair<std::string, std::weak_ptr<p2p_internal::WriteTaskGroup>>;

    void    cleanup();
    int64_t cleanupLocked(int64_t now_ms);
    void    runCleanupLoop();
    void    stopCleanupThread();

    P2PConnectorWorkerConfig                                             config_;
    std::shared_ptr<LayerBlockConverter>                                 converter_;
    transfer::IKVCacheSenderPtr                                          sender_;
    std::shared_ptr<WritebackStepTracker>                                step_tracker_;
    std::shared_ptr<autil::ThreadPoolBase>                               sender_pool_;
    std::mutex                                                           mutex_;
    std::condition_variable                                              cleanup_cv_;
    bool                                                                 stopping_{false};
    std::map<std::string, std::shared_ptr<p2p_internal::WriteTaskGroup>> groups_;
    std::unordered_set<std::string>                                      active_groups_;
    std::multimap<int64_t, ExpiringEntry>                                expiring_groups_;
    std::thread                                                          cleanup_thread_;
};

}  // namespace rtp_llm
