#pragma once

#include "rtp_llm/cpp/cache/connector/p2p/P2PWriteWorkerUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/IKVCacheSender.h"
#include "autil/LoopThread.h"
#include "autil/ThreadPool.h"

namespace rtp_llm {

class P2PWorkerDecodeWrite {
public:
    P2PWorkerDecodeWrite(P2PConnectorWorkerConfig             config,
                         std::shared_ptr<LayerBlockConverter> converter,
                         transfer::IKVCacheSenderPtr          sender);
    ~P2PWorkerDecodeWrite();
    bool      init(size_t sender_thread_count = 4, size_t sender_queue_size = 10000);
    ErrorInfo write(int64_t                   request_id,
                    const std::string&        unique_key,
                    int64_t                   deadline_ms,
                    const P2PWorkerRoutePlan& worker_plan);
    bool      cancelWrite(const std::string& unique_key, int64_t deadline_ms);
    bool      queryWriteStatus(const std::string& unique_key, WriteTaskStatus& status);

private:
    void                                                                 cleanup();
    P2PConnectorWorkerConfig                                             config_;
    std::shared_ptr<LayerBlockConverter>                                 converter_;
    transfer::IKVCacheSenderPtr                                          sender_;
    std::shared_ptr<autil::ThreadPoolBase>                               sender_pool_;
    std::mutex                                                           mutex_;
    std::map<std::string, std::shared_ptr<p2p_internal::WriteTaskGroup>> groups_;
    autil::LoopThreadPtr                                                 cleanup_thread_;
};

}  // namespace rtp_llm
