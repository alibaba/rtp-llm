#pragma once

#include "rtp_llm/cpp/cache/connector/p2p/P2PWriteWorkerUtil.h"
#include "autil/LoopThread.h"

namespace rtp_llm {

class P2PWorkerPrefillWrite {
public:
    P2PWorkerPrefillWrite(P2PConnectorWorkerConfig             config,
                          std::shared_ptr<LayerBlockConverter> converter,
                          transfer::IKVCacheReceiverPtr        receiver);
    ~P2PWorkerPrefillWrite();
    bool      init();
    ErrorInfo handleWrite(int64_t                    request_id,
                          const std::string&         unique_key,
                          int64_t                    deadline_ms,
                          const P2PWorkerRoutePlan&  worker_plan,
                          std::function<ErrorInfo()> on_registered = {});
    bool      cancelWrite(const std::string& unique_key, int64_t deadline_ms);
    bool
    queryLeaseStatus(const std::string& unique_key, bool& sealed, int& started_ops, int& finished_ops, bool& stopped);

private:
    void                                 cleanup();
    void                                 removeRecvTasks(const std::shared_ptr<p2p_internal::WriteTaskGroup>& group);
    P2PConnectorWorkerConfig             config_;
    std::shared_ptr<LayerBlockConverter> converter_;
    transfer::IKVCacheReceiverPtr        receiver_;
    std::mutex                           mutex_;
    std::map<std::string, std::shared_ptr<p2p_internal::WriteTaskGroup>> groups_;
    autil::LoopThreadPtr                                                 cleanup_thread_;
};

}  // namespace rtp_llm
