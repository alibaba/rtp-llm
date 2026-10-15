#pragma once

#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorConfig.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerBlockConverter.h"
#include "rtp_llm/cpp/cache/connector/p2p/DecodeTargetWriteLease.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PWorkerRoute.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/IKVCacheReceiver.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include <map>
#include <mutex>
#include <condition_variable>

namespace rtp_llm {

namespace p2p_internal {

struct WriteTransferUnit {
    std::string               key;
    transfer::KeyBlockInfoMap blocks;
    std::string               ip;
    uint32_t                  port = 0;
};

// Local transfer tasks; Decode rank 0 also retains the source until scheduler cleanup.
struct WriteTaskGroup: public std::enable_shared_from_this<WriteTaskGroup> {
    explicit WriteTaskGroup(int64_t deadline, KVCacheResourcePtr source_resource = nullptr):
        lease(std::make_shared<DecodeTargetWriteLease>()),
        deadline_ms(deadline),
        source_resource_(std::move(source_resource)) {}
    std::mutex                                           mutex;
    std::map<std::string, transfer::IKVCacheRecvTaskPtr> tasks;
    std::shared_ptr<DecodeTargetWriteLease>              lease;
    int64_t                                              deadline_ms;
    bool                                                 started   = false;
    bool                                                 cancelled = false;
    ErrorInfo                                            error;
    std::condition_variable                              cv;

    void      cancel();
    void      watchTask(const std::string& key, const transfer::IKVCacheRecvTaskPtr& task);
    ErrorInfo waitDone();
    void      waitStopped();
    void      releaseStoppedTasks();
    KVCacheResourcePtr sourceResource();
    bool               beginSourceTransfer(int64_t deadline);
    void               releaseSourceResource();

private:
    KVCacheResourcePtr                                     source_resource_;
    bool                                                   source_submitted_{false};
};

ErrorInfo buildWriteUnits(const std::string&                          unique_key,
                          int64_t                                     deadline_ms,
                          const P2PWorkerRoutePlan&                   worker_plan,
                          const P2PConnectorWorkerConfig&             config,
                          const std::shared_ptr<LayerBlockConverter>& converter,
                          bool                                        sending,
                          std::vector<WriteTransferUnit>&             units);

}  // namespace p2p_internal
}  // namespace rtp_llm
