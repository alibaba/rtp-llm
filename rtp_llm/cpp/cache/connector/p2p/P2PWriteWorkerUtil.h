#pragma once

#include <c10/core/Event.h>

#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorConfig.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerBlockConverter.h"
#include "rtp_llm/cpp/cache/connector/p2p/DecodeTargetWriteLease.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PWorkerRoute.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PWriteRpcUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/IKVCacheReceiver.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include <map>
#include <mutex>
#include <condition_variable>
#include <deque>

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
        stop_at_ms(writebackSteadyDeadline(deadline)),
        source_resource_(std::move(source_resource)) {}
    std::mutex                                           mutex;
    std::map<std::string, transfer::IKVCacheRecvTaskPtr> tasks;
    std::shared_ptr<DecodeTargetWriteLease>              lease;
    int64_t                                              stop_at_ms;
    bool                                                 started   = false;
    bool                                                 cancelled = false;
    ErrorInfo                                            error;
    std::condition_variable                              cv;

    void               cancel();
    void               watchTask(const std::string& key, const transfer::IKVCacheRecvTaskPtr& task);
    void               queryLeaseStatus(bool& sealed, int& started_ops, int& finished_ops, bool& stopped);
    ErrorInfo          waitDone();
    void               releaseStoppedTasks();
    KVCacheResourcePtr sourceResource();
    bool               beginSourceTransfer(int64_t deadline);
    void               releaseSourceResource();

private:
    KVCacheResourcePtr source_resource_;
    bool               source_submitted_{false};
};

ErrorInfo buildWriteUnits(const std::string&                          unique_key,
                          int64_t                                     deadline_ms,
                          const P2PWorkerRoutePlan&                   worker_plan,
                          const P2PConnectorWorkerConfig&             config,
                          const std::shared_ptr<LayerBlockConverter>& converter,
                          bool                                        sending,
                          std::vector<WriteTransferUnit>&             units);

}  // namespace p2p_internal

// A per-rank watermark on the Decode compute stream. PP retains a distinct
// event for each in-flight step so a sender never waits for a newer batch.
class WritebackStepTracker {
public:
    void      publish(uint64_t step, const std::shared_ptr<c10::Event>& event);
    ErrorInfo wait(uint64_t step, p2p_internal::WriteTaskGroup& group);

private:
    void advanceCompleted(uint64_t through_step);

    std::mutex                                                   mutex_;
    uint64_t                                                     completed_step_ = 0;
    std::deque<std::pair<uint64_t, std::shared_ptr<c10::Event>>> pending_;
};

}  // namespace rtp_llm
