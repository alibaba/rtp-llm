#pragma once

#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorConfig.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerBlockConverter.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PTransferLease.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PWorkerRoute.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/IKVCacheReceiver.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include <map>
#include <mutex>
#include <optional>

namespace rtp_llm {

struct WriteTaskStatus {
    bool      sealed        = false;
    int       started_ops   = 0;
    int       finished_ops  = 0;
    bool      stopped       = false;
    bool      write_success = false;
    ErrorInfo error;
};

namespace p2p_internal {

struct WriteTransferUnit {
    std::string               key;
    transfer::KeyBlockInfoMap blocks;
    std::string               ip;
    uint32_t                  port = 0;
};

// All addresses are borrowed. The scheduler must hold their KV resources until stopped.
struct WriteTaskGroup {
    explicit WriteTaskGroup(int64_t deadline): lease(std::make_shared<P2PTransferLease>()), deadline_ms(deadline) {}
    std::mutex                                           mutex;
    std::map<std::string, transfer::IKVCacheRecvTaskPtr> tasks;
    std::shared_ptr<P2PTransferLease>                    lease;
    int64_t                                              deadline_ms;
    bool                                                 cancelled = false;
    ErrorInfo                                            error;
    std::optional<bool>                                  terminal_success;

    void cancel();
    bool fillStatus(WriteTaskStatus& status);
    void releaseStoppedTasks();
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
