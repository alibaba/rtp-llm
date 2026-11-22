#pragma once

#include <map>
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/FrozenTask.h"

namespace rtp_llm {

class FrozenTaskReclaimer {
public:
    struct Trace {
        TransferCompletionKey key;
        TransferCompletionTarget participants;
        const TransferResult& result;
        std::string backend{"local"};
    };

    explicit FrozenTaskReclaimer(std::shared_ptr<TransferCompletionCleanup> cleanup):
        cleanup_(std::move(cleanup)) {}

    const std::shared_ptr<TransferCompletionCleanup>& cleanup() const { return cleanup_; }

    void freezeTargetBlocksWithTraceIDAndStillIOWorkerID(const Trace& trace_id,
                                                        std::vector<FrozenTask::TargetBlock> target_blocks,
                                                        std::vector<size_t> still_io_worker_ids);
    bool unfreezeTargetBlocksInOneWorkerWithTraceID(const TransferCompletionKey& trace_id, size_t worker_id);
    std::vector<std::shared_ptr<FrozenTask>> frozenTasks() const;

private:
    std::shared_ptr<TransferCompletionCleanup> cleanup_;
    mutable std::mutex mutex_;
    mutable std::map<std::string, std::shared_ptr<FrozenTask>> tasks_;
};

}
