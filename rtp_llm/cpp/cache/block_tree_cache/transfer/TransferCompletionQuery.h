#pragma once

#include <functional>
#include <memory>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/FrozenTask.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferCompletionTypes.h"

namespace rtp_llm {

class BroadcastManager;

// One read-only observation of the execution's original participants. Results
// are independent of its frozen failure snapshot and never release references.
class TransferCompletionQuery {
public:
    // The local reader only observes the selected registry; it must not perform
    // remote I/O or mutate admission/completion state.
    using LocalReader = std::function<TransferTaskRegistry::State(const TransferExecutionTrace&)>;

    explicit TransferCompletionQuery(std::shared_ptr<BroadcastManager> broadcast_manager,
                                     LocalReader local_reader = {});

    // Positive timeout only. Missing participants, malformed responses and
    // transport errors never prove completion. Rows are indexed by original rank.
    TransferCompletionQueryResult query(const FrozenTask::ExecutionInfo& execution, int timeout_ms) const;

private:
    std::shared_ptr<BroadcastManager> broadcast_manager_;
    LocalReader                      local_reader_;
};

}  // namespace rtp_llm
