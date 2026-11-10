#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/FrozenTask.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTaskRegistry.h"

namespace rtp_llm {

class BroadcastManager;

struct TransferCompletionRankResult {
    size_t                      rank{0};
    std::string                 server_address;
    TransferTaskRegistry::State state{TransferTaskRegistry::State::UNKNOWN};
    std::optional<int>           grpc_code;
    std::string                 error;
    bool                        cq_received{false};
    bool                        cq_ok{false};
};

struct TransferCompletionQueryResult {
    TransferExecutionTrace                    trace_id{uint64_t{0}};
    std::vector<TransferCompletionRankResult> ranks;
    bool                                      all_safe{false};
    std::string                               error;
};

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
