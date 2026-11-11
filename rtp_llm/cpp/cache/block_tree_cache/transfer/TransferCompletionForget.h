#pragma once

#include <functional>
#include <memory>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferCompletionTypes.h"

namespace rtp_llm {

class BroadcastManager;

// Deletes only worker evidence which is already BUFFER_SAFE. It never changes
// frozen references, business results or a worker's RUNNING execution.
class TransferCompletionForget {
public:
    using LocalForget = std::function<TransferTaskRegistry::ForgetResult(const TransferExecutionTrace&)>;

    explicit TransferCompletionForget(std::shared_ptr<BroadcastManager> broadcast_manager,
                                       LocalForget local_forget = {});

    // One bounded batch, with a positive timeout and the original participants.
    // Entries preserve input order; rows are indexed by original rank. Invalid
    // keys are rejected individually without preventing other keys from finishing.
    TransferCompletionForgetResult forget(const TransferCompletionTarget& target,
                                            const std::vector<TransferExecutionTrace>& traces,
                                            int timeout_ms) const;

private:
    std::shared_ptr<BroadcastManager> broadcast_manager_;
    LocalForget                      local_forget_;
};

}  // namespace rtp_llm
