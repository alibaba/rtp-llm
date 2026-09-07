#pragma once

#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {

class BroadcastManager;

class MultiRankBlockTransferEngine {
public:
    MultiRankBlockTransferEngine(std::vector<GroupSetPtr>          group_sets,
                                 std::shared_ptr<BroadcastManager> broadcast_manager);

    std::shared_ptr<AsyncContext> execute(TransferTask task) const;

private:
    std::vector<GroupSetPtr>          group_sets_;
    std::shared_ptr<BroadcastManager> broadcast_manager_;
};

using MultiRankBlockTransferEnginePtr = std::shared_ptr<MultiRankBlockTransferEngine>;

}  // namespace rtp_llm
