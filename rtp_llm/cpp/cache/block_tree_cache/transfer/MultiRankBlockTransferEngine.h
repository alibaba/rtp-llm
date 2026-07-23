#pragma once

#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/ComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {

class BroadcastManager;

class MultiRankBlockTransferEngine {
public:
    MultiRankBlockTransferEngine(std::vector<ComponentGroupPtr>    component_groups,
                                 std::shared_ptr<BroadcastManager> broadcast_manager);

    bool execute(const std::vector<TransferDescriptor>& descriptors, int timeout_ms) const;

private:
    std::vector<ComponentGroupPtr>    component_groups_;
    std::shared_ptr<BroadcastManager> broadcast_manager_;
};

using MultiRankBlockTransferEnginePtr = std::shared_ptr<MultiRankBlockTransferEngine>;

}  // namespace rtp_llm
