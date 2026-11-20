#pragma once

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {

// Shared result and transfer protection for one node/GroupSet load generation.
// Completion is published after tree settlement, outside the cache mutex.
class LoadOperation final {
public:
    LoadOperation(TransferDescriptor descriptor, GroupSetPtr group_set, uint64_t workflow_id);

    const TransferDescriptor& descriptor() const {
        return descriptor_;
    }
    uint64_t workflowId() const {
        return workflow_id_;
    }
    void unreferenceSource();
    void complete(bool success) noexcept;
    void subscribe(uint64_t subscriber_id, std::function<void(ErrorInfo)> callback);
    void unsubscribe(uint64_t subscriber_id);

private:
    const TransferDescriptor                           descriptor_;
    const GroupSetPtr                                  group_set_;
    const uint64_t                                     workflow_id_;
    std::mutex                                         mutex_;
    bool                                               done_{false};
    ErrorInfo                                          error_{ErrorInfo::OkStatus()};
    std::map<uint64_t, std::function<void(ErrorInfo)>> callbacks_;
};

}  // namespace rtp_llm
