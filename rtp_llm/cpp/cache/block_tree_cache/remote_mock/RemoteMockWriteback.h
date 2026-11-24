#pragma once
#include <functional>
#include <memory>
#include "rtp_llm/cpp/cache/block_tree_cache/remote_mock/RemoteMockTypes.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {
class RemoteMockAdapter;
class PerRankBlockTransferEngine;
class GroupSet;
// Physical per-rank writer. The adapter owns this object and outlives every
// callback. Its engine and GroupSets remain alive through stop/drain. This API
// does not decide request completion or count logical requests.
class RemoteMockWriteback {
public:
    using DoneCallback = std::function<void(ErrorInfo)>;
    RemoteMockWriteback(const std::shared_ptr<RemoteMockAdapter>& adapter,
                        PerRankBlockTransferEngine*               engine,
                        std::vector<std::shared_ptr<GroupSet>>    groups,
                        bool                                      owns_device_allocations,
                        size_t                                    max_descriptors);
    ~RemoteMockWriteback();
    // Admission reserves the bounded record window before pinning/allocating.
    // Rejection is not submitted; its callback may be inline. Accepted geometry
    // stays frozen and source refs survive all D2H copies and final ACKs.
    bool submit(TransferTask completed_geometry, DoneCallback callback);
    void stopAccepting();
    // Invoke the returned callable exactly once while this provider is alive.
    std::function<void()>    prepareDrain(std::function<void()> callback);
    void                     drain(std::function<void()> callback);
    RemoteMockWritebackStats stats() const;
    size_t                   maxDescriptors() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace rtp_llm
