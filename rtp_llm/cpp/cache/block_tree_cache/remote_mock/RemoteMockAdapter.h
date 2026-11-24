#pragma once

#include <functional>
#include <memory>
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/cache/block_tree_cache/remote_mock/RemoteMockTypes.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {
class PerRankBlockTransferEngine;
class TransferTask;

// The external engine owner outlives callbacks. Shutdown order (outside cache
// locks): stopAccepting(), drain(callback notifying the shutdown owner), then
// destroy on that external thread before destroying GroupSets/transfer engine.
// Callbacks may reenter cancel/drain but must not destroy the last adapter owner
// or block waiting for other callbacks on the same transport/continuation pool.
class RemoteMockAdapter {
public:
    using DoneCallback = std::function<void(ErrorInfo)>;
    // Startup only, outside scheduler/cache locks. Throws on invalid or unreachable
    // configuration; all-rank READY and the reverse Host probe precede return.
    static std::shared_ptr<RemoteMockAdapter> create(const KVCacheConfig&                        config,
                                                     const ParallelismConfig&                    parallel,
                                                     std::vector<GroupSetPtr>                    groups,
                                                     std::shared_ptr<PerRankBlockTransferEngine> transfer_engine);
    ~RemoteMockAdapter();
    RemoteMockSourceHandle freezeSource(size_t group, uint64_t admission_offset, uint64_t logical_index) const;
    // Callback is the sole terminal signal, including a zero operation ID.
    // A rejected or failed submission may still settle asynchronously.
    uint64_t load(const RemoteMockLoadPlan& plan, DoneCallback callback);
    // Reusable bounded Host snapshot transport. No request-completion policy or
    // GPU capture; the callback follows publisher completion, outer RPC drain
    // and release of every borrowed transport alias of the Host backing.
    uint64_t transferHost(const RemoteMockHostTransfer& plan, DoneCallback callback);
    // Startup-only role attachment. Direct Host-transfer users need no writer.
    void                     initializeWriteback(const std::shared_ptr<RemoteMockAdapter>& self,
                                                 bool                                      owns_device_allocations,
                                                 size_t                                    max_descriptors);
    bool                     writeback(TransferTask task, DoneCallback callback);
    RemoteMockWritebackStats writebackStats() const;
    size_t                   maxWritebackDescriptors() const;
    bool                     cancel(uint64_t operation);
    void                     stopAccepting();
    // Invoke the returned callable exactly once while this provider is alive.
    std::function<void()>                prepareDrain(std::function<void()> callback);
    void                                 drain(std::function<void()> callback);
    RemoteMockRuntimeSnapshot            runtimeSnapshot() const;
    RemoteMockSession                    session() const;
    RemoteMockRankCredits                credits() const;
    RemoteMockAdapterStats               stats() const;
    RemoteMockLimits                     limits() const;
    const std::vector<RemoteMockLayout>& layouts() const;

private:
    struct Impl;
    explicit RemoteMockAdapter(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};
}  // namespace rtp_llm
