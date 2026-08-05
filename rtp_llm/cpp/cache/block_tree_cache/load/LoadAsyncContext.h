#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {

class LoadContextCoordinator;

class LoadAsyncContext: public AsyncContext {
public:
    enum class State : int {
        PENDING          = 0,
        CANCEL_REQUESTED = 1,
        SUCCEEDED        = 2,
        FAILED           = 3,
        CANCELLED        = 4
    };

    LoadAsyncContext(std::vector<TransferDescriptor>                load_descs,
                     std::vector<bool>                              joined_load,
                     size_t                                         matched_blocks,
                     uint64_t                                       context_id,
                     const std::shared_ptr<LoadContextCoordinator>& coordinator);
    ~LoadAsyncContext() override;

    bool     empty() const;
    uint64_t contextId() const;
    size_t   matchedBlocks() const;
    size_t   matchedBlocks(Tier tier) const;

    void setTargetBlocks(size_t desc_index, std::vector<BlockIdxType> target_blocks);

    const std::vector<TransferDescriptor>& loadDescs() const;
    const std::vector<bool>& joinedLoads() const;

    bool commit();
    void abort();

    bool requestCancel();
    bool isRequestCanceled() const;
    bool completeOne(bool success);
    bool onTaskFail();
    void waitDone() override;
    bool done() const override;
    bool success() const override;

private:
    void markAborted();
    void rebuildMatchedBlocksByTier();

    // Keep the coordinator alive so a registered context can perform RAII abort.
    std::shared_ptr<LoadContextCoordinator> coordinator_;
    const uint64_t                          context_id_;

    std::vector<TransferDescriptor> load_descs_;
    std::vector<bool>     joined_load_;
    const size_t                    matched_blocks_{0};
    std::array<size_t, 3>           matched_blocks_by_tier_{};

    std::atomic<State>      state_{State::PENDING};
    mutable std::mutex      mutex_;
    std::condition_variable cv_;
    size_t                  remaining_transfer_count_;
    bool                    has_failure_{false};
};

class LoadContextCoordinator: public std::enable_shared_from_this<LoadContextCoordinator> {
public:
    using CommitCallback = std::function<bool(const std::shared_ptr<LoadAsyncContext>& context)>;
    using AbortCallback  = std::function<void(LoadAsyncContext& context)>;

    LoadContextCoordinator(CommitCallback commit_callback, AbortCallback abort_callback);

    std::shared_ptr<LoadAsyncContext>
    create(std::vector<TransferDescriptor> load_descs, std::vector<bool> joined_load, size_t matched_blocks);
    bool                              registerContext(const std::shared_ptr<LoadAsyncContext>& context);
    bool                              commit(uint64_t context_id);
    bool                              abort(LoadAsyncContext& context) noexcept;
    void                              shutdown();

private:
    // Pending registration must not keep a context alive; its destructor performs RAII abort.
    using PendingContextMap = std::unordered_map<uint64_t, std::weak_ptr<LoadAsyncContext>>;

    void retireActiveCallback();

    std::mutex              mutex_;
    std::condition_variable cv_;
    bool                    accepting_{true};
    uint64_t                next_context_id_{1};
    size_t                  active_callbacks_{0};
    PendingContextMap       pending_contexts_;
    CommitCallback          commit_callback_;
    AbortCallback           abort_callback_;
    std::function<void()>   shutdown_wait_observer_for_test_;
};

}  // namespace rtp_llm
