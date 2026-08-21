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
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/StorageBackend.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {

class LoadContextCoordinator;

struct LoadMatchResult {
    // Keep bool-returning callbacks source-compatible. A callback failure that
    // does not provide a capacity verdict is terminal rather than eligible for
    // PREFILL's post-allocation transfer fallback.
    LoadMatchResult(bool success, MallocStatus malloc_status = MallocStatus::NONE):
        success(success),
        malloc_status(success ? MallocStatus::NONE :
                      malloc_status == MallocStatus::NONE ? MallocStatus::INTERNAL_ERROR :
                                                           malloc_status) {}

    bool         success;
    MallocStatus malloc_status;
};

class LoadAsyncContext: public AsyncContext, public std::enable_shared_from_this<LoadAsyncContext> {
public:
    enum class State : int {
        PENDING,
        SUCCEEDED,
        FAILED
    };
    using MatchCallback = std::function<LoadMatchResult(LoadAsyncContext&, size_t matched_blocks)>;

    LoadAsyncContext(std::vector<TransferDescriptor>                load_descs,
                     std::vector<bool>                              joined_load,
                     size_t                                         local_matched_blocks,
                     uint64_t                                       context_id,
                     const std::shared_ptr<LoadContextCoordinator>& coordinator,
                     std::shared_ptr<StorageBackend>                storage_backend = nullptr,
                     StorageRequest                                 storage_request = {});
    ~LoadAsyncContext() override;

    bool     empty() const;
    uint64_t contextId() const;
    size_t   localMatchedBlocks() const;
    size_t   matchedBlocks() const;
    size_t   matchedBlocks(Tier tier) const;
    bool     needBackendMatch() const;

    void setMatchCallback(MatchCallback callback);
    void startBackendMatch();
    void setTargetBlocks(size_t desc_index, std::vector<BlockIdxType> target_blocks);
    void setBackendTargetBlock(size_t key_index, size_t handle_index, BlockIdxType target_block);

    const std::vector<TransferDescriptor>&              loadDescs() const;
    const std::vector<bool>&                            joinedLoads() const;
    const std::vector<std::vector<StorageBlockHandle>>& backendHandles() const;

    bool commit();

    bool abortPending();
    bool completeOne(bool success);
    bool onTaskFail();
    void waitDone() override;
    bool done() const override;
    bool success() const override;
    MallocStatus mallocStatus() const;

private:
    void markAborted();
    void rebuildMatchedBlocksByTier();
    void onBackendMatch(size_t matched_blocks_num, std::shared_ptr<StorageBackendMatchMeta> match_meta, bool success);
    void onBackendRead(bool success);
    void failBeforeCommit();
    void failCommit();
    void finishIfReadyLocked(bool& notify);

    std::shared_ptr<LoadContextCoordinator> coordinator_;
    const uint64_t                          context_id_;
    std::vector<TransferDescriptor>         load_descs_;
    std::vector<bool>                       joined_load_;
    const size_t                            local_matched_blocks_{0};
    size_t                                  matched_blocks_{0};
    size_t                                  backend_matched_blocks_{0};
    std::array<size_t, 3>                   matched_blocks_by_tier_{};

    std::shared_ptr<StorageBackend> storage_backend_;
    StorageRequest                  storage_request_;
    MatchCallback                   match_callback_;
    const bool                      need_backend_match_{false};
    bool                            backend_started_{false};
    bool                            backend_pending_{false};
    std::atomic<bool>               commit_started_{false};
    bool                            committed_{false};

    std::mutex backend_match_mutex_;

    std::atomic<State>        state_{State::PENDING};
    std::atomic<MallocStatus> malloc_status_{MallocStatus::NONE};
    mutable std::mutex        mutex_;
    std::condition_variable   cv_;
    size_t                    remaining_transfer_count_{0};
    bool                      has_failure_{false};

    friend class LoadContextCoordinator;
};

class LoadContextCoordinator: public std::enable_shared_from_this<LoadContextCoordinator> {
public:
    using CommitCallback = std::function<bool(const std::shared_ptr<LoadAsyncContext>& context)>;
    using AbortCallback  = std::function<void(LoadAsyncContext& context)>;

    LoadContextCoordinator(CommitCallback commit_callback, AbortCallback abort_callback);

    std::shared_ptr<LoadAsyncContext> create(std::vector<TransferDescriptor> load_descs,
                                             std::vector<bool>               joined_load,
                                             size_t                          matched_blocks,
                                             std::shared_ptr<StorageBackend> storage_backend = nullptr,
                                             StorageRequest                  storage_request = {});
    bool                              registerContext(const std::shared_ptr<LoadAsyncContext>& context);
    bool                              commit(uint64_t context_id);
    bool                              abort(LoadAsyncContext& context) noexcept;
    void                              shutdown();

private:
    using PendingContextMap = std::unordered_map<uint64_t, std::weak_ptr<LoadAsyncContext>>;

    void retireActiveCallback();
    bool beginActiveCallback();

    friend class LoadAsyncContext;

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
