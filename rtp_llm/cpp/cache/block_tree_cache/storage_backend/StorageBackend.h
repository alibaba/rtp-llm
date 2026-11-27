#pragma once

#include <unordered_map>
#include <atomic>

#include <condition_variable>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/BlockIndex.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeKey.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/StorageBackendExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

class CacheObserver;
class FrozenTask;
class IBlockPool;
enum class AsyncProfileKind;
class TransferStageTimer;
class SharedTransferStageTimer;

struct StorageBlockHandle {
    std::string  tag;
    BlockIdxType block{NULL_BLOCK_IDX};
    Tier         tier{Tier::DEVICE};
    uint64_t     end_token{0};
};

struct StorageBackendMatchMeta {
    virtual ~StorageBackendMatchMeta() = default;
    virtual std::string replicaDescriptor(size_t, const std::vector<std::string>&) const { return {}; }
};

enum class StorageQuarantineReason {
    SDK_TIMEOUT,
    RPC_TIMEOUT,
    RPC_ERROR,
    INVALID_RESPONSE,
    SDK_ERROR,
};

// Describes who protects local targets after UNKNOWN, not whether I/O is safe.
enum class StorageLocalTargetProtection {
    LEGACY,            // StorageBackend hands its pins off to the block pool's FROZEN references.
    TARGETS_HELD,      // The backend already owns protection for every local target.
    NO_LOCAL_TARGETS,  // The operation only reads local sources.
};

// The backend accepted an operation but cannot prove that DMA/RDMA has
// stopped touching its buffers. By default StorageBackend replaces request
// pins with FROZEN references; a backend may explicitly own target protection.
class StorageOperationCompletionUnknown: public std::runtime_error {
public:
    StorageOperationCompletionUnknown(
        const std::string&           message,
        StorageQuarantineReason      reason,
        StorageLocalTargetProtection local_target_protection = StorageLocalTargetProtection::LEGACY,
        std::shared_ptr<const FrozenTask> recovery = {}):
        std::runtime_error(message), reason_(reason), local_target_protection_(local_target_protection),
        recovery_(std::move(recovery)) {}

    StorageQuarantineReason reason() const noexcept {
        return reason_;
    }

    StorageLocalTargetProtection localTargetProtection() const noexcept {
        return local_target_protection_;
    }
    const std::shared_ptr<const FrozenTask>& recovery() const noexcept { return recovery_; }

private:
    StorageQuarantineReason      reason_;
    StorageLocalTargetProtection local_target_protection_;
    std::shared_ptr<const FrozenTask> recovery_;
};

// An explicit timeout with unknown buffer safety preserves the timeout error code.
class StorageOperationTimeout: public StorageOperationCompletionUnknown {
public:
    using StorageOperationCompletionUnknown::StorageOperationCompletionUnknown;
};

// A shared workflow budget expired before any payload I/O was submitted.
class StorageDeadlineExceeded: public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Derived code may use this only with explicit not-submitted or completed
// payload proof. A generic SDK error is not such a proof.
class StorageOperationBufferSafeFailure: public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct StorageMatchResult {
    size_t                                   matched_blocks_num{0};
    std::shared_ptr<StorageBackendMatchMeta> match_meta;
};

struct StorageTransferOutcome {
    ErrorInfo error;
    bool safe;
    std::shared_ptr<const FrozenTask> recovery;
};

struct StorageRequest {
    std::shared_ptr<const CacheKeysType> keys;
    // handles[i] contains every group block associated with (*keys)[i].
    std::vector<std::vector<StorageBlockHandle>> handles;
    // Match requests expose the complete key sequence. Keys before this
    // boundary are already available from Device/Host/Disk.
    size_t                      local_matched_blocks_num{0};
    std::optional<int64_t>      request_id;
    std::optional<TransferTask> transfer_task;
    // Shared-tree queries return exact replica descriptors, and writes check
    // those same descriptors for visibility after handoff to the backend.
    bool positional_replicas{false};
    // A planned read selects exact group replicas per keys row.
    std::vector<std::vector<std::string>> replica_descriptors;
    // Shared-tree transfer identity spans the coordinator and all ranks.
    std::string payload_trace_id;
    // Keep the native integer fast path and support complete opaque keys.
    // Exactly one sequence supplies the positions in handles.
    std::shared_ptr<const BlockTreeKeys> opaque_keys;
    // Optional early progress for the exact cold prefetch that owns this I/O.
    // True terminates the scheduler wait while physical ownership remains held.
    std::function<void(bool)> prefetch_notice{};

    size_t cacheKeyCount() const noexcept {
        return keys ? keys->size() : opaque_keys ? opaque_keys->size() : 0;
    }

    void sliceKeys(size_t first, size_t last) {
        if ((!keys && !opaque_keys) || first > last || last > cacheKeyCount()) {
            throw std::out_of_range("invalid storage request key slice");
        }
        if (keys) {
            keys = std::make_shared<CacheKeysType>(keys->begin() + first, keys->begin() + last);
        } else {
            opaque_keys = std::make_shared<BlockTreeKeys>(opaque_keys->begin() + first, opaque_keys->begin() + last);
        }
    }

    bool empty() const {
        for (const auto& key_handles : handles) {
            if (!key_handles.empty()) {
                return false;
            }
        }
        return true;
    }
};

namespace storage_backend_detail {
struct StorageTaskState;
}  // namespace storage_backend_detail

class StorageWriteTask {
public:
    StorageWriteTask()                                       = default;
    StorageWriteTask(StorageWriteTask&&) noexcept            = default;
    StorageWriteTask& operator=(StorageWriteTask&&) noexcept = default;

    StorageWriteTask(const StorageWriteTask&)            = delete;
    StorageWriteTask& operator=(const StorageWriteTask&) = delete;

    explicit operator bool() const {
        return state_ != nullptr;
    }

private:
    explicit StorageWriteTask(std::shared_ptr<storage_backend_detail::StorageTaskState> state);
    std::shared_ptr<storage_backend_detail::StorageTaskState> state_;
    friend class StorageBackend;
};

// Asynchronous facade for derived I/O. Owners must call shutdown
// before derived backend state starts destruction.
class StorageBackend {
public:
    using MatchDone      = std::function<void(
        size_t matched_blocks_num, std::shared_ptr<StorageBackendMatchMeta> match_meta, bool success)>;
    using Done           = std::function<void(ErrorInfo error)>;
    using TransferDone   = std::function<void(StorageTransferOutcome)>;
    struct PageAccess {
        std::shared_ptr<IBlockPool> pool;
        std::function<std::function<void()>(BlockIdxType)> pin;
    };
    struct GroupAccess {
        PageAccess device, host;
        std::function<bool(size_t key_index, size_t matched_keys)> required;

        const PageAccess& forTier(Tier tier) const;
    };
    using Groups = std::unordered_map<std::string, GroupAccess>;

    // An injected executor may be observed by its owner but belongs to only
    // one backend; init rejects binding the same instance a second time.
    explicit StorageBackend(std::shared_ptr<StorageBackendExecutor>        executor         = nullptr,
                            std::shared_ptr<CacheObserver>                  metrics_reporter = nullptr,
                            int64_t                                        world_rank       = -1,
                            std::optional<std::chrono::milliseconds>       timeout          = std::chrono::seconds(10));
    virtual ~StorageBackend();
    // Atomic replacement allows attaching the cache reporter before or after init.
    void                        setStageMetricsReporter(std::shared_ptr<CacheObserver> reporter);
    virtual BlockTreeQueueSizes queueSizes() const {
        return executor_ ? executor_->queueSizes() : BlockTreeQueueSizes{};
    }

    // Initialization is single-attempt. A failed start may permanently stop
    // an injected executor; create a fresh backend/executor to retry.
    bool init(Groups groups);
    // Runtime admission only. Disabling must not tear down clients or interrupt
    // a payload batch that rank 0 has already admitted.
    void setRemoteCacheEnabled(bool enabled) noexcept {
        remote_cache_enabled_.store(enabled);
    }
    bool remoteCacheEnabled() const noexcept {
        return remote_cache_enabled_.load();
    }
    void             match(StorageRequest request, MatchDone done);
    void             read(StorageRequest request, std::shared_ptr<StorageBackendMatchMeta> match_meta, Done done);
    void             readOwned(StorageRequest request, std::shared_ptr<StorageBackendMatchMeta> match_meta,
                               TransferDone done);
    virtual bool supportsPositionalReplicas() const noexcept { return false; }
    virtual bool supportsWriteTier(Tier tier) const noexcept { return tier == Tier::DEVICE; }
    StorageWriteTask prepareWrite(StorageRequest request);
    // Returns admission only, not the I/O result; never waits for completion.
    // Source pins live through payload completion and publication.
    bool write(StorageWriteTask task, TransferDone done = {});
    // Must not be called from backend I/O or completion callbacks.
    void shutdown();

protected:
    using Task = StorageBackendExecutor::Task;
    using MatchResult     = std::function<StorageMatchResult()>;
    using MatchCompletion = std::function<void(MatchResult)>;
    // The completion callable only checks results and updates local state; it
    // must not start blocking I/O on the RPC completion/settlement thread.
    using ReadCompletion = std::function<void(Task)>;
    // Only a pool worker may publish or abort a write. On queue rejection the
    // result can still check the rank responses and settle the source pins.
    using WriteResult     = std::function<void(bool on_worker)>;
    using WriteCompletion = std::function<void(WriteResult)>;
    using WritePinFilter  = std::function<void(const std::vector<size_t>&)>;
    bool initializationAttempted() const noexcept { return init_attempted_; }
    // Match queries contain every possible group handle. Derived matchers use
    // this predicate for each candidate prefix; the core applies the same rule
    // before allocating read targets.
    bool isHandleRequired(size_t key_index, size_t matched_key_count, const std::string& tag) const;
    size_t coveredPrefix(const StorageRequest& request,
                         const std::function<bool(size_t row, const StorageBlockHandle&)>& available) const;

    virtual bool               initImpl()                                                           = 0;
    virtual StorageMatchResult matchImpl(const StorageRequest& request)                             = 0;
    virtual bool usesAsyncMatch(const StorageRequest&) const noexcept { return false; }
    virtual bool matchAsyncImpl(const StorageRequest&, MatchCompletion) { return false; }
    virtual void               readImpl(const StorageRequest&                           request,
                                        const std::shared_ptr<StorageBackendMatchMeta>& match_meta) = 0;
    // Synchronous payloads run on one worker. A deferred payload spans dispatch through completion.
    virtual bool usesAsyncPayload(const StorageRequest&, bool /*is_write*/) const noexcept {
        return false;
    }
    // False preserves the synchronous path. True transfers completion to done;
    // the request and pins remain alive until its callable has been settled.
    virtual bool readAsyncImpl(const StorageRequest&, const std::shared_ptr<StorageBackendMatchMeta>&, ReadCompletion) {
        return false;
    }
    virtual void writeImpl(const StorageRequest& request) = 0;
    // Filter validated handle rows once on this worker before dispatch or done.
    // True defers completion, retaining the request and remaining pins until done settles.
    virtual bool startWriteImpl(const StorageRequest& request, WriteCompletion, WritePinFilter) {
        writeImpl(request);
        return false;
    }
    virtual void shutdownImpl() noexcept {}
    virtual void onRemoteCacheSkipped(const char*) noexcept {}

private:
    enum class Lifecycle {
        CREATED,
        ACCEPTING,
        STOPPING,
        FINALIZING,
        STOPPED
    };
    using Operation = std::function<void(Lifecycle outcome, Task finished)>;

    std::shared_ptr<storage_backend_detail::StorageTaskState>
                             prepare(StorageRequest request, bool for_write = false, uint64_t trace_id = 0);
    SharedTransferStageTimer stageTimer(const char*           stage,
                                        const StorageRequest& request,
                                        const char*           operation,
                                        uint64_t              trace_id,
                                        AsyncProfileKind      kind) const noexcept;
    TransferStageTimer       stageValue(const char*           stage,
                                        const StorageRequest& request,
                                        const char*           operation,
                                        uint64_t              trace_id) const noexcept;
    void                     validateRequest(const StorageRequest& request, bool allow_null_blocks) const;
    void                     applyDeadline(StorageRequest& request) const;
    std::optional<StorageBackendExecutor::Clock::time_point> deadline(const StorageRequest& request) const;
    bool                     expired(const StorageRequest& request) const;
    bool                     dispatch(Operation                                                operation,
                                      BlockTreeTaskClass                                       task_class,
                                      bool                                                     run_inline = false,
                                      std::optional<StorageBackendExecutor::Clock::time_point> io_deadline = std::nullopt);
    void                     taskFinished();
    std::atomic<bool>                              remote_cache_enabled_{true};
    Groups                                         groups_;
    std::shared_ptr<StorageBackendExecutor>        executor_;
    std::shared_ptr<CacheObserver>                  stage_metrics_reporter_;
    std::atomic<bool>                              stage_metrics_enabled_{false};
    const int64_t                                  world_rank_;
    const std::optional<std::chrono::milliseconds> timeout_;
    bool                                           init_attempted_{false};
    bool                                           initialized_{false};

    std::mutex              lifecycle_mutex_;
    std::condition_variable lifecycle_cv_;
    Lifecycle               lifecycle_{Lifecycle::CREATED};
    size_t                  in_flight_{0};

    friend class LoadAsyncContext;
};

}  // namespace rtp_llm
