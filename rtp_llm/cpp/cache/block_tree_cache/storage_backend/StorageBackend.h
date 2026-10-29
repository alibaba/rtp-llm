#pragma once

#include <array>
#include <unordered_map>

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "rtp_llm/cpp/cache/BlockInfo.h"
#include "rtp_llm/cpp/cache/CacheTopology.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/cache/DecodeReuseSources.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/StorageCompletionAdmission.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/StorageBackendExecutor.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

class BatchKVCacheResource;
class BlockTreeCacheMetricsReporter;
class TransferStageTimer;
class SharedTransferStageTimer;

struct StorageBlockHandle {
    std::string  tag;
    BlockIdxType block{NULL_BLOCK_IDX};
    // A Decode source owns all target P peers locally; replicas can name the
    // same page. Only completed-Decode writes may carry this projection.
    std::vector<BlockIdxType> peer_blocks;
};

struct StorageBackendMatchMeta {
    virtual ~StorageBackendMatchMeta() = default;
    DecodeReuseSourcesPtr decode_sources;
};

enum class StorageQuarantineReason {
    SDK_TIMEOUT,
    RPC_TIMEOUT,
    RPC_ERROR,
    INVALID_RESPONSE,
    SDK_ERROR,
};

struct StorageQuarantineCounts {
    size_t task_count{0};
    size_t block_count{0};
};

// Read/write rows and StorageQuarantineReason columns share one state snapshot.
using StorageQuarantineBreakdown = std::array<std::array<StorageQuarantineCounts, 5>, 2>;

// The backend accepted an operation but cannot prove that DMA/RDMA has
// stopped touching its buffers. StorageBackend replaces request pins with
// permanent FROZEN references for the lifetime of the block pool.
class StorageOperationCompletionUnknown: public std::runtime_error {
public:
    StorageOperationCompletionUnknown(const std::string& message, StorageQuarantineReason reason):
        std::runtime_error(message), reason_(reason) {}

    StorageQuarantineReason reason() const noexcept {
        return reason_;
    }

private:
    StorageQuarantineReason reason_;
};

// The transport reported an explicit timeout. Buffer ownership is still
// unknown, so this remains a completion-unknown failure for quarantine while
// allowing callers to preserve the timeout error code.
class StorageOperationTimeout: public StorageOperationCompletionUnknown {
public:
    using StorageOperationCompletionUnknown::StorageOperationCompletionUnknown;
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

struct StorageRequest {
    std::shared_ptr<const CacheKeysType> keys;
    // handles[i] contains every group block associated with (*keys)[i].
    std::vector<std::vector<StorageBlockHandle>> handles;
    // Match requests expose the complete key sequence. Keys before this
    // boundary are already available from Device/Host/Disk.
    size_t                 local_matched_blocks_num{0};
    std::optional<int64_t> request_id;
    bool decode_completion{false};
    size_t computed_tokens{0};
    size_t published_tokens{0};
    std::optional<int64_t> prefill_computed_tokens;
    std::shared_ptr<void> completion_budget_lease;
    std::shared_ptr<StorageCompletionAdmission> completion_admission;

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
    StorageWriteTask()                            = default;
    StorageWriteTask(StorageWriteTask&&) noexcept = default;
    StorageWriteTask& operator=(StorageWriteTask&&) noexcept = default;

    StorageWriteTask(const StorageWriteTask&) = delete;
    StorageWriteTask& operator=(const StorageWriteTask&) = delete;

    explicit operator bool() const {
        return state_ != nullptr;
    }

private:
    explicit StorageWriteTask(std::shared_ptr<storage_backend_detail::StorageTaskState> state);
    std::shared_ptr<storage_backend_detail::StorageTaskState> state_;
    friend class StorageBackend;
};

// Asynchronous facade for synchronous derived I/O. Owners must call shutdown
// before derived backend state starts destruction.
class StorageBackend {
public:
    class ExternalIoGuard {
    public:
        ExternalIoGuard() = default;
        ExternalIoGuard(ExternalIoGuard&& other) noexcept;
        ExternalIoGuard& operator=(ExternalIoGuard&& other) noexcept;
        ~ExternalIoGuard();
        ExternalIoGuard(const ExternalIoGuard&) = delete;
        ExternalIoGuard& operator=(const ExternalIoGuard&) = delete;
        explicit operator bool() const { return backend_ != nullptr; }
    private:
        explicit ExternalIoGuard(StorageBackend* backend): backend_(backend) {}
        StorageBackend* backend_{nullptr};
        friend class StorageBackend;
    };
    using MatchDone      = std::function<void(
        size_t matched_blocks_num, std::shared_ptr<StorageBackendMatchMeta> match_meta, bool success)>;
    using Done           = std::function<void(ErrorInfo error)>;
    using PoolsByTag     = std::unordered_map<std::string, DeviceBlockPoolPtr>;
    using BufferResolver = std::function<std::vector<BlockInfo>(int layer_id, const std::string& tag, int block_id)>;

    // An injected executor may be observed by its owner but belongs to only
    // one backend; init rejects binding the same instance a second time.
    explicit StorageBackend(std::shared_ptr<StorageBackendExecutor>        executor         = nullptr,
                            std::shared_ptr<BlockTreeCacheMetricsReporter> metrics_reporter = nullptr,
                            int64_t                                        world_rank       = -1);
    virtual ~StorageBackend();
    // Atomic replacement allows attaching the cache reporter before or after init.
    void setStageMetricsReporter(std::shared_ptr<BlockTreeCacheMetricsReporter> reporter);

    // Initialization is single-attempt. A failed start may permanently stop
    // an injected executor; create a fresh backend/executor to retry.
    bool init(std::shared_ptr<const CacheTopology> topology, PoolsByTag pools_by_tag, BufferResolver buffer_resolver);
    void match(StorageRequest request, MatchDone done);
    void read(StorageRequest request, std::shared_ptr<StorageBackendMatchMeta> match_meta, Done done);
    StorageWriteTask prepareWrite(StorageRequest request);
    // Returns admission only, not the I/O result; never waits for completion.
    // Source pins live until execution or rejection.
    bool write(StorageWriteTask task);
    // Optional EOS publication. Non-supporting backends keep their original
    // behavior; admission is not a successful remote commit.
    virtual bool captureCompletedDecode(const BatchKVCacheResource&, size_t, bool = true) {
        return false;
    }
    // Must not be called from backend I/O or completion callbacks.
    void shutdown();

protected:
    // Worker RPC payloads bypass the executor. Count their whole transaction,
    // including publishing an UNKNOWN outcome, before shutdown may finalize.
    ExternalIoGuard trackExternalIo();
    const CacheTopology&      topology() const;
    const DeviceBlockPoolPtr& devicePool(const std::string& tag) const;
    std::vector<BlockInfo>    convertIndexToBuffer(int layer_id, const std::string& tag, int block_id) const;
    // Match queries contain every possible group handle. Derived matchers use
    // this predicate for each candidate prefix; the core applies the same rule
    // before allocating read targets.
    bool isHandleRequired(size_t key_index, size_t matched_key_count, std::string_view tag) const;

    virtual bool               initImpl()                                                           = 0;
    virtual StorageMatchResult matchImpl(const StorageRequest& request)                             = 0;
    virtual void               readImpl(const StorageRequest&                           request,
                                        const std::shared_ptr<StorageBackendMatchMeta>& match_meta) = 0;
    virtual void               writeImpl(const StorageRequest& request)                             = 0;
    virtual void               shutdownImpl() noexcept {}
    virtual void               onQuarantined(bool, StorageQuarantineReason, size_t) noexcept {}
    virtual void               onQuarantineChanged(uint64_t, size_t, size_t,
                                                   const StorageQuarantineBreakdown&) noexcept {}
    virtual bool protectUnknownCompletion() const noexcept { return false; }
    virtual std::shared_ptr<void> completionResourceOwner() const { return {}; }
    virtual bool hasUnresolvedExternalIo() const noexcept { return false; }
    // Called only after application I/O callbacks have joined. True requires
    // actual transport quiescence, not merely SDK method/worker completion.
    // This is a nonblocking predicate under the quarantine mutex; it must not
    // invoke notifyCompletionQuiescence or initiate an SDK fence synchronously.
    virtual bool quiesceUnknownOperations() noexcept { return false; }
    // Call after publishing proof, without holding transport locks. Taking
    // this mutex prevents a notification racing the wait predicate from being
    // lost before the waiter atomically releases the mutex.
    void notifyCompletionQuiescence() noexcept {
        std::lock_guard<std::mutex> lock(quarantine_mutex_);
        quarantine_cv_.notify_all();
    }

private:
    enum class Lifecycle {
        CREATED,
        ACCEPTING,
        STOPPING,
        FINALIZING,
        STOPPED
    };
    using Operation = std::function<void(Lifecycle outcome)>;

    std::shared_ptr<storage_backend_detail::StorageTaskState>
    prepare(StorageRequest request, bool for_write = false, uint64_t trace_id = 0);
    SharedTransferStageTimer stageTimer(const char*           stage,
                                        const StorageRequest& request,
                                        const char*           operation,
                                        uint64_t              trace_id) const noexcept;
    TransferStageTimer       stageValue(const char*           stage,
                                        const StorageRequest& request,
                                        const char*           operation,
                                        uint64_t              trace_id) const noexcept;
    void validateRequest(const StorageRequest& request, bool allow_null_blocks, bool for_write = false) const;
    bool                                 dispatch(Operation operation);
    void                                 taskFinished();
    void quarantineTask(const std::shared_ptr<storage_backend_detail::StorageTaskState>& state,
                        bool                                                             is_write,
                        StorageQuarantineReason                                          reason);
    std::shared_ptr<const CacheTopology> topology_;
    PoolsByTag                           pools_by_tag_;
    BufferResolver                       buffer_resolver_;
    std::shared_ptr<StorageBackendExecutor> executor_;
    std::shared_ptr<BlockTreeCacheMetricsReporter> stage_metrics_reporter_;
    std::atomic<bool>                              stage_metrics_enabled_{false};
    const int64_t                                  world_rank_;
    bool                                    init_attempted_{false};
    bool                                    initialized_{false};

    std::mutex              lifecycle_mutex_;
    std::condition_variable lifecycle_cv_;
    Lifecycle               lifecycle_{Lifecycle::CREATED};
    size_t                  in_flight_{0};

    std::mutex quarantine_mutex_;
    std::condition_variable quarantine_cv_;
    std::shared_ptr<storage_backend_detail::StorageTaskState> protected_quarantine_;
    // Preserve the per-operation quarantine metrics and abort threshold.
    size_t                     quarantined_task_count_{0};
    size_t                     quarantined_block_count_{0};
    uint64_t                   quarantine_generation_{0};
    StorageQuarantineBreakdown quarantined_breakdown_{};

    friend class LoadAsyncContext;
};

}  // namespace rtp_llm
