#pragma once

#include <unordered_map>
#include <atomic>

#include <condition_variable>
#include <chrono>
#include <array>
#include <cstddef>
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
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/CacheTopology.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/StorageBackendExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

class BatchKVCacheResource;
class BlockTreeCacheMetricsReporter;
enum class AsyncProfileKind;
class TransferStageTimer;
class SharedTransferStageTimer;

struct StorageBlockHandle {
    std::string  tag;
    BlockIdxType block{NULL_BLOCK_IDX};
};

struct StorageBackendMatchMeta {
    virtual ~StorageBackendMatchMeta() = default;
};

enum class StorageQuarantineReason {
    SDK_TIMEOUT,
    RPC_TIMEOUT,
    RPC_ERROR,
    INVALID_RESPONSE,
};

struct StorageQuarantineCounts {
    size_t task_count{0};
    size_t block_count{0};
};
using StorageQuarantineBreakdown = std::array<std::array<StorageQuarantineCounts, 4>, 2>;

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

// A shared workflow budget expired before any payload I/O was submitted.
class StorageDeadlineExceeded: public std::runtime_error {
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
    std::optional<TransferTask> transfer_task;

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
    using MatchDone      = std::function<void(
        size_t matched_blocks_num, std::shared_ptr<StorageBackendMatchMeta> match_meta, bool success)>;
    using Done           = std::function<void(ErrorInfo error)>;
    using PoolsByTag     = std::unordered_map<std::string, DeviceBlockPoolPtr>;
    // The resolver receives module-local layer IDs; the backend validates
    // local layer membership through the tag owner before invoking it.
    using StorageBufferResolver =
        std::function<std::vector<BlockInfo>(int module_local_layer_id, const std::string& tag, int block_id)>;

    // An injected executor may be observed by its owner but belongs to only
    // one backend; init rejects binding the same instance a second time.
    explicit StorageBackend(std::shared_ptr<StorageBackendExecutor>        executor         = nullptr,
                            std::shared_ptr<BlockTreeCacheMetricsReporter> metrics_reporter = nullptr,
                            int64_t                                        world_rank       = -1,
                            std::chrono::milliseconds                      timeout = std::chrono::seconds(10));
    virtual ~StorageBackend();
    // Atomic replacement allows attaching the cache reporter before or after init.
    void setStageMetricsReporter(std::shared_ptr<BlockTreeCacheMetricsReporter> reporter);

    // Initialization is single-attempt. A failed start may permanently stop
    // an injected executor; create a fresh backend/executor to retry.
    bool init(std::shared_ptr<const CacheConfig> cache_config,
              PoolsByTag                         pools_by_tag,
              StorageBufferResolver              buffer_resolver);
    // Runtime admission only. Disabling must not tear down clients or interrupt
    // a payload batch that rank 0 has already admitted.
    void setRemoteCacheEnabled(bool enabled) noexcept {
        remote_cache_enabled_.store(enabled);
    }
    bool remoteCacheEnabled() const noexcept {
        return remote_cache_enabled_.load();
    }
    void match(StorageRequest request, MatchDone done);
    void read(StorageRequest request, std::shared_ptr<StorageBackendMatchMeta> match_meta, Done done);
    StorageWriteTask prepareWrite(StorageRequest request);
    // Returns admission only, not the I/O result; never waits for completion.
    // Source pins live until execution or rejection.
    bool write(StorageWriteTask task);
    // Must not be called from backend I/O or completion callbacks.
    void shutdown();

protected:
    const DeviceBlockPoolPtr& devicePool(const std::string& tag) const;
    std::vector<BlockInfo>    convertIndexToBuffer(int layer_id, const std::string& tag, int block_id) const;
    const CacheConfig&        cacheConfig() const;
    std::vector<BlockInfo> resolveStorageBuffer(int module_local_layer_id, const std::string& tag, int block_id) const;
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
    virtual void               onRemoteCacheSkipped(const char*) noexcept {}
    virtual void               onQuarantined(bool, StorageQuarantineReason, size_t) noexcept {}
    virtual void               onQuarantineChanged(uint64_t, size_t, size_t, const StorageQuarantineBreakdown&) noexcept {}

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
                                        uint64_t              trace_id,
                                        AsyncProfileKind      kind) const noexcept;
    TransferStageTimer       stageValue(const char*           stage,
                                        const StorageRequest& request,
                                        const char*           operation,
                                        uint64_t              trace_id) const noexcept;
    void validateRequest(const StorageRequest& request, bool allow_null_blocks) const;
    bool dispatch(Operation operation, BlockTreeTaskClass task_class, bool run_inline = false);
    void                                 taskFinished();
    void quarantineTask(const std::shared_ptr<storage_backend_detail::StorageTaskState>& state,
                        bool                                                             is_write,
                        StorageQuarantineReason                                          reason);
std::shared_ptr<const CacheConfig>             cache_config_;
    std::atomic<bool>                              remote_cache_enabled_{true};
    PoolsByTag                                     pools_by_tag_;
    StorageBufferResolver                          storage_buffer_resolver_;
    std::shared_ptr<StorageBackendExecutor>        executor_;
    std::shared_ptr<BlockTreeCacheMetricsReporter> stage_metrics_reporter_;
    std::atomic<bool>                              stage_metrics_enabled_{false};
    const int64_t                                  world_rank_;
    const std::chrono::milliseconds                 timeout_;
    bool                                    init_attempted_{false};
    bool                                    initialized_{false};

    std::mutex              lifecycle_mutex_;
    std::condition_variable lifecycle_cv_;
    Lifecycle               lifecycle_{Lifecycle::CREATED};
    size_t                  in_flight_{0};

    std::mutex quarantine_mutex_;
    // Count quarantined operations; frozen capacity is read from the pools.
    size_t   quarantined_task_count_{0};
    uint64_t quarantine_generation_{0};
    StorageQuarantineBreakdown quarantined_breakdown_{};

    friend class LoadAsyncContext;
};

}  // namespace rtp_llm
