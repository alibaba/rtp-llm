#pragma once

#include <unordered_map>

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "rtp_llm/cpp/cache/BlockInfo.h"
#include "rtp_llm/cpp/cache/CacheTopology.h"
#include "rtp_llm/cpp/cache/CacheDomainConfig.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/StorageBackendExecutor.h"

namespace rtp_llm {

struct StorageBlockHandle {
    ModuleGroupKey key;
    BlockIdxType block{NULL_BLOCK_IDX};

    StorageBlockHandle() = default;
    StorageBlockHandle(ModuleGroupKey key, BlockIdxType block): key(std::move(key)), block(block) {}
    StorageBlockHandle(std::string tag, BlockIdxType block):
        StorageBlockHandle(ModuleGroupKey{CacheModuleId::main(), std::move(tag)}, block) {}
};

struct StorageBackendMatchMeta {
    virtual ~StorageBackendMatchMeta() = default;
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
    size_t local_matched_blocks_num{0};

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
    using Done           = std::function<void(bool success)>;
    using PoolsByTag     = std::unordered_map<std::string, DeviceBlockPoolPtr>;
    using PoolsByGroup   = std::unordered_map<ModuleGroupKey, DeviceBlockPoolPtr>;
    using LeafBufferResolver =
        std::function<std::vector<BlockInfo>(int layer_id, const std::string& tag, int block_id)>;
    using BufferResolver = std::function<std::vector<BlockInfo>(
        CacheModuleId module_id, int local_layer_id, const std::string& tag, int block_id)>;

    // An injected executor may be observed by its owner but belongs to only
    // one backend; init rejects binding the same instance a second time.
    explicit StorageBackend(std::shared_ptr<StorageBackendExecutor> executor = nullptr);
    virtual ~StorageBackend();

    // Initialization is single-attempt. A failed start may permanently stop
    // an injected executor; create a fresh backend/executor to retry.
    bool
    init(std::shared_ptr<const CacheTopology> topology, PoolsByTag pools_by_tag, LeafBufferResolver buffer_resolver);
    bool             init(const CacheDomainConfig& domain, PoolsByGroup pools, BufferResolver buffer_resolver);
    void match(StorageRequest request, MatchDone done);
    void read(StorageRequest request, std::shared_ptr<StorageBackendMatchMeta> match_meta, Done done);
    StorageWriteTask prepareWrite(StorageRequest request);
    // Returns admission only, not the I/O result; never waits for completion.
    // Source pins live until execution or rejection.
    bool write(StorageWriteTask task);
    // Must not be called from backend I/O or completion callbacks.
    void shutdown();

protected:
    const CacheTopology&      topology() const;
    const CacheTopology&      topology(CacheModuleId module_id) const;
    const GroupBase&          group(const ModuleGroupKey& key) const;
    const DeviceBlockPoolPtr& devicePool(const ModuleGroupKey& key) const;
    const DeviceBlockPoolPtr& devicePool(const std::string& tag) const {
        return devicePool(ModuleGroupKey{CacheModuleId::main(), tag});
    }
    std::vector<BlockInfo>
    convertIndexToBuffer(CacheModuleId module_id, int layer_id, const std::string& tag, int block_id) const;
    std::vector<BlockInfo> convertIndexToBuffer(int layer_id, const std::string& tag, int block_id) const {
        return convertIndexToBuffer(CacheModuleId::main(), layer_id, tag, block_id);
    }
    // Match queries contain every possible group handle. Derived matchers use
    // this predicate for each candidate prefix; the core applies the same rule
    // before allocating read targets.
    bool isHandleRequired(size_t key_index, size_t matched_key_count, const ModuleGroupKey& key) const;

    virtual bool               initImpl()                                                           = 0;
    virtual StorageMatchResult matchImpl(const StorageRequest& request)                             = 0;
    virtual void               readImpl(const StorageRequest&                           request,
                                        const std::shared_ptr<StorageBackendMatchMeta>& match_meta) = 0;
    virtual void               writeImpl(const StorageRequest& request)                             = 0;
    virtual void               shutdownImpl() noexcept {}

private:
    enum class Lifecycle {
        CREATED,
        ACCEPTING,
        STOPPING,
        FINALIZING,
        STOPPED
    };
    using Operation = std::function<void(Lifecycle outcome)>;

    std::shared_ptr<storage_backend_detail::StorageTaskState> prepare(StorageRequest request);
    void                                 validateRequest(const StorageRequest& request, bool allow_null_blocks) const;
    bool                                 dispatch(Operation operation);
    void                                 taskFinished();
    using ModuleTopologies = std::unordered_map<CacheModuleId, std::shared_ptr<const CacheTopology>>;
    bool             initMembers(ModuleTopologies topologies, PoolsByGroup pools, BufferResolver buffer_resolver);
    ModuleTopologies topologies_;
    PoolsByGroup     pools_by_group_;
    BufferResolver                       buffer_resolver_;
    std::shared_ptr<StorageBackendExecutor> executor_;
    bool                                    init_attempted_{false};
    bool                                    initialized_{false};

    std::mutex              lifecycle_mutex_;
    std::condition_variable lifecycle_cv_;
    Lifecycle               lifecycle_{Lifecycle::CREATED};
    size_t                  in_flight_{0};

    friend class LoadAsyncContext;
};

}  // namespace rtp_llm
