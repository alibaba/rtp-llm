#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/StorageBackend.h"

#include <algorithm>
#include <mutex>
#include <exception>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm::storage_backend_detail {

thread_local const StorageBackend* completing_backend = nullptr;

template<typename Callback>
void invokeCallback(const StorageBackend* backend, Callback&& callback) noexcept {
    struct CompletionScope {
        const StorageBackend* previous = completing_backend;
        ~CompletionScope() {
            completing_backend = previous;
        }
    } scope;
    completing_backend = backend;
    try {
        callback();
    } catch (const std::exception& error) {
        RTP_LLM_LOG_ERROR("StorageBackend completion failed: %s", error.what());
    } catch (...) { RTP_LLM_LOG_ERROR("StorageBackend completion failed with an unknown exception"); }
}

struct StorageTaskState {
    struct Pin {
        DeviceBlockPoolPtr pool;
        BlockIdxType       block;
    };

    StorageRequest   request;
    std::vector<Pin> pins;
    std::once_flag   finish_once;
    void             finish() {
        std::call_once(finish_once, [this] {
            for (const Pin& pin : pins) {
                pin.pool->decRef(pin.block);
            }
            pins.clear();
        });
    }

    ~StorageTaskState() {
        finish();
    }
};

}  // namespace rtp_llm::storage_backend_detail

namespace rtp_llm {
namespace {

struct BlockKey {
    DeviceBlockPool* pool;
    BlockIdxType     block;
    bool             operator==(const BlockKey& other) const {
        return pool == other.pool && block == other.block;
    }
};

struct BlockKeyHash {
    size_t operator()(const BlockKey& key) const {
        return std::hash<DeviceBlockPool*>{}(key.pool) ^ (std::hash<BlockIdxType>{}(key.block) << 1U);
    }
};

}  // namespace

StorageWriteTask::StorageWriteTask(std::shared_ptr<storage_backend_detail::StorageTaskState> state):
    state_(std::move(state)) {}

StorageBackend::StorageBackend(std::shared_ptr<StorageBackendExecutor> executor): executor_(std::move(executor)) {}

StorageBackend::~StorageBackend() {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    RTP_LLM_CHECK_WITH_INFO(lifecycle_ == Lifecycle::CREATED || lifecycle_ == Lifecycle::STOPPED,
                            "StorageBackend must be shutdown before derived destruction");
}

bool StorageBackend::init(std::shared_ptr<const CacheTopology> topology,
                          PoolsByTag                           pools_by_tag,
                          LeafBufferResolver                   buffer_resolver) {
    PoolsByGroup pools;
    for (auto& [tag, pool] : pools_by_tag) {
        pools.emplace(ModuleGroupKey{CacheModuleId::main(), tag}, std::move(pool));
    }
    RTP_LLM_CHECK(static_cast<bool>(buffer_resolver));
    return initMembers(
        {{CacheModuleId::main(), std::move(topology)}},
        std::move(pools),
        [resolver = std::move(buffer_resolver)](CacheModuleId module_id, int layer, const std::string& tag, int block) {
            RTP_LLM_CHECK_WITH_INFO(module_id.isMain(), "leaf storage resolver received module=%zu", module_id.value());
            return resolver(layer, tag, block);
        });
}

bool StorageBackend::init(const CacheDomainConfig& domain, PoolsByGroup pools, BufferResolver buffer_resolver) {
    ModuleTopologies topologies;
    for (size_t index = 0; index < domain.activeModuleCount(); ++index) {
        topologies.emplace(CacheModuleId(index), domain.module(CacheModuleId(index)).topologyPtr());
    }
    return initMembers(std::move(topologies), std::move(pools), std::move(buffer_resolver));
}

bool StorageBackend::initMembers(ModuleTopologies topologies, PoolsByGroup pools, BufferResolver buffer_resolver) {
    if (init_attempted_) {
        RTP_LLM_LOG_ERROR("StorageBackend initialization has already been attempted");
        return false;
    }
    RTP_LLM_CHECK(!topologies.empty() && buffer_resolver);
    size_t group_count = 0;
    for (const auto& [module_id, topology] : topologies) {
        (void)module_id;
        RTP_LLM_CHECK(topology != nullptr);
        group_count += topology->groups().size();
    }
    RTP_LLM_CHECK(pools.size() == group_count);
    std::unordered_set<const DeviceBlockPool*> seen_pools;
    for (const auto& [key, pool] : pools) {
        (void)topologies.at(key.module_id)->group(key.tag);
        RTP_LLM_CHECK_WITH_INFO(pool != nullptr, "null storage pool for %s", key.debugString().c_str());
        RTP_LLM_CHECK_WITH_INFO(seen_pools.emplace(pool.get()).second,
                                "storage members must own distinct pools: %s",
                                key.debugString().c_str());
    }
    init_attempted_ = true;
    if (executor_ == nullptr) {
        executor_ = makeDefaultStorageBackendExecutor();
    }
    if (executor_->bound_to_backend_.exchange(true)) {
        RTP_LLM_LOG_ERROR("StorageBackend executor cannot be shared between backends");
        return false;
    }
    topologies_          = std::move(topologies);
    pools_by_group_      = std::move(pools);
    buffer_resolver_     = std::move(buffer_resolver);
    const auto fail_init = [this] {
        shutdownImpl();
        buffer_resolver_ = {};
        pools_by_group_.clear();
        topologies_.clear();
        return false;
    };
    bool impl_initialized = false;
    try {
        impl_initialized = initImpl();
    } catch (...) {}
    if (!impl_initialized) {
        // The executor has never started and may still be used by another backend.
        executor_->bound_to_backend_.store(false);
        return fail_init();
    }
    bool started = false;
    try {
        started = executor_->start();
    } catch (...) {}
    if (!started) {
        executor_->shutdown();
        // Keep the binding: shutdown executors are terminal and must not be reused.
        return fail_init();
    }
    initialized_ = true;
    {
        std::lock_guard<std::mutex> lock(lifecycle_mutex_);
        lifecycle_ = Lifecycle::ACCEPTING;
    }
    return true;
}

bool StorageBackend::dispatch(Operation operation) {
    const auto once = std::make_shared<std::once_flag>();
    Lifecycle  outcome;
    {
        std::lock_guard<std::mutex> lock(lifecycle_mutex_);
        outcome = lifecycle_;
        if (outcome == Lifecycle::FINALIZING) {
            outcome = Lifecycle::STOPPED;
        } else if (outcome != Lifecycle::STOPPED) {
            RTP_LLM_CHECK(outcome != Lifecycle::CREATED);
            ++in_flight_;
        }
    }
    if (outcome == Lifecycle::STOPPED) {
        try {
            operation(outcome);
        } catch (...) {}
        return false;
    }
    auto complete = [this, once, operation = std::move(operation)](Lifecycle result) mutable {
        std::call_once(*once, [&] {
            storage_backend_detail::invokeCallback(this, [&] { operation(result); });
            // invokeCallback isolates all exceptions, so every admitted task
            // reaches this exactly-once accounting boundary.
            taskFinished();
        });
    };
    if (outcome != Lifecycle::ACCEPTING) {
        complete(Lifecycle::STOPPING);
        return false;
    }
    try {
        if (executor_->submit([complete]() mutable { complete(Lifecycle::ACCEPTING); })) {
            return true;
        }
    } catch (...) {}
    complete(Lifecycle::STOPPING);
    return false;
}

void StorageBackend::taskFinished() {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    RTP_LLM_CHECK(in_flight_ > 0);
    if (--in_flight_ == 0) {
        lifecycle_cv_.notify_all();
    }
}

void StorageBackend::shutdown() {
    RTP_LLM_CHECK_WITH_INFO(storage_backend_detail::completing_backend != this,
                            "StorageBackend shutdown cannot run from its callback");
    std::shared_ptr<StorageBackendExecutor> executor;
    {
        std::unique_lock<std::mutex> lock(lifecycle_mutex_);
        if (lifecycle_ == Lifecycle::CREATED || lifecycle_ == Lifecycle::STOPPED) {
            return;
        }
        if (lifecycle_ != Lifecycle::ACCEPTING) {
            lifecycle_cv_.wait(lock, [this] { return lifecycle_ == Lifecycle::STOPPED; });
            return;
        }
        lifecycle_ = Lifecycle::STOPPING;
        executor   = executor_;
    }
    executor->shutdown();
    {
        std::unique_lock<std::mutex> lock(lifecycle_mutex_);
        lifecycle_cv_.wait(lock, [this] { return in_flight_ == 0; });
        lifecycle_ = Lifecycle::FINALIZING;
    }
    shutdownImpl();
    {
        std::lock_guard<std::mutex> lock(lifecycle_mutex_);
        lifecycle_ = Lifecycle::STOPPED;
    }
    lifecycle_cv_.notify_all();
}

std::shared_ptr<storage_backend_detail::StorageTaskState> StorageBackend::prepare(StorageRequest request) {
    validateRequest(request, /*allow_null_blocks=*/false);
    auto state     = std::make_shared<storage_backend_detail::StorageTaskState>();
    state->request = std::move(request);
    RTP_LLM_CHECK(initialized_);

    std::unordered_set<BlockKey, BlockKeyHash> pinned;
    for (const auto& key_handles : state->request.handles) {
        for (const StorageBlockHandle& handle : key_handles) {
            const auto&    pool = devicePool(handle.key);
            const BlockKey key{pool.get(), handle.block};
            if (pinned.insert(key).second) {
                pool->incRef(handle.block);
                state->pins.push_back({pool, handle.block});
            }
        }
    }
    return state;
}

void StorageBackend::validateRequest(const StorageRequest& request, bool allow_null_blocks) const {
    RTP_LLM_CHECK_WITH_INFO(request.keys != nullptr, "storage request requires cache keys");
    RTP_LLM_CHECK_WITH_INFO(request.handles.size() == request.keys->size(),
                            "storage request key/handle count mismatch: keys=%zu handles=%zu",
                            request.keys->size(),
                            request.handles.size());
    for (size_t key_index = 0; key_index < request.handles.size(); ++key_index) {
        std::unordered_set<ModuleGroupKey> seen_tags;
        for (const auto& handle : request.handles[key_index]) {
            RTP_LLM_CHECK_WITH_INFO(!handle.key.tag.empty(), "storage handle has empty tag at key=%zu", key_index);
            (void)group(handle.key);
            RTP_LLM_CHECK_WITH_INFO(seen_tags.emplace(handle.key).second,
                                    "storage request has duplicate tag=%s at key=%zu",
                                    handle.key.debugString().c_str(),
                                    key_index);
            RTP_LLM_CHECK_WITH_INFO(allow_null_blocks || !isNullBlockIdx(handle.block),
                                    "storage request has null block for tag=%s at key=%zu",
                                    handle.key.debugString().c_str(),
                                    key_index);
        }
    }
}

const CacheTopology& StorageBackend::topology() const {
    return topology(CacheModuleId::main());
}

const CacheTopology& StorageBackend::topology(CacheModuleId module_id) const {
    const auto it = topologies_.find(module_id);
    RTP_LLM_CHECK_WITH_INFO(
        it != topologies_.end() && it->second, "storage backend missing module=%zu", module_id.value());
    return *it->second;
}

const GroupBase& StorageBackend::group(const ModuleGroupKey& key) const {
    return topology(key.module_id).group(key.tag);
}

const DeviceBlockPoolPtr& StorageBackend::devicePool(const ModuleGroupKey& key) const {
    return pools_by_group_.at(key);
}

std::vector<BlockInfo> StorageBackend::convertIndexToBuffer(CacheModuleId      module_id,
                                                            int                layer_id,
                                                            const std::string& tag,
                                                            int                block_id) const {
    RTP_LLM_CHECK(static_cast<bool>(buffer_resolver_));
    (void)topology(module_id).groupForLayer(layer_id, tag);
    return buffer_resolver_(module_id, layer_id, tag, block_id);
}

bool StorageBackend::isHandleRequired(size_t key_index, size_t matched_key_count, const ModuleGroupKey& key) const {
    RTP_LLM_CHECK(key_index < matched_key_count);
    const size_t reuse_count = group(key).reuseBlockCount(matched_key_count);
    return matched_key_count - key_index <= reuse_count;
}

void StorageBackend::match(StorageRequest request, MatchDone done) {
    RTP_LLM_CHECK(initialized_);
    validateRequest(request, /*allow_null_blocks=*/true);
    dispatch([this, request = std::move(request), done = std::move(done)](Lifecycle outcome) mutable {
        StorageMatchResult result;
        bool               success = outcome == Lifecycle::ACCEPTING;
        if (success) {
            try {
                result = matchImpl(request);
            } catch (...) { success = false; }
        }
        if (done) {
            done(success ? result.matched_blocks_num : 0, success ? std::move(result.match_meta) : nullptr, success);
        }
    });
}

void StorageBackend::read(StorageRequest request, std::shared_ptr<StorageBackendMatchMeta> match_meta, Done done) {
    auto state = prepare(std::move(request));
    dispatch([this, state = std::move(state), match_meta = std::move(match_meta), done = std::move(done)](
                 Lifecycle outcome) mutable {
        bool success = outcome == Lifecycle::ACCEPTING;
        if (success) {
            try {
                readImpl(state->request, match_meta);
            } catch (...) { success = false; }
        }
        state->finish();
        if (done) {
            done(success);
        }
    });
}

StorageWriteTask StorageBackend::prepareWrite(StorageRequest request) {
    RTP_LLM_CHECK(initialized_);
    if (request.empty()) {
        return {};
    }
    return StorageWriteTask(prepare(std::move(request)));
}

bool StorageBackend::write(StorageWriteTask task) {
    RTP_LLM_CHECK(initialized_);
    RTP_LLM_CHECK(task.state_ != nullptr);
    auto state = std::move(task.state_);
    return dispatch([this, state](Lifecycle outcome) {
        if (outcome == Lifecycle::ACCEPTING) {
            try {
                writeImpl(state->request);
            } catch (...) {}
        }
        state->finish();
    });
}

}  // namespace rtp_llm
