#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/StorageBackend.h"

#include <mutex>
#include <unordered_set>
#include <utility>

#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm::storage_backend_detail {

thread_local const StorageBackend* completing_backend = nullptr;

template<typename Callback>
void invokeCallback(const StorageBackend* backend, Callback&& callback) noexcept {
    const auto* previous = completing_backend;
    completing_backend   = backend;
    callback();
    completing_backend = previous;
}

struct StorageTaskState {
    struct Pin {
        std::shared_ptr<IBlockPool> pool;
        BlockIdxType                block;
    };

    StorageRequest   request;
    std::vector<Pin> pins;
    std::once_flag   finish_once;

    void finish() {
        std::call_once(finish_once, [this] {
            for (const Pin& pin : pins) {
                pin.pool->decRef(pin.block, BlockRefType::STORAGE_BACKEND);
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
    IBlockPool*  pool;
    BlockIdxType block;
    bool         operator==(const BlockKey& other) const {
        return pool == other.pool && block == other.block;
    }
};

struct BlockKeyHash {
    size_t operator()(const BlockKey& key) const {
        return std::hash<IBlockPool*>{}(key.pool) ^ (std::hash<BlockIdxType>{}(key.block) << 1U);
    }
};

}  // namespace

StorageWriteTask::StorageWriteTask(std::shared_ptr<storage_backend_detail::StorageTaskState> state):
    state_(std::move(state)) {}

StorageBackend::StorageBackend(std::shared_ptr<StorageBackendExecutor> executor):
    executor_(std::move(executor)) {}

StorageBackend::~StorageBackend() {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    RTP_LLM_CHECK_WITH_INFO(lifecycle_ == Lifecycle::CREATED || lifecycle_ == Lifecycle::STOPPED,
                            "StorageBackend must be shutdown before derived destruction");
}

bool StorageBackend::init(std::shared_ptr<const CacheTopology>     topology,
                          std::vector<std::shared_ptr<IBlockPool>> device_pools,
                          BufferResolver                           buffer_resolver) {
    RTP_LLM_CHECK_WITH_INFO(!topology_, "StorageBackend is already initialized");
    RTP_LLM_CHECK(topology && device_pools.size() == topology->groups().size() && buffer_resolver);
    for (const auto& pool : device_pools) {
        RTP_LLM_CHECK(pool != nullptr);
    }
    topology_               = std::move(topology);
    device_pools_           = std::move(device_pools);
    buffer_resolver_        = std::move(buffer_resolver);
    if (!initImpl()) {
        return false;
    }
    if (executor_ == nullptr) {
        executor_ = makeDefaultStorageBackendExecutor();
    }
    bool started = false;
    try {
        started = executor_ != nullptr && executor_->start();
    } catch (...) {}
    if (!started) {
        executor_->shutdown();
        return false;
    }
    initialized_ = true;
    {
        std::lock_guard<std::mutex> lock(lifecycle_mutex_);
        lifecycle_ = Lifecycle::ACCEPTING;
    }
    return true;
}

void StorageBackend::dispatch(Operation operation) {
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
        return;
    }
    auto complete = [this, once, operation = std::move(operation)](Lifecycle result) mutable {
        std::call_once(*once, [&] {
            storage_backend_detail::invokeCallback(this, [&] { operation(result); });
            taskFinished();
        });
    };
    if (outcome != Lifecycle::ACCEPTING) {
        complete(Lifecycle::STOPPING);
        return;
    }
    try {
        if (executor_->submit([complete]() mutable { complete(Lifecycle::ACCEPTING); })) {
            return;
        }
    } catch (...) {}
    complete(Lifecycle::STOPPING);
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
    {
        std::lock_guard<std::mutex> lock(lifecycle_mutex_);
        lifecycle_ = Lifecycle::STOPPED;
    }
    lifecycle_cv_.notify_all();
}

std::shared_ptr<storage_backend_detail::StorageTaskState> StorageBackend::prepare(StorageRequest request) {
    auto state     = std::make_shared<storage_backend_detail::StorageTaskState>();
    state->request = std::move(request);
    RTP_LLM_CHECK(initialized_);

    std::unordered_set<BlockKey, BlockKeyHash> pinned;
    for (const auto& key_handles : state->request.handles) {
        for (const StorageBlockHandle& handle : key_handles) {
            RTP_LLM_CHECK(handle.group_id < device_pools_.size() && !isNullBlockIdx(handle.block));
            const auto&    pool = device_pools_[handle.group_id];
            const BlockKey key{pool.get(), handle.block};
            if (pinned.insert(key).second) {
                pool->incRef(handle.block, BlockRefType::STORAGE_BACKEND);
                state->pins.push_back({pool, handle.block});
            }
        }
    }
    return state;
}

const CacheTopology& StorageBackend::topology() const {
    RTP_LLM_CHECK(topology_ != nullptr);
    return *topology_;
}

std::vector<BlockInfo> StorageBackend::convertIndexToBuffer(int layer_id, int group_id, int block_id) const {
    RTP_LLM_CHECK(buffer_resolver_ && group_id >= 0 && static_cast<size_t>(group_id) < device_pools_.size());
    return buffer_resolver_(layer_id, group_id, block_id);
}

bool StorageBackend::isHandleRequired(size_t key_index, size_t matched_key_count, size_t group_id) const {
    RTP_LLM_CHECK(key_index < matched_key_count);
    const size_t reuse_count = topology().groupById(group_id).reuseBlockCount(matched_key_count);
    return matched_key_count - key_index <= reuse_count;
}

void StorageBackend::match(StorageRequest request, MatchDone done) {
    RTP_LLM_CHECK(initialized_);
    dispatch([this, request = std::move(request), done = std::move(done)](Lifecycle outcome) mutable {
        StorageMatchResult result;
        bool               success = outcome == Lifecycle::ACCEPTING;
        if (success) {
            try {
                result = matchImpl(request);
            } catch (...) {
                success = false;
            }
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
            } catch (...) {
                success = false;
            }
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

void StorageBackend::write(StorageWriteTask task) {
    RTP_LLM_CHECK(initialized_);
    RTP_LLM_CHECK(task.state_ != nullptr);
    auto state = std::move(task.state_);
    dispatch([this, state = std::move(state)](Lifecycle outcome) {
        if (outcome == Lifecycle::ACCEPTING) {
            try {
                writeImpl(state->request);
            } catch (...) {}
        }
        state->finish();
    });
}

}  // namespace rtp_llm
