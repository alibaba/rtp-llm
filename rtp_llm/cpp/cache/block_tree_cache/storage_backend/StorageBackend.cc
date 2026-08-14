#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/StorageBackend.h"

#include <mutex>
#include <unordered_map>
#include <utility>

#include "rtp_llm/cpp/cache/BlockReleaseBatch.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm::storage_backend_detail {

struct StorageTaskState {
    struct Pin {
        std::shared_ptr<IBlockPool> pool;
        BlockIdxType                block;
    };

    struct Target {
        size_t pin_index{0};
        size_t group_id{0};
    };

    StorageRequest                  request;
    std::vector<Pin>                pins;
    std::vector<Target>             targets;
    StorageBackend::ReleaseCallback release;
    std::once_flag                  finish_once;

    void finish() {
        std::call_once(finish_once, [this] {
            BlockReleaseBatch                            releases;
            std::vector<std::vector<BlockRefTransition>> transitions;
            transitions.reserve(pins.size());
            for (const Pin& pin : pins) {
                transitions.push_back(pin.pool->decRefWithResult({pin.block}, BlockRefType::STORAGE_BACKEND));
            }
            for (const Target& target : targets) {
                releases.append(target.group_id, transitions[target.pin_index]);
            }
            pins.clear();
            targets.clear();
            auto callback = std::move(release);
            auto receipts = releases.finish();
            if (callback && !receipts.empty()) {
                callback(receipts);
            }
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

bool StorageBackend::init(std::shared_ptr<const CacheTopology>     topology,
                          std::vector<std::shared_ptr<IBlockPool>> device_pools,
                          BufferResolver                           buffer_resolver,
                          ReleaseCallback                          release_callback) {
    RTP_LLM_CHECK_WITH_INFO(!topology_, "StorageBackend is already initialized");
    RTP_LLM_CHECK(topology && device_pools.size() == topology->groups().size() && buffer_resolver && release_callback);
    for (const auto& pool : device_pools) {
        RTP_LLM_CHECK(pool != nullptr);
    }
    topology_         = std::move(topology);
    device_pools_     = std::move(device_pools);
    buffer_resolver_  = std::move(buffer_resolver);
    release_callback_ = std::move(release_callback);
    initialized_      = initImpl();
    return initialized_;
}

std::shared_ptr<storage_backend_detail::StorageTaskState> StorageBackend::prepare(StorageRequest request) {
    auto state     = std::make_shared<storage_backend_detail::StorageTaskState>();
    state->request = std::move(request);
    state->release = release_callback_;
    RTP_LLM_CHECK(initialized_);

    std::unordered_map<BlockKey, size_t, BlockKeyHash> pin_indexes;
    for (const auto& key_handles : state->request.handles) {
        for (const StorageBlockHandle& handle : key_handles) {
            RTP_LLM_CHECK(handle.group_id < device_pools_.size() && !isNullBlockIdx(handle.block));
            const auto&    pool = device_pools_[handle.group_id];
            const BlockKey key{pool.get(), handle.block};
            const auto [pin, inserted] = pin_indexes.emplace(key, state->pins.size());
            if (inserted) {
                pool->incRef(handle.block, BlockRefType::STORAGE_BACKEND);
                state->pins.push_back({pool, handle.block});
            }
            state->targets.push_back({pin->second, handle.group_id});
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
    matchImpl(std::move(request), std::move(done));
}

void StorageBackend::read(StorageRequest request, std::shared_ptr<StorageBackendMatchMeta> match_meta, Done done) {
    auto state            = prepare(std::move(request));
    auto prepared_request = std::move(state->request);
    readImpl(std::move(prepared_request), std::move(match_meta), [state = std::move(state), done = std::move(done)] {
        state->finish();
        done();
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
    auto state            = std::move(task.state_);
    auto prepared_request = std::move(state->request);
    writeImpl(std::move(prepared_request), [state = std::move(state)] { state->finish(); });
}

}  // namespace rtp_llm
