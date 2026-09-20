#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <future>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "autil/EnvUtil.h"
#include "autil/legacy/jsonizable.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/ClientWrapper.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/KVCMStorageBackend.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test/MockKVCMClient.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

namespace rtp_llm {
namespace {

using ::testing::_;
using ::testing::Invoke;
using ::testing::Return;

// Test-only observation of backend-owned I/O completion; not a write mode or
// a BlockTree task-pool workflow barrier. The fixtures submit no concurrent I/O.
[[maybe_unused]] bool waitForBackendOperationsForTest(StorageBackend& backend) {
    std::unique_lock<std::mutex> lock(backend.lifecycle_mutex_);
    return backend.lifecycle_cv_.wait_for(lock, std::chrono::seconds(5), [&] { return backend.in_flight_ == 0; });
}

struct KVCMBroadcastState {
    std::mutex                            mutex;
    std::vector<RemoteOperationRequestPB> requests;
    bool                                  fail{false};
};

class KVCMBroadcastRpcService final: public RpcService::Service {
public:
    KVCMBroadcastRpcService(size_t rank, std::shared_ptr<KVCMBroadcastState> state):
        rank_(rank), state_(std::move(state)) {}

    grpc::Status
    ExecuteFunction(grpc::ServerContext*, const FunctionRequestPB* request, FunctionResponsePB* response) override {
        if (!request->has_remote_request()) {
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "missing remote request");
        }
        const auto& remote_request = request->remote_request();
        {
            std::lock_guard<std::mutex> lock(state_->mutex);
            state_->requests.push_back(remote_request);
        }
        if (state_->fail) {
            return grpc::Status(grpc::StatusCode::INTERNAL, "injected KVCM broadcast failure");
        }
        if (remote_request.op() == REMOTE_OPERATION_WRITE) {
            auto* remote_response = response->mutable_remote_response();
            for (int index = 0; index < remote_request.uris_size(); ++index) {
                remote_response->add_actual_uris("actual_rank_" + std::to_string(rank_) + "_" + std::to_string(index));
            }
        }
        return grpc::Status::OK;
    }

private:
    size_t                              rank_;
    std::shared_ptr<KVCMBroadcastState> state_;
};

class KVCMBroadcastRpcServer {
public:
    KVCMBroadcastRpcServer(size_t rank, std::shared_ptr<KVCMBroadcastState> state):
        service_(std::make_unique<KVCMBroadcastRpcService>(rank, std::move(state))) {}

    ~KVCMBroadcastRpcServer() {
        if (server_) {
            server_->Shutdown();
            server_->Wait();
        }
    }

    bool start() {
        grpc::ServerBuilder builder;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &listen_port_);
        builder.RegisterService(service_.get());
        server_ = builder.BuildAndStart();
        return server_ != nullptr && listen_port_ > 0;
    }

    std::string address() const {
        return "127.0.0.1:" + std::to_string(listen_port_);
    }

private:
    std::unique_ptr<KVCMBroadcastRpcService> service_;
    std::unique_ptr<grpc::Server>            server_;
    int                                      listen_port_{0};
};

[[maybe_unused]] std::vector<RemoteOperationRequestPB>
snapshotRequests(const std::shared_ptr<KVCMBroadcastState>& state) {
    std::lock_guard<std::mutex> lock(state->mutex);
    return state->requests;
}

class MockClientWrapper final: public kvcm::ClientWrapper {
public:
    MockClientWrapper(): ClientWrapper(std::make_unique<kvcm::MockClientFactory>()) {}

    MOCK_METHOD(bool,
                initForPools,
                (const ConfigMap&,
                 kv_cache_manager::RoleType,
                 const std::vector<PoolRegistration>&,
                 const std::vector<std::string>&),
                (override));
    MOCK_METHOD(void, shutdown, (), (noexcept, override));
    MOCK_METHOD((std::pair<bool, kv_cache_manager::Locations>),
                match,
                (const std::string&,
                 const std::string&,
                 kv_cache_manager::QueryType,
                 const std::vector<int64_t>&,
                 const kv_cache_manager::BlockMask&,
                 const kv_cache_manager::ForwardContext&),
                (override));
    MOCK_METHOD((std::pair<bool, kv_cache_manager::WriteLocation>),
                getWriteLocation,
                (const std::string&,
                 const std::string&,
                 const std::vector<int64_t>&,
                 const std::vector<int64_t>&,
                 const std::vector<std::string>&,
                 int64_t),
                (override));
    MOCK_METHOD(bool,
                finishWrite,
                (const std::string&,
                 const std::string&,
                 const std::string&,
                 const kv_cache_manager::BlockMask&,
                 const kv_cache_manager::Locations&),
                (override));
    MOCK_METHOD(bool,
                loadKvCachesForTag,
                (const std::string&,
                 const kv_cache_manager::UriStrVec&,
                 kv_cache_manager::BlockBuffers&,
                 const std::shared_ptr<kv_cache_manager::TransferTraceInfo>&),
                (override));
    MOCK_METHOD((std::pair<bool, kv_cache_manager::UriStrVec>),
                saveKvCachesForTag,
                (const std::string&,
                 const kv_cache_manager::UriStrVec&,
                 const kv_cache_manager::BlockBuffers&,
                 const std::shared_ptr<kv_cache_manager::TransferTraceInfo>&),
                (override));
};

struct BackendEnvironment {
    CacheConfig                cache_config;
    DeviceBlockPoolPtr         device_pool;
    StorageBackend::PoolsByTag pools_by_tag;
    BlockIdxType               block_id{NULL_BLOCK_IDX};

    BackendEnvironment()                              = default;
    BackendEnvironment(BackendEnvironment&&) noexcept = default;
    BackendEnvironment& operator=(BackendEnvironment&&) = delete;
    BackendEnvironment(const BackendEnvironment&)       = delete;
    BackendEnvironment& operator=(const BackendEnvironment&) = delete;
    ~BackendEnvironment() {
        for (const auto& [tag, pool] : pools_by_tag) {
            if (pool->isAllocated(block_id)) {
                pool->decRef(block_id);
            }
        }
    }
};

struct BackendHandle {
    std::unique_ptr<KVCMStorageBackend> backend;

    BackendHandle() = default;
    explicit BackendHandle(std::unique_ptr<KVCMStorageBackend> value): backend(std::move(value)) {}
    BackendHandle(BackendHandle&&) noexcept = default;
    BackendHandle& operator=(BackendHandle&&) noexcept = default;
    BackendHandle(const BackendHandle&)                = delete;
    BackendHandle& operator=(const BackendHandle&) = delete;
    ~BackendHandle() {
        if (backend) {
            backend->shutdown();
        }
    }

    KVCMStorageBackend* operator->() const {
        return backend.get();
    }
};

class ScopedReferencedBlocks {
public:
    ScopedReferencedBlocks(DeviceBlockPoolPtr pool, size_t count): pool_(std::move(pool)) {
        const auto blocks = pool_->malloc(count);
        RTP_LLM_CHECK(blocks.has_value());
        blocks_ = *blocks;
        pool_->incRef(blocks_);
    }

    ScopedReferencedBlocks(const BackendEnvironment& environment, size_t count) {
        for (const auto& tag : environment.cache_config.groupTags()) {
            const auto& pool   = environment.pools_by_tag.at(tag);
            const auto  blocks = pool->malloc(count);
            RTP_LLM_CHECK(blocks.has_value());
            pool->incRef(*blocks);
            if (blocks_.empty()) {
                blocks_ = *blocks;
            }
            RTP_LLM_CHECK(blocks_ == *blocks);
            pools_.push_back(pool);
        }
    }
    ~ScopedReferencedBlocks() {
        if (pool_) {
            pool_->decRef(blocks_);
        }
        for (const auto& pool : pools_) {
            pool->decRef(blocks_);
        }
    }

    const std::vector<BlockIdxType>& get() const {
        return blocks_;
    }

private:
    DeviceBlockPoolPtr              pool_;
    std::vector<DeviceBlockPoolPtr> pools_;
    std::vector<BlockIdxType>       blocks_;
};

struct MatchObservation {
    size_t                                   matched_blocks_num{0};
    std::shared_ptr<StorageBackendMatchMeta> match_meta;
    bool                                     success{false};
};

template<typename T>
T await(std::future<T>& future) {
    if (future.wait_for(std::chrono::seconds(30)) != std::future_status::ready) {
        ADD_FAILURE() << "timed out waiting for KVCM storage callback";
        return T{};
    }
    return future.get();
}

void initializeEnvironmentPools(BackendEnvironment& environment, const std::string& name, size_t count) {
    for (const auto& group : environment.cache_config.topology().groups()) {
        std::vector<block_tree_cache_test::DeviceLayerBufferSpec> layers(
            environment.cache_config.layerIdsForGroup(group.tag).size(),
            {group.kvBlockStrideBytes(), group.kvScaleStrideBytes()});
        auto       pool  = block_tree_cache_test::makeDevicePool(layers, count, name + "_" + group.tag);
        const auto block = pool->malloc();
        RTP_LLM_CHECK(block.has_value());
        pool->incRef(*block);
        if (!environment.device_pool) {
            environment.device_pool = pool;
            environment.block_id    = *block;
        }
        RTP_LLM_CHECK(*block == environment.block_id);
        RTP_LLM_CHECK(environment.pools_by_tag.emplace(group.tag, pool).second);
    }
}

std::vector<BlockInfo>
environmentBuffers(const BackendEnvironment& environment, int layer, const std::string& tag, int block) {
    const auto layers = environment.cache_config.layerIdsForGroup(tag);
    const auto it     = std::find(layers.begin(), layers.end(), layer);
    RTP_LLM_CHECK(it != layers.end());
    return environment.pools_by_tag.at(tag)->convertIndexToBuffer(std::distance(layers.begin(), it), block);
}

[[maybe_unused]] BackendEnvironment makeBackendEnvironment(const std::string& pool_name) {
    BackendEnvironment result;
    result.cache_config = test::makeSimpleMhaCacheConfig(/*layer_num=*/1,
                                                         /*block_num=*/5,
                                                         /*tokens_per_block=*/8,
                                                         DataType::TYPE_FP16,
                                                         /*local_head_num_kv=*/1,
                                                         /*size_per_head=*/2);
    initializeEnvironmentPools(result, pool_name, 8);
    return result;
}

[[maybe_unused]] BackendEnvironment makeHybridBackendEnvironment(const std::string& pool_name) {
    BackendEnvironment result;
    result.cache_config = test::makeSimpleHybridMhaCacheConfig(/*layer_num=*/4,
                                                               /*block_num=*/8,
                                                               /*tokens_per_block=*/8,
                                                               DataType::TYPE_FP16,
                                                               /*group_layer_num=*/2,
                                                               /*local_head_num_kv=*/1,
                                                               /*size_per_head=*/2);
    initializeEnvironmentPools(result, pool_name, 8);
    return result;
}

[[maybe_unused]] BackendEnvironment makeMultiGroupBackendEnvironment(const std::string& pool_name,
                                                                     size_t             full_group_count,
                                                                     size_t             linear_group_count,
                                                                     int                linear_step = 2) {
    RTP_LLM_CHECK(full_group_count > 0 && linear_group_count > 0);
    BackendEnvironment result;
    const size_t       group_count         = full_group_count + linear_group_count;
    result.cache_config.dtype              = DataType::TYPE_FP16;
    result.cache_config.layer_num          = static_cast<uint32_t>(group_count);
    result.cache_config.seq_size_per_block = 8;
    result.cache_config.linear_step        = linear_step;

    std::vector<KVCacheSpecPtr>   specs;
    std::vector<std::vector<int>> layers_by_group;
    std::vector<CacheGroupType>   types;
    std::vector<std::string>      tags;
    specs.reserve(group_count);
    layers_by_group.reserve(group_count);
    types.reserve(group_count);
    tags.reserve(group_count);
    for (size_t group_id = 0; group_id < group_count; ++group_id) {
        const bool        is_full = group_id < full_group_count;
        const std::string tag =
            (is_full ? "full" : "linear") + std::to_string(is_full ? group_id : group_id - full_group_count);
        specs.push_back(is_full ? test::makeMhaSpec(tag, 8, DataType::TYPE_FP16, 1, 2) :
                                  test::makeLinearSpec(tag, 8, DataType::TYPE_FP16, 1, 2));
        layers_by_group.push_back({static_cast<int>(group_id)});
        types.push_back(is_full ? CacheGroupType::FULL : CacheGroupType::LINEAR);
        tags.push_back(tag);
    }
    result.cache_config.fromGroupedSpecs(specs, layers_by_group, types, tags);
    result.cache_config.finalizeBlockNums(/*baseline_block_num=*/8, RuntimeConfig{});

    initializeEnvironmentPools(result, pool_name, 12);
    return result;
}

[[maybe_unused]] BackendHandle makeBackend(const BackendEnvironment&                 environment,
                                           const ParallelismConfig&                  parallelism_config,
                                           const std::shared_ptr<MockClientWrapper>& client_wrapper,
                                           std::shared_ptr<BroadcastManager>         broadcast_manager = nullptr) {
    KVCacheConfig kv_cache_config;
    kv_cache_config.kvcm_server_address = "unused-test-address";
    RuntimeConfig runtime_config;
    runtime_config.model_name = "kvcm_test_model";
    return BackendHandle(std::make_unique<KVCMStorageBackend>(environment.cache_config,
                                                              kv_cache_config,
                                                              runtime_config,
                                                              parallelism_config,
                                                              SpeculativeExecutionConfig{},
                                                              std::move(broadcast_manager),
                                                              client_wrapper));
}

[[maybe_unused]] StorageRequest makeStorageRequest(const BackendEnvironment& environment,
                                                   CacheKeysType             keys                 = {101},
                                                   size_t                    local_matched_blocks = 0,
                                                   std::vector<BlockIdxType> block_ids            = {}) {
    StorageRequest request;
    request.keys = std::make_shared<const CacheKeysType>(std::move(keys));
    request.handles.resize(request.keys->size());
    RTP_LLM_CHECK(block_ids.empty() || block_ids.size() == request.keys->size());
    for (size_t key_index = 0; key_index < request.handles.size(); ++key_index) {
        const auto block_id = block_ids.empty() ? environment.block_id : block_ids[key_index];
        request.handles[key_index].push_back({environment.cache_config.groupTags().front(), block_id});
    }
    request.local_matched_blocks_num = local_matched_blocks;
    return request;
}

[[maybe_unused]] StorageRequest makeGroupedStorageRequest(const BackendEnvironment&             environment,
                                                          CacheKeysType                         keys,
                                                          size_t                                local_matched_blocks,
                                                          const std::vector<BlockIdxType>&      block_ids,
                                                          std::vector<std::vector<std::string>> groups_by_key = {}) {
    RTP_LLM_CHECK(keys.size() == block_ids.size());
    if (groups_by_key.empty()) {
        groups_by_key.resize(keys.size());
        for (auto& groups : groups_by_key) {
            groups = environment.cache_config.groupTags();
        }
    }
    RTP_LLM_CHECK(groups_by_key.size() == keys.size());

    StorageRequest request;
    request.keys = std::make_shared<const CacheKeysType>(std::move(keys));
    request.handles.resize(request.keys->size());
    for (size_t key_index = 0; key_index < request.handles.size(); ++key_index) {
        for (const auto& tag : groups_by_key[key_index]) {
            environment.cache_config.group(tag);
            request.handles[key_index].push_back({tag, block_ids[key_index]});
        }
    }
    request.local_matched_blocks_num = local_matched_blocks;
    return request;
}

[[maybe_unused]] void* blockBase(const BackendEnvironment& environment, const std::string& tag, BlockIdxType block_id) {
    const auto layer_ids = environment.cache_config.layerIdsForGroup(tag);
    RTP_LLM_CHECK(layer_ids.size() == 1u);
    const auto info = environmentBuffers(environment, layer_ids.front(), tag, block_id);
    RTP_LLM_CHECK(info.size() == 1u);
    return info.front().addr;
}

[[maybe_unused]] MatchObservation match(KVCMStorageBackend& backend, StorageRequest request) {
    auto promise = std::make_shared<std::promise<MatchObservation>>();
    auto future  = promise->get_future();
    backend.match(
        std::move(request),
        [promise](size_t matched_blocks_num, std::shared_ptr<StorageBackendMatchMeta> match_meta, bool success) {
            promise->set_value({matched_blocks_num, std::move(match_meta), success});
        });
    return await(future);
}

[[maybe_unused]] bool
read(KVCMStorageBackend& backend, StorageRequest request, std::shared_ptr<StorageBackendMatchMeta> match_meta) {
    auto promise = std::make_shared<std::promise<bool>>();
    auto future  = promise->get_future();
    backend.read(std::move(request), std::move(match_meta), [promise](bool success) { promise->set_value(success); });
    return await(future);
}

[[maybe_unused]] ParallelismConfig singleRankConfig() {
    ParallelismConfig config;
    config.tp_size    = 1;
    config.tp_rank    = 0;
    config.local_rank = 0;
    return config;
}

[[maybe_unused]] bool initSingleRank(KVCMStorageBackend& backend, const BackendEnvironment& environment) {
    return backend.init(environment.cache_config.topologyPtr(),
                        environment.pools_by_tag,
                        [&](int layer_id, const std::string& tag, int block_id) {
                            return environmentBuffers(environment, layer_id, tag, block_id);
                        });
}

}  // namespace
}  // namespace rtp_llm
