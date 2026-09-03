#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <chrono>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <variant>
#include <vector>

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

struct KVCMBroadcastState {
    std::mutex                            mutex;
    std::vector<RemoteOperationRequestPB> requests;
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

std::vector<RemoteOperationRequestPB> snapshotRequests(const std::shared_ptr<KVCMBroadcastState>& state) {
    std::lock_guard<std::mutex> lock(state->mutex);
    return state->requests;
}

class MockClientWrapper final: public kvcm::ClientWrapper {
public:
    MockClientWrapper(): ClientWrapper(std::make_unique<kvcm::MockClientFactory>()) {}

    MOCK_METHOD(bool, init, (const ConfigMap& config_map, const kv_cache_manager::InitParams& init_params), (override));
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
                loadKvCaches,
                (const kv_cache_manager::UriStrVec&,
                 kv_cache_manager::BlockBuffers&,
                 const std::shared_ptr<kv_cache_manager::TransferTraceInfo>&),
                (override));
    MOCK_METHOD((std::pair<bool, kv_cache_manager::UriStrVec>),
                saveKvCaches,
                (const kv_cache_manager::UriStrVec&,
                 const kv_cache_manager::BlockBuffers&,
                 const std::shared_ptr<kv_cache_manager::TransferTraceInfo>&),
                (override));
};

struct BackendEnvironment {
    CacheConfig        cache_config;
    DeviceBlockPoolPtr device_pool;
    BlockIdxType       block_id{NULL_BLOCK_IDX};

    BackendEnvironment()                                     = default;
    BackendEnvironment(BackendEnvironment&&) noexcept        = default;
    BackendEnvironment& operator=(BackendEnvironment&&)      = delete;
    BackendEnvironment(const BackendEnvironment&)            = delete;
    BackendEnvironment& operator=(const BackendEnvironment&) = delete;
    ~BackendEnvironment() {
        if (device_pool && device_pool->isAllocated(block_id)) {
            device_pool->decRef(block_id);
        }
    }
};

struct BackendHandle {
    std::unique_ptr<KVCMStorageBackend> backend;

    BackendHandle() = default;
    explicit BackendHandle(std::unique_ptr<KVCMStorageBackend> value): backend(std::move(value)) {}
    BackendHandle(BackendHandle&&) noexcept            = default;
    BackendHandle& operator=(BackendHandle&&) noexcept = default;
    BackendHandle(const BackendHandle&)                = delete;
    BackendHandle& operator=(const BackendHandle&)     = delete;
    ~BackendHandle() {
        if (backend) {
            backend->shutdown();
        }
    }

    KVCMStorageBackend* operator->() const {
        return backend.get();
    }
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

BackendEnvironment makeBackendEnvironment(const std::string& pool_name) {
    BackendEnvironment result;
    result.cache_config = test::makeSimpleMhaCacheConfig(/*layer_num=*/1,
                                                         /*block_num=*/5,
                                                         /*tokens_per_block=*/8,
                                                         DataType::TYPE_FP16,
                                                         /*local_head_num_kv=*/1,
                                                         /*size_per_head=*/2);
    result.device_pool  = block_tree_cache_test::makeDevicePool(
        {{result.cache_config.kv_block_stride_bytes, result.cache_config.kv_scale_stride_bytes}},
        /*usable_count=*/4,
        pool_name);
    const auto block = result.device_pool->malloc();
    RTP_LLM_CHECK(block.has_value());
    result.block_id = *block;
    result.device_pool->incRef(result.block_id);
    return result;
}

BackendHandle makeBackend(const BackendEnvironment&                 environment,
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

StorageRequest
makeStorageRequest(const BackendEnvironment& environment, CacheKeysType keys = {101}, size_t local_matched_blocks = 0) {
    StorageRequest request;
    request.keys = std::make_shared<const CacheKeysType>(std::move(keys));
    request.handles.resize(request.keys->size());
    for (auto& handles : request.handles) {
        handles.push_back({/*group_id=*/0, environment.block_id});
    }
    request.local_matched_blocks_num = local_matched_blocks;
    return request;
}

MatchObservation match(KVCMStorageBackend& backend, StorageRequest request) {
    auto promise = std::make_shared<std::promise<MatchObservation>>();
    auto future  = promise->get_future();
    backend.match(
        std::move(request),
        [promise](size_t matched_blocks_num, std::shared_ptr<StorageBackendMatchMeta> match_meta, bool success) {
            promise->set_value({matched_blocks_num, std::move(match_meta), success});
        });
    return await(future);
}

bool read(KVCMStorageBackend& backend, StorageRequest request, std::shared_ptr<StorageBackendMatchMeta> match_meta) {
    auto promise = std::make_shared<std::promise<bool>>();
    auto future  = promise->get_future();
    backend.read(std::move(request), std::move(match_meta), [promise](bool success) { promise->set_value(success); });
    return await(future);
}

ParallelismConfig singleRankConfig() {
    ParallelismConfig config;
    config.tp_size    = 1;
    config.tp_rank    = 0;
    config.local_rank = 0;
    return config;
}

bool initSingleRank(KVCMStorageBackend& backend, const BackendEnvironment& environment) {
    return backend.init(
        environment.cache_config.topologyPtr(), {environment.device_pool}, [&](int layer_id, int, int block_id) {
            return environment.device_pool->convertIndexToBuffer(layer_id, block_id);
        });
}

TEST(KVCMStorageBackendTest, TP2WorkerRegistersItsRankAndExecutesLocalPayload) {
    auto environment    = makeBackendEnvironment("kvcm_storage_backend_tp2_worker");
    auto client_wrapper = std::make_shared<MockClientWrapper>();

    ParallelismConfig parallelism_config;
    parallelism_config.tp_size    = 2;
    parallelism_config.tp_rank    = 1;
    parallelism_config.local_rank = 0;

    EXPECT_CALL(*client_wrapper, init(_, _))
        .WillOnce(Invoke(
            [&](const kvcm::ClientWrapper::ConfigMap& config_map, const kv_cache_manager::InitParams& init_params) {
                EXPECT_EQ(init_params.role_type, kv_cache_manager::RoleType::WORKER);
                EXPECT_EQ(init_params.self_location_spec_name, "tp1_Fdefault");
                EXPECT_NE(init_params.regist_span, nullptr);
                if (init_params.regist_span != nullptr) {
                    EXPECT_EQ(init_params.regist_span->base, environment.device_pool->getBaseAddress());
                    EXPECT_EQ(init_params.regist_span->size, environment.device_pool->getTotalSizeBytes());
                }
                const auto config = config_map.find("");
                EXPECT_NE(config, config_map.end());
                if (config != config_map.end()) {
                    const std::string json = autil::legacy::ToJsonString(config->second, /*isCompact=*/true);
                    EXPECT_NE(json.find("tp0_Fdefault"), std::string::npos);
                    EXPECT_NE(json.find("tp1_Fdefault"), std::string::npos);
                    EXPECT_NE(json.find("\"tp_size\":2"), std::string::npos);
                }
                return true;
            }));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);

    auto backend = makeBackend(environment, parallelism_config, client_wrapper);
    ASSERT_TRUE(backend->init(environment.cache_config.topologyPtr(),
                              {environment.device_pool},
                              [&](int layer_id, int group_id, int block_id) {
                                  EXPECT_EQ(group_id, 0);
                                  return environment.device_pool->convertIndexToBuffer(layer_id, block_id);
                              }));

    const kv_cache_manager::UriStrVec expected_read_uris{"read_uri"};
    EXPECT_CALL(*client_wrapper, loadKvCaches(expected_read_uris, _, _))
        .WillOnce(Invoke([](const kv_cache_manager::UriStrVec&,
                            kv_cache_manager::BlockBuffers& buffers,
                            const std::shared_ptr<kv_cache_manager::TransferTraceInfo>&) {
            EXPECT_EQ(buffers.size(), 1u);
            if (!buffers.empty()) {
                EXPECT_EQ(buffers.front().iovs.size(), 1u);
            }
            return true;
        }));
    RemoteOperationRequestPB read_request;
    read_request.set_op(REMOTE_OPERATION_READ);
    read_request.add_group_tags("default");
    read_request.add_block_ids(environment.block_id);
    read_request.add_uris(expected_read_uris.front());
    RemoteOperationResponsePB read_response;
    EXPECT_TRUE(backend->execute(read_request, read_response));

    const kv_cache_manager::UriStrVec expected_write_uris{"write_uri"};
    EXPECT_CALL(*client_wrapper, saveKvCaches(expected_write_uris, _, _))
        .WillOnce(Return(std::make_pair(true, kv_cache_manager::UriStrVec{"actual_write_uri"})));
    RemoteOperationRequestPB write_request;
    write_request.set_op(REMOTE_OPERATION_WRITE);
    write_request.add_group_tags("default");
    write_request.add_block_ids(environment.block_id);
    write_request.add_uris(expected_write_uris.front());
    RemoteOperationResponsePB write_response;
    ASSERT_TRUE(backend->execute(write_request, write_response));
    ASSERT_EQ(write_response.actual_uris_size(), 1);
    EXPECT_EQ(write_response.actual_uris(0), "actual_write_uri");
}

TEST(KVCMStorageBackendTest, TP2CoordinatorRejectsMissingBroadcastManager) {
    auto environment    = makeBackendEnvironment("kvcm_storage_backend_tp2_coordinator");
    auto client_wrapper = std::make_shared<MockClientWrapper>();

    ParallelismConfig parallelism_config;
    parallelism_config.tp_size    = 2;
    parallelism_config.tp_rank    = 0;
    parallelism_config.local_rank = 0;

    EXPECT_CALL(*client_wrapper, init(_, _)).Times(0);
    auto backend = makeBackend(environment, parallelism_config, client_wrapper);
    EXPECT_FALSE(backend->init(
        environment.cache_config.topologyPtr(), {environment.device_pool}, [&](int layer_id, int, int block_id) {
            return environment.device_pool->convertIndexToBuffer(layer_id, block_id);
        }));
}

TEST(KVCMStorageBackendTest, TP2CoordinatorBroadcastsRankOrderedReadAndWritePayloads) {
    auto environment    = makeBackendEnvironment("kvcm_storage_backend_tp2_broadcast");
    auto client_wrapper = std::make_shared<MockClientWrapper>();

    std::vector<std::shared_ptr<KVCMBroadcastState>>     states;
    std::vector<std::unique_ptr<KVCMBroadcastRpcServer>> servers;
    std::vector<std::string>                             addresses;
    for (size_t rank = 0; rank < 2; ++rank) {
        auto state  = std::make_shared<KVCMBroadcastState>();
        auto server = std::make_unique<KVCMBroadcastRpcServer>(rank, state);
        ASSERT_TRUE(server->start());
        states.push_back(std::move(state));
        addresses.push_back(server->address());
        servers.push_back(std::move(server));
    }
    auto broadcast_manager = std::make_shared<BroadcastManager>(addresses);
    ASSERT_TRUE(broadcast_manager->init());

    ParallelismConfig parallelism_config;
    parallelism_config.tp_size    = 2;
    parallelism_config.tp_rank    = 0;
    parallelism_config.local_rank = 0;
    EXPECT_CALL(*client_wrapper, init(_, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment, parallelism_config, client_wrapper, broadcast_manager);
    ASSERT_TRUE(backend->init(
        environment.cache_config.topologyPtr(), {environment.device_pool}, [&](int layer_id, int, int block_id) {
            return environment.device_pool->convertIndexToBuffer(layer_id, block_id);
        }));

    const kv_cache_manager::Locations read_locations{kv_cache_manager::Location{
        kv_cache_manager::LocationSpecUnit{"tp1_Fdefault", "read_rank_1"},
        kv_cache_manager::LocationSpecUnit{"tp0_Fdefault", "read_rank_0"},
    }};
    EXPECT_CALL(*client_wrapper, match(_, _, _, _, _, _)).WillOnce(Return(std::make_pair(true, read_locations)));
    auto observation = match(*backend.backend, makeStorageRequest(environment));
    ASSERT_TRUE(observation.success);
    ASSERT_EQ(observation.matched_blocks_num, 1u);
    ASSERT_NE(observation.match_meta, nullptr);
    EXPECT_TRUE(read(*backend.backend, makeStorageRequest(environment), std::move(observation.match_meta)));

    for (size_t rank = 0; rank < states.size(); ++rank) {
        const auto requests = snapshotRequests(states[rank]);
        ASSERT_EQ(requests.size(), 1u);
        EXPECT_EQ(requests[0].op(), REMOTE_OPERATION_READ);
        ASSERT_EQ(requests[0].group_tags_size(), 1);
        ASSERT_EQ(requests[0].block_ids_size(), 1);
        ASSERT_EQ(requests[0].uris_size(), 1);
        EXPECT_EQ(requests[0].group_tags(0), "default");
        EXPECT_EQ(requests[0].block_ids(0), environment.block_id);
        EXPECT_EQ(requests[0].uris(0), "read_rank_" + std::to_string(rank));
    }

    kv_cache_manager::WriteLocation write_location;
    write_location.write_session_id = "tp2_broadcast_write";
    write_location.block_mask       = kv_cache_manager::BlockMaskOffset{0};
    write_location.locations        = {kv_cache_manager::Location{
        kv_cache_manager::LocationSpecUnit{"tp1_Fdefault", "write_rank_1"},
        kv_cache_manager::LocationSpecUnit{"tp0_Fdefault", "write_rank_0"},
    }};
    EXPECT_CALL(*client_wrapper, getWriteLocation(_, _, _, _, _, _))
        .WillOnce(Return(std::make_pair(true, write_location)));
    EXPECT_CALL(*client_wrapper, finishWrite(_, _, "tp2_broadcast_write", _, _))
        .WillOnce(Invoke([](const std::string&,
                            const std::string&,
                            const std::string&,
                            const kv_cache_manager::BlockMask&,
                            const kv_cache_manager::Locations& locations) {
            if (locations.size() != 1u || locations[0].size() != 2u) {
                return false;
            }
            EXPECT_EQ(locations[0][0].spec_name, "tp1_Fdefault");
            EXPECT_EQ(locations[0][0].uri, "actual_rank_1_0");
            EXPECT_EQ(locations[0][1].spec_name, "tp0_Fdefault");
            EXPECT_EQ(locations[0][1].uri, "actual_rank_0_0");
            return true;
        }));
    EXPECT_TRUE(backend->write(backend->prepareWrite(makeStorageRequest(environment)), /*synchronous=*/true));

    for (size_t rank = 0; rank < states.size(); ++rank) {
        const auto requests = snapshotRequests(states[rank]);
        ASSERT_EQ(requests.size(), 2u);
        EXPECT_EQ(requests[1].op(), REMOTE_OPERATION_WRITE);
        ASSERT_EQ(requests[1].group_tags_size(), 1);
        ASSERT_EQ(requests[1].block_ids_size(), 1);
        ASSERT_EQ(requests[1].uris_size(), 1);
        EXPECT_EQ(requests[1].group_tags(0), "default");
        EXPECT_EQ(requests[1].block_ids(0), environment.block_id);
        EXPECT_EQ(requests[1].uris(0), "write_rank_" + std::to_string(rank));
    }
}

TEST(KVCMStorageBackendTest, RejectsMismatchedTransferVectorsBeforeClientIO) {
    auto environment    = makeBackendEnvironment("kvcm_storage_backend_bad_shape");
    auto client_wrapper = std::make_shared<MockClientWrapper>();

    ParallelismConfig parallelism_config;
    parallelism_config.tp_size    = 1;
    parallelism_config.tp_rank    = 0;
    parallelism_config.local_rank = 0;

    EXPECT_CALL(*client_wrapper, init(_, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, loadKvCaches(_, _, _)).Times(0);
    EXPECT_CALL(*client_wrapper, saveKvCaches(_, _, _)).Times(0);
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment, parallelism_config, client_wrapper);
    ASSERT_TRUE(backend->init(
        environment.cache_config.topologyPtr(), {environment.device_pool}, [&](int layer_id, int, int block_id) {
            return environment.device_pool->convertIndexToBuffer(layer_id, block_id);
        }));

    RemoteOperationRequestPB request;
    request.set_op(REMOTE_OPERATION_READ);
    request.add_group_tags("default");
    request.add_block_ids(environment.block_id);
    RemoteOperationResponsePB response;
    EXPECT_FALSE(backend->execute(request, response));
}

TEST(KVCMStorageBackendTest, MatchAndReadUseReturnedLocation) {
    auto environment    = makeBackendEnvironment("kvcm_storage_backend_match_read");
    auto client_wrapper = std::make_shared<MockClientWrapper>();

    EXPECT_CALL(*client_wrapper, init(_, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), client_wrapper);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));

    kv_cache_manager::Locations locations{
        kv_cache_manager::Location{kv_cache_manager::LocationSpecUnit{"tp0_Fdefault", "read_uri"}}};
    EXPECT_CALL(*client_wrapper,
                match(_, _, kv_cache_manager::QueryType::QT_PREFIX_MATCH, std::vector<int64_t>{101}, _, _))
        .WillOnce(Return(std::make_pair(true, locations)));
    auto observation = match(*backend.backend, makeStorageRequest(environment));
    ASSERT_TRUE(observation.success);
    ASSERT_EQ(observation.matched_blocks_num, 1u);
    ASSERT_NE(observation.match_meta, nullptr);

    EXPECT_CALL(*client_wrapper, loadKvCaches(kv_cache_manager::UriStrVec{"read_uri"}, _, _))
        .WillOnce(Invoke([](const kv_cache_manager::UriStrVec&,
                            kv_cache_manager::BlockBuffers& buffers,
                            const std::shared_ptr<kv_cache_manager::TransferTraceInfo>&) {
            EXPECT_EQ(buffers.size(), 1u);
            return true;
        }));
    EXPECT_TRUE(read(*backend.backend, makeStorageRequest(environment), std::move(observation.match_meta)));
}

TEST(KVCMStorageBackendTest, MatchFailurePreservesLocalPrefix) {
    auto environment    = makeBackendEnvironment("kvcm_storage_backend_match_fallback");
    auto client_wrapper = std::make_shared<MockClientWrapper>();

    EXPECT_CALL(*client_wrapper, init(_, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), client_wrapper);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));

    EXPECT_CALL(*client_wrapper,
                match(_, _, kv_cache_manager::QueryType::QT_PREFIX_MATCH, std::vector<int64_t>({101, 102}), _, _))
        .WillOnce(Return(std::make_pair(false, kv_cache_manager::Locations{})));
    auto observation =
        match(*backend.backend, makeStorageRequest(environment, /*keys=*/{101, 102}, /*local_matched_blocks=*/1));
    EXPECT_TRUE(observation.success);
    EXPECT_EQ(observation.matched_blocks_num, 1u);
    EXPECT_EQ(observation.match_meta, nullptr);
}

TEST(KVCMStorageBackendTest, InvalidMatchLocationReportsNoHitAndDoesNotDispatchPayload) {
    auto environment    = makeBackendEnvironment("kvcm_storage_backend_invalid_location");
    auto client_wrapper = std::make_shared<MockClientWrapper>();

    EXPECT_CALL(*client_wrapper, init(_, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), client_wrapper);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));

    kv_cache_manager::Locations locations{
        kv_cache_manager::Location{kv_cache_manager::LocationSpecUnit{"unknown_spec", "read_uri"}}};
    EXPECT_CALL(*client_wrapper, match(_, _, _, _, _, _)).WillOnce(Return(std::make_pair(true, locations)));
    auto observation = match(*backend.backend, makeStorageRequest(environment));
    EXPECT_CALL(*client_wrapper, loadKvCaches(_, _, _)).Times(0);
    EXPECT_FALSE(observation.success);
    EXPECT_EQ(observation.matched_blocks_num, 0u);
    EXPECT_EQ(observation.match_meta, nullptr);
}

TEST(KVCMStorageBackendTest, TP2DuplicateRankMatchReportsNoHitAndDoesNotDispatchPayload) {
    auto environment    = makeBackendEnvironment("kvcm_storage_backend_tp2_duplicate_rank");
    auto client_wrapper = std::make_shared<MockClientWrapper>();

    ParallelismConfig parallelism_config;
    parallelism_config.tp_size    = 2;
    parallelism_config.tp_rank    = 0;
    parallelism_config.local_rank = 0;

    EXPECT_CALL(*client_wrapper, init(_, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    EXPECT_CALL(*client_wrapper, loadKvCaches(_, _, _)).Times(0);
    auto broadcast_manager =
        std::make_shared<BroadcastManager>(std::vector<std::string>{"unused-rank-0", "unused-rank-1"});
    auto backend = makeBackend(environment, parallelism_config, client_wrapper, std::move(broadcast_manager));
    ASSERT_TRUE(backend->init(
        environment.cache_config.topologyPtr(), {environment.device_pool}, [&](int layer_id, int, int block_id) {
            return environment.device_pool->convertIndexToBuffer(layer_id, block_id);
        }));

    kv_cache_manager::Locations locations{kv_cache_manager::Location{
        kv_cache_manager::LocationSpecUnit{"tp0_Fdefault", "rank0_uri"},
        kv_cache_manager::LocationSpecUnit{"tp0_Fdefault", "duplicate_rank0_uri"},
    }};
    EXPECT_CALL(*client_wrapper, match(_, _, _, _, _, _)).WillOnce(Return(std::make_pair(true, locations)));
    const auto observation = match(*backend.backend, makeStorageRequest(environment));
    EXPECT_FALSE(observation.success);
    EXPECT_EQ(observation.matched_blocks_num, 0u);
    EXPECT_EQ(observation.match_meta, nullptr);
}

TEST(KVCMStorageBackendTest, PayloadReadFailurePropagatesToCompletion) {
    auto environment    = makeBackendEnvironment("kvcm_storage_backend_read_failure");
    auto client_wrapper = std::make_shared<MockClientWrapper>();

    EXPECT_CALL(*client_wrapper, init(_, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), client_wrapper);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));

    kv_cache_manager::Locations locations{
        kv_cache_manager::Location{kv_cache_manager::LocationSpecUnit{"tp0_Fdefault", "read_uri"}}};
    EXPECT_CALL(*client_wrapper, match(_, _, _, _, _, _)).WillOnce(Return(std::make_pair(true, locations)));
    auto observation = match(*backend.backend, makeStorageRequest(environment));
    ASSERT_TRUE(observation.success);
    ASSERT_NE(observation.match_meta, nullptr);

    EXPECT_CALL(*client_wrapper, loadKvCaches(kv_cache_manager::UriStrVec{"read_uri"}, _, _)).WillOnce(Return(false));
    EXPECT_FALSE(read(*backend.backend, makeStorageRequest(environment), std::move(observation.match_meta)));
}

TEST(KVCMStorageBackendTest, WritePublishesActualUri) {
    auto environment    = makeBackendEnvironment("kvcm_storage_backend_write");
    auto client_wrapper = std::make_shared<MockClientWrapper>();

    EXPECT_CALL(*client_wrapper, init(_, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), client_wrapper);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));

    kv_cache_manager::WriteLocation write_location;
    write_location.write_session_id = "write_session";
    write_location.block_mask       = kv_cache_manager::BlockMaskOffset{0};
    write_location.locations        = {
        kv_cache_manager::Location{kv_cache_manager::LocationSpecUnit{"tp0_Fdefault", "write_uri"}}};
    EXPECT_CALL(*client_wrapper, getWriteLocation(_, _, std::vector<int64_t>{101}, std::vector<int64_t>{}, _, 600))
        .WillOnce(Return(std::make_pair(true, write_location)));
    EXPECT_CALL(*client_wrapper, saveKvCaches(kv_cache_manager::UriStrVec{"write_uri"}, _, _))
        .WillOnce(Return(std::make_pair(true, kv_cache_manager::UriStrVec{"actual_uri"})));
    EXPECT_CALL(*client_wrapper, finishWrite(_, _, "write_session", _, _))
        .WillOnce(Invoke([](const std::string&,
                            const std::string&,
                            const std::string&,
                            const kv_cache_manager::BlockMask&,
                            const kv_cache_manager::Locations& locations) {
            if (locations.size() != 1u || locations.front().size() != 1u) {
                ADD_FAILURE() << "finishWrite received an invalid location shape";
                return false;
            }
            EXPECT_EQ(locations.front().front().uri, "actual_uri");
            return true;
        }));

    EXPECT_TRUE(backend->write(backend->prepareWrite(makeStorageRequest(environment)), /*synchronous=*/true));
}

TEST(KVCMStorageBackendTest, StartWriteFailureDoesNotFinishSession) {
    auto environment    = makeBackendEnvironment("kvcm_storage_backend_start_write_failure");
    auto client_wrapper = std::make_shared<MockClientWrapper>();

    EXPECT_CALL(*client_wrapper, init(_, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), client_wrapper);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));

    EXPECT_CALL(*client_wrapper, getWriteLocation(_, _, _, _, _, _))
        .WillOnce(Return(std::make_pair(false, kv_cache_manager::WriteLocation{})));
    EXPECT_CALL(*client_wrapper, saveKvCaches(_, _, _)).Times(0);
    EXPECT_CALL(*client_wrapper, finishWrite(_, _, _, _, _)).Times(0);
    EXPECT_FALSE(backend->write(backend->prepareWrite(makeStorageRequest(environment)), /*synchronous=*/true));
}

TEST(KVCMStorageBackendTest, TransferFailureAbortsWriteSession) {
    auto environment    = makeBackendEnvironment("kvcm_storage_backend_abort_write");
    auto client_wrapper = std::make_shared<MockClientWrapper>();

    EXPECT_CALL(*client_wrapper, init(_, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), client_wrapper);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));

    kv_cache_manager::WriteLocation write_location;
    write_location.write_session_id = "abort_session";
    write_location.block_mask       = kv_cache_manager::BlockMaskOffset{0};
    write_location.locations        = {
        kv_cache_manager::Location{kv_cache_manager::LocationSpecUnit{"tp0_Fdefault", "write_uri"}}};
    EXPECT_CALL(*client_wrapper, getWriteLocation(_, _, _, _, _, _))
        .WillOnce(Return(std::make_pair(true, write_location)));
    EXPECT_CALL(*client_wrapper, saveKvCaches(kv_cache_manager::UriStrVec{"write_uri"}, _, _))
        .WillOnce(Return(std::make_pair(false, kv_cache_manager::UriStrVec{})));
    EXPECT_CALL(*client_wrapper, finishWrite(_, _, "abort_session", _, _))
        .WillOnce(Invoke([](const std::string&,
                            const std::string&,
                            const std::string&,
                            const kv_cache_manager::BlockMask& block_mask,
                            const kv_cache_manager::Locations& locations) {
            const auto* offset = std::get_if<kv_cache_manager::BlockMaskOffset>(&block_mask);
            EXPECT_NE(offset, nullptr);
            if (offset != nullptr) {
                EXPECT_EQ(*offset, 0u);
            }
            EXPECT_TRUE(locations.empty());
            return true;
        }));
    EXPECT_FALSE(backend->write(backend->prepareWrite(makeStorageRequest(environment)), /*synchronous=*/true));
}

TEST(KVCMStorageBackendTest, FinishWriteFailureIsNotRetried) {
    auto environment    = makeBackendEnvironment("kvcm_storage_backend_finish_write_failure");
    auto client_wrapper = std::make_shared<MockClientWrapper>();

    EXPECT_CALL(*client_wrapper, init(_, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), client_wrapper);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));

    kv_cache_manager::WriteLocation write_location;
    write_location.write_session_id = "finish_session";
    write_location.block_mask       = kv_cache_manager::BlockMaskOffset{0};
    write_location.locations        = {
        kv_cache_manager::Location{kv_cache_manager::LocationSpecUnit{"tp0_Fdefault", "write_uri"}}};
    EXPECT_CALL(*client_wrapper, getWriteLocation(_, _, _, _, _, _))
        .WillOnce(Return(std::make_pair(true, write_location)));
    EXPECT_CALL(*client_wrapper, saveKvCaches(kv_cache_manager::UriStrVec{"write_uri"}, _, _))
        .WillOnce(Return(std::make_pair(true, kv_cache_manager::UriStrVec{})));
    EXPECT_CALL(*client_wrapper, finishWrite(_, _, "finish_session", _, _))
        .WillOnce(Invoke([](const std::string&,
                            const std::string&,
                            const std::string&,
                            const kv_cache_manager::BlockMask&,
                            const kv_cache_manager::Locations& locations) {
            EXPECT_TRUE(locations.empty());
            return false;
        }));
    EXPECT_FALSE(backend->write(backend->prepareWrite(makeStorageRequest(environment)), /*synchronous=*/true));
}

}  // namespace
}  // namespace rtp_llm
