#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test/KVCMMockTestBase.h"

#include <optional>
#include <stdexcept>
#include <tuple>
#include <cuda_runtime.h>

namespace rtp_llm {
namespace {

bool acknowledgeAbort(const std::string&,
                      const std::string& trace_id,
                      const std::string&,
                      const kv_cache_manager::BlockMask& block_mask,
                      const kv_cache_manager::Locations& locations) {
    EXPECT_EQ(trace_id.find("block_tree_abort_write_"), 0u);
    const auto* offset = std::get_if<kv_cache_manager::BlockMaskOffset>(&block_mask);
    EXPECT_NE(offset, nullptr);
    if (offset != nullptr) {
        EXPECT_EQ(*offset, 0u);
    }
    EXPECT_TRUE(locations.empty());
    return true;
}

class KVCMLocalWriteTest: public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(initialize(singleRankConfig()));
    }

    bool initialize(const ParallelismConfig& parallel, std::shared_ptr<BroadcastManager> broadcast = nullptr) {
        EXPECT_CALL(*client_wrapper, initForPools(_, _, _, _)).WillOnce(Return(true));
        EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
        backend = makeBackend(environment, parallel, client_wrapper, std::move(broadcast), 1000);
        write_location.write_session_id = "session";
        write_location.block_mask       = kv_cache_manager::BlockMaskOffset{0};
        write_location.locations        = {{{"tp0_Fdefault", "write_uri"}}};
        return initSingleRank(*backend.backend, environment);
    }

    void expectWriteLocation() {
        EXPECT_CALL(*client_wrapper, getWriteLocation(_, _, _, _, _, _))
            .WillOnce(Return(std::make_pair(true, write_location)));
    }

    bool write(StorageRequest request) {
        return backend->write(backend->prepareWrite(std::move(request)))
               && waitForBackendOperationsForTest(*backend.backend);
    }
    bool write() {
        return write(makeStorageRequest(environment));
    }
    void expectBlockState(bool frozen) {
        EXPECT_EQ(environment.device_pool->isFrozen(environment.block_id), frozen);
        EXPECT_EQ(environment.device_pool->refCount(environment.block_id), frozen ? 2u : 1u);
        EXPECT_EQ(environment.device_pool->referencedBlocksNum(BlockTreeRefType::FROZEN), frozen ? 1u : 0u);
        EXPECT_EQ(environment.device_pool->referencedBlocksNum(BlockTreeRefType::STORE), 0u);
    }

    BackendEnvironment                 environment    = makeBackendEnvironment("kvcm_full_test");
    std::shared_ptr<MockClientWrapper> client_wrapper = std::make_shared<MockClientWrapper>();
    BackendHandle                      backend;
    kv_cache_manager::WriteLocation    write_location;
};

class KVCMTP2Test: public KVCMLocalWriteTest {
protected:
    void SetUp() override {
        std::vector<std::string> addresses;
        for (size_t rank = 0; rank < 2; ++rank) {
            auto state  = std::make_shared<KVCMBroadcastState>();
            auto server = std::make_unique<KVCMBroadcastRpcServer>(rank, state);
            ASSERT_TRUE(server->start());
            addresses.push_back(server->address());
            states.push_back(std::move(state));
            servers.push_back(std::move(server));
        }
        auto broadcast = std::make_shared<BroadcastManager>(addresses);
        ASSERT_TRUE(broadcast->init());
        auto parallel    = singleRankConfig();
        parallel.tp_size = 2;
        ASSERT_TRUE(initialize(parallel, broadcast));
        write_location.locations = {{{"tp0_Fdefault", "rank0"}, {"tp1_Fdefault", "rank1"}}};
    }
    void TearDown() override {
        // Drain handlers while both backends and their registered memory still exist.
        servers.clear();
    }
    void initializeWorker() {
        worker_environment = std::make_unique<BackendEnvironment>(makeBackendEnvironment("kvcm_full_worker"));
        ASSERT_EQ(environment.block_id, worker_environment->block_id);
        worker_client = std::make_shared<MockClientWrapper>();
        EXPECT_CALL(*worker_client, initForPools(_, _, _, _)).WillOnce(Return(true));
        EXPECT_CALL(*worker_client, shutdown()).Times(1);
        auto parallel    = singleRankConfig();
        parallel.tp_size = 2;
        parallel.tp_rank = 1;
        worker           = makeBackend(*worker_environment, parallel, worker_client);
        ASSERT_TRUE(initSingleRank(*worker.backend, *worker_environment));
    }
    void failTransfer(bool writing, StorageRequest request) {
        if (writing) {
            expectWriteLocation();
            ASSERT_TRUE(write(std::move(request)));
        } else {
            EXPECT_CALL(*client_wrapper, match(_, _, _, _, _, _))
                .WillOnce(Return(std::make_pair(true, write_location.locations)));
            auto matched = match(*backend.backend, request);
            ASSERT_TRUE(matched.success);
            EXPECT_FALSE(read(*backend.backend, request, matched.match_meta));
        }
    }

    std::vector<std::shared_ptr<KVCMBroadcastState>>     states;
    std::vector<std::unique_ptr<KVCMBroadcastRpcServer>> servers;
    std::unique_ptr<BackendEnvironment>                  worker_environment;
    std::shared_ptr<MockClientWrapper>                   worker_client;
    BackendHandle                                        worker;
};

TEST(KVCMMockOnlyFullTest, TP2WorkerRegistersItsRankAndExecutesLocalPayload) {
    auto environment    = makeBackendEnvironment("kvcm_storage_backend_tp2_worker");
    auto client_wrapper = std::make_shared<MockClientWrapper>();

    ParallelismConfig parallelism_config;
    parallelism_config.tp_size    = 2;
    parallelism_config.tp_rank    = 1;
    parallelism_config.local_rank = 0;

    EXPECT_CALL(*client_wrapper, initForPools(_, _, _, _))
        .WillOnce(Invoke([&](const kvcm::ClientWrapper::ConfigMap&                     config_map,
                             kv_cache_manager::RoleType                                role,
                             const std::vector<kvcm::ClientWrapper::PoolRegistration>& registrations,
                             const std::vector<std::string>&                           tags) {
            EXPECT_EQ(role, kv_cache_manager::RoleType::WORKER);
            EXPECT_EQ(tags, (std::vector<std::string>{"default"}));
            EXPECT_EQ(registrations.size(), 1u);
            EXPECT_EQ(registrations.front().location_spec_name, "tp1_Fdefault");
            EXPECT_EQ(registrations.front().span.base, environment.device_pool->getBaseAddress());
            EXPECT_EQ(registrations.front().span.size, environment.device_pool->getTotalSizeBytes());
            EXPECT_EQ(config_map.size(), 1u);
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
                              environment.pools_by_tag,
                              [&](int layer_id, const std::string& tag, int block_id) {
                                  EXPECT_EQ(tag, "default");
                                  return environment.device_pool->convertIndexToBuffer(layer_id, block_id);
                              }));

    const kv_cache_manager::UriStrVec expected_read_uris{"read_uri"};
    EXPECT_CALL(*client_wrapper, loadKvCachesForTag("default", expected_read_uris, _, _))
        .WillOnce(Invoke([](const std::string&,
                            const kv_cache_manager::UriStrVec&,
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
    EXPECT_CALL(*client_wrapper, saveKvCachesForTag("default", expected_write_uris, _, _))
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

TEST(KVCMMockOnlyFullTest, TP2CoordinatorRejectsMissingBroadcastManager) {
    auto environment    = makeBackendEnvironment("kvcm_storage_backend_tp2_coordinator");
    auto client_wrapper = std::make_shared<MockClientWrapper>();

    ParallelismConfig parallelism_config;
    parallelism_config.tp_size    = 2;
    parallelism_config.tp_rank    = 0;
    parallelism_config.local_rank = 0;

    EXPECT_CALL(*client_wrapper, initForPools(_, _, _, _)).Times(0);
    auto backend = makeBackend(environment, parallelism_config, client_wrapper);
    EXPECT_FALSE(backend->init(environment.cache_config.topologyPtr(),
                               environment.pools_by_tag,
                               [&](int layer_id, const std::string&, int block_id) {
                                   return environment.device_pool->convertIndexToBuffer(layer_id, block_id);
                               }));
}

TEST(KVCMMockOnlyFullTest, TP2CoordinatorBroadcastsRankOrderedReadAndWritePayloads) {
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
    EXPECT_CALL(*client_wrapper, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment, parallelism_config, client_wrapper, broadcast_manager);
    ASSERT_TRUE(backend->init(environment.cache_config.topologyPtr(),
                              environment.pools_by_tag,
                              [&](int layer_id, const std::string&, int block_id) {
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
    backend->write(backend->prepareWrite(makeStorageRequest(environment)));
    ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));

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

TEST(KVCMMockOnlyFullTest, RejectsMismatchedTransferVectorsBeforeClientIO) {
    auto environment    = makeBackendEnvironment("kvcm_storage_backend_bad_shape");
    auto client_wrapper = std::make_shared<MockClientWrapper>();

    ParallelismConfig parallelism_config;
    parallelism_config.tp_size    = 1;
    parallelism_config.tp_rank    = 0;
    parallelism_config.local_rank = 0;

    EXPECT_CALL(*client_wrapper, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, loadKvCachesForTag("default", _, _, _)).Times(0);
    EXPECT_CALL(*client_wrapper, saveKvCachesForTag("default", _, _, _)).Times(0);
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment, parallelism_config, client_wrapper);
    ASSERT_TRUE(backend->init(environment.cache_config.topologyPtr(),
                              environment.pools_by_tag,
                              [&](int layer_id, const std::string&, int block_id) {
                                  return environment.device_pool->convertIndexToBuffer(layer_id, block_id);
                              }));

    RemoteOperationRequestPB request;
    request.set_op(REMOTE_OPERATION_READ);
    request.add_group_tags("default");
    request.add_block_ids(environment.block_id);
    RemoteOperationResponsePB response;
    EXPECT_FALSE(backend->execute(request, response));
}

class KVCMSdkCheckTest: public ::testing::TestWithParam<std::tuple<const char*, const char*, bool>> {};

TEST_P(KVCMSdkCheckTest, TracesReadAndWriteBlockIdsWithLegacyFallback) {
    const auto [canonical, legacy, enabled] = GetParam();
    autil::EnvGuard sdk_check("KVCM_SDK_CHECK", canonical ? canonical : "0");
    autil::EnvGuard legacy_sdk_check("RECO_SDK_CHECK", legacy ? legacy : "0");
    if (!canonical) {
        autil::EnvUtil::unsetEnv("KVCM_SDK_CHECK");
    }
    if (!legacy) {
        autil::EnvUtil::unsetEnv("RECO_SDK_CHECK");
    }
    auto environment    = makeBackendEnvironment("kvcm_storage_backend_sdk_check");
    auto client_wrapper = std::make_shared<MockClientWrapper>();

    EXPECT_CALL(*client_wrapper, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), client_wrapper);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));

    const std::vector<std::string> expected_block_ids{std::to_string(environment.block_id)};
    EXPECT_CALL(*client_wrapper, loadKvCachesForTag("default", kv_cache_manager::UriStrVec{"read_uri"}, _, _))
        .WillOnce(Invoke([&](const std::string&,
                             const kv_cache_manager::UriStrVec&,
                             kv_cache_manager::BlockBuffers&,
                             const std::shared_ptr<kv_cache_manager::TransferTraceInfo>& trace_info) {
            if (trace_info == nullptr) {
                EXPECT_FALSE(enabled);
                return !enabled;
            }
            EXPECT_TRUE(enabled);
            EXPECT_TRUE(trace_info->need_print);
            EXPECT_EQ(trace_info->block_ids, expected_block_ids);
            return true;
        }));
    RemoteOperationRequestPB read_request;
    read_request.set_op(REMOTE_OPERATION_READ);
    read_request.add_group_tags("default");
    read_request.add_block_ids(environment.block_id);
    read_request.add_uris("read_uri");
    RemoteOperationResponsePB read_response;
    EXPECT_TRUE(backend->execute(read_request, read_response));

    EXPECT_CALL(*client_wrapper, saveKvCachesForTag("default", kv_cache_manager::UriStrVec{"write_uri"}, _, _))
        .WillOnce(Invoke([&](const std::string&,
                             const kv_cache_manager::UriStrVec&,
                             const kv_cache_manager::BlockBuffers&,
                             const std::shared_ptr<kv_cache_manager::TransferTraceInfo>& trace_info) {
            if (trace_info == nullptr) {
                EXPECT_FALSE(enabled);
                return (!enabled ? completedSave(kv_cache_manager::UriStrVec{}) :
                                   std::make_pair(false, kv_cache_manager::UriStrVec{}));
            }
            EXPECT_TRUE(enabled);
            EXPECT_TRUE(trace_info->need_print);
            EXPECT_EQ(trace_info->block_ids, expected_block_ids);
            return completedSave(kv_cache_manager::UriStrVec{});
        }));
    RemoteOperationRequestPB write_request;
    write_request.set_op(REMOTE_OPERATION_WRITE);
    write_request.add_group_tags("default");
    write_request.add_block_ids(environment.block_id);
    write_request.add_uris("write_uri");
    RemoteOperationResponsePB write_response;
    EXPECT_TRUE(backend->execute(write_request, write_response));
}

INSTANTIATE_TEST_SUITE_P(Compatibility,
                         KVCMSdkCheckTest,
                         ::testing::Values(std::make_tuple("1", "0", true),
                                           std::make_tuple(nullptr, "1", true),
                                           std::make_tuple("0", "1", false),
                                           std::make_tuple(nullptr, nullptr, false)));

TEST(KVCMMockOnlyFullTest, WritePublishesActualUri) {
    auto environment    = makeBackendEnvironment("kvcm_storage_backend_write");
    auto client_wrapper = std::make_shared<MockClientWrapper>();

    EXPECT_CALL(*client_wrapper, initForPools(_, _, _, _)).WillOnce(Return(true));
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
    EXPECT_CALL(*client_wrapper, saveKvCachesForTag("default", kv_cache_manager::UriStrVec{"write_uri"}, _, _))
        .WillOnce(Return(completedSave(kv_cache_manager::UriStrVec{"actual_uri"})));
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

    backend->write(backend->prepareWrite(makeStorageRequest(environment)));
    ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
}

TEST(KVCMMockOnlyFullTest, WriteHonorsOffsetBlockMask) {
    auto environment    = makeBackendEnvironment("kvcm_storage_backend_offset_mask");
    auto client_wrapper = std::make_shared<MockClientWrapper>();

    EXPECT_CALL(*client_wrapper, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), client_wrapper);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));
    ScopedReferencedBlocks source_blocks(environment.device_pool, 3);
    const auto&            block_ids         = source_blocks.get();
    const auto             second_block_info = environment.device_pool->convertIndexToBuffer(0, block_ids[1]);
    const auto             third_block_info  = environment.device_pool->convertIndexToBuffer(0, block_ids[2]);
    ASSERT_EQ(second_block_info.size(), 1u);
    ASSERT_EQ(third_block_info.size(), 1u);

    kv_cache_manager::WriteLocation write_location;
    write_location.write_session_id = "offset_session";
    write_location.block_mask       = kv_cache_manager::BlockMaskOffset{1};
    write_location.locations        = {
        kv_cache_manager::Location{kv_cache_manager::LocationSpecUnit{"tp0_Fdefault", "write_uri_102"}},
        kv_cache_manager::Location{kv_cache_manager::LocationSpecUnit{"tp0_Fdefault", "write_uri_103"}},
    };
    EXPECT_CALL(*client_wrapper,
                getWriteLocation(_, _, std::vector<int64_t>({101, 102, 103}), std::vector<int64_t>{}, _, 600))
        .WillOnce(Return(std::make_pair(true, write_location)));
    EXPECT_CALL(*client_wrapper,
                saveKvCachesForTag("default", kv_cache_manager::UriStrVec({"write_uri_102", "write_uri_103"}), _, _))
        .WillOnce(Invoke([second_base = second_block_info.front().addr, third_base = third_block_info.front().addr](
                             const std::string&,
                             const kv_cache_manager::UriStrVec&,
                             const kv_cache_manager::BlockBuffers& buffers,
                             const std::shared_ptr<kv_cache_manager::TransferTraceInfo>&) {
            if (buffers.size() != 2u || buffers[0].iovs.size() != 1u || buffers[1].iovs.size() != 1u) {
                ADD_FAILURE() << "KVCM offset write received an invalid block-buffer shape";
                return std::make_pair(false, kv_cache_manager::UriStrVec{});
            }
            EXPECT_EQ(buffers[0].iovs[0].base, second_base);
            EXPECT_EQ(buffers[1].iovs[0].base, third_base);
            return completedSave(kv_cache_manager::UriStrVec({"actual_uri_102", "actual_uri_103"}));
        }));
    EXPECT_CALL(*client_wrapper, finishWrite(_, _, "offset_session", _, _))
        .WillOnce(Invoke([](const std::string&,
                            const std::string&,
                            const std::string&,
                            const kv_cache_manager::BlockMask& block_mask,
                            const kv_cache_manager::Locations& locations) {
            const auto* offset = std::get_if<kv_cache_manager::BlockMaskOffset>(&block_mask);
            EXPECT_NE(offset, nullptr);
            if (offset != nullptr) {
                EXPECT_EQ(*offset, 2u);
            }
            EXPECT_EQ(locations.size(), 2u);
            return true;
        }));

    auto request =
        makeStorageRequest(environment, /*keys=*/{101, 102, 103}, /*local_matched_blocks=*/0, /*block_ids=*/block_ids);
    backend->write(backend->prepareWrite(std::move(request)));
    ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
}

TEST(KVCMMockOnlyFullTest, WriteHonorsSparseBlockMask) {
    auto environment    = makeBackendEnvironment("kvcm_storage_backend_sparse_mask");
    auto client_wrapper = std::make_shared<MockClientWrapper>();

    EXPECT_CALL(*client_wrapper, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), client_wrapper);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));
    ScopedReferencedBlocks source_blocks(environment.device_pool, 4);
    const auto&            block_ids         = source_blocks.get();
    const auto             second_block_info = environment.device_pool->convertIndexToBuffer(0, block_ids[1]);
    const auto             fourth_block_info = environment.device_pool->convertIndexToBuffer(0, block_ids[3]);
    ASSERT_EQ(second_block_info.size(), 1u);
    ASSERT_EQ(fourth_block_info.size(), 1u);

    kv_cache_manager::WriteLocation write_location;
    write_location.write_session_id = "sparse_session";
    write_location.block_mask       = std::vector<bool>{true, false, true, false};
    write_location.locations        = {
        kv_cache_manager::Location{kv_cache_manager::LocationSpecUnit{"tp0_Fdefault", "write_uri_102"}},
        kv_cache_manager::Location{kv_cache_manager::LocationSpecUnit{"tp0_Fdefault", "write_uri_104"}},
    };
    EXPECT_CALL(*client_wrapper, getWriteLocation(_, _, _, _, _, _))
        .WillOnce(Return(std::make_pair(true, write_location)));
    EXPECT_CALL(*client_wrapper,
                saveKvCachesForTag("default", kv_cache_manager::UriStrVec({"write_uri_102", "write_uri_104"}), _, _))
        .WillOnce(Invoke([second_base = second_block_info.front().addr, fourth_base = fourth_block_info.front().addr](
                             const std::string&,
                             const kv_cache_manager::UriStrVec&,
                             const kv_cache_manager::BlockBuffers& buffers,
                             const std::shared_ptr<kv_cache_manager::TransferTraceInfo>&) {
            if (buffers.size() != 2u || buffers[0].iovs.size() != 1u || buffers[1].iovs.size() != 1u) {
                ADD_FAILURE() << "KVCM sparse write received an invalid block-buffer shape";
                return std::make_pair(false, kv_cache_manager::UriStrVec{});
            }
            EXPECT_EQ(buffers[0].iovs[0].base, second_base);
            EXPECT_EQ(buffers[1].iovs[0].base, fourth_base);
            return completedSave(kv_cache_manager::UriStrVec{});
        }));
    EXPECT_CALL(*client_wrapper, finishWrite(_, _, "sparse_session", _, _))
        .WillOnce(Invoke([](const std::string&,
                            const std::string&,
                            const std::string&,
                            const kv_cache_manager::BlockMask& block_mask,
                            const kv_cache_manager::Locations& locations) {
            const auto* offset = std::get_if<kv_cache_manager::BlockMaskOffset>(&block_mask);
            EXPECT_NE(offset, nullptr);
            if (offset != nullptr) {
                EXPECT_EQ(*offset, 2u);
            }
            EXPECT_TRUE(locations.empty());
            return true;
        }));

    auto request = makeStorageRequest(
        environment, /*keys=*/{101, 102, 103, 104}, /*local_matched_blocks=*/0, /*block_ids=*/block_ids);
    backend->write(backend->prepareWrite(std::move(request)));
    ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
}

TEST_F(KVCMLocalWriteTest, EmptyWriteLocationsCompleteWithoutPayloadOrFinish) {
    write_location.locations.clear();
    write_location.block_mask = kv_cache_manager::BlockMaskOffset{3};
    expectWriteLocation();
    EXPECT_CALL(*client_wrapper, saveKvCachesForTag("default", _, _, _)).Times(0);
    EXPECT_CALL(*client_wrapper, finishWrite(_, _, _, _, _)).Times(0);
    ASSERT_TRUE(write(makeStorageRequest(environment, {101, 102, 103})));
}

TEST_F(KVCMLocalWriteTest, UnchangedActualUrisAreNotRepublished) {
    expectWriteLocation();
    EXPECT_CALL(*client_wrapper, saveKvCachesForTag("default", kv_cache_manager::UriStrVec{"write_uri"}, _, _))
        .WillOnce(Return(completedSave(kv_cache_manager::UriStrVec{"write_uri"})));
    EXPECT_CALL(*client_wrapper, finishWrite(_, _, write_location.write_session_id, _, ::testing::IsEmpty()))
        .WillOnce(Return(true));
    ASSERT_TRUE(write());
}

TEST_F(KVCMLocalWriteTest, PayloadFailureAbortsSessionWithoutFreezing) {
    for (const auto& result : {std::make_pair(false, kv_cache_manager::UriStrVec{}),
                               completedSave(kv_cache_manager::UriStrVec{"actual_uri", "unexpected_extra_uri"})}) {
        SCOPED_TRACE(result.first ? "invalid actual URIs" : "SDK failure");
        expectWriteLocation();
        EXPECT_CALL(*client_wrapper, saveKvCachesForTag("default", kv_cache_manager::UriStrVec{"write_uri"}, _, _))
            .WillOnce(Return(result));
        EXPECT_CALL(*client_wrapper, finishWrite(_, _, write_location.write_session_id, _, _))
            .WillOnce(Invoke(acknowledgeAbort));
        ASSERT_TRUE(write());
        expectBlockState(false);
    }
}

TEST_F(KVCMLocalWriteTest, StartWriteFailureDoesNotFinishSession) {
    EXPECT_CALL(*client_wrapper, getWriteLocation(_, _, _, _, _, _))
        .WillOnce(Return(std::make_pair(false, kv_cache_manager::WriteLocation{})));
    EXPECT_CALL(*client_wrapper, saveKvCachesForTag("default", _, _, _)).Times(0);
    EXPECT_CALL(*client_wrapper, finishWrite(_, _, _, _, _)).Times(0);
    ASSERT_TRUE(write());
}

TEST_F(KVCMLocalWriteTest, FinishWriteFailureIsNotRetried) {
    for (bool response_lost : {false, true}) {
        SCOPED_TRACE(response_lost);
        expectWriteLocation();
        EXPECT_CALL(*client_wrapper, saveKvCachesForTag("default", _, _, _))
            .WillOnce(Return(completedSave(kv_cache_manager::UriStrVec{})));
        EXPECT_CALL(*client_wrapper, finishWrite(_, _, write_location.write_session_id, _, ::testing::IsEmpty()))
            .WillOnce(Invoke([response_lost](const auto&...) {
                if (response_lost) {
                    throw std::runtime_error("FinishWrite response lost after publication");
                }
                return false;
            }));
        ASSERT_TRUE(write());
        expectBlockState(false);
    }
}

TEST_F(KVCMLocalWriteTest, AbortFailurePreservesPreparationExceptionOrPayloadError) {
    for (bool preparation_fails : {false, true}) {
        for (bool abort_throws : {false, true}) {
            SCOPED_TRACE(::testing::Message()
                         << "preparation_fails=" << preparation_fails << " abort_throws=" << abort_throws);
            write_location.locations.front().front().spec_name = preparation_fails ? "unknown_spec" : "tp0_Fdefault";
            expectWriteLocation();
            if (preparation_fails) {
                EXPECT_CALL(*client_wrapper, saveKvCachesForTag("default", _, _, _)).Times(0);
            } else {
                EXPECT_CALL(*client_wrapper, saveKvCachesForTag("default", _, _, _))
                    .WillOnce(Return(std::make_pair(false, kv_cache_manager::UriStrVec{})));
            }
            EXPECT_CALL(*client_wrapper, finishWrite(_, _, write_location.write_session_id, _, _))
                .WillOnce(Invoke([abort_throws](const auto&... args) {
                    acknowledgeAbort(args...);
                    if (abort_throws) {
                        throw std::logic_error("injected abort failure");
                    }
                    return false;
                }));
            if (preparation_fails) {
                try {
                    (void)backend->writeImpl(makeStorageRequest(environment));
                    FAIL() << "request preparation should throw";
                } catch (const std::runtime_error& error) {
                    EXPECT_THAT(error.what(), ::testing::HasSubstr("unknown spec"));
                }
            } else {
                const auto error = backend->writeImpl(makeStorageRequest(environment));
                EXPECT_EQ(error.code(), ErrorCode::EXECUTION_EXCEPTION);
                EXPECT_EQ(error.ToString(), "KVCM local transfer failed");
            }
        }
    }
}

TEST_F(KVCMLocalWriteTest, PayloadExceptionPropagatesWithoutAbortingSession) {
    expectWriteLocation();
    EXPECT_CALL(*client_wrapper, saveKvCachesForTag("default", _, _, _))
        .WillOnce(::testing::Throw(std::logic_error("injected payload exception")));
    EXPECT_CALL(*client_wrapper, finishWrite(_, _, _, _, _)).Times(0);
    EXPECT_THROW(backend->writeImpl(makeStorageRequest(environment)), std::logic_error);
}

struct WorkerReplyCase {
    const char*         name;
    grpc::StatusCode    rpc_code;
    bool                has_response;
    std::optional<bool> success;
    int                 uri_count;
    bool                frozen;
};
using TP2ResultParam = std::tuple<bool, WorkerReplyCase>;

std::vector<TP2ResultParam> tp2ResultCases() {
    const WorkerReplyCase replies[] = {
        {"completed", grpc::StatusCode::OK, true, true, 0, false},
        {"rpc failure", grpc::StatusCode::INTERNAL, true, true, 0, true},
        {"missing response", grpc::StatusCode::OK, false, std::nullopt, 0, true},
        {"missing success", grpc::StatusCode::OK, true, std::nullopt, 0, true},
        {"short URIs", grpc::StatusCode::OK, true, true, 1, true},
        {"extra URIs", grpc::StatusCode::OK, true, true, 3, true},
    };
    std::vector<TP2ResultParam> cases;
    for (bool writing : {false, true}) {
        for (const auto& reply : replies) {
            if (writing || reply.uri_count < 3) {
                cases.emplace_back(writing, reply);
            }
        }
    }
    return cases;
}
class KVCMTP2ResultTest: public KVCMTP2Test, public ::testing::WithParamInterface<TP2ResultParam> {};

TEST_P(KVCMTP2ResultTest, ClassifiesCompletionAndSettlesWriteSession) {
    const auto [writing, reply] = GetParam();
    SCOPED_TRACE(::testing::Message() << "writing=" << writing << " reply=" << reply.name);
    write_location.locations.push_back(write_location.locations.front());
    // Rank 0 always reports a completed business failure; UNKNOWN from rank 1 must take precedence.
    states[0]->handler = [](const auto&, auto& response) {
        response.mutable_remote_response()->mutable_success()->set_value(false);
        return grpc::Status::OK;
    };
    states[1]->handler = [reply](const auto&, auto& response) {
        if (reply.has_response) {
            auto* remote = response.mutable_remote_response();
            if (reply.success.has_value()) {
                remote->mutable_success()->set_value(*reply.success);
            }
            for (int index = 0; index < reply.uri_count; ++index) {
                remote->add_actual_uris("actual");
            }
        }
        return reply.rpc_code == grpc::StatusCode::OK ? grpc::Status::OK : grpc::Status(reply.rpc_code, reply.name);
    };
    if (writing && !reply.frozen) {
        EXPECT_CALL(*client_wrapper, finishWrite(_, _, write_location.write_session_id, _, _))
            .WillOnce(Invoke(acknowledgeAbort));
    } else {
        EXPECT_CALL(*client_wrapper, finishWrite(_, _, write_location.write_session_id, _, _)).Times(0);
    }
    ASSERT_NO_FATAL_FAILURE(failTransfer(writing, makeStorageRequest(environment, {101, 102})));
    expectBlockState(reply.frozen);
    if (writing && reply.rpc_code != grpc::StatusCode::OK) {
        for (const auto& state : states) {
            state->handler = nullptr;
        }
        ScopedReferencedBlocks next_source(environment.device_pool, 1);
        ASSERT_NE(next_source.get().front(), environment.block_id);
        write_location.write_session_id = "next_session";
        write_location.locations.resize(1);
        expectWriteLocation();
        EXPECT_CALL(*client_wrapper, finishWrite(_, _, "next_session", _, _)).WillOnce(Return(true));
        ASSERT_TRUE(write(makeStorageRequest(environment, {103}, 0, next_source.get())));
        expectBlockState(true);
        EXPECT_FALSE(environment.device_pool->isFrozen(next_source.get().front()));
        EXPECT_EQ(environment.device_pool->refCount(next_source.get().front()), 1u);
    }
}
INSTANTIATE_TEST_SUITE_P(WorkerResponses, KVCMTP2ResultTest, ::testing::ValuesIn(tp2ResultCases()));

TEST(KVCMMockOnlyFullTest, BroadcastRejectionAbortsSessionAndReleasesPins) {
    auto environment = makeBackendEnvironment("broadcast_rejected");
    auto client      = std::make_shared<MockClientWrapper>();
    // All nullptr returns in BroadcastManager occur before AsyncExecuteFunction.
    auto broadcast = std::make_shared<BroadcastManager>(std::vector<std::string>{"unused0", "unused1"});
    EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client, shutdown()).Times(1);
    auto parallel    = singleRankConfig();
    parallel.tp_size = 2;
    auto backend     = makeBackend(environment, parallel, client, broadcast);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));
    kv_cache_manager::WriteLocation proposal;
    proposal.write_session_id = "unsubmitted";
    proposal.block_mask       = kv_cache_manager::BlockMaskOffset{0};
    proposal.locations        = {{{"tp0_Fdefault", "rank0"}, {"tp1_Fdefault", "rank1"}}};
    EXPECT_CALL(*client, getWriteLocation(_, _, _, _, _, _)).WillOnce(Return(std::make_pair(true, proposal)));
    EXPECT_CALL(*client, finishWrite(_, _, "unsubmitted", _, _)).WillOnce(Invoke(acknowledgeAbort));
    ASSERT_TRUE(backend->write(backend->prepareWrite(makeStorageRequest(environment))));
    ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
    EXPECT_FALSE(environment.device_pool->isFrozen(environment.block_id));
    EXPECT_EQ(environment.device_pool->refCount(environment.block_id), 1u);
    EXPECT_CALL(*client, match(_, _, _, _, _, _)).WillOnce(Return(std::make_pair(true, proposal.locations)));
    auto matched = match(*backend.backend, makeStorageRequest(environment));
    ASSERT_TRUE(matched.success);
    EXPECT_FALSE(read(*backend.backend, makeStorageRequest(environment), matched.match_meta));
    EXPECT_FALSE(environment.device_pool->isFrozen(environment.block_id));
    EXPECT_EQ(environment.device_pool->refCount(environment.block_id), 1u);
}

class KVCMTP2DeadlineTest: public KVCMTP2Test, public ::testing::WithParamInterface<bool> {};

TEST_P(KVCMTP2DeadlineTest, LateWorkerCannotAccessReallocatedBlocks) {
    const bool writing = GetParam();
    ASSERT_NO_FATAL_FAILURE(initializeWorker());
    std::promise<void>   entered;
    auto                 entered_future  = entered.get_future();
    auto                 release         = std::make_shared<std::promise<void>>();
    auto                 released        = release->get_future().share();
    auto                 finished        = std::make_shared<std::promise<void>>();
    auto                 finished_future = finished->get_future().share();
    std::vector<uint8_t> observed_source;
    const auto           access_old = [&, writing](const kv_cache_manager::BlockBuffers& buffers) {
        EXPECT_EQ(cudaSetDevice(0), cudaSuccess);
        entered.set_value();
        released.wait();
        for (const auto& buffer : buffers) {
            for (const auto& iov : buffer.iovs) {
                if (writing) {
                    const auto offset = observed_source.size();
                    observed_source.resize(offset + iov.size);
                    EXPECT_EQ(cudaMemcpy(observed_source.data() + offset, iov.base, iov.size, cudaMemcpyDeviceToHost),
                              cudaSuccess);
                } else {
                    EXPECT_EQ(cudaMemset(iov.base, 0x11, iov.size), cudaSuccess);
                }
            }
        }
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    };
    struct DrainWorker {
        std::shared_ptr<std::promise<void>> release;
        std::shared_future<void>            finished;
        bool                                opened = false;
        std::future_status                  finish() {
            if (!opened) {
                opened = true;
                release->set_value();
            }
            return finished.wait_for(std::chrono::seconds(10));
        }
        ~DrainWorker() {
            (void)finish();
        }
    } drain{release, finished_future};
    states[0]->handler = [this](const auto& request, auto& response) {
        auto*      remote  = response.mutable_remote_response();
        const bool success = backend->execute(request, *remote);
        remote->mutable_success()->set_value(success);
        return grpc::Status::OK;
    };
    states[1]->handler = [this, finished](const auto& request, auto& response) {
        auto*      remote  = response.mutable_remote_response();
        const bool success = worker->execute(request, *remote);
        remote->mutable_success()->set_value(success);
        finished->set_value();
        return grpc::Status::OK;
    };
    const auto pools = {environment.device_pool, worker_environment->device_pool};
    if (writing) {
        for (const auto& pool : pools) {
            for (const auto& buffer : pool->convertIndexToBuffer(0, environment.block_id)) {
                EXPECT_EQ(cudaMemset(buffer.addr, 0x11, buffer.size_bytes), cudaSuccess);
            }
        }
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        EXPECT_CALL(*client_wrapper, saveKvCachesForTag("default", _, _, _))
            .WillOnce(Return(std::make_pair(false, kv_cache_manager::UriStrVec{})));
        EXPECT_CALL(*worker_client, saveKvCachesForTag("default", _, _, _))
            .WillOnce(Invoke([access_old](const auto&, const auto&, const auto& buffers, const auto&) {
                access_old(buffers);
                return completedSave(kv_cache_manager::UriStrVec{});
            }));
    } else {
        EXPECT_CALL(*client_wrapper, loadKvCachesForTag("default", _, _, _)).WillOnce(Return(false));
        EXPECT_CALL(*worker_client, loadKvCachesForTag("default", _, _, _))
            .WillOnce(Invoke([access_old](const auto&, const auto&, auto& buffers, const auto&) {
                access_old(buffers);
                return true;
            }));
    }
    EXPECT_CALL(*client_wrapper, finishWrite(_, _, _, _, _)).Times(0);
    ASSERT_NO_FATAL_FAILURE(failTransfer(writing, makeStorageRequest(environment)));
    ASSERT_EQ(entered_future.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_EQ(finished_future.wait_for(std::chrono::milliseconds(0)), std::future_status::timeout);
    const auto old_block = environment.block_id;
    expectBlockState(true);
    EXPECT_FALSE(worker_environment->device_pool->isFrozen(old_block));
    environment.device_pool->decRef(old_block);
    environment.block_id = NULL_BLOCK_IDX;
    ScopedReferencedBlocks replacements(environment.device_pool, environment.device_pool->freeBlocksNum());
    for (const auto block : replacements.get()) {
        EXPECT_NE(block, old_block);
        for (const auto& pool : pools) {
            for (const auto& buffer : pool->convertIndexToBuffer(0, block)) {
                EXPECT_EQ(cudaMemset(buffer.addr, 0x22, buffer.size_bytes), cudaSuccess);
            }
        }
    }
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(drain.finish(), std::future_status::ready);
    if (writing) {
        EXPECT_FALSE(observed_source.empty());
        EXPECT_TRUE(std::all_of(observed_source.begin(), observed_source.end(), [](uint8_t v) { return v == 0x11; }));
    }
    for (const auto block : replacements.get()) {
        for (const auto& pool : pools) {
            for (const auto& buffer : pool->convertIndexToBuffer(0, block)) {
                std::vector<uint8_t> bytes(buffer.size_bytes);
                EXPECT_EQ(cudaMemcpy(bytes.data(), buffer.addr, bytes.size(), cudaMemcpyDeviceToHost), cudaSuccess);
                EXPECT_TRUE(std::all_of(bytes.begin(), bytes.end(), [](uint8_t v) { return v == 0x22; }));
            }
        }
    }
}
INSTANTIATE_TEST_SUITE_P(PayloadDirections, KVCMTP2DeadlineTest, ::testing::Bool());

}  // namespace
}  // namespace rtp_llm
