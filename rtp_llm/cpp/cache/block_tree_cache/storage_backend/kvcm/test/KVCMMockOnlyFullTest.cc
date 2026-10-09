#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test/KVCMMockTestBase.h"

#include <tuple>
#include <thread>

namespace rtp_llm {
namespace {

TEST(KVCMMockOnlyFullTest, TPFollowerUsesControllerBlockIdsWithoutLocalAllocationMetadata) {
    auto environment = makeBackendEnvironment("kvcm_follower_physical_blocks");
    auto client = std::make_shared<MockClientWrapper>();
    ParallelismConfig parallelism;
    parallelism.tp_size = 2;
    parallelism.tp_rank = 1;
    parallelism.local_rank = 0;
    EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client, shutdown()).Times(1);
    auto backend = makeBackend(environment, parallelism, client);
    ASSERT_TRUE(backend->init(environment.cache_config.topologyPtr(), environment.pools_by_tag,
        [&](int layer, const std::string&, int block) {
            return environment.device_pool->convertIndexToBuffer(layer, block);
        }));
    environment.device_pool->decRef(environment.block_id);
    ASSERT_FALSE(environment.device_pool->isAllocated(environment.block_id));
    EXPECT_CALL(*client, saveKvCachesForTag("default", kv_cache_manager::UriStrVec{"write_uri"}, _, _))
        .WillOnce(Return(std::make_pair(true, kv_cache_manager::UriStrVec{})));
    RemoteOperationRequestPB request;
    request.set_op(REMOTE_OPERATION_WRITE);
    request.add_group_tags("default");
    request.add_block_ids(environment.block_id);
    request.add_uris("write_uri");
    RemoteOperationResponsePB response;
    EXPECT_TRUE(backend->execute(request, response));
    EXPECT_FALSE(environment.device_pool->isAllocated(environment.block_id));
    request.set_block_ids(0, static_cast<int32_t>(environment.device_pool->totalBlocksNum() + 1));
    EXPECT_FALSE(backend->execute(request, response));
}

TEST(KVCMMockOnlyFullTest, TPWriteTimeoutReturnsWithControllerPinsQuarantined) {
    auto environment = makeBackendEnvironment("kvcm_tp_write_quarantine");
    auto client = std::make_shared<MockClientWrapper>();
    std::promise<void> peer_entered;
    auto entered = peer_entered.get_future();
    std::promise<void> release_peer;
    auto release = release_peer.get_future().share();
    std::vector<std::shared_ptr<KVCMBroadcastState>> states;
    std::vector<std::unique_ptr<KVCMBroadcastRpcServer>> servers;
    std::vector<std::string> addresses;
    for (size_t rank = 0; rank < 2; ++rank) {
        auto state = std::make_shared<KVCMBroadcastState>();
        if (rank == 1) {
            state->before_reply = [&peer_entered, release] {
                peer_entered.set_value();
                EXPECT_EQ(release.wait_for(std::chrono::seconds(5)), std::future_status::ready);
            };
        }
        auto server = std::make_unique<KVCMBroadcastRpcServer>(rank, state);
        ASSERT_TRUE(server->start());
        addresses.push_back(server->address());
        states.push_back(std::move(state));
        servers.push_back(std::move(server));
    }
    auto broadcaster = std::make_shared<BroadcastManager>(addresses);
    ASSERT_TRUE(broadcaster->init());
    ParallelismConfig parallelism;
    parallelism.tp_size = 2;
    parallelism.tp_rank = 0;
    parallelism.local_rank = 0;
    KVCacheConfig config;
    config.kvcm_server_address = "unused-test-address";
    config.kvcm_put_broadcast_timeout = 20;
    RuntimeConfig runtime;
    runtime.model_name = "kvcm_test_model";
    EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client, shutdown()).Times(0);
    ::testing::Mock::AllowLeak(client.get());
    BackendHandle backend(std::make_unique<KVCMStorageBackend>(environment.cache_config, config, runtime,
        parallelism, SpeculativeExecutionConfig{}, broadcaster, client));
    ASSERT_TRUE(backend->init(environment.cache_config.topologyPtr(), environment.pools_by_tag,
        [&](int layer, const std::string&, int block) {
            return environment.device_pool->convertIndexToBuffer(layer, block);
        }));
    kv_cache_manager::WriteLocation location;
    location.write_session_id = "drained_timeout";
    location.block_mask = kv_cache_manager::BlockMaskOffset{0};
    location.locations = {{{"tp0_Fdefault", "rank0_uri"}, {"tp1_Fdefault", "rank1_uri"}}};
    EXPECT_CALL(*client, getWriteLocation(_, _, _, _, _, _, 0))
        .WillOnce(Return(std::make_pair(true, location)));
    EXPECT_CALL(*client, finishWrite(_, _, "drained_timeout", _, _)).Times(0);
    backend->write(backend->prepareWrite(makeStorageRequest(environment)));
    environment.device_pool->decRef(environment.block_id);
    const bool peer_started = entered.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
    EXPECT_TRUE(peer_started);
    // The frontend and shutdown must finish before the peer is released.
    EXPECT_TRUE(waitForBackendOperationsForTest(*backend.backend));
    EXPECT_TRUE(environment.device_pool->isAllocated(environment.block_id));
    backend->shutdown();
    release_peer.set_value();
    // Unknown completion never makes these physical IDs reusable again.
    EXPECT_TRUE(environment.device_pool->isAllocated(environment.block_id));
    // Restore the fixture's own reference so its destructor does not consume a quarantine pin.
    environment.device_pool->incRef(environment.block_id);
    EXPECT_TRUE(::testing::Mock::VerifyAndClearExpectations(client.get()));
}

TEST(KVCMMockOnlyFullTest, MetadataLengthRpcUsesTheInstanceDefaultAndReturnsTheSdkResult) {
    auto environment = makeBackendEnvironment("kvcm_metadata_length");
    auto client_wrapper = std::make_shared<MockClientWrapper>();
    EXPECT_CALL(*client_wrapper, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment, ParallelismConfig{}, client_wrapper);
    ASSERT_TRUE(backend->init(environment.cache_config.topologyPtr(), environment.pools_by_tag,
        [&](int layer, const std::string&, int block) {
            return environment.device_pool->convertIndexToBuffer(layer, block);
        }));
    EXPECT_CALL(*client_wrapper, matchLocationLen("", "length", kv_cache_manager::QueryType::QT_PREFIX_MATCH,
                                                 std::vector<int64_t>({101, 102}), _, 0))
        .WillOnce(Return(std::make_pair(true, int64_t{2})));
    RemoteOperationRequestPB request;
    request.set_op(REMOTE_OPERATION_MATCH_LOCATION_LEN);
    request.set_trace_id("length");
    request.mutable_metadata()->add_block_keys(101);
    request.mutable_metadata()->add_block_keys(102);
    RemoteOperationResponsePB response;
    EXPECT_TRUE(backend->execute(request, response));
    EXPECT_EQ(response.matched_blocks(), 2);
}

TEST(KVCMMockOnlyFullTest, MetadataRpcPreservesFiltersMasksAndHitResponses) {
    auto environment = makeBackendEnvironment("kvcm_metadata_hits");
    auto client = std::make_shared<MockClientWrapper>();
    EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client, shutdown()).Times(1);
    auto backend = makeBackend(environment, ParallelismConfig{}, client);
    ASSERT_TRUE(backend->init(environment.cache_config.topologyPtr(), environment.pools_by_tag,
        [&](int layer, const std::string&, int block) {
            return environment.device_pool->convertIndexToBuffer(layer, block);
        }));

    const std::vector<int64_t> keys{101, 102};
    const std::vector<int64_t> tokens{7, 8};
    const std::vector<std::string> names{"tp0_Fdefault", "tp0_Fdefault"};
    const kv_cache_manager::BlockMask mask = kv_cache_manager::BlockMaskVector{false, true};
    const kv_cache_manager::Locations locations{{{names[0], "pace://hit"}}, {}};
    RemoteOperationRequestPB request;
    request.set_trace_id("metadata");
    auto* query = request.mutable_metadata();
    query->set_query_type(1);
    for (auto key : keys) {
        query->add_block_keys(key);
    }
    for (auto token : tokens) {
        query->add_token_ids(token);
    }
    for (const auto& name : names) {
        query->add_location_spec_names(name);
    }
    query->mutable_block_mask()->mutable_bool_masks()->add_values(false);
    query->mutable_block_mask()->mutable_bool_masks()->add_values(true);
    RemoteOperationResponsePB response;

    EXPECT_CALL(*client, queryLocations("", "metadata", kv_cache_manager::QueryType::QT_BATCH_GET,
                                       keys, tokens, mask, 0, names))
        .WillOnce(Return(std::make_pair(true, locations)));
    request.set_op(REMOTE_OPERATION_MATCH_LOCATION);
    ASSERT_TRUE(backend->execute(request, response));
    ASSERT_EQ(response.locations_size(), 2);
    ASSERT_EQ(response.locations(0).specs_size(), 1);
    EXPECT_EQ(response.locations(0).specs(0).name(), names[0]);
    EXPECT_EQ(response.locations(0).specs(0).uri(), "pace://hit");
    EXPECT_EQ(response.locations(1).specs_size(), 0);

    const kv_cache_manager::Metas metas{locations, {R"({"key":101})", ""}};
    EXPECT_CALL(*client, matchMeta("", "metadata", keys, tokens, mask, 1))
        .WillOnce(Return(std::make_pair(true, metas)));
    request.set_op(REMOTE_OPERATION_MATCH_META);
    query->set_detail_level(1);
    response.Clear();
    ASSERT_TRUE(backend->execute(request, response));
    ASSERT_EQ(response.locations_size(), 2);
    ASSERT_EQ(response.locations(0).specs_size(), 1);
    EXPECT_EQ(response.locations(0).specs(0).uri(), "pace://hit");
    EXPECT_EQ(response.locations(1).specs_size(), 0);
    ASSERT_EQ(response.metas_size(), 2);
    EXPECT_EQ(response.metas(0), metas.metas[0]);
    EXPECT_EQ(response.metas(1), "");

    const kv_cache_manager::BackendLocations by_backend{
        {{kv_cache_manager::StorageType::ST_TAIRMEMPOOL, 4096, locations[0]}}, {}};
    EXPECT_CALL(*client, getCacheLocationsByBackend("", "metadata", keys, tokens, mask, names,
                                                   kv_cache_manager::StorageType::ST_TAIRMEMPOOL))
        .WillOnce(Return(std::make_pair(true, by_backend)));
    request.set_op(REMOTE_OPERATION_GET_LOCATIONS_BY_BACKEND);
    query->set_backend_type(3);
    response.Clear();
    ASSERT_TRUE(backend->execute(request, response));
    ASSERT_EQ(response.backend_locations_size(), 2);
    ASSERT_EQ(response.backend_locations(0).locations_size(), 1);
    const auto& hit = response.backend_locations(0).locations(0);
    EXPECT_EQ(hit.backend_type(), 3);
    EXPECT_EQ(hit.spec_size(), 4096);
    ASSERT_EQ(hit.specs_size(), 1);
    EXPECT_EQ(hit.specs(0).name(), names[0]);
    EXPECT_EQ(hit.specs(0).uri(), "pace://hit");
    EXPECT_EQ(response.backend_locations(1).locations_size(), 0);

    const kv_cache_manager::HostCacheState hosts{{"192.0.2.1:1234", 2, 1, 3}};
    EXPECT_CALL(*client, getHostCacheState("", "metadata", kv_cache_manager::QueryType::QT_PREFIX_MATCH,
                                         keys, std::vector<std::string>{"hbm"}, 1))
        .WillOnce(Return(std::make_pair(true, hosts)));
    request.set_op(REMOTE_OPERATION_GET_HOST_CACHE_STATE);
    query->set_query_type(2);
    query->add_medium("hbm");
    query->set_p2p_host_count(1);
    response.Clear();
    ASSERT_TRUE(backend->execute(request, response));
    ASSERT_EQ(response.hosts_size(), 1);
    EXPECT_EQ(response.hosts(0).host_ip_port(), hosts[0].host_ip_port);
    EXPECT_EQ(response.hosts(0).local(), 2);
    EXPECT_EQ(response.hosts(0).p2p_1_fetch(), 1);
    EXPECT_EQ(response.hosts(0).p2p_1_total_match(), 3);

    const kv_cache_manager::BlockMask offset = kv_cache_manager::BlockMaskOffset{1};
    EXPECT_CALL(*client, removeCache("", "metadata", keys, tokens, offset)).WillOnce(Return(true));
    request.set_op(REMOTE_OPERATION_REMOVE_CACHE);
    query->mutable_block_mask()->set_offset(1);
    response.Clear();
    EXPECT_TRUE(backend->execute(request, response));
}

TEST(KVCMMockOnlyFullTest, SwaPayloadMatchUsesBatchLocationsForFullPrefix) {
    BackendEnvironment environment;
    environment.cache_config.dtype = DataType::TYPE_FP16;
    environment.cache_config.layer_num = 2;
    environment.cache_config.seq_size_per_block = 8;
    auto window = defaultCacheGroupPolicy(CacheGroupType::SWA);
    window.sliding_window_size = 16;
    environment.cache_config.fromGroupedSpecs(
        {test::makeMhaSpec("full0", 8, DataType::TYPE_FP16, 1, 2),
         test::makeMhaSpec("window0", 8, DataType::TYPE_FP16, 1, 2)},
        {{0}, {1}}, {CacheGroupType::FULL, CacheGroupType::SWA}, {"full0", "window0"},
        {defaultCacheGroupPolicy(CacheGroupType::FULL), window});
    environment.cache_config.finalizeBlockNums(8, RuntimeConfig{});
    initializeEnvironmentPools(environment, "kvcm_swa_query", 12);
    auto client = std::make_shared<MockClientWrapper>();
    KVCacheConfig config;
    config.kvcm_server_address = "unused-test-address";
    config.kvcm_query_type = 3;
    config.kvcm_sw_size = 2;
    RuntimeConfig runtime;
    runtime.model_name = "kvcm_test_model";
    EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client, shutdown()).Times(1);
    BackendHandle backend(std::make_unique<KVCMStorageBackend>(environment.cache_config, config, runtime,
        ParallelismConfig{}, SpeculativeExecutionConfig{}, nullptr, client));
    ASSERT_TRUE(backend->init(environment.cache_config.topologyPtr(), environment.pools_by_tag,
        [&](int layer, const std::string& tag, int block) {
            return environmentBuffers(environment, layer, tag, block);
        }));
    // Batch mode returns the FULL prefix, including blocks outside the SWA
    // window. A reverse-window query would leave the first location empty.
    const kv_cache_manager::Locations locations{
        {{"tp0_Ffull0", "full_101"}},
        {{"tp0_Ffull0", "full_102"}, {"tp0_Lwindow0", "window_102"}},
        {{"tp0_Ffull0", "full_103"}, {"tp0_Lwindow0", "window_103"}}};
    EXPECT_CALL(*client, match(_, _, kv_cache_manager::QueryType::QT_BATCH_GET,
                              std::vector<int64_t>({101, 102, 103}), _, _))
        .WillOnce(Return(std::make_pair(true, locations)));
    const auto result = match(*backend.backend, makeStorageRequest(environment, {101, 102, 103}));
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.matched_blocks_num, 3u);
}

TEST(KVCMMockOnlyFullTest, BatchMatchStopsAtAMissAndPreservesReadUriPosition) {
    auto environment = makeBackendEnvironment("kvcm_batch_prefix");
    auto client_wrapper = std::make_shared<MockClientWrapper>();
    KVCacheConfig config;
    config.kvcm_server_address = "unused-test-address";
    config.kvcm_query_type = 1;
    RuntimeConfig runtime;
    runtime.model_name = "kvcm_test_model";
    EXPECT_CALL(*client_wrapper, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    BackendHandle backend(std::make_unique<KVCMStorageBackend>(environment.cache_config, config, runtime,
        ParallelismConfig{}, SpeculativeExecutionConfig{}, nullptr, client_wrapper));
    ASSERT_TRUE(backend->init(environment.cache_config.topologyPtr(), environment.pools_by_tag,
        [&](int layer, const std::string&, int block) {
            return environment.device_pool->convertIndexToBuffer(layer, block);
        }));
    const kv_cache_manager::Locations locations{
        {{"tp0_Fdefault", "uri_101"}}, {{"tp0_Fdefault", ""}}, {{"tp0_Fdefault", "uri_103"}}};
    EXPECT_CALL(*client_wrapper, match(_, _, kv_cache_manager::QueryType::QT_BATCH_GET, _, _, _))
        .WillOnce(Return(std::make_pair(true, locations)));
    auto matched = match(*backend.backend, makeStorageRequest(environment, {101, 102, 103}));
    ASSERT_TRUE(matched.success);
    ASSERT_EQ(matched.matched_blocks_num, 1u);
    EXPECT_CALL(*client_wrapper, loadKvCachesForTag("default", kv_cache_manager::UriStrVec{"uri_101"}, _, _))
        .WillOnce(Return(true));
    EXPECT_TRUE(read(*backend.backend, makeStorageRequest(environment, {101}), matched.match_meta));
}

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
    EXPECT_CALL(*client_wrapper, getWriteLocation(_, _, _, _, _, _, 0))
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

TEST(KVCMMockOnlyFullTest, TP2BroadcastFailureQuarantinesWriteSession) {
    auto environment    = makeBackendEnvironment("kvcm_storage_backend_tp2_broadcast_failure");
    auto client_wrapper = std::make_shared<MockClientWrapper>();

    std::vector<std::shared_ptr<KVCMBroadcastState>>     states;
    std::vector<std::unique_ptr<KVCMBroadcastRpcServer>> servers;
    std::vector<std::string>                             addresses;
    for (size_t rank = 0; rank < 2; ++rank) {
        auto state  = std::make_shared<KVCMBroadcastState>();
        state->fail = rank == 1;
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
    EXPECT_CALL(*client_wrapper, shutdown()).Times(0);
    ::testing::Mock::AllowLeak(client_wrapper.get());
    auto backend = makeBackend(environment, parallelism_config, client_wrapper, broadcast_manager);
    ASSERT_TRUE(backend->init(environment.cache_config.topologyPtr(),
                              environment.pools_by_tag,
                              [&](int layer_id, const std::string&, int block_id) {
                                  return environment.device_pool->convertIndexToBuffer(layer_id, block_id);
                              }));

    kv_cache_manager::WriteLocation write_location;
    write_location.write_session_id = "tp2_failed_broadcast_write";
    write_location.block_mask       = kv_cache_manager::BlockMaskOffset{0};
    write_location.locations        = {kv_cache_manager::Location{
        kv_cache_manager::LocationSpecUnit{"tp0_Fdefault", "write_rank_0"},
        kv_cache_manager::LocationSpecUnit{"tp1_Fdefault", "write_rank_1"},
    }};
    EXPECT_CALL(*client_wrapper, getWriteLocation(_, _, _, _, _, _, 0))
        .WillOnce(Return(std::make_pair(true, write_location)));
    // A failed RPC may leave a peer writing; do not recycle its remote destination.
    EXPECT_CALL(*client_wrapper, finishWrite(_, _, "tp2_failed_broadcast_write", _, _)).Times(0);

    const auto source_ref_count = environment.device_pool->refCount(environment.block_id);
    EXPECT_EQ(environment.device_pool->referencedBlocksNum(BlockTreeRefType::STORE), 0u);
    backend->write(backend->prepareWrite(makeStorageRequest(environment)));
    ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
    EXPECT_GT(environment.device_pool->refCount(environment.block_id), source_ref_count);
    EXPECT_EQ(environment.device_pool->referencedBlocksNum(BlockTreeRefType::STORE), 0u);
    for (const auto& state : states) {
        const auto requests = snapshotRequests(state);
        ASSERT_EQ(requests.size(), 1u);
        EXPECT_EQ(requests.front().op(), REMOTE_OPERATION_WRITE);
    }
    backend->shutdown();
    EXPECT_TRUE(::testing::Mock::VerifyAndClearExpectations(client_wrapper.get()));
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
                return std::make_pair(!enabled, kv_cache_manager::UriStrVec{});
            }
            EXPECT_TRUE(enabled);
            EXPECT_TRUE(trace_info->need_print);
            EXPECT_EQ(trace_info->block_ids, expected_block_ids);
            return std::make_pair(true, kv_cache_manager::UriStrVec{});
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
    EXPECT_CALL(*client_wrapper, getWriteLocation(_, _, std::vector<int64_t>{101}, std::vector<int64_t>{}, _, 600, 0))
        .WillOnce(Return(std::make_pair(true, write_location)));
    EXPECT_CALL(*client_wrapper, saveKvCachesForTag("default", kv_cache_manager::UriStrVec{"write_uri"}, _, _))
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
                getWriteLocation(_, _, std::vector<int64_t>({101, 102, 103}), std::vector<int64_t>{}, _, 600, 0))
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
            return std::make_pair(true, kv_cache_manager::UriStrVec({"actual_uri_102", "actual_uri_103"}));
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
    EXPECT_CALL(*client_wrapper, getWriteLocation(_, _, _, _, _, _, 0))
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
            return std::make_pair(true, kv_cache_manager::UriStrVec{});
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

TEST(KVCMMockOnlyFullTest, EmptyWriteSessionCloseFailureIsAdvisory) {
    auto environment    = makeBackendEnvironment("kvcm_storage_backend_empty_write");
    auto client_wrapper = std::make_shared<MockClientWrapper>();

    EXPECT_CALL(*client_wrapper, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), client_wrapper);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));

    kv_cache_manager::WriteLocation write_location;
    write_location.write_session_id = "empty_session";
    write_location.block_mask       = kv_cache_manager::BlockMaskOffset{3};
    EXPECT_CALL(*client_wrapper, getWriteLocation(_, _, _, _, _, _, 0))
        .WillOnce(Return(std::make_pair(true, write_location)));
    EXPECT_CALL(*client_wrapper, saveKvCachesForTag("default", _, _, _)).Times(0);
    EXPECT_CALL(*client_wrapper, finishWrite(_, _, "empty_session", _, _)).WillOnce(Return(false));

    auto request = makeStorageRequest(environment, /*keys=*/{101, 102, 103});
    backend->write(backend->prepareWrite(std::move(request)));
    ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
}

TEST(KVCMMockOnlyFullTest, UnchangedActualUrisAreNotRepublished) {
    auto environment    = makeBackendEnvironment("kvcm_storage_backend_unchanged_uri");
    auto client_wrapper = std::make_shared<MockClientWrapper>();

    EXPECT_CALL(*client_wrapper, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), client_wrapper);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));

    kv_cache_manager::WriteLocation write_location;
    write_location.write_session_id = "unchanged_session";
    write_location.block_mask       = kv_cache_manager::BlockMaskOffset{0};
    write_location.locations        = {
        kv_cache_manager::Location{kv_cache_manager::LocationSpecUnit{"tp0_Fdefault", "write_uri"}}};
    EXPECT_CALL(*client_wrapper, getWriteLocation(_, _, _, _, _, _, 0))
        .WillOnce(Return(std::make_pair(true, write_location)));
    EXPECT_CALL(*client_wrapper, saveKvCachesForTag("default", kv_cache_manager::UriStrVec{"write_uri"}, _, _))
        .WillOnce(Return(std::make_pair(true, kv_cache_manager::UriStrVec{"write_uri"})));
    EXPECT_CALL(*client_wrapper, finishWrite(_, _, "unchanged_session", _, _))
        .WillOnce(Invoke([](const std::string&,
                            const std::string&,
                            const std::string&,
                            const kv_cache_manager::BlockMask&,
                            const kv_cache_manager::Locations& locations) {
            EXPECT_TRUE(locations.empty());
            return true;
        }));

    backend->write(backend->prepareWrite(makeStorageRequest(environment)));
    ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
}

TEST(KVCMMockOnlyFullTest, MismatchedActualUriCountAbortsWriteSession) {
    auto environment    = makeBackendEnvironment("kvcm_storage_backend_actual_uri_shape");
    auto client_wrapper = std::make_shared<MockClientWrapper>();

    EXPECT_CALL(*client_wrapper, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), client_wrapper);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));

    kv_cache_manager::WriteLocation write_location;
    write_location.write_session_id = "shape_session";
    write_location.block_mask       = kv_cache_manager::BlockMaskOffset{0};
    write_location.locations        = {
        kv_cache_manager::Location{kv_cache_manager::LocationSpecUnit{"tp0_Fdefault", "write_uri"}}};
    EXPECT_CALL(*client_wrapper, getWriteLocation(_, _, _, _, _, _, 0))
        .WillOnce(Return(std::make_pair(true, write_location)));
    EXPECT_CALL(*client_wrapper, saveKvCachesForTag("default", kv_cache_manager::UriStrVec{"write_uri"}, _, _))
        .WillOnce(Return(std::make_pair(true, kv_cache_manager::UriStrVec({"actual_uri", "unexpected_extra_uri"}))));
    EXPECT_CALL(*client_wrapper, finishWrite(_, _, "shape_session", _, _))
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

    backend->write(backend->prepareWrite(makeStorageRequest(environment)));
    ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
}

TEST(KVCMMockOnlyFullTest, StartWriteFailureDoesNotFinishSession) {
    auto environment    = makeBackendEnvironment("kvcm_storage_backend_start_write_failure");
    auto client_wrapper = std::make_shared<MockClientWrapper>();

    EXPECT_CALL(*client_wrapper, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), client_wrapper);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));

    EXPECT_CALL(*client_wrapper, getWriteLocation(_, _, _, _, _, _, 0))
        .WillOnce(Return(std::make_pair(false, kv_cache_manager::WriteLocation{})));
    EXPECT_CALL(*client_wrapper, saveKvCachesForTag("default", _, _, _)).Times(0);
    EXPECT_CALL(*client_wrapper, finishWrite(_, _, _, _, _)).Times(0);
    backend->write(backend->prepareWrite(makeStorageRequest(environment)));
    ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
}

TEST(KVCMMockOnlyFullTest, TransferFailureQuarantinesWriteSession) {
    auto environment    = makeBackendEnvironment("kvcm_storage_backend_abort_write");
    auto client_wrapper = std::make_shared<MockClientWrapper>();

    EXPECT_CALL(*client_wrapper, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(0);
    ::testing::Mock::AllowLeak(client_wrapper.get());
    auto backend = makeBackend(environment, singleRankConfig(), client_wrapper);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));

    kv_cache_manager::WriteLocation write_location;
    write_location.write_session_id = "abort_session";
    write_location.block_mask       = kv_cache_manager::BlockMaskOffset{0};
    write_location.locations        = {
        kv_cache_manager::Location{kv_cache_manager::LocationSpecUnit{"tp0_Fdefault", "write_uri"}}};
    EXPECT_CALL(*client_wrapper, getWriteLocation(_, _, _, _, _, _, 0))
        .WillOnce(Return(std::make_pair(true, write_location)));
    EXPECT_CALL(*client_wrapper, saveKvCachesForTag("default", kv_cache_manager::UriStrVec{"write_uri"}, _, _))
        .WillOnce(Return(std::make_pair(false, kv_cache_manager::UriStrVec{})));
    EXPECT_CALL(*client_wrapper, finishWrite(_, _, "abort_session", _, _)).Times(0);
    const auto source_refs = environment.device_pool->refCount(environment.block_id);
    backend->write(backend->prepareWrite(makeStorageRequest(environment)));
    ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
    EXPECT_GT(environment.device_pool->refCount(environment.block_id), source_refs);
    backend->shutdown();
    EXPECT_TRUE(::testing::Mock::VerifyAndClearExpectations(client_wrapper.get()));
}

TEST(KVCMMockOnlyFullTest, FinishWriteFailureIsNotRetried) {
    auto environment    = makeBackendEnvironment("kvcm_storage_backend_finish_write_failure");
    auto client_wrapper = std::make_shared<MockClientWrapper>();

    EXPECT_CALL(*client_wrapper, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), client_wrapper);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));

    kv_cache_manager::WriteLocation write_location;
    write_location.write_session_id = "finish_session";
    write_location.block_mask       = kv_cache_manager::BlockMaskOffset{0};
    write_location.locations        = {
        kv_cache_manager::Location{kv_cache_manager::LocationSpecUnit{"tp0_Fdefault", "write_uri"}}};
    EXPECT_CALL(*client_wrapper, getWriteLocation(_, _, _, _, _, _, 0))
        .WillOnce(Return(std::make_pair(true, write_location)));
    EXPECT_CALL(*client_wrapper, saveKvCachesForTag("default", kv_cache_manager::UriStrVec{"write_uri"}, _, _))
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
    backend->write(backend->prepareWrite(makeStorageRequest(environment)));
    ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
}

}  // namespace
}  // namespace rtp_llm
