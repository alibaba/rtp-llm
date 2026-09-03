#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test/KVCMMockTestBase.h"

namespace rtp_llm {
namespace {

TEST(KVCMMockOnlyFullTest, TP2WorkerRegistersItsRankAndExecutesLocalPayload) {
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
                EXPECT_EQ(config_map.size(), 1u);
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

TEST(KVCMMockOnlyFullTest, TP2CoordinatorRejectsMissingBroadcastManager) {
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

TEST(KVCMMockOnlyFullTest, TP2BroadcastFailureAbortsWriteSession) {
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
    EXPECT_CALL(*client_wrapper, init(_, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment, parallelism_config, client_wrapper, broadcast_manager);
    ASSERT_TRUE(backend->init(
        environment.cache_config.topologyPtr(), {environment.device_pool}, [&](int layer_id, int, int block_id) {
            return environment.device_pool->convertIndexToBuffer(layer_id, block_id);
        }));

    kv_cache_manager::WriteLocation write_location;
    write_location.write_session_id = "tp2_failed_broadcast_write";
    write_location.block_mask       = kv_cache_manager::BlockMaskOffset{0};
    write_location.locations        = {kv_cache_manager::Location{
        kv_cache_manager::LocationSpecUnit{"tp0_Fdefault", "write_rank_0"},
        kv_cache_manager::LocationSpecUnit{"tp1_Fdefault", "write_rank_1"},
    }};
    EXPECT_CALL(*client_wrapper, getWriteLocation(_, _, _, _, _, _))
        .WillOnce(Return(std::make_pair(true, write_location)));
    EXPECT_CALL(*client_wrapper, finishWrite(_, _, "tp2_failed_broadcast_write", _, _))
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

    const auto source_ref_count = environment.device_pool->refCount(environment.block_id);
    EXPECT_EQ(environment.device_pool->referencedBlocksNum(BlockTreeRefType::STORE), 0u);
    EXPECT_FALSE(backend->write(backend->prepareWrite(makeStorageRequest(environment)), /*synchronous=*/true));
    EXPECT_EQ(environment.device_pool->refCount(environment.block_id), source_ref_count);
    EXPECT_EQ(environment.device_pool->referencedBlocksNum(BlockTreeRefType::STORE), 0u);
    for (const auto& state : states) {
        const auto requests = snapshotRequests(state);
        ASSERT_EQ(requests.size(), 1u);
        EXPECT_EQ(requests.front().op(), REMOTE_OPERATION_WRITE);
    }
}

TEST(KVCMMockOnlyFullTest, RejectsMismatchedTransferVectorsBeforeClientIO) {
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

TEST(KVCMMockOnlyFullTest, SDKCheckTracesReadAndWriteBlockIds) {
    autil::EnvGuard sdk_check("KVCM_SDK_CHECK", "1");
    auto            environment    = makeBackendEnvironment("kvcm_storage_backend_sdk_check");
    auto            client_wrapper = std::make_shared<MockClientWrapper>();

    EXPECT_CALL(*client_wrapper, init(_, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), client_wrapper);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));

    const std::vector<std::string> expected_block_ids{std::to_string(environment.block_id)};
    EXPECT_CALL(*client_wrapper, loadKvCaches(kv_cache_manager::UriStrVec{"read_uri"}, _, _))
        .WillOnce(Invoke([&](const kv_cache_manager::UriStrVec&,
                             kv_cache_manager::BlockBuffers&,
                             const std::shared_ptr<kv_cache_manager::TransferTraceInfo>& trace_info) {
            if (trace_info == nullptr) {
                ADD_FAILURE() << "KVCM_SDK_CHECK read omitted transfer trace info";
                return false;
            }
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

    EXPECT_CALL(*client_wrapper, saveKvCaches(kv_cache_manager::UriStrVec{"write_uri"}, _, _))
        .WillOnce(Invoke([&](const kv_cache_manager::UriStrVec&,
                             const kv_cache_manager::BlockBuffers&,
                             const std::shared_ptr<kv_cache_manager::TransferTraceInfo>& trace_info) {
            if (trace_info == nullptr) {
                ADD_FAILURE() << "KVCM_SDK_CHECK write omitted transfer trace info";
                return std::make_pair(false, kv_cache_manager::UriStrVec{});
            }
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

TEST(KVCMMockOnlyFullTest, WritePublishesActualUri) {
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

TEST(KVCMMockOnlyFullTest, WriteHonorsOffsetBlockMask) {
    auto environment    = makeBackendEnvironment("kvcm_storage_backend_offset_mask");
    auto client_wrapper = std::make_shared<MockClientWrapper>();

    EXPECT_CALL(*client_wrapper, init(_, _)).WillOnce(Return(true));
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
    EXPECT_CALL(*client_wrapper, saveKvCaches(kv_cache_manager::UriStrVec({"write_uri_102", "write_uri_103"}), _, _))
        .WillOnce(Invoke([second_base = second_block_info.front().addr, third_base = third_block_info.front().addr](
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
    EXPECT_TRUE(backend->write(backend->prepareWrite(std::move(request)), /*synchronous=*/true));
}

TEST(KVCMMockOnlyFullTest, WriteHonorsSparseBlockMask) {
    auto environment    = makeBackendEnvironment("kvcm_storage_backend_sparse_mask");
    auto client_wrapper = std::make_shared<MockClientWrapper>();

    EXPECT_CALL(*client_wrapper, init(_, _)).WillOnce(Return(true));
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
    EXPECT_CALL(*client_wrapper, saveKvCaches(kv_cache_manager::UriStrVec({"write_uri_102", "write_uri_104"}), _, _))
        .WillOnce(Invoke([second_base = second_block_info.front().addr, fourth_base = fourth_block_info.front().addr](
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
    EXPECT_TRUE(backend->write(backend->prepareWrite(std::move(request)), /*synchronous=*/true));
}

TEST(KVCMMockOnlyFullTest, EmptyWriteLocationsCompleteWithoutPayloadOrFinish) {
    auto environment    = makeBackendEnvironment("kvcm_storage_backend_empty_write");
    auto client_wrapper = std::make_shared<MockClientWrapper>();

    EXPECT_CALL(*client_wrapper, init(_, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), client_wrapper);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));

    kv_cache_manager::WriteLocation write_location;
    write_location.write_session_id = "empty_session";
    write_location.block_mask       = kv_cache_manager::BlockMaskOffset{3};
    EXPECT_CALL(*client_wrapper, getWriteLocation(_, _, _, _, _, _))
        .WillOnce(Return(std::make_pair(true, write_location)));
    EXPECT_CALL(*client_wrapper, saveKvCaches(_, _, _)).Times(0);
    EXPECT_CALL(*client_wrapper, finishWrite(_, _, _, _, _)).Times(0);

    auto request = makeStorageRequest(environment, /*keys=*/{101, 102, 103});
    EXPECT_TRUE(backend->write(backend->prepareWrite(std::move(request)), /*synchronous=*/true));
}

TEST(KVCMMockOnlyFullTest, UnchangedActualUrisAreNotRepublished) {
    auto environment    = makeBackendEnvironment("kvcm_storage_backend_unchanged_uri");
    auto client_wrapper = std::make_shared<MockClientWrapper>();

    EXPECT_CALL(*client_wrapper, init(_, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), client_wrapper);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));

    kv_cache_manager::WriteLocation write_location;
    write_location.write_session_id = "unchanged_session";
    write_location.block_mask       = kv_cache_manager::BlockMaskOffset{0};
    write_location.locations        = {
        kv_cache_manager::Location{kv_cache_manager::LocationSpecUnit{"tp0_Fdefault", "write_uri"}}};
    EXPECT_CALL(*client_wrapper, getWriteLocation(_, _, _, _, _, _))
        .WillOnce(Return(std::make_pair(true, write_location)));
    EXPECT_CALL(*client_wrapper, saveKvCaches(kv_cache_manager::UriStrVec{"write_uri"}, _, _))
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

    EXPECT_TRUE(backend->write(backend->prepareWrite(makeStorageRequest(environment)), /*synchronous=*/true));
}

TEST(KVCMMockOnlyFullTest, MismatchedActualUriCountAbortsWriteSession) {
    auto environment    = makeBackendEnvironment("kvcm_storage_backend_actual_uri_shape");
    auto client_wrapper = std::make_shared<MockClientWrapper>();

    EXPECT_CALL(*client_wrapper, init(_, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), client_wrapper);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));

    kv_cache_manager::WriteLocation write_location;
    write_location.write_session_id = "shape_session";
    write_location.block_mask       = kv_cache_manager::BlockMaskOffset{0};
    write_location.locations        = {
        kv_cache_manager::Location{kv_cache_manager::LocationSpecUnit{"tp0_Fdefault", "write_uri"}}};
    EXPECT_CALL(*client_wrapper, getWriteLocation(_, _, _, _, _, _))
        .WillOnce(Return(std::make_pair(true, write_location)));
    EXPECT_CALL(*client_wrapper, saveKvCaches(kv_cache_manager::UriStrVec{"write_uri"}, _, _))
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

    EXPECT_FALSE(backend->write(backend->prepareWrite(makeStorageRequest(environment)), /*synchronous=*/true));
}

TEST(KVCMMockOnlyFullTest, StartWriteFailureDoesNotFinishSession) {
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

TEST(KVCMMockOnlyFullTest, TransferFailureAbortsWriteSession) {
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

TEST(KVCMMockOnlyFullTest, FinishWriteFailureIsNotRetried) {
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
