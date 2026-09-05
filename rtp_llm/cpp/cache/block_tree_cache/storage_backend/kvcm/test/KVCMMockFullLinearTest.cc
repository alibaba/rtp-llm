#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test/KVCMMockTestBase.h"

namespace rtp_llm {
namespace {

kv_cache_manager::Location makeFullLinearLocation(size_t             full_group_count,
                                                  size_t             linear_group_count,
                                                  int64_t            key,
                                                  bool               include_linear,
                                                  const std::string& uri_prefix = "uri") {
    kv_cache_manager::Location location;
    for (size_t group_id = 0; group_id < full_group_count + (include_linear ? linear_group_count : 0); ++group_id) {
        const bool        is_full = group_id < full_group_count;
        const std::string group_name =
            (is_full ? "Ffull" : "Llinear") + std::to_string(is_full ? group_id : group_id - full_group_count);
        location.emplace_back(kv_cache_manager::LocationSpecUnit{
            "tp0_" + group_name, uri_prefix + "_" + std::to_string(key) + "_" + group_name});
    }
    return location;
}

kv_cache_manager::UriStrVec flattenUris(const kv_cache_manager::Locations& locations) {
    kv_cache_manager::UriStrVec uris;
    for (const auto& location : locations) {
        for (const auto& spec : location) {
            uris.push_back(spec.uri);
        }
    }
    return uris;
}

std::vector<void*> expectedBases(const BackendEnvironment&               environment,
                                 const std::vector<BlockIdxType>&        block_ids,
                                 const std::vector<std::vector<size_t>>& groups_by_key) {
    RTP_LLM_CHECK(block_ids.size() == groups_by_key.size());
    std::vector<void*> bases;
    for (size_t key_idx = 0; key_idx < block_ids.size(); ++key_idx) {
        for (const size_t group_id : groups_by_key[key_idx]) {
            bases.push_back(blockBase(environment, group_id, block_ids[key_idx]));
        }
    }
    return bases;
}

void expectBufferBases(const kv_cache_manager::BlockBuffers& buffers, const std::vector<void*>& bases) {
    ASSERT_EQ(buffers.size(), bases.size());
    for (size_t index = 0; index < bases.size(); ++index) {
        ASSERT_EQ(buffers[index].iovs.size(), 1u) << "buffer index=" << index;
        EXPECT_EQ(buffers[index].iovs.front().base, bases[index]) << "buffer index=" << index;
    }
}

TEST(KVCMMockFullLinearTest, ReadsCompleteMultiGroupLocationWithoutLocalReuse) {
    auto environment = makeMultiGroupBackendEnvironment(
        "kvcm_storage_backend_full_linear_read_no_local", /*full_group_count=*/1, /*linear_group_count=*/2);
    auto client_wrapper = std::make_shared<MockClientWrapper>();
    EXPECT_CALL(*client_wrapper, init(_, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), client_wrapper);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));

    ScopedReferencedBlocks      source_blocks(environment.device_pool, 1);
    const auto&                 block_ids = source_blocks.get();
    kv_cache_manager::Locations locations{
        makeFullLinearLocation(/*full_group_count=*/1, /*linear_group_count=*/2, /*key=*/101, true)};
    EXPECT_CALL(*client_wrapper,
                match(_, _, kv_cache_manager::QueryType::QT_PREFIX_MATCH, std::vector<int64_t>{101}, _, _))
        .WillOnce(Invoke([locations](const std::string&,
                                     const std::string&,
                                     kv_cache_manager::QueryType,
                                     const std::vector<int64_t>&,
                                     const kv_cache_manager::BlockMask& block_mask,
                                     const kv_cache_manager::ForwardContext&) {
            const auto* offset = std::get_if<kv_cache_manager::BlockMaskOffset>(&block_mask);
            EXPECT_NE(offset, nullptr);
            if (offset != nullptr) {
                EXPECT_EQ(*offset, 0u);
            }
            return std::make_pair(true, locations);
        }));
    auto request = makeGroupedStorageRequest(environment, {101}, /*local_matched_blocks=*/0, block_ids);
    auto result  = match(*backend.backend, request);
    ASSERT_TRUE(result.success);
    ASSERT_EQ(result.matched_blocks_num, 1u);
    ASSERT_NE(result.match_meta, nullptr);

    const auto uris  = flattenUris(locations);
    const auto bases = expectedBases(environment, block_ids, {{0, 1, 2}});
    EXPECT_CALL(*client_wrapper, loadKvCaches(uris, _, _))
        .WillOnce(Invoke([bases](const auto&, auto& buffers, const auto&) {
            expectBufferBases(buffers, bases);
            return true;
        }));
    EXPECT_TRUE(read(*backend.backend, std::move(request), std::move(result.match_meta)));
}

TEST(KVCMMockFullLinearTest, ReadsOnlyThroughNewestCompleteLinearStateAfterLocalReuse) {
    auto environment = makeMultiGroupBackendEnvironment(
        "kvcm_storage_backend_full_linear_read_local", /*full_group_count=*/1, /*linear_group_count=*/2);
    auto client_wrapper = std::make_shared<MockClientWrapper>();
    EXPECT_CALL(*client_wrapper, init(_, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), client_wrapper);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));

    ScopedReferencedBlocks      source_blocks(environment.device_pool, 4);
    const auto&                 block_ids = source_blocks.get();
    kv_cache_manager::Locations locations{
        makeFullLinearLocation(/*full_group_count=*/1, /*linear_group_count=*/2, /*key=*/102, false),
        makeFullLinearLocation(/*full_group_count=*/1, /*linear_group_count=*/2, /*key=*/103, true),
        makeFullLinearLocation(/*full_group_count=*/1, /*linear_group_count=*/2, /*key=*/104, false),
    };
    EXPECT_CALL(
        *client_wrapper,
        match(_, _, kv_cache_manager::QueryType::QT_PREFIX_MATCH, std::vector<int64_t>({101, 102, 103, 104}), _, _))
        .WillOnce(Invoke([locations](const std::string&,
                                     const std::string&,
                                     kv_cache_manager::QueryType,
                                     const std::vector<int64_t>&,
                                     const kv_cache_manager::BlockMask& block_mask,
                                     const kv_cache_manager::ForwardContext&) {
            const auto* offset = std::get_if<kv_cache_manager::BlockMaskOffset>(&block_mask);
            EXPECT_NE(offset, nullptr);
            if (offset != nullptr) {
                EXPECT_EQ(*offset, 1u);
            }
            return std::make_pair(true, locations);
        }));
    auto request = makeGroupedStorageRequest(environment, {101, 102, 103, 104}, /*local_matched_blocks=*/1, block_ids);
    auto result  = match(*backend.backend, request);
    ASSERT_TRUE(result.success);
    ASSERT_EQ(result.matched_blocks_num, 3u);
    ASSERT_NE(result.match_meta, nullptr);

    const CacheKeysType matched_keys(request.keys->begin(), request.keys->begin() + result.matched_blocks_num);
    request.keys = std::make_shared<const CacheKeysType>(matched_keys);
    request.handles.resize(result.matched_blocks_num);
    request.handles.front().clear();

    const kv_cache_manager::Locations selected_locations{locations[0], locations[1]};
    const auto                        uris = flattenUris(selected_locations);
    const auto bases                       = expectedBases(environment, {block_ids[1], block_ids[2]}, {{0}, {0, 1, 2}});
    EXPECT_CALL(*client_wrapper, loadKvCaches(uris, _, _))
        .WillOnce(Invoke([bases](const auto&, auto& buffers, const auto&) {
            expectBufferBases(buffers, bases);
            return true;
        }));
    EXPECT_TRUE(read(*backend.backend, std::move(request), std::move(result.match_meta)));
}

TEST(KVCMMockFullLinearTest, FullLinearWriteRoutesEachGroupToItsOwnLayerBuffers) {
    auto environment    = makeHybridBackendEnvironment("kvcm_storage_backend_full_linear_write");
    auto client_wrapper = std::make_shared<MockClientWrapper>();

    EXPECT_CALL(*client_wrapper, init(_, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), client_wrapper);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));

    std::vector<void*> expected_full_bases;
    std::vector<void*> expected_linear_bases;
    for (const int layer_id : environment.cache_config.topology().groupById(1).layer_ids) {
        const auto block_info = environment.device_pool->convertIndexToBuffer(layer_id, environment.block_id);
        ASSERT_EQ(block_info.size(), 1u);
        expected_full_bases.push_back(block_info.front().addr);
    }
    for (const int layer_id : environment.cache_config.topology().groupById(0).layer_ids) {
        const auto block_info = environment.device_pool->convertIndexToBuffer(layer_id, environment.block_id);
        ASSERT_EQ(block_info.size(), 1u);
        expected_linear_bases.push_back(block_info.front().addr);
    }

    kv_cache_manager::WriteLocation write_location;
    write_location.write_session_id = "full_linear_session";
    write_location.block_mask       = kv_cache_manager::BlockMaskOffset{0};
    write_location.locations        = {kv_cache_manager::Location{
        kv_cache_manager::LocationSpecUnit{"tp0_Ffull1", "full_uri"},
        kv_cache_manager::LocationSpecUnit{"tp0_Llinear", "linear_uri"},
    }};
    EXPECT_CALL(*client_wrapper, getWriteLocation(_, _, std::vector<int64_t>{101}, _, std::vector<std::string>{}, 600))
        .WillOnce(Return(std::make_pair(true, write_location)));
    EXPECT_CALL(*client_wrapper, saveKvCaches(kv_cache_manager::UriStrVec({"full_uri", "linear_uri"}), _, _))
        .WillOnce(Invoke(
            [expected_full_bases, expected_linear_bases](const kv_cache_manager::UriStrVec&,
                                                         const kv_cache_manager::BlockBuffers& buffers,
                                                         const std::shared_ptr<kv_cache_manager::TransferTraceInfo>&) {
                if (buffers.size() != 2u || buffers[0].iovs.size() != expected_full_bases.size()
                    || buffers[1].iovs.size() != expected_linear_bases.size()) {
                    ADD_FAILURE() << "KVCM full-linear write received an invalid block-buffer shape";
                    return std::make_pair(false, kv_cache_manager::UriStrVec{});
                }
                for (size_t index = 0; index < expected_full_bases.size(); ++index) {
                    EXPECT_EQ(buffers[0].iovs[index].base, expected_full_bases[index]);
                }
                for (size_t index = 0; index < expected_linear_bases.size(); ++index) {
                    EXPECT_EQ(buffers[1].iovs[index].base, expected_linear_bases[index]);
                }
                return std::make_pair(true, kv_cache_manager::UriStrVec{});
            }));
    EXPECT_CALL(*client_wrapper, finishWrite(_, _, "full_linear_session", _, _))
        .WillOnce(Invoke([](const std::string&,
                            const std::string&,
                            const std::string&,
                            const kv_cache_manager::BlockMask& block_mask,
                            const kv_cache_manager::Locations& locations) {
            const auto* offset = std::get_if<kv_cache_manager::BlockMaskOffset>(&block_mask);
            EXPECT_NE(offset, nullptr);
            if (offset != nullptr) {
                EXPECT_EQ(*offset, 1u);
            }
            EXPECT_TRUE(locations.empty());
            return true;
        }));

    StorageRequest request;
    request.keys    = std::make_shared<const CacheKeysType>(CacheKeysType{101});
    request.handles = {{{/*group_id=*/0, environment.block_id}, {/*group_id=*/1, environment.block_id}}};
    EXPECT_TRUE(backend->write(backend->prepareWrite(std::move(request)), /*synchronous=*/true));
    EXPECT_EQ(environment.device_pool->referencedBlocksNum(BlockTreeRefType::STORE), 0u);
}

TEST(KVCMMockFullLinearTest, TwoFullTwoLinearWritePreservesMaskOrderAndActualUris) {
    auto environment = makeMultiGroupBackendEnvironment(
        "kvcm_storage_backend_two_full_two_linear_write", /*full_group_count=*/2, /*linear_group_count=*/2);
    auto client_wrapper = std::make_shared<MockClientWrapper>();
    EXPECT_CALL(*client_wrapper, init(_, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), client_wrapper);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));

    ScopedReferencedBlocks                 source_blocks(environment.device_pool, 3);
    const auto&                            block_ids = source_blocks.get();
    const std::vector<std::vector<size_t>> groups_by_key{{0, 1, 2, 3}, {0, 1}, {0, 1, 2, 3}};
    const std::vector<std::string>         expected_write_groups{
        "Ffull0Ffull1Llinear0Llinear1", "Ffull0Ffull1", "Ffull0Ffull1Llinear0Llinear1"};

    kv_cache_manager::WriteLocation write_location;
    write_location.write_session_id = "two_full_two_linear_session";
    write_location.block_mask       = kv_cache_manager::BlockMaskOffset{1};
    write_location.locations        = {
        makeFullLinearLocation(/*full_group_count=*/2, /*linear_group_count=*/2, /*key=*/102, false),
        makeFullLinearLocation(/*full_group_count=*/2, /*linear_group_count=*/2, /*key=*/103, true),
    };
    EXPECT_CALL(*client_wrapper,
                getWriteLocation(
                    _, _, std::vector<int64_t>({101, 102, 103}), std::vector<int64_t>{}, expected_write_groups, 600))
        .WillOnce(Return(std::make_pair(true, write_location)));

    const auto expected_uris = flattenUris(write_location.locations);
    const auto bases         = expectedBases(environment, {block_ids[1], block_ids[2]}, {{0, 1}, {0, 1, 2, 3}});
    kv_cache_manager::UriStrVec actual_uris;
    actual_uris.reserve(expected_uris.size());
    for (size_t index = 0; index < expected_uris.size(); ++index) {
        actual_uris.push_back("actual_" + std::to_string(index));
    }
    EXPECT_CALL(*client_wrapper, saveKvCaches(expected_uris, _, _))
        .WillOnce(Invoke([bases, actual_uris](const auto&, const auto& buffers, const auto&) {
            expectBufferBases(buffers, bases);
            return std::make_pair(true, actual_uris);
        }));
    EXPECT_CALL(*client_wrapper, finishWrite(_, _, "two_full_two_linear_session", _, _))
        .WillOnce(Invoke([actual_uris](const std::string&,
                                       const std::string&,
                                       const std::string&,
                                       const kv_cache_manager::BlockMask& block_mask,
                                       const kv_cache_manager::Locations& locations) {
            const auto* offset = std::get_if<kv_cache_manager::BlockMaskOffset>(&block_mask);
            EXPECT_NE(offset, nullptr);
            if (offset != nullptr) {
                EXPECT_EQ(*offset, 2u);
            }
            EXPECT_EQ(flattenUris(locations), actual_uris);
            return true;
        }));

    auto request =
        makeGroupedStorageRequest(environment, {101, 102, 103}, /*local_matched_blocks=*/0, block_ids, groups_by_key);
    EXPECT_TRUE(backend->write(backend->prepareWrite(std::move(request)), /*synchronous=*/true));
    EXPECT_EQ(environment.device_pool->referencedBlocksNum(BlockTreeRefType::STORE), 0u);
}

TEST(KVCMMockFullLinearTest, AllMissingLinearGroupsWriteOnlyFullPayloads) {
    auto environment = makeMultiGroupBackendEnvironment(
        "kvcm_storage_backend_all_missing_linear", /*full_group_count=*/1, /*linear_group_count=*/2);
    auto client_wrapper = std::make_shared<MockClientWrapper>();
    EXPECT_CALL(*client_wrapper, init(_, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), client_wrapper);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));

    ScopedReferencedBlocks                 source_blocks(environment.device_pool, 3);
    const auto&                            block_ids = source_blocks.get();
    const std::vector<std::vector<size_t>> groups_by_key{{0}, {0}, {0}};
    kv_cache_manager::WriteLocation        write_location;
    write_location.write_session_id = "full_only_session";
    write_location.block_mask       = kv_cache_manager::BlockMaskOffset{0};
    write_location.locations        = {
        makeFullLinearLocation(1, 2, 101, false),
        makeFullLinearLocation(1, 2, 102, false),
        makeFullLinearLocation(1, 2, 103, false),
    };
    EXPECT_CALL(*client_wrapper,
                getWriteLocation(_, _, _, _, std::vector<std::string>({"Ffull0", "Ffull0", "Ffull0"}), 600))
        .WillOnce(Return(std::make_pair(true, write_location)));
    const auto uris  = flattenUris(write_location.locations);
    const auto bases = expectedBases(environment, block_ids, groups_by_key);
    EXPECT_CALL(*client_wrapper, saveKvCaches(uris, _, _))
        .WillOnce(Invoke([bases](const auto&, const auto& buffers, const auto&) {
            expectBufferBases(buffers, bases);
            return std::make_pair(true, kv_cache_manager::UriStrVec{});
        }));
    EXPECT_CALL(*client_wrapper, finishWrite(_, _, "full_only_session", _, _))
        .WillOnce(Invoke([](const std::string&,
                            const std::string&,
                            const std::string&,
                            const kv_cache_manager::BlockMask& block_mask,
                            const kv_cache_manager::Locations& locations) {
            const auto* offset = std::get_if<kv_cache_manager::BlockMaskOffset>(&block_mask);
            EXPECT_NE(offset, nullptr);
            if (offset != nullptr) {
                EXPECT_EQ(*offset, 3u);
            }
            EXPECT_TRUE(locations.empty());
            return true;
        }));

    auto request =
        makeGroupedStorageRequest(environment, {101, 102, 103}, /*local_matched_blocks=*/0, block_ids, groups_by_key);
    EXPECT_TRUE(backend->write(backend->prepareWrite(std::move(request)), /*synchronous=*/true));
}

TEST(KVCMMockFullLinearTest, IncompleteLinearGroupSetFailsBeforeClientIO) {
    auto environment = makeMultiGroupBackendEnvironment(
        "kvcm_storage_backend_incomplete_linear", /*full_group_count=*/1, /*linear_group_count=*/2);
    auto client_wrapper = std::make_shared<MockClientWrapper>();
    EXPECT_CALL(*client_wrapper, init(_, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    EXPECT_CALL(*client_wrapper, getWriteLocation(_, _, _, _, _, _)).Times(0);
    EXPECT_CALL(*client_wrapper, saveKvCaches(_, _, _)).Times(0);
    EXPECT_CALL(*client_wrapper, finishWrite(_, _, _, _, _)).Times(0);
    auto backend = makeBackend(environment, singleRankConfig(), client_wrapper);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));

    ScopedReferencedBlocks source_blocks(environment.device_pool, 1);
    const auto&            block_ids = source_blocks.get();
    auto                   request   = makeGroupedStorageRequest(
        environment, {101}, /*local_matched_blocks=*/0, block_ids, /*groups_by_key=*/{{0, 1}});
    EXPECT_FALSE(backend->write(backend->prepareWrite(std::move(request)), /*synchronous=*/true));
    EXPECT_EQ(environment.device_pool->referencedBlocksNum(BlockTreeRefType::STORE), 0u);
}

}  // namespace
}  // namespace rtp_llm
