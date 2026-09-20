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

std::vector<void*> expectedBases(const BackendEnvironment&                    environment,
                                 const std::vector<BlockIdxType>&             block_ids,
                                 const std::vector<std::vector<std::string>>& groups_by_key) {
    RTP_LLM_CHECK(block_ids.size() == groups_by_key.size());
    std::vector<void*> bases;
    for (size_t key_idx = 0; key_idx < block_ids.size(); ++key_idx) {
        for (const auto& tag : groups_by_key[key_idx]) {
            bases.push_back(blockBase(environment, tag, block_ids[key_idx]));
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

void expectTaggedTransfers(MockClientWrapper&                 client,
                           const kv_cache_manager::Locations& locations,
                           const std::vector<void*>&          bases,
                           bool                               write,
                           const kv_cache_manager::UriStrVec& actual_uris = {}) {
    std::map<std::string, std::vector<size_t>> indices;
    const auto                                 uris  = flattenUris(locations);
    size_t                                     index = 0;
    for (const auto& location : locations) {
        for (const auto& spec : location) {
            const auto tag = spec.spec_name.substr(spec.spec_name.find('_') + 2);
            indices[tag].push_back(index++);
        }
    }
    ASSERT_EQ(index, bases.size());
    for (const auto& [tag, slots] : indices) {
        kv_cache_manager::UriStrVec batch_uris, batch_actual;
        std::vector<void*>          batch_bases;
        for (size_t slot : slots) {
            batch_uris.push_back(uris.at(slot));
            batch_bases.push_back(bases.at(slot));
            if (!actual_uris.empty()) {
                batch_actual.push_back(actual_uris.at(slot));
            }
        }
        if (write) {
            EXPECT_CALL(client, saveKvCachesForTag(tag, batch_uris, _, _))
                .WillOnce(
                    Invoke([batch_bases, batch_actual](const auto&, const auto&, const auto& buffers, const auto&) {
                        expectBufferBases(buffers, batch_bases);
                        return std::make_pair(true, batch_actual);
                    }));
        } else {
            EXPECT_CALL(client, loadKvCachesForTag(tag, batch_uris, _, _))
                .WillOnce(Invoke([batch_bases](const auto&, const auto&, auto& buffers, const auto&) {
                    expectBufferBases(buffers, batch_bases);
                    return true;
                }));
        }
    }
}

TEST(KVCMMockFullLinearTest, ReadsCompleteMultiGroupLocationWithoutLocalReuse) {
    auto environment = makeMultiGroupBackendEnvironment(
        "kvcm_storage_backend_full_linear_read_no_local", /*full_group_count=*/1, /*linear_group_count=*/2);
    auto client_wrapper = std::make_shared<MockClientWrapper>();
    EXPECT_CALL(*client_wrapper, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), client_wrapper);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));

    ScopedReferencedBlocks      source_blocks(environment, 1);
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

    const auto bases = expectedBases(environment, block_ids, {{"full0", "linear0", "linear1"}});
    expectTaggedTransfers(*client_wrapper, locations, bases, false);
    EXPECT_TRUE(read(*backend.backend, std::move(request), std::move(result.match_meta)));
}

TEST(KVCMMockFullLinearTest, ReadsOnlyThroughNewestCompleteLinearStateAfterLocalReuse) {
    auto environment = makeMultiGroupBackendEnvironment(
        "kvcm_storage_backend_full_linear_read_local", /*full_group_count=*/1, /*linear_group_count=*/2);
    auto client_wrapper = std::make_shared<MockClientWrapper>();
    EXPECT_CALL(*client_wrapper, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), client_wrapper);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));

    ScopedReferencedBlocks      source_blocks(environment, 4);
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
    const auto                        bases =
        expectedBases(environment, {block_ids[1], block_ids[2]}, {{"full0"}, {"full0", "linear0", "linear1"}});
    expectTaggedTransfers(*client_wrapper, selected_locations, bases, false);
    EXPECT_TRUE(read(*backend.backend, std::move(request), std::move(result.match_meta)));
}

TEST(KVCMMockFullLinearTest, FullLinearWriteRoutesEachGroupToItsOwnLayerBuffers) {
    auto environment    = makeHybridBackendEnvironment("kvcm_storage_backend_full_linear_write");
    auto client_wrapper = std::make_shared<MockClientWrapper>();

    EXPECT_CALL(*client_wrapper, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), client_wrapper);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));

    std::vector<void*> expected_full_bases;
    std::vector<void*> expected_linear_bases;
    for (const int layer_id : environment.cache_config.layerIdsForGroup("full1")) {
        const auto block_info = environmentBuffers(environment, layer_id, "full1", environment.block_id);
        ASSERT_EQ(block_info.size(), 1u);
        expected_full_bases.push_back(block_info.front().addr);
    }
    for (const int layer_id : environment.cache_config.layerIdsForGroup("linear")) {
        const auto block_info = environmentBuffers(environment, layer_id, "linear", environment.block_id);
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
    for (const auto& tag : std::vector<std::string>{"full1", "linear"}) {
        const auto                        bases = tag == "full1" ? expected_full_bases : expected_linear_bases;
        const kv_cache_manager::UriStrVec uris{tag == "full1" ? "full_uri" : "linear_uri"};
        EXPECT_CALL(*client_wrapper, saveKvCachesForTag(tag, uris, _, _))
            .WillOnce(Invoke([bases](const auto&, const auto&, const auto& buffers, const auto&) {
                EXPECT_EQ(buffers.size(), 1u);
                EXPECT_EQ(buffers.front().iovs.size(), bases.size());
                for (size_t i = 0; i < bases.size(); ++i) {
                    EXPECT_EQ(buffers.front().iovs.at(i).base, bases[i]);
                }
                return std::make_pair(true, kv_cache_manager::UriStrVec{});
            }));
    }
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
    request.handles = {{{environment.cache_config.groupTags().at(0), environment.block_id},
                        {environment.cache_config.groupTags().at(1), environment.block_id}}};
    backend->write(backend->prepareWrite(std::move(request)));
    ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
    for (const auto& [tag, pool] : environment.pools_by_tag) {
        EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::STORE), 0u);
        EXPECT_EQ(pool->refCount(environment.block_id), 1u);
    }
}

TEST(KVCMMockFullLinearTest, TwoFullTwoLinearWritePreservesMaskOrderAndActualUris) {
    auto environment = makeMultiGroupBackendEnvironment(
        "kvcm_storage_backend_two_full_two_linear_write", /*full_group_count=*/2, /*linear_group_count=*/2);
    auto client_wrapper = std::make_shared<MockClientWrapper>();
    EXPECT_CALL(*client_wrapper, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), client_wrapper);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));

    ScopedReferencedBlocks                      source_blocks(environment, 3);
    const auto&                                 block_ids = source_blocks.get();
    const std::vector<std::vector<std::string>> groups_by_key{
        {"full0", "full1", "linear0", "linear1"}, {"full0", "full1"}, {"full0", "full1", "linear0", "linear1"}};
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
    const auto bases         = expectedBases(
        environment, {block_ids[1], block_ids[2]}, {{"full0", "full1"}, {"full0", "full1", "linear0", "linear1"}});
    kv_cache_manager::UriStrVec actual_uris;
    actual_uris.reserve(expected_uris.size());
    for (size_t index = 0; index < expected_uris.size(); ++index) {
        actual_uris.push_back("actual_" + std::to_string(index));
    }
    expectTaggedTransfers(*client_wrapper, write_location.locations, bases, true, actual_uris);
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
    backend->write(backend->prepareWrite(std::move(request)));
    ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
    for (const auto& [tag, pool] : environment.pools_by_tag) {
        EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::STORE), 0u);
        EXPECT_EQ(pool->refCount(environment.block_id), 1u);
    }
}

TEST(KVCMMockFullLinearTest, AllMissingLinearGroupsWriteOnlyFullPayloads) {
    auto environment = makeMultiGroupBackendEnvironment(
        "kvcm_storage_backend_all_missing_linear", /*full_group_count=*/1, /*linear_group_count=*/2);
    auto client_wrapper = std::make_shared<MockClientWrapper>();
    EXPECT_CALL(*client_wrapper, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), client_wrapper);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));

    ScopedReferencedBlocks                      source_blocks(environment, 3);
    const auto&                                 block_ids = source_blocks.get();
    const std::vector<std::vector<std::string>> groups_by_key{{"full0"}, {"full0"}, {"full0"}};
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
    const auto bases = expectedBases(environment, block_ids, groups_by_key);
    expectTaggedTransfers(*client_wrapper, write_location.locations, bases, true);
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
    backend->write(backend->prepareWrite(std::move(request)));
    ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
}

TEST(KVCMMockFullLinearTest, IncompleteLinearGroupSetFailsBeforeClientIO) {
    auto environment = makeMultiGroupBackendEnvironment(
        "kvcm_storage_backend_incomplete_linear", /*full_group_count=*/1, /*linear_group_count=*/2);
    auto client_wrapper = std::make_shared<MockClientWrapper>();
    EXPECT_CALL(*client_wrapper, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    EXPECT_CALL(*client_wrapper, getWriteLocation(_, _, _, _, _, _)).Times(0);
    EXPECT_CALL(*client_wrapper, saveKvCachesForTag(_, _, _, _)).Times(0);
    EXPECT_CALL(*client_wrapper, finishWrite(_, _, _, _, _)).Times(0);
    auto backend = makeBackend(environment, singleRankConfig(), client_wrapper);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));

    ScopedReferencedBlocks source_blocks(environment, 1);
    const auto&            block_ids = source_blocks.get();
    auto                   request   = makeGroupedStorageRequest(
        environment, {101}, /*local_matched_blocks=*/0, block_ids, /*groups_by_key=*/{{"full0", "linear0"}});
    backend->write(backend->prepareWrite(std::move(request)));
    ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
    for (const auto& [tag, pool] : environment.pools_by_tag) {
        EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::STORE), 0u);
        EXPECT_EQ(pool->refCount(environment.block_id), 1u);
    }
}

}  // namespace
}  // namespace rtp_llm
