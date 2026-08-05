#include <gtest/gtest.h>

#include "rtp_llm/cpp/model_rpc/DecodeRpcServer.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/cache/OpaqueKVCacheSpec.h"
#include "rtp_llm/cpp/testing/TestLogCapture.h"

namespace rtp_llm {

namespace {

DecodeRpcServer::LoadKVCacheContext makeLoadContext(const std::string&               request_key,
                                                    const std::vector<std::string>&  peer_addrs,
                                                    const std::vector<CacheKeyType>& cache_keys,
                                                    const GroupBlockIds&             group_block_ids,
                                                    int32_t                          prefill_cp_size,
                                                    int64_t                          reuse_block_size = 0) {
    return {/*request_id=*/42,
            request_key,
            peer_addrs,
            cache_keys,
            group_block_ids,
            reuse_block_size,
            /*timeout_ms=*/1000,
            /*partition_count=*/1,
            /*partition_id=*/0,
            /*server_context=*/nullptr,
            prefill_cp_size};
}

GroupBase makeRpcGroup(std::string      tag,
                       std::vector<int> layer_ids,
                       uint32_t         physical_tokens = 8,
                       uint32_t         kernel_tokens   = 8,
                       CacheGroupType   group_type      = CacheGroupType::FULL,
                       CpBlockSliceMode cp_slice        = CpBlockSliceMode::NONE) {
    auto spec = std::make_shared<MHAKVCacheSpec>(physical_tokens, kernel_tokens);
    spec->tag = tag;

    GroupBase group;
    group.tag                       = std::move(tag);
    group.spec                      = std::move(spec);
    group.policy                    = defaultCacheGroupPolicy(group_type);
    group.policy.explicit_block_num = 8;
    group.policy.cp_slice           = cp_slice;
    group.layer_ids                 = std::move(layer_ids);
    return group;
}

CacheConfig makeRpcConfig(std::vector<GroupBase> groups, bool use_mla = false) {
    CacheConfig config;
    config.seq_size_per_block = 8;
    config.layer_num          = static_cast<uint32_t>(groups.size());
    config.use_mla            = use_mla;
    std::vector<LayerBase> layers;
    layers.reserve(groups.size());
    for (size_t layer_id = 0; layer_id < groups.size(); ++layer_id) {
        groups[layer_id].layer_ids = {static_cast<int>(layer_id)};
        layers.push_back({static_cast<int>(layer_id), {groups[layer_id].tag}});
    }
    config.setTopology(std::move(groups), std::move(layers));
    return config;
}

GroupBase makeOpaqueRpcGroup(std::string tag) {
    auto spec = std::make_shared<OpaqueKVCacheSpec>(8, 2);
    spec->tag = tag;

    GroupBase group;
    group.tag    = std::move(tag);
    group.spec   = std::move(spec);
    group.policy = defaultCacheGroupPolicy(CacheGroupType::FULL);
    return group;
}

GroupBlockIds makeRpcBlockIds(std::string tag, BlockIndicesType block_ids) {
    GroupBlockIds result;
    auto          holder = std::make_shared<BlockIds>();
    holder->assign(std::move(block_ids));
    result.emplace(std::move(tag), std::move(holder));
    return result;
}

}  // namespace

TEST(DecodeRpcServerTest, PageLevelRoutingRequiresExactCpPeerCount) {
    EXPECT_TRUE(DecodeRpcServer::isPageLevelRouting(/*prefill_cp_size=*/2, /*peer_addr_count=*/2));
    EXPECT_FALSE(DecodeRpcServer::isPageLevelRouting(/*prefill_cp_size=*/2, /*peer_addr_count=*/1));
    EXPECT_FALSE(DecodeRpcServer::isPageLevelRouting(/*prefill_cp_size=*/1, /*peer_addr_count=*/1));
}

TEST(DecodeRpcServerTest, PrefillCpPeerCountMismatchFailsLoad) {
    EXPECT_TRUE(DecodeRpcServer::validatePrefillCpPeerCount(/*prefill_cp_size=*/1, /*peer_addr_count=*/1).ok());
    EXPECT_TRUE(DecodeRpcServer::validatePrefillCpPeerCount(/*prefill_cp_size=*/2, /*peer_addr_count=*/2).ok());

    for (const size_t peer_addr_count : {1u, 3u}) {
        const auto error = DecodeRpcServer::validatePrefillCpPeerCount(/*prefill_cp_size=*/2, peer_addr_count);
        EXPECT_EQ(error.code(), ErrorCode::LOAD_KV_CACHE_FAILED);
        EXPECT_EQ(error.ToString(),
                  "prefill_cp_size 2 does not match peer addr count " + std::to_string(peer_addr_count));
    }
}

TEST(DecodeRpcServerTest, WholeBlockTransferCoversCacheTopologyAndPageRouting) {
    const auto single = makeRpcConfig({makeRpcGroup("full", {0})});
    EXPECT_FALSE(DecodeRpcServer::requiresWholeBlockTransfer(
        single, DecodeRpcServer::isPageLevelRouting(/*prefill_cp_size=*/2, /*peer_addr_count=*/1)));
    EXPECT_TRUE(DecodeRpcServer::requiresWholeBlockTransfer(
        single, DecodeRpcServer::isPageLevelRouting(/*prefill_cp_size=*/2, /*peer_addr_count=*/2)));

    const auto mla = makeRpcConfig({makeRpcGroup("full", {0})}, /*use_mla=*/true);
    EXPECT_TRUE(DecodeRpcServer::requiresWholeBlockTransfer(mla, /*page_level_routing=*/false));

    const auto opaque = makeRpcConfig({makeOpaqueRpcGroup("opaque")});
    EXPECT_TRUE(DecodeRpcServer::requiresWholeBlockTransfer(opaque, /*page_level_routing=*/false));

    const auto grouped = makeRpcConfig({makeRpcGroup("full", {0}), makeRpcGroup("second", {1})});
    EXPECT_TRUE(DecodeRpcServer::requiresWholeBlockTransfer(grouped, /*page_level_routing=*/false));
}

TEST(DecodeRpcServerTest, LoadPredicatesAreUnrestrictedWithoutPageLevelRouting) {
    const auto config =
        makeRpcConfig({makeRpcGroup("full", {0}), makeRpcGroup("linear", {1}, 8, 8, CacheGroupType::LINEAR)});

    for (int peer_idx = 0; peer_idx < 2; ++peer_idx) {
        for (const auto& [tag, group_type] : std::vector<std::pair<std::string, CacheGroupType>>{
                 {"full", CacheGroupType::FULL}, {"linear", CacheGroupType::LINEAR}}) {
            EXPECT_TRUE(DecodeRpcServer::shouldLoadGroupFromPeer(
                config, group_type, tag, peer_idx, /*page_level_routing=*/false, /*prefill_cp_size=*/2))
                << "tag=" << tag << " peer=" << peer_idx;
            EXPECT_TRUE(DecodeRpcServer::shouldLoadBlockFromPeer(
                group_type, /*block_pos=*/1, peer_idx, /*page_level_routing=*/false, /*prefill_cp_size=*/2))
                << "tag=" << tag << " peer=" << peer_idx;
        }
    }
}

TEST(DecodeRpcServerTest, PageLevelRoutingSplitsFullBlocksAcrossPeers) {
    const auto config = makeRpcConfig({makeRpcGroup("full", {0})});

    for (int peer_idx = 0; peer_idx < 2; ++peer_idx) {
        EXPECT_TRUE(DecodeRpcServer::shouldLoadGroupFromPeer(config,
                                                             CacheGroupType::FULL,
                                                             "full",
                                                             peer_idx,
                                                             /*page_level_routing=*/true,
                                                             /*prefill_cp_size=*/2));
        for (size_t block_pos = 0; block_pos < 4; ++block_pos) {
            const bool expected = static_cast<int>(block_pos % 2) == peer_idx;
            EXPECT_EQ(DecodeRpcServer::shouldLoadBlockFromPeer(CacheGroupType::FULL,
                                                               block_pos,
                                                               peer_idx,
                                                               /*page_level_routing=*/true,
                                                               /*prefill_cp_size=*/2),
                      expected)
                << "block_pos=" << block_pos << " peer=" << peer_idx;
        }
    }
}

TEST(DecodeRpcServerTest, PageLevelRoutingKeepsUnslicedLinearGroupOnFirstPeerOnly) {
    const auto config = makeRpcConfig({makeRpcGroup("linear", {0}, 8, 8, CacheGroupType::LINEAR)});
    ASSERT_FALSE(DecodeRpcServer::groupUsesCpSlice(config, "linear", /*prefill_cp_size=*/2));

    EXPECT_TRUE(DecodeRpcServer::shouldLoadGroupFromPeer(config,
                                                         CacheGroupType::LINEAR,
                                                         "linear",
                                                         /*peer_idx=*/0,
                                                         /*page_level_routing=*/true,
                                                         /*prefill_cp_size=*/2));
    EXPECT_FALSE(DecodeRpcServer::shouldLoadGroupFromPeer(config,
                                                          CacheGroupType::LINEAR,
                                                          "linear",
                                                          /*peer_idx=*/1,
                                                          /*page_level_routing=*/true,
                                                          /*prefill_cp_size=*/2));
    for (int peer_idx = 0; peer_idx < 2; ++peer_idx) {
        EXPECT_TRUE(DecodeRpcServer::shouldLoadBlockFromPeer(CacheGroupType::LINEAR,
                                                             /*block_pos=*/1,
                                                             peer_idx,
                                                             /*page_level_routing=*/true,
                                                             /*prefill_cp_size=*/2));
    }
}

TEST(DecodeRpcServerTest, PageLevelRoutingGathersCpSlicedGroupFromEveryPeer) {
    const auto config =
        makeRpcConfig({makeRpcGroup("state", {0}, 8, 8, CacheGroupType::LINEAR, CpBlockSliceMode::EQUAL_BYTES)});
    ASSERT_TRUE(DecodeRpcServer::groupUsesCpSlice(config, "state", /*prefill_cp_size=*/2));
    EXPECT_FALSE(DecodeRpcServer::groupUsesCpSlice(config, "state", /*prefill_cp_size=*/1));

    for (int peer_idx = 0; peer_idx < 2; ++peer_idx) {
        EXPECT_TRUE(DecodeRpcServer::shouldLoadGroupFromPeer(config,
                                                             CacheGroupType::LINEAR,
                                                             "state",
                                                             peer_idx,
                                                             /*page_level_routing=*/true,
                                                             /*prefill_cp_size=*/2))
            << "peer=" << peer_idx;
    }
}

TEST(ModelRpcProtoTest, GroupedCacheFieldsPreserveLegacyNumbers) {
    const auto* broadcast = BroadcastLoadRequestPB::descriptor();
    ASSERT_NE(broadcast, nullptr);
    EXPECT_TRUE(broadcast->IsReservedNumber(5));
    EXPECT_TRUE(broadcast->IsReservedNumber(12));
    EXPECT_EQ(broadcast->FindFieldByName("block_num")->number(), 6);
    EXPECT_EQ(broadcast->FindFieldByName("reuse_block_size")->number(), 7);
    EXPECT_EQ(broadcast->FindFieldByName("timeout_ms")->number(), 8);
    EXPECT_EQ(broadcast->FindFieldByName("dp_rank")->number(), 9);
    EXPECT_EQ(broadcast->FindFieldByName("partition_count")->number(), 10);
    EXPECT_EQ(broadcast->FindFieldByName("partition_id")->number(), 11);
    EXPECT_EQ(broadcast->FindFieldByName("prefill_cp_size")->number(), 13);
    EXPECT_EQ(broadcast->FindFieldByName("tagged_group_block_ids")->number(), 14);

    const auto* remote = RemoteOperationRequestPB::descriptor();
    ASSERT_NE(remote, nullptr);
    EXPECT_TRUE(remote->IsReservedNumber(3));
    EXPECT_EQ(remote->FindFieldByName("group_ids"), nullptr);
    EXPECT_EQ(remote->FindFieldByName("block_ids")->number(), 4);
    EXPECT_EQ(remote->FindFieldByName("uris")->number(), 5);
    EXPECT_EQ(remote->FindFieldByName("group_tags")->number(), 6);
    EXPECT_EQ(RemoteOperationResponsePB::descriptor()->FindFieldByName("actual_uris")->number(), 2);
}

TEST(DecodeRpcServerTest, CPShardedLoadRequestReadsFromEveryPrefillPeer) {
    DecodeRpcServer server;
    server.resource_.workers = {"decode-0", "decode-1"};

    const std::string               request_key     = "request";
    const std::vector<std::string>  peer_addrs      = {"prefill-0", "prefill-1"};
    const std::vector<CacheKeyType> cache_keys      = {101, 102};
    const GroupBlockIds             group_block_ids = makeRpcBlockIds("full", {7, 9});
    const auto                      load_context =
        makeLoadContext(request_key, peer_addrs, cache_keys, group_block_ids, /*cp_size=*/2, /*reuse=*/3);

    const auto request = server.constructRemoteLoadRequest(load_context, /*index=*/0, peer_addrs);

    EXPECT_EQ(request.prefill_cp_size(), 2);
    EXPECT_EQ(request.partition_count(), 1);
    EXPECT_EQ(request.partition_id(), 0);
    EXPECT_EQ(request.reuse_block_size(), 3);
    ASSERT_EQ(request.peer_addrs_size(), 2);
    EXPECT_EQ(request.peer_addrs(0), "prefill-0");
    EXPECT_EQ(request.peer_addrs(1), "prefill-1");
    ASSERT_EQ(request.cache_keys_size(), 2);
    EXPECT_EQ(request.cache_keys(0), 101);
    EXPECT_EQ(request.cache_keys(1), 102);
    ASSERT_EQ(request.tagged_group_block_ids_size(), 1);
    EXPECT_EQ(request.tagged_group_block_ids(0).tag(), "full");
    EXPECT_EQ(request.tagged_group_block_ids(0).block_ids_size(), 2);
}

TEST(DecodeRpcServerTest, CPShardedMlaLoadRequestReadsFromEveryPrefillPeer) {
    DecodeRpcServer server;
    server.resource_.workers = {"decode-0", "decode-1"};

    const std::string               request_key     = "request";
    const std::vector<std::string>  peer_addrs      = {"prefill-0", "prefill-1"};
    const std::vector<CacheKeyType> cache_keys      = {101};
    const GroupBlockIds             group_block_ids = makeRpcBlockIds("full", {7});
    const auto                      load_context =
        makeLoadContext(request_key, peer_addrs, cache_keys, group_block_ids, /*cp_size=*/2, /*reuse=*/3);

    const auto request = server.constructRemoteLoadRequestForMla(load_context, /*index=*/1, peer_addrs);

    EXPECT_EQ(request.prefill_cp_size(), 2);
    EXPECT_EQ(request.partition_count(), 1);
    EXPECT_EQ(request.partition_id(), 0);
    EXPECT_EQ(request.reuse_block_size(), 3);
    ASSERT_EQ(request.peer_addrs_size(), 2);
    EXPECT_EQ(request.peer_addrs(0), "prefill-0");
    EXPECT_EQ(request.peer_addrs(1), "prefill-1");
    ASSERT_EQ(request.tagged_group_block_ids_size(), 1);
    EXPECT_EQ(request.tagged_group_block_ids(0).tag(), "full");
}

TEST(DecodeRpcServerTest, LoadRequestBuildersShareCommonFields) {
    DecodeRpcServer server;
    server.resource_.workers                            = {"decode-0", "decode-1"};
    server.maga_init_params_.parallelism_config.dp_rank = 3;

    const std::string               request_key     = "shared-request";
    const std::vector<std::string>  peer_addrs      = {"prefill-0", "prefill-1"};
    const std::vector<CacheKeyType> cache_keys      = {101, 102};
    const GroupBlockIds             group_block_ids = makeRpcBlockIds("full", {7, 9});
    const auto                      load_context =
        makeLoadContext(request_key, peer_addrs, cache_keys, group_block_ids, /*cp_size=*/2, /*reuse=*/3);

    const auto split = server.constructRemoteLoadRequest(load_context, /*index=*/0, peer_addrs);
    const auto whole = server.constructRemoteLoadRequestForMla(load_context, /*index=*/0, peer_addrs);

    EXPECT_EQ(split.request_id(), whole.request_id());
    EXPECT_EQ(split.request_key(), whole.request_key());
    EXPECT_EQ(split.dp_rank(), whole.dp_rank());
    EXPECT_EQ(split.prefill_cp_size(), whole.prefill_cp_size());
    EXPECT_EQ(split.reuse_block_size(), whole.reuse_block_size());
    EXPECT_EQ(split.timeout_ms(), whole.timeout_ms());
    ASSERT_EQ(split.cache_keys_size(), whole.cache_keys_size());
    for (int i = 0; i < split.cache_keys_size(); ++i) {
        EXPECT_EQ(split.cache_keys(i), whole.cache_keys(i));
    }
    ASSERT_EQ(split.tagged_group_block_ids_size(), whole.tagged_group_block_ids_size());
    EXPECT_EQ(split.tagged_group_block_ids(0).SerializeAsString(), whole.tagged_group_block_ids(0).SerializeAsString());
}

TEST(DecodeRpcServerTest, WholeBlockLoadRequestMapsMoreDecodeRanksToPrefillPeers) {
    DecodeRpcServer server;
    server.resource_.workers = {"decode-0", "decode-1", "decode-2", "decode-3"};

    const std::vector<std::string>  peer_addrs      = {"prefill-0", "prefill-1"};
    const std::vector<CacheKeyType> cache_keys      = {101};
    const GroupBlockIds             group_block_ids = makeRpcBlockIds("full", {7});
    const auto load_context = makeLoadContext("request", peer_addrs, cache_keys, group_block_ids, /*cp_size=*/1);

    for (int index = 0; index < 4; ++index) {
        const auto request = server.constructRemoteLoadRequestForMla(load_context, index, peer_addrs);
        EXPECT_EQ(request.partition_count(), 1);
        EXPECT_EQ(request.partition_id(), 0);
        ASSERT_EQ(request.peer_addrs_size(), 1);
        EXPECT_EQ(request.peer_addrs(0), peer_addrs[index / 2]) << "decode index=" << index;
    }
}

TEST(DecodeRpcServerTest, WholeBlockLoadRequestMapsMorePrefillRanksToDecodeRanks) {
    DecodeRpcServer server;
    server.resource_.workers = {"decode-0", "decode-1"};

    const std::vector<std::string>  peer_addrs      = {"prefill-0", "prefill-1", "prefill-2", "prefill-3"};
    const std::vector<CacheKeyType> cache_keys      = {101};
    const GroupBlockIds             group_block_ids = makeRpcBlockIds("full", {7});
    const auto load_context = makeLoadContext("request", peer_addrs, cache_keys, group_block_ids, /*cp_size=*/1);

    for (int index = 0; index < 2; ++index) {
        const auto request = server.constructRemoteLoadRequestForMla(load_context, index, peer_addrs);
        EXPECT_EQ(request.partition_count(), 1);
        EXPECT_EQ(request.partition_id(), 0);
        ASSERT_EQ(request.peer_addrs_size(), 1);
        EXPECT_EQ(request.peer_addrs(0), peer_addrs[index * 2]) << "decode index=" << index;
    }
}

TEST(DecodeRpcServerTest, LoadRequestRejectsEmptyGroupBlockIds) {
    DecodeRpcServer server;
    server.resource_.workers                    = {"decode-0"};
    const std::string               request_key = "request";
    const std::vector<std::string>  peer_addrs  = {"prefill-0"};
    const std::vector<CacheKeyType> cache_keys  = {101};
    const GroupBlockIds             group_block_ids;
    const auto                      load_context =
        makeLoadContext(request_key, peer_addrs, cache_keys, group_block_ids, /*cp_size=*/1, /*reuse=*/0);

    try {
        (void)server.constructRemoteLoadRequest(load_context, /*index=*/0, peer_addrs);
        FAIL() << "empty group_block_ids must be rejected";
    } catch (const std::exception& e) {
        EXPECT_NE(std::string(e.what()).find("remote load request requires non-empty group_block_ids"),
                  std::string::npos);
    }
}

TEST(DecodeRpcServerTest, MlaLoadRequestRejectsEmptyGroupBlockIds) {
    DecodeRpcServer server;
    server.resource_.workers                    = {"decode-0"};
    const std::string               request_key = "request";
    const std::vector<std::string>  peer_addrs  = {"prefill-0"};
    const std::vector<CacheKeyType> cache_keys  = {101};
    const GroupBlockIds             group_block_ids;
    const auto                      load_context =
        makeLoadContext(request_key, peer_addrs, cache_keys, group_block_ids, /*cp_size=*/1, /*reuse=*/0);

    try {
        (void)server.constructRemoteLoadRequestForMla(load_context, /*index=*/0, peer_addrs);
        FAIL() << "empty MLA group_block_ids must be rejected";
    } catch (const std::exception& e) {
        EXPECT_NE(std::string(e.what()).find("remote load request requires non-empty group_block_ids"),
                  std::string::npos);
    }
}

TEST(DecodeRpcServerTest, ConstructAndDecodePreserveTaggedBlockGeometry) {
    DecodeRpcServer server;
    server.resource_.workers = {"decode-0"};

    const std::string               request_key     = "request";
    const std::vector<std::string>  peer_addrs      = {"prefill-0"};
    const std::vector<CacheKeyType> cache_keys      = {101, 102};
    const auto                      group_block_ids = makeRpcBlockIds("full", {7, 9});
    const auto                      load_context =
        makeLoadContext(request_key, peer_addrs, cache_keys, group_block_ids, /*cp_size=*/1, /*reuse=*/0);

    const auto request = server.constructRemoteLoadRequest(load_context, /*index=*/0, peer_addrs);
    ASSERT_EQ(request.tagged_group_block_ids_size(), 1);
    EXPECT_EQ(request.tagged_group_block_ids(0).tag(), "full");
    EXPECT_EQ(request.tagged_group_block_ids(0).block_ids(0), 7);
    EXPECT_EQ(request.tagged_group_block_ids(0).block_ids(1), 9);

    const auto topology = CacheTopology::create({makeRpcGroup("full", {0}, 8, 2)}, {{0, {"full"}}});
    const auto decoded  = DecodeRpcServer::decodeGroupBlockIds(request, *topology).at("full");
    EXPECT_EQ(decoded->blocks(), (BlockIndicesType{7, 9}));
    EXPECT_EQ(decoded->kernelBlocksPerKvBlock(), 4u);
    EXPECT_EQ(decoded->kernelBlocks(), (BlockIndicesType{28, 29, 30, 31, 36, 37, 38, 39}));
}

TEST(DecodeRpcServerTest, TaggedBlockRowsResolveByLocalTagOrder) {
    auto                   topology = CacheTopology::create({makeRpcGroup("linear", {0}), makeRpcGroup("full", {1})},
                                                            {{0, {"linear"}}, {1, {"full"}}});
    BroadcastLoadRequestPB request;
    auto*                  full = request.add_tagged_group_block_ids();
    full->set_tag("full");
    full->add_block_ids(10);
    auto* linear = request.add_tagged_group_block_ids();
    linear->set_tag("linear");
    linear->add_block_ids(20);

    const auto blocks = DecodeRpcServer::decodeGroupBlockIds(request, *topology);
    EXPECT_EQ(blocks.at("full")->blocks(), (BlockIndicesType{10}));
    EXPECT_EQ(blocks.at("linear")->blocks(), (BlockIndicesType{20}));

    auto       reordered        = CacheTopology::create({makeRpcGroup("full", {1}), makeRpcGroup("linear", {0})},
                                                        {{0, {"linear"}}, {1, {"full"}}});
    const auto reordered_blocks = DecodeRpcServer::decodeGroupBlockIds(request, *reordered);
    EXPECT_EQ(reordered_blocks.at("full")->blocks(), blocks.at("full")->blocks());
    EXPECT_EQ(reordered_blocks.at("linear")->blocks(), blocks.at("linear")->blocks());
    EXPECT_EQ(DecodeRpcServer::makeGroupRequestKey(42, 1, topology->group("full").tag),
              DecodeRpcServer::makeGroupRequestKey(42, 1, reordered->group("full").tag));
}

TEST(DecodeRpcServerTest, TaggedBlockRowsPreservePhysicalAndKernelGeometry) {
    auto                   topology = CacheTopology::create({makeRpcGroup("full", {0}, 8, 2)}, {{0, {"full"}}});
    BroadcastLoadRequestPB request;
    auto*                  row = request.add_tagged_group_block_ids();
    row->set_tag("full");
    row->add_block_ids(7);
    row->add_block_ids(9);

    const auto blocks = DecodeRpcServer::decodeGroupBlockIds(request, *topology).at("full");
    EXPECT_EQ(blocks->blocks(), (BlockIndicesType{7, 9}));
    EXPECT_EQ(blocks->kernelBlocksPerKvBlock(), 4u);
    EXPECT_EQ(blocks->kernelBlocks(), (BlockIndicesType{28, 29, 30, 31, 36, 37, 38, 39}));
}

TEST(DecodeRpcServerTest, EmptyTaggedBlockRowsAreRejected) {
    auto                   topology = CacheTopology::create({makeRpcGroup("full", {0})}, {{0, {"full"}}});
    BroadcastLoadRequestPB request;
    EXPECT_ANY_THROW(DecodeRpcServer::decodeGroupBlockIds(request, *topology));
}

TEST(DecodeRpcServerTest, TaggedBlockRowsRejectTopologyMismatch) {
    auto topology =
        CacheTopology::create({makeRpcGroup("full", {0}), makeRpcGroup("linear", {0})}, {{0, {"full", "linear"}}});
    BroadcastLoadRequestPB missing_tag;
    auto*                  row = missing_tag.add_tagged_group_block_ids();
    row->set_tag("full");
    row->add_block_ids(1);

    EXPECT_ANY_THROW(DecodeRpcServer::decodeGroupBlockIds(missing_tag, *topology));

    BroadcastLoadRequestPB unknown_tag;
    auto*                  unknown = unknown_tag.add_tagged_group_block_ids();
    unknown->set_tag("unknown");
    unknown->add_block_ids(1);
    EXPECT_ANY_THROW(DecodeRpcServer::decodeGroupBlockIds(unknown_tag, *topology));

    BroadcastLoadRequestPB duplicate_tag;
    for (int i = 0; i < 2; ++i) {
        auto* duplicate = duplicate_tag.add_tagged_group_block_ids();
        duplicate->set_tag("full");
        duplicate->add_block_ids(i + 1);
    }
    EXPECT_ANY_THROW(DecodeRpcServer::decodeGroupBlockIds(duplicate_tag, *topology));
}

TEST(DecodeRpcServerTest, MtpCacheKeyUsesSharedBaseModelIdForEverySlot) {
    constexpr size_t mtp_base_model_id = 17;

    for (size_t mtp_model_id = 0; mtp_model_id < 2; ++mtp_model_id) {
        EXPECT_EQ(DecodeRpcServer::makeMTPModuleCacheKey(mtp_base_model_id, "101", /*layer_id=*/0),
                  "model_id_17_token_id_str_101_layer_id_0")
            << "mtp_model_id=" << mtp_model_id;
    }
}

TEST(DecodeRpcServerTest, MtpGroupBlockIdsResolveSubsetByTag) {
    GroupBlockIds main_group_block_ids;
    const auto    full   = makeRpcBlockIds("full", {7, 9}).at("full");
    const auto    linear = makeRpcBlockIds("linear", {11}).at("linear");
    main_group_block_ids.emplace("full", full);
    main_group_block_ids.emplace("linear", linear);

    const auto selected = DecodeRpcServer::mtpGroupBlockIdsForTag(main_group_block_ids, "full");
    EXPECT_EQ(selected, full);
    EXPECT_NE(selected, linear);
    EXPECT_EQ(selected->blocks(), (BlockIndicesType{7, 9}));
}

TEST(DecodeRpcServerTest, MtpGroupBlockIdsRejectMissingAndNullTags) {
    GroupBlockIds main_group_block_ids;
    main_group_block_ids.emplace("null_mtp", nullptr);

    try {
        (void)DecodeRpcServer::mtpGroupBlockIdsForTag(main_group_block_ids, "unknown_mtp");
        FAIL() << "missing MTP tag must be rejected";
    } catch (const std::exception& e) {
        EXPECT_NE(std::string(e.what()).find("missing MTP RPC cache blocks for tag=unknown_mtp"), std::string::npos);
    }

    try {
        (void)DecodeRpcServer::mtpGroupBlockIdsForTag(main_group_block_ids, "null_mtp");
        FAIL() << "null MTP block ids must be rejected";
    } catch (const std::exception& e) {
        EXPECT_NE(std::string(e.what()).find("null MTP group_block for tag=null_mtp"), std::string::npos);
    }
}

TEST(DecodeRpcServerTest, MtpLoadPlanContainsOnlyModule0) {
    auto module0          = std::make_unique<EngineInitParams>();
    module0->model_id     = 17;
    auto module1          = std::make_unique<EngineInitParams>();
    module1->model_id     = 23;
    auto mtp_model_params = std::make_unique<std::vector<std::unique_ptr<EngineInitParams>>>();
    mtp_model_params->push_back(std::move(module0));
    mtp_model_params->push_back(std::move(module1));
    ProposeModelEngineInitParams propose_params(SP_TYPE_MTP, /*gen_num_per_cycle=*/2, std::move(mtp_model_params));

    const auto plan = DecodeRpcServer::makeMTPModuleLoadPlan(&propose_params);

    ASSERT_EQ(plan.size(), 1);
    EXPECT_EQ(plan[0].module_index, 0);
    EXPECT_EQ(plan[0].engine_init_params, propose_params.mtp_model_params_->at(0).get());
    EXPECT_EQ(plan[0].cache_model_id, 17);
}

TEST(DecodeRpcServerTest, MtpLoadPlanRejectsMissingModule0) {
    EXPECT_TRUE(DecodeRpcServer::makeMTPModuleLoadPlan(nullptr).empty());

    ProposeModelEngineInitParams missing_params;
    EXPECT_TRUE(DecodeRpcServer::makeMTPModuleLoadPlan(&missing_params).empty());

    auto                         empty_params = std::make_unique<std::vector<std::unique_ptr<EngineInitParams>>>();
    ProposeModelEngineInitParams no_modules(SP_TYPE_MTP, /*gen_num_per_cycle=*/2, std::move(empty_params));
    EXPECT_TRUE(DecodeRpcServer::makeMTPModuleLoadPlan(&no_modules).empty());

    auto mtp_model_params = std::make_unique<std::vector<std::unique_ptr<EngineInitParams>>>();
    mtp_model_params->push_back(nullptr);
    mtp_model_params->push_back(std::make_unique<EngineInitParams>());
    ProposeModelEngineInitParams null_module0(SP_TYPE_MTP, /*gen_num_per_cycle=*/2, std::move(mtp_model_params));
    EXPECT_TRUE(DecodeRpcServer::makeMTPModuleLoadPlan(&null_module0).empty());
}

TEST(DecodeRpcServerTest, MtpLoadPlanIgnoresInactiveModules) {
    auto mtp_model_params = std::make_unique<std::vector<std::unique_ptr<EngineInitParams>>>();
    mtp_model_params->push_back(std::make_unique<EngineInitParams>());
    mtp_model_params->push_back(nullptr);
    ProposeModelEngineInitParams propose_params(SP_TYPE_MTP, /*gen_num_per_cycle=*/2, std::move(mtp_model_params));

    const auto plan = DecodeRpcServer::makeMTPModuleLoadPlan(&propose_params);

    ASSERT_EQ(plan.size(), 1);
    EXPECT_EQ(plan[0].engine_init_params, propose_params.mtp_model_params_->at(0).get());
}

TEST(DecodeRpcServerTest, MtpLayerNumMismatchReportsModuleContext) {
    EXPECT_NO_THROW(DecodeRpcServer::validateMTPModuleLayerNum(/*engine_layer_num=*/3,
                                                               /*cache_layer_num=*/3,
                                                               /*module_index=*/0));
    try {
        DecodeRpcServer::validateMTPModuleLayerNum(/*engine_layer_num=*/3,
                                                   /*cache_layer_num=*/2,
                                                   /*module_index=*/7);
        FAIL() << "MTP layer mismatch must be rejected";
    } catch (const std::exception& e) {
        EXPECT_NE(std::string(e.what()).find("mtp layer_num mismatch: engine=3 cache_cfg=2 (mtp_model_id=7)"),
                  std::string::npos);
    }
}

TEST(DecodeRpcServerTest, ReadFailureLogContainsPeerErrorAndEveryBlockKey) {
    test::TestLogCapture log_capture("read_cache_failure");
    DecodeRpcServer::logReadFailures(/*request_id=*/42,
                                     "127.0.0.1:1:2",
                                     ErrorCode::CACHE_STORE_LOAD_CONNECT_FAILED,
                                     "connect failed",
                                     {"blocks={kv_key_0,kv_key_1}"});

    const auto log_content = log_capture.content();
    EXPECT_NE(log_content.find("PD_CACHE_KEY_READ_FAILED"), std::string::npos);
    EXPECT_NE(log_content.find("127.0.0.1:1:2"), std::string::npos);
    EXPECT_NE(log_content.find("kv_key_0"), std::string::npos);
    EXPECT_NE(log_content.find("kv_key_1"), std::string::npos);
}

TEST(DecodeRpcServerTest, ReadTimeoutLogsKeysAndCancellationIsSilent) {
    test::TestLogCapture log_capture("read_cache_timeout_cancel");
    DecodeRpcServer::logReadFailures(
        /*request_id=*/43, "peer", ErrorCode::LOAD_CACHE_TIMEOUT, "timeout", {"blocks={timeout_key}"});
    DecodeRpcServer::logReadFailures(
        /*request_id=*/44, "peer", ErrorCode::CANCELLED, "cancelled", {"blocks={cancelled_key}"});

    const auto log_content = log_capture.content();
    EXPECT_NE(log_content.find("timeout_key"), std::string::npos);
    EXPECT_EQ(log_content.find("cancelled_key"), std::string::npos);
}

}  // namespace rtp_llm
