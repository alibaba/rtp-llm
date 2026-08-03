#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache/KVCacheSpecDesc.h"
#include "rtp_llm/cpp/cache/connector/Meta.h"
#include "rtp_llm/cpp/cache/connector/remote_connector/test/RemoteConnectorMockTestBase.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"

#include <stdio.h>
#include <string_view>

using namespace kv_cache_manager;
using namespace ::testing;
using namespace rtp_llm;
using namespace rtp_llm::remote_connector;

namespace rtp_llm {
namespace test {
namespace {

KVCacheSpecPtr makeTestMhaSpec(const std::string& tag, uint32_t seq_size_per_block) {
    AttentionConfigs attn_config;
    attn_config.kv_head_num      = 8;
    attn_config.size_per_head    = 128;
    attn_config.tokens_per_block = seq_size_per_block;

    ParallelismConfig parallelism_config;
    parallelism_config.tp_size = 1;

    KVCacheSpecDesc desc;
    desc.tag        = tag;
    desc.cache_type = KVCacheSpecType::MultiHeadAttention;
    desc.dtype      = rtp_llm::DataType::TYPE_FP16;

    SpecBuildContext ctx;
    ctx.dtype              = rtp_llm::DataType::TYPE_FP16;
    ctx.seq_size_per_block = seq_size_per_block;
    ctx.attn_config        = &attn_config;
    ctx.parallelism_config = &parallelism_config;
    return SpecBuilder::build(desc, ctx);
}

void initializeResourceTopology(KVCacheResource&        resource,
                                const CacheConfig&      config,
                                std::string_view        tag,
                                const BlockIndicesType& blocks) {
    resource.initGroups(config.topologyPtr());
    ASSERT_EQ(resource.groupNums(), 1);
    resource.mutableBlockIds(tag).assign(blocks);
    CacheKeysType keys;
    keys.reserve(blocks.size());
    for (size_t i = 0; i < blocks.size(); ++i) {
        keys.push_back(static_cast<CacheKeyType>(i + 1));
    }
    resource.cacheKeys(tag)   = std::move(keys);
    const size_t         span = config.seqSizePerBlockForGroup(tag);
    std::vector<int32_t> tokens((blocks.size() + 1) * span, 1);
    resource.requestPrefix().rebuild(tokens.data(), tokens.size());
}

RequestPrefixMatchView requestMatchView(const KVCacheResourcePtr& resource) {
    return resource->requestPrefix().matchView();
}

CacheKeysType remoteMatchKeys(const KVCacheResourcePtr& resource) {
    const auto view      = requestMatchView(resource);
    const auto key_count = view.matchLimitTokens() / view.matchSpanTokens();
    return CacheKeysType(view.keys().begin(), view.keys().begin() + key_count);
}

}  // namespace
void waitAsyncContextDone(const std::shared_ptr<rtp_llm::AsyncContext>& ctx) {
    ASSERT_NE(ctx, nullptr);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (std::chrono::steady_clock::now() < deadline) {
        if (ctx->done()) {
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    FAIL() << "AsyncContext timeout waiting done()";
}

class MetaImpl: public Meta {
public:
    MetaImpl(bool enable_memory_cache, bool enable_remote_cache, std::string trace_id):
        enable_memory_cache_(enable_memory_cache), enable_remote_cache_(enable_remote_cache), trace_id_(trace_id) {}
    virtual ~MetaImpl() = default;

public:
    bool enableMemoryCache() const override {
        return enable_memory_cache_;
    }
    bool enableRemoteCache() const override {
        return enable_remote_cache_;
    }
    const std::string& trace_id() const override {
        return trace_id_;
    }
    const std::string& unique_id() const override {
        return unique_id_;
    }
    const std::vector<int64_t>& tokens() const override {
        return tokens_;
    }

protected:
    bool                 enable_memory_cache_{false};
    bool                 enable_remote_cache_{false};
    std::string          trace_id_;
    std::string          unique_id_ = "";
    std::vector<int64_t> tokens_;  // TODO : get tokens (remote connector)
};

class RemoteConnectorMockOnlyFullTest: public RemoteConnectorMockTestBase {
public:
    void SetUp() override {
        RemoteConnectorMockTestBase::SetUp();
        initConnector();
    }

    void TearDown() override {
        RemoteConnectorMockTestBase::TearDown();
    }

private:
    std::shared_ptr<AsyncContext> asyncReadBlocks(size_t                                    tp_rank,
                                                  const KVCacheResourcePtr&                 resource,
                                                  const std::shared_ptr<Meta>&              meta,
                                                  const std::shared_ptr<AsyncMatchContext>& match_context,
                                                  size_t                                    start_block,
                                                  size_t                                    block_count) {
        const size_t span = cache_config_.seqSizePerBlockForGroup("default");
        return remote_connectors_[tp_rank]->asyncRead(
            resource, meta, match_context, start_block * span, block_count * span);
    }

    void initConnector() {
        int block_num          = 10;
        int seq_size_per_block = 8;
        initCacheConfig(kFakeLayerNum, block_num, seq_size_per_block);
        for (int i = 0; i < tp_size_; i++) {
            auto meta_client = std::make_unique<kv_cache_manager::MockMetaClient>();
            meta_clients_.push_back(meta_client.get());
            EXPECT_CALL(*mock_client_factory_, CreateMetaClient(_, _))
                .WillOnce(Invoke(
                    [&](const std::string&, const kv_cache_manager::InitParams&) { return std::move(meta_client); }));
            auto allocator = std::make_shared<KVCacheAllocator>(cache_config_);
            ASSERT_TRUE(allocator->init());
            remote_connectors_.push_back(std::make_shared<RemoteConnector>(cache_config_,
                                                                           kv_cache_config_,
                                                                           runtime_config_,
                                                                           parallelism_config_,
                                                                           sp_config_,
                                                                           nullptr,
                                                                           0,
                                                                           allocator));
            ASSERT_TRUE(remote_connectors_[i]->init());
            servers_[i]->set_remote_connector(remote_connectors_[i]);
        }
    }

    void initCacheConfig(int layer_num = 4, int block_num = 10, int seq_size_per_block = 8) {
        cache_config_.layer_num     = layer_num;
        cache_config_.layer_all_num = layer_num;
        auto mha_spec               = makeTestMhaSpec("default", static_cast<uint32_t>(seq_size_per_block));
        cache_config_.dtype         = rtp_llm::DataType::TYPE_FP16;
        std::vector<int> layer_ids(layer_num);
        for (int i = 0; i < layer_num; ++i) {
            layer_ids[i] = i;
        }
        setTestTopology(cache_config_,
                        {makeTestGroupForConfig(mha_spec, std::move(layer_ids), CacheGroupType::FULL, "default")});
        const auto             topology_groups = cache_config_.topology().groups();
        std::vector<GroupBase> groups(topology_groups.begin(), topology_groups.end());
        groups[0].block_num             = static_cast<uint32_t>(block_num);
        groups[0].kv_block_stride_bytes = mha_spec->block_size_bytes();
        groups[0].kv_scale_stride_bytes = 0;
        cache_config_.setTopology(std::move(groups), cache_config_.topology().layers());
    }
};

class RemoteConnectorMockOnlyFullCPTest: public RemoteConnectorMockOnlyFullTest {
public:
    void SetUp() override {
        tp_size_                                               = 2;
        parallelism_config_.tp_size                            = 2;
        parallelism_config_.tp_rank                            = 0;
        parallelism_config_.prefill_cp_config.kv_cache_sharded = true;
        RemoteConnectorMockTestBase::SetUp();
        initConnector();
    }
};

TEST_F(RemoteConnectorMockOnlyFullCPTest, CanonicalMatchReadAndWriteUseLogicalSpan) {
    auto resource = std::make_shared<KVCacheResource>();
    resource->initGroups(cache_config_.topologyPtr());
    resource->mutableBlockIds("default").assign({1, 2, 3});
    std::vector<int32_t> tokens(/*six complete physical keys plus a partial tail=*/7 * 8, 1);
    resource->requestPrefix().rebuild(tokens.data(), tokens.size());
    const auto view = resource->requestPrefix().matchView();
    ASSERT_EQ(view.matchSpanTokens(), 8u);
    ASSERT_EQ(view.matchLimitTokens(), 48u);
    const CacheKeysType canonical_keys{view.keys()[1], view.keys()[3], view.keys()[5]};
    resource->setCacheKeys("default", canonical_keys);
    resource->setCacheKeysAreCpCanonical("default", true);
    resource->setLastBlockAligned("default", true);

    auto      meta               = std::make_shared<MetaImpl>(false, true, "cp_trace");
    Locations expected_locations = genFullotherLocations(canonical_keys);
    EXPECT_CALL(*meta_clients_[0],
                MatchLocation(Eq("match_cp_trace"), _, canonical_keys, _, Eq(BlockMask(static_cast<size_t>(0))), _, _))
        .WillOnce(Return(MatchLocationReturnType({ClientErrorCode::ER_OK, expected_locations})));

    auto match_context = remote_connectors_[0]->asyncMatch(view, meta);
    waitAsyncContextDone(match_context);
    ASSERT_TRUE(match_context->success());
    EXPECT_EQ(match_context->matchedTokenCount(), 48u);

    EXPECT_EQ(remote_connectors_[0]->asyncRead(resource, meta, match_context, /*start_token=*/8, /*token_count=*/16),
              nullptr);

    EXPECT_CALL(*transfer_client_, LoadKvCaches(_, _, _)).Times(2).WillRepeatedly(Return(ClientErrorCode::ER_OK));
    auto read_context =
        remote_connectors_[0]->asyncRead(resource, meta, match_context, /*start_token=*/0, /*token_count=*/48);
    waitAsyncContextDone(read_context);
    ASSERT_TRUE(read_context->success());
    EXPECT_EQ(resource->remoteReuseTokenNum(), 48u);

    const std::string write_session_id = "cp_write_session";
    WriteLocation     empty_write_location({write_session_id, static_cast<size_t>(3), {}});
    EXPECT_CALL(*meta_clients_[0], StartWrite(Eq("start_write_cp_trace"), canonical_keys, _, _, _))
        .WillOnce(Return(StartWriteReturnType({ClientErrorCode::ER_OK, empty_write_location})));
    EXPECT_CALL(*transfer_client_, SaveKvCaches(_, _, _)).Times(0);
    auto write_context = remote_connectors_[0]->asyncWrite(resource, meta);
    waitAsyncContextDone(write_context);
    EXPECT_TRUE(write_context->success());
}

TEST_F(RemoteConnectorMockOnlyFullTest, rejectsMultiGroupTopologyAtConstruction) {
    CacheConfig multi_group = cache_config_;
    auto        first_spec  = makeTestMhaSpec("first", 8);
    auto        second_spec = makeTestMhaSpec("second", 8);
    setTestTopology(multi_group,
                    {makeTestGroupForConfig(first_spec, {0, 1}, CacheGroupType::FULL, "first"),
                     makeTestGroupForConfig(second_spec, {2, 3}, CacheGroupType::FULL, "second")});

    EXPECT_ANY_THROW(std::make_shared<RemoteConnector>(
        multi_group, kv_cache_config_, runtime_config_, parallelism_config_, sp_config_, nullptr, 0, nullptr));
}

// 初始reuse_len = 0
TEST_F(RemoteConnectorMockOnlyFullTest, test_async_match_and_async_read_with_gpu_reuse_len_zero) {
    // match
    auto kv_cache_resouce = std::make_shared<KVCacheResource>();
    initializeResourceTopology(*kv_cache_resouce, cache_config_, "default", {1, 2, 3, 4});
    auto       meta               = std::make_shared<MetaImpl>(false, true, "trace_1");
    size_t     tp_rank            = 0;
    Locations  expected_locations = genFullotherLocations({1, 2, 3});
    const auto request_keys       = remoteMatchKeys(kv_cache_resouce);
    EXPECT_CALL(*meta_clients_[tp_rank],
                MatchLocation(Eq("match_trace_1"),                    // trace_id
                              _,                                      // query_type
                              request_keys,                           // keys
                              _,                                      // tokens
                              Eq(BlockMask(static_cast<size_t>(0))),  // block_mask
                              _,                                      // sw_size
                              _                                       // location_spec_names
                              ))
        .WillOnce(Return(MatchLocationReturnType({ClientErrorCode::ER_OK, expected_locations})));
    auto match_context = remote_connectors_[tp_rank]->asyncMatch(requestMatchView(kv_cache_resouce), meta);
    waitAsyncContextDone(match_context);
    ASSERT_TRUE(match_context->success());
    ASSERT_EQ((match_context->matchedTokenCount() / cache_config_.seqSizePerBlockForGroup("default")), 3);

    // read
    {
        // 没有其他connector
        UriStrVec          expected_uris        = genUris({1, 2, 3});
        BlockBuffersExpect block_buffers_expect = {
            3, kFakeLayerNum, cache_config_.specForGroup("default")->block_size_bytes()};
        std::vector<std::string> expect_block_ids({"1", "2", "3"});
        EXPECT_CALL(*transfer_client_,
                    LoadKvCaches(Eq(expected_uris),
                                 BlockBuffersMatcher(block_buffers_expect),
                                 TransferTraceInfoMatcher(expect_block_ids)))
            .WillOnce(Return(ClientErrorCode::ER_OK));

        const int gpu_reuse_num = static_cast<int>(kv_cache_resouce->reuseTokenNum() / 8);  // 0
        const int matched_num   = static_cast<int>(
            (match_context->matchedTokenCount() / cache_config_.seqSizePerBlockForGroup("default")));  // 3
        // auto      meta     = std::make_shared<TestReadMeta>(gpu_reuse_num, matched_num - gpu_reuse_num);
        int  start_read_block_index = gpu_reuse_num;
        int  read_block_num         = matched_num - gpu_reuse_num;
        auto read_context =
            asyncReadBlocks(tp_rank, kv_cache_resouce, meta, match_context, start_read_block_index, read_block_num);
        waitAsyncContextDone(read_context);
        ASSERT_TRUE(read_context->success());
        ASSERT_EQ(kv_cache_resouce->remoteReuseTokenNum() / 8, 3);
        ASSERT_EQ(kv_cache_resouce->reuseTokenNum() / 8, 3);

        kv_cache_resouce->setRemoteReuseTokenNum(0 * 8);
    }

    {
        // 其他connector也命中了部分
        UriStrVec          expected_uris        = genUris({2, 3});
        BlockBuffersExpect block_buffers_expect = {
            2, kFakeLayerNum, cache_config_.specForGroup("default")->block_size_bytes()};
        std::vector<std::string> expect_block_ids({"2", "3"});
        EXPECT_CALL(*transfer_client_,
                    LoadKvCaches(Eq(expected_uris),
                                 BlockBuffersMatcher(block_buffers_expect),
                                 TransferTraceInfoMatcher(expect_block_ids)))
            .WillOnce(Return(ClientErrorCode::ER_OK));

        const int gpu_reuse_num = static_cast<int>(kv_cache_resouce->reuseTokenNum() / 8);  // 0
        const int matched_num   = static_cast<int>(
            (match_context->matchedTokenCount() / cache_config_.seqSizePerBlockForGroup("default")));  // 3
        // auto      meta     = std::make_shared<TestReadMeta>(gpu_reuse_num + 1, matched_num - gpu_reuse_num - 1);
        int  start_read_block_index = gpu_reuse_num + 1;
        int  read_block_num         = matched_num - gpu_reuse_num - 1;
        auto read_context =
            asyncReadBlocks(tp_rank, kv_cache_resouce, meta, match_context, start_read_block_index, read_block_num);
        waitAsyncContextDone(read_context);
        ASSERT_TRUE(read_context->success());
        ASSERT_EQ(kv_cache_resouce->remoteReuseTokenNum() / 8, 2);
        kv_cache_resouce->setRemoteReuseTokenNum(0 * 8);
    }

    {
        // 其他connector也命中了部分,超出了remote
        const int matched_num = static_cast<int>(
            (match_context->matchedTokenCount() / cache_config_.seqSizePerBlockForGroup("default")));  // 3
        // auto      meta     = std::make_shared<TestReadMeta>(gpu_reuse_num + 4, matched_num - gpu_reuse_num - 4);
        int  start_read_block_index = matched_num;
        int  read_block_num         = 0;
        auto read_context =
            asyncReadBlocks(tp_rank, kv_cache_resouce, meta, match_context, start_read_block_index, read_block_num);
        ASSERT_EQ(read_context, nullptr);
        ASSERT_EQ(kv_cache_resouce->remoteReuseTokenNum() / 8, 0);
    }
}

// 初始reuse_len = 1
TEST_F(RemoteConnectorMockOnlyFullTest, test_async_match_and_async_read_with_gpu_reuse_len_not_zero) {
    // match
    auto kv_cache_resouce = std::make_shared<KVCacheResource>();
    initializeResourceTopology(*kv_cache_resouce, cache_config_, "default", {1, 2, 3, 4});
    kv_cache_resouce->setDeviceReuseTokenNum(1 * 8);
    auto       meta               = std::make_shared<MetaImpl>(false, true, "trace_1");
    size_t     tp_rank            = 0;
    Locations  expected_locations = genFullotherLocations({2, 3});
    const auto request_keys       = remoteMatchKeys(kv_cache_resouce);
    EXPECT_CALL(*meta_clients_[tp_rank],
                MatchLocation(Eq("match_trace_1"),                    // trace_id
                              _,                                      // query_type
                              request_keys,                           // keys
                              _,                                      // tokens
                              Eq(BlockMask(static_cast<size_t>(1))),  // block_mask
                              _,                                      // sw_size
                              _                                       // location_spec_names
                              ))
        .WillOnce(Return(MatchLocationReturnType({ClientErrorCode::ER_OK, expected_locations})));
    auto match_context = remote_connectors_[tp_rank]->asyncMatch(requestMatchView(kv_cache_resouce), meta);
    waitAsyncContextDone(match_context);
    ASSERT_TRUE(match_context->success());
    ASSERT_EQ((match_context->matchedTokenCount() / cache_config_.seqSizePerBlockForGroup("default")), 3);

    // read
    {
        // 没有其他connector
        UriStrVec          expected_uris        = genUris({2, 3});
        BlockBuffersExpect block_buffers_expect = {
            2, kFakeLayerNum, cache_config_.specForGroup("default")->block_size_bytes()};
        std::vector<std::string> expect_block_ids({"2", "3"});
        EXPECT_CALL(*transfer_client_,
                    LoadKvCaches(Eq(expected_uris),
                                 BlockBuffersMatcher(block_buffers_expect),
                                 TransferTraceInfoMatcher(expect_block_ids)))
            .WillOnce(Return(ClientErrorCode::ER_OK));

        const int gpu_reuse_num = static_cast<int>(kv_cache_resouce->reuseTokenNum() / 8);  // 1
        const int matched_num   = static_cast<int>(
            (match_context->matchedTokenCount() / cache_config_.seqSizePerBlockForGroup("default")));  // 3
        // auto      meta     = std::make_shared<TestReadMeta>(gpu_reuse_num, matched_num - gpu_reuse_num);
        int  start_read_block_index = gpu_reuse_num;
        int  read_block_num         = matched_num - gpu_reuse_num;
        auto read_context =
            asyncReadBlocks(tp_rank, kv_cache_resouce, meta, match_context, start_read_block_index, read_block_num);
        waitAsyncContextDone(read_context);
        ASSERT_TRUE(read_context->success());
        ASSERT_EQ(kv_cache_resouce->remoteReuseTokenNum() / 8, 2);
        kv_cache_resouce->setRemoteReuseTokenNum(0 * 8);
    }
    {
        // 有其他connector
        UriStrVec          expected_uris        = genUris({3});
        BlockBuffersExpect block_buffers_expect = {
            1, kFakeLayerNum, cache_config_.specForGroup("default")->block_size_bytes()};
        std::vector<std::string> expect_block_ids({"3"});
        EXPECT_CALL(*transfer_client_,
                    LoadKvCaches(Eq(expected_uris),
                                 BlockBuffersMatcher(block_buffers_expect),
                                 TransferTraceInfoMatcher(expect_block_ids)))
            .WillOnce(Return(ClientErrorCode::ER_OK));

        const int gpu_reuse_num   = static_cast<int>(kv_cache_resouce->reuseTokenNum() / 8);  // 1
        const int other_reuse_num = 1;                                                        // other connector
        const int matched_num     = static_cast<int>(
            (match_context->matchedTokenCount() / cache_config_.seqSizePerBlockForGroup("default")));  // 3
        // auto      meta       = std::make_shared<TestReadMeta>(gpu_reuse_num + other_reuse_num,
        //                                                 matched_num - gpu_reuse_num - other_reuse_num);
        int  start_read_block_index = gpu_reuse_num + other_reuse_num;
        int  read_block_num         = matched_num - gpu_reuse_num - other_reuse_num;
        auto read_context =
            asyncReadBlocks(tp_rank, kv_cache_resouce, meta, match_context, start_read_block_index, read_block_num);
        waitAsyncContextDone(read_context);
        ASSERT_TRUE(read_context->success());
        ASSERT_EQ(kv_cache_resouce->remoteReuseTokenNum() / 8, 1);
        kv_cache_resouce->setRemoteReuseTokenNum(0 * 8);
    }
    {
        // 有其他connector,覆盖了
        const int gpu_reuse_num   = static_cast<int>(kv_cache_resouce->reuseTokenNum() / 8);  // 1
        const int other_reuse_num = 2;                                                        // other connector
        const int matched_num     = static_cast<int>(
            (match_context->matchedTokenCount() / cache_config_.seqSizePerBlockForGroup("default")));  // 3
        // auto      meta       = std::make_shared<TestReadMeta>(gpu_reuse_num + other_reuse_num,
        //                                                 matched_num - gpu_reuse_num - other_reuse_num);
        int  start_read_block_index = gpu_reuse_num + other_reuse_num;
        int  read_block_num         = matched_num - gpu_reuse_num - other_reuse_num;
        auto read_context =
            asyncReadBlocks(tp_rank, kv_cache_resouce, meta, match_context, start_read_block_index, read_block_num);
        ASSERT_EQ(read_context, nullptr);
        ASSERT_EQ(kv_cache_resouce->remoteReuseTokenNum() / 8, 0);
    }
}

TEST_F(RemoteConnectorMockOnlyFullTest, test_write_success_broadcast_success_actual_locations_different) {
    auto kv_cache_resouce = std::make_shared<KVCacheResource>();
    initializeResourceTopology(*kv_cache_resouce, cache_config_, "default", {1, 2, 3});
    kv_cache_resouce->setLastBlockAligned("default", true);
    auto          meta    = std::make_shared<MetaImpl>(false, true, "trace_1");
    size_t        tp_rank = 0;
    std::string   write_session_id("write_session_id_1");
    Locations     expected_write_locations = genFullotherLocations({1, 2, 3});
    WriteLocation write_location({write_session_id, static_cast<size_t>(0), expected_write_locations});
    EXPECT_CALL(*meta_clients_[tp_rank],
                StartWrite(Eq("start_write_trace_1"),        // trace_id
                           std::vector<int64_t>({1, 2, 3}),  // keys
                           _,                                // tokens
                           Eq(std::vector<std::string>()),   // location_spec_group_names
                           _                                 // write_timeout_seconds
                           ))
        .WillOnce(Return(StartWriteReturnType({ClientErrorCode::ER_OK, write_location})));

    UriStrVec expected_uris = genUris({1, 2, 3});
    UriStrVec actual_uris   = genUris({1, 2, 3}, {}, "actual_");

    BlockBuffersExpect block_buffers_expect = {
        3, kFakeLayerNum, cache_config_.specForGroup("default")->block_size_bytes()};
    std::vector<std::string> expect_block_ids({"1", "2", "3"});
    EXPECT_CALL(*transfer_client_,
                SaveKvCaches(Eq(expected_uris),
                             BlockBuffersMatcher(block_buffers_expect),
                             TransferTraceInfoMatcher(expect_block_ids)))
        .WillOnce(Return(SaveKvCachesReturnType({ClientErrorCode::ER_OK, actual_uris})));

    Locations expected_actual_locations = genFullotherLocations({1, 2, 3}, {}, "actual_");
    EXPECT_CALL(*meta_clients_[tp_rank],
                FinishWrite(Eq("finish_write_trace_1"),             // trace_id
                            write_session_id,                       // write_session_id
                            Eq(BlockMask(static_cast<size_t>(3))),  // success_block
                            Eq(expected_actual_locations)           // locations
                            ))
        .WillOnce(Return(ClientErrorCode::ER_OK));

    auto async_context = remote_connectors_[tp_rank]->asyncWrite(kv_cache_resouce, meta);
    waitAsyncContextDone(async_context);
    ASSERT_TRUE(async_context->success());
}

TEST_F(RemoteConnectorMockOnlyFullTest,
       test_write_success_broadcast_success_actual_locations_different_with_block_mask) {
    auto kv_cache_resouce = std::make_shared<KVCacheResource>();
    initializeResourceTopology(*kv_cache_resouce, cache_config_, "default", {1, 2, 3});
    kv_cache_resouce->setLastBlockAligned("default", true);

    auto          meta    = std::make_shared<MetaImpl>(false, true, "trace_1");
    size_t        tp_rank = 0;
    std::string   write_session_id("write_session_id_1");
    Locations     expected_write_locations = genFullotherLocations({2, 3});
    WriteLocation write_location({write_session_id, static_cast<size_t>(1), expected_write_locations});
    EXPECT_CALL(*meta_clients_[tp_rank],
                StartWrite(Eq("start_write_trace_1"),        // trace_id
                           std::vector<int64_t>({1, 2, 3}),  // keys
                           _,                                // tokens
                           Eq(std::vector<std::string>()),   // location_spec_group_names
                           _                                 // write_timeout_seconds
                           ))
        .WillOnce(Return(StartWriteReturnType({ClientErrorCode::ER_OK, write_location})));

    UriStrVec expected_uris = genUris({2, 3});
    UriStrVec actual_uris   = genUris({2, 3}, {}, "actual_");

    BlockBuffersExpect block_buffers_expect = {
        2, kFakeLayerNum, cache_config_.specForGroup("default")->block_size_bytes()};
    std::vector<std::string> expect_block_ids({"2", "3"});
    EXPECT_CALL(*transfer_client_,
                SaveKvCaches(Eq(expected_uris),
                             BlockBuffersMatcher(block_buffers_expect),
                             TransferTraceInfoMatcher(expect_block_ids)))
        .WillOnce(Return(SaveKvCachesReturnType({ClientErrorCode::ER_OK, actual_uris})));

    Locations expected_actual_locations = genFullotherLocations({2, 3}, {}, "actual_");
    EXPECT_CALL(*meta_clients_[tp_rank],
                FinishWrite(Eq("finish_write_trace_1"),             // trace_id
                            write_session_id,                       // write_session_id
                            Eq(BlockMask(static_cast<size_t>(2))),  // success_block
                            Eq(expected_actual_locations)           // locations
                            ))
        .WillOnce(Return(ClientErrorCode::ER_OK));

    auto async_context = remote_connectors_[tp_rank]->asyncWrite(kv_cache_resouce, meta);
    waitAsyncContextDone(async_context);
    ASSERT_TRUE(async_context->success());
}

TEST_F(RemoteConnectorMockOnlyFullTest,
       test_write_success_broadcast_success_actual_locations_different_with_block_mask_vec) {
    auto kv_cache_resouce = std::make_shared<KVCacheResource>();
    initializeResourceTopology(*kv_cache_resouce, cache_config_, "default", {1, 2, 3, 4});
    kv_cache_resouce->setLastBlockAligned("default", true);

    auto          meta    = std::make_shared<MetaImpl>(false, true, "trace_1");
    size_t        tp_rank = 0;
    std::string   write_session_id("write_session_id_1");
    Locations     expected_write_locations = genFullotherLocations({2, 4});
    WriteLocation write_location(
        {write_session_id, std::vector<bool>({true, false, true, false}), expected_write_locations});
    EXPECT_CALL(*meta_clients_[tp_rank],
                StartWrite(Eq("start_write_trace_1"),           // trace_id
                           std::vector<int64_t>({1, 2, 3, 4}),  // keys
                           _,                                   // tokens
                           Eq(std::vector<std::string>()),      // location_spec_group_names
                           _                                    // write_timeout_seconds
                           ))
        .WillOnce(Return(StartWriteReturnType({ClientErrorCode::ER_OK, write_location})));

    UriStrVec expected_uris = genUris({2, 4});
    UriStrVec actual_uris   = genUris({2, 4}, {}, "actual_");

    BlockBuffersExpect block_buffers_expect = {
        2, kFakeLayerNum, cache_config_.specForGroup("default")->block_size_bytes()};
    std::vector<std::string> expect_block_ids({"2", "4"});
    EXPECT_CALL(*transfer_client_,
                SaveKvCaches(Eq(expected_uris),
                             BlockBuffersMatcher(block_buffers_expect),
                             TransferTraceInfoMatcher(expect_block_ids)))
        .WillOnce(Return(SaveKvCachesReturnType({ClientErrorCode::ER_OK, actual_uris})));

    Locations expected_actual_locations = genFullotherLocations({2, 4}, {}, "actual_");
    EXPECT_CALL(*meta_clients_[tp_rank],
                FinishWrite(Eq("finish_write_trace_1"),             // trace_id
                            write_session_id,                       // write_session_id
                            Eq(BlockMask(static_cast<size_t>(2))),  // success_block
                            Eq(expected_actual_locations)           // locations
                            ))
        .WillOnce(Return(ClientErrorCode::ER_OK));

    auto async_context = remote_connectors_[tp_rank]->asyncWrite(kv_cache_resouce, meta);
    waitAsyncContextDone(async_context);
    ASSERT_TRUE(async_context->success());
}

TEST_F(RemoteConnectorMockOnlyFullTest,
       test_write_success_broadcast_success_actual_locations_different_with_empty_write_locations) {
    auto kv_cache_resouce = std::make_shared<KVCacheResource>();
    initializeResourceTopology(*kv_cache_resouce, cache_config_, "default", {1, 2, 3});
    kv_cache_resouce->setLastBlockAligned("default", true);

    auto          meta    = std::make_shared<MetaImpl>(false, true, "trace_1");
    size_t        tp_rank = 0;
    std::string   write_session_id("write_session_id_1");
    Locations     expected_write_locations = genFullotherLocations({}, {});
    WriteLocation write_location({write_session_id, static_cast<size_t>(3), expected_write_locations});
    EXPECT_CALL(*meta_clients_[tp_rank],
                StartWrite(Eq("start_write_trace_1"),        // trace_id
                           std::vector<int64_t>({1, 2, 3}),  // keys
                           _,                                // tokens
                           Eq(std::vector<std::string>()),   // location_spec_group_names
                           _                                 // write_timeout_seconds
                           ))
        .WillOnce(Return(StartWriteReturnType({ClientErrorCode::ER_OK, write_location})));

    EXPECT_CALL(*transfer_client_, SaveKvCaches(_, _, _)).Times(0);
    EXPECT_CALL(*meta_clients_[tp_rank], FinishWrite(_, _, _, _)).Times(0);

    auto async_context = remote_connectors_[tp_rank]->asyncWrite(kv_cache_resouce, meta);
    waitAsyncContextDone(async_context);
    ASSERT_TRUE(async_context->success());
}

TEST_F(RemoteConnectorMockOnlyFullTest, test_write_success_broadcast_success_actual_locations_same) {
    auto kv_cache_resouce = std::make_shared<KVCacheResource>();
    initializeResourceTopology(*kv_cache_resouce, cache_config_, "default", {1, 2, 3});
    kv_cache_resouce->setLastBlockAligned("default", true);
    auto          meta    = std::make_shared<MetaImpl>(false, true, "trace_2");
    size_t        tp_rank = 0;
    std::string   write_session_id("write_session_id_2");
    Locations     expected_locations = genFullotherLocations({1, 2, 3});
    WriteLocation write_location({write_session_id, static_cast<size_t>(0), expected_locations});
    EXPECT_CALL(*meta_clients_[tp_rank],
                StartWrite(Eq("start_write_trace_2"),        // trace_id
                           std::vector<int64_t>({1, 2, 3}),  // keys
                           _,                                // tokens
                           Eq(std::vector<std::string>()),   // location_spec_group_names
                           _                                 // write_timeout_seconds
                           ))
        .WillOnce(Return(StartWriteReturnType({ClientErrorCode::ER_OK, write_location})));

    UriStrVec expected_uris = genUris({1, 2, 3});

    BlockBuffersExpect block_buffers_expect = {
        3, kFakeLayerNum, cache_config_.specForGroup("default")->block_size_bytes()};
    std::vector<std::string> expect_block_ids({"1", "2", "3"});
    EXPECT_CALL(*transfer_client_,
                SaveKvCaches(Eq(expected_uris),
                             BlockBuffersMatcher(block_buffers_expect),
                             TransferTraceInfoMatcher(expect_block_ids)))
        .WillOnce(Return(SaveKvCachesReturnType({ClientErrorCode::ER_OK, expected_uris})));

    EXPECT_CALL(*meta_clients_[tp_rank],
                FinishWrite(Eq("finish_write_trace_2"),             // trace_id
                            write_session_id,                       // write_session_id
                            Eq(BlockMask(static_cast<size_t>(3))),  // success_block
                            Eq(Locations({}))                       // locations
                            ))
        .WillOnce(Return(ClientErrorCode::ER_OK));

    auto async_context = remote_connectors_[tp_rank]->asyncWrite(kv_cache_resouce, meta);
    waitAsyncContextDone(async_context);
    ASSERT_TRUE(async_context->success());
}

TEST_F(RemoteConnectorMockOnlyFullTest, MatchClientFailureIsReported) {
    auto resource = std::make_shared<KVCacheResource>();
    initializeResourceTopology(*resource, cache_config_, "default", {1, 2, 3, 4});
    auto       meta         = std::make_shared<MetaImpl>(false, true, "trace_1");
    const auto request_keys = remoteMatchKeys(resource);
    EXPECT_CALL(*meta_clients_[0], MatchLocation(_, _, request_keys, _, _, _, _))
        .WillOnce(Return(MatchLocationReturnType({ClientErrorCode::ER_INVALID_GRPCSTATUS, {}})));
    EXPECT_CALL(*transfer_client_, LoadKvCaches(_, _, _)).Times(0);

    auto context = remote_connectors_[0]->asyncMatch(requestMatchView(resource), meta);
    waitAsyncContextDone(context);
    EXPECT_FALSE(context->success());
    EXPECT_EQ(std::dynamic_pointer_cast<RemoteAsyncMatchContext>(context)->state(),
              RemoteConnectorState::State::RCS_READ_MATCH_ERROR);
}

TEST_F(RemoteConnectorMockOnlyFullTest, StartWriteFailureIsReported) {
    auto resource = std::make_shared<KVCacheResource>();
    initializeResourceTopology(*resource, cache_config_, "default", {1, 2, 3});
    resource->setLastBlockAligned("default", true);
    auto meta = std::make_shared<MetaImpl>(false, true, "trace_1");
    EXPECT_CALL(*meta_clients_[0], StartWrite(_, std::vector<int64_t>({1, 2, 3}), _, _, _))
        .WillOnce(Return(StartWriteReturnType({ClientErrorCode::ER_INVALID_GRPCSTATUS, {}})));
    EXPECT_CALL(*transfer_client_, SaveKvCaches(_, _, _)).Times(0);
    EXPECT_CALL(*meta_clients_[0], FinishWrite(_, _, _, _)).Times(0);

    auto context = remote_connectors_[0]->asyncWrite(resource, meta);
    waitAsyncContextDone(context);
    EXPECT_FALSE(context->success());
}

TEST_F(RemoteConnectorMockOnlyFullTest, InvalidBlockIdRejectsWriteBeforeRemoteCalls) {
    auto resource = std::make_shared<KVCacheResource>();
    initializeResourceTopology(*resource, cache_config_, "default", {1, NULL_BLOCK_IDX, 3});
    resource->setLastBlockAligned("default", true);
    auto meta = std::make_shared<MetaImpl>(false, true, "trace_1");
    EXPECT_CALL(*meta_clients_[0], StartWrite(_, _, _, _, _)).Times(0);
    EXPECT_CALL(*transfer_client_, SaveKvCaches(_, _, _)).Times(0);
    EXPECT_CALL(*meta_clients_[0], FinishWrite(_, _, _, _)).Times(0);

    auto context = remote_connectors_[0]->asyncWrite(resource, meta);
    waitAsyncContextDone(context);
    EXPECT_FALSE(context->success());
}

TEST_F(RemoteConnectorMockOnlyFullTest, SaveFailureFinishesSessionAsFailed) {
    auto resource = std::make_shared<KVCacheResource>();
    initializeResourceTopology(*resource, cache_config_, "default", {1, 2, 3});
    resource->setLastBlockAligned("default", true);
    auto          meta = std::make_shared<MetaImpl>(false, true, "trace_2");
    std::string   session("write_session_id_2");
    auto          locations = genFullotherLocations({1, 2, 3});
    WriteLocation write_location({session, static_cast<size_t>(0), locations});
    EXPECT_CALL(*meta_clients_[0], StartWrite(_, std::vector<int64_t>({1, 2, 3}), _, _, _))
        .WillOnce(Return(StartWriteReturnType({ClientErrorCode::ER_OK, write_location})));
    EXPECT_CALL(*transfer_client_, SaveKvCaches(_, _, _))
        .WillOnce(Return(SaveKvCachesReturnType({ClientErrorCode::ER_SDK_TIMEOUT, {}})));
    EXPECT_CALL(*meta_clients_[0], FinishWrite(_, session, Eq(BlockMask(static_cast<size_t>(0))), Eq(Locations({}))))
        .WillOnce(Return(ClientErrorCode::ER_OK));

    auto context = remote_connectors_[0]->asyncWrite(resource, meta);
    waitAsyncContextDone(context);
    EXPECT_FALSE(context->success());
}

TEST_F(RemoteConnectorMockOnlyFullTest, FinishWriteFailureIsReported) {
    auto resource = std::make_shared<KVCacheResource>();
    initializeResourceTopology(*resource, cache_config_, "default", {1, 2, 3});
    resource->setLastBlockAligned("default", true);
    auto          meta = std::make_shared<MetaImpl>(false, true, "trace_2");
    std::string   session("write_session_id_2");
    auto          locations = genFullotherLocations({1, 2, 3});
    auto          uris      = genUris({1, 2, 3});
    WriteLocation write_location({session, static_cast<size_t>(0), locations});
    EXPECT_CALL(*meta_clients_[0], StartWrite(_, std::vector<int64_t>({1, 2, 3}), _, _, _))
        .WillOnce(Return(StartWriteReturnType({ClientErrorCode::ER_OK, write_location})));
    EXPECT_CALL(*transfer_client_, SaveKvCaches(Eq(uris), _, _))
        .WillOnce(Return(SaveKvCachesReturnType({ClientErrorCode::ER_OK, uris})));
    EXPECT_CALL(*meta_clients_[0], FinishWrite(_, session, Eq(BlockMask(static_cast<size_t>(3))), _))
        .WillOnce(Return(ClientErrorCode::ER_INVALID_GRPCSTATUS));

    auto context = remote_connectors_[0]->asyncWrite(resource, meta);
    waitAsyncContextDone(context);
    EXPECT_FALSE(context->success());
}

}  // namespace test
}  // namespace rtp_llm
