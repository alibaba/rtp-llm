#include "rtp_llm/cpp/cache/CoordinatorCacheManager.h"
#include "rtp_llm/cpp/cache/KVCacheSpecDesc.h"
#include "rtp_llm/cpp/cache/connector/Meta.h"
#include "rtp_llm/cpp/cache/connector/remote_connector/test/RemoteConnectorMockTestBase.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/config/StaticConfig.h"

#ifdef USE_REMOTE_KV_CACHE
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/connector/KVCacheConnectorCoordinator.h"
#include "rtp_llm/cpp/cache/connector/remote_connector/GroupPolicy.h"
#endif

#include <stdio.h>

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
    return SpecBuilder::build(desc, ctx).spec;
}

void initializeResourceTopology(KVCacheResource& resource, const CacheConfig& config, BlockIndicesType blocks) {
    resource.initGroups(config);
    resource.mutableBlockIds("default").assign(std::move(blocks));
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

private:
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
            auto coordinator_cache_manager = std::make_shared<CoordinatorCacheManager>(cache_config_);
            ASSERT_TRUE(coordinator_cache_manager->init());
            remote_connectors_.push_back(std::make_shared<RemoteConnector>(cache_config_,
                                                                           kv_cache_config_,
                                                                           runtime_config_,
                                                                           parallelism_config_,
                                                                           sp_config_,
                                                                           nullptr,
                                                                           0,
                                                                           coordinator_cache_manager));
            ASSERT_TRUE(remote_connectors_[i]->init());
            servers_[i]->set_remote_connector(remote_connectors_[i]);
        }
    }

    void initCacheConfig(int layer_num = 4, int block_num = 10, int seq_size_per_block = 8) {
        cache_config_.block_num          = block_num;
        cache_config_.seq_size_per_block = seq_size_per_block;

        auto mha_spec       = makeTestMhaSpec("default", static_cast<uint32_t>(seq_size_per_block));
        cache_config_.dtype = rtp_llm::DataType::TYPE_FP16;
        std::vector<int> layer_ids(layer_num);
        for (int i = 0; i < layer_num; ++i) {
            layer_ids[i] = i;
        }
        rtp_llm::test::assignCacheConfigFromGroupedSpecs(cache_config_,
                                                         static_cast<uint32_t>(layer_num),
                                                         {mha_spec},
                                                         {layer_ids},
                                                         {CacheGroupType::FULL},
                                                         {"default"});
        cache_config_.finalizeBlockNums(static_cast<uint32_t>(block_num), runtime_config_);
    }
};

TEST_F(RemoteConnectorMockOnlyFullTest, SparseWireWritePreservesActualUriPositions) {
    RemoteOperationRequestPB request;
    request.set_op(::RemoteOpType::REMOTE_OPERATION_WRITE);
    request.set_trace_id("sparse_wire_write");
    for (const auto block_id : {0, 2, 0, 3}) {
        request.add_group_tags("default");
        request.add_block_ids(block_id);
    }
    for (const auto* uri : {"uri-0", "uri-1", "uri-2", "uri-3"}) {
        request.add_uris(uri);
    }

    const UriStrVec          expected_uris        = {"uri-1", "uri-3"};
    const UriStrVec          actual_uris          = {"actual-1", "actual-3"};
    const BlockBuffersExpect block_buffers_expect = {
        2, kFakeLayerNum, cache_config_.group("default").spec->block_size_bytes()};
    EXPECT_CALL(*transfer_client_, SaveKvCaches(Eq(expected_uris), BlockBuffersMatcher(block_buffers_expect), _))
        .WillOnce(Return(SaveKvCachesReturnType({ClientErrorCode::ER_OK, actual_uris})));

    RemoteOperationResponsePB response;
    ASSERT_TRUE(remote_connectors_[0]->copyCache(request, response));
    ASSERT_EQ(response.actual_uris_size(), 4);
    EXPECT_TRUE(response.actual_uris(0).empty());
    EXPECT_EQ(response.actual_uris(1), "actual-1");
    EXPECT_TRUE(response.actual_uris(2).empty());
    EXPECT_EQ(response.actual_uris(3), "actual-3");
}

TEST_F(RemoteConnectorMockOnlyFullTest, AllZeroWireWriteSucceedsWithEmptyPositionedUris) {
    RemoteOperationRequestPB request;
    request.set_op(::RemoteOpType::REMOTE_OPERATION_WRITE);
    request.set_trace_id("all_zero_wire_write");
    for (const auto* uri : {"uri-0", "uri-1"}) {
        request.add_group_tags("default");
        request.add_block_ids(0);
        request.add_uris(uri);
    }
    EXPECT_CALL(*transfer_client_, SaveKvCaches(_, _, _)).Times(0);

    RemoteOperationResponsePB response;
    ASSERT_TRUE(remote_connectors_[0]->copyCache(request, response));
    ASSERT_EQ(response.actual_uris_size(), 2);
    EXPECT_TRUE(response.actual_uris(0).empty());
    EXPECT_TRUE(response.actual_uris(1).empty());
}

#ifdef USE_REMOTE_KV_CACHE
TEST_F(RemoteConnectorMockOnlyFullTest, CoordinatorRegistersTheSoleCoordinatorAndBuildsFullPolicy) {
    CacheConfig hybrid_config = std::move(cache_config_);
    setGroupBlockLayout(hybrid_config,
                        {hybrid_config.block_num},
                        {hybrid_config.group("default").spec->block_size_bytes()},
                        {hybrid_config.group("default").spec->scale_block_size_bytes()});

    auto coordinator_cache_manager = std::make_shared<CoordinatorCacheManager>(hybrid_config, AllocationType::DEVICE);
    ASSERT_TRUE(coordinator_cache_manager->init());
    const auto sole_pool = coordinator_cache_manager->blockPool("default");
    ASSERT_NE(sole_pool, nullptr);

    auto meta_client = std::make_unique<kv_cache_manager::MockMetaClient>();
    meta_clients_.push_back(meta_client.get());
    EXPECT_CALL(*mock_client_factory_, CreateMetaClient(_, _))
        .WillOnce(
            Invoke([&](const std::string&, const kv_cache_manager::InitParams&) { return std::move(meta_client); }));

    KVCacheConfig coordinator_kv_config       = kv_cache_config_;
    coordinator_kv_config.reuse_cache         = true;
    coordinator_kv_config.enable_remote_cache = true;
    auto coordinator                          = std::make_shared<KVCacheConnectorCoordinator>(hybrid_config,
                                                                     coordinator_kv_config,
                                                                     runtime_config_,
                                                                     parallelism_config_,
                                                                     sp_config_,
                                                                     coordinator_cache_manager);

    ASSERT_TRUE(coordinator->init());
    ASSERT_NE(coordinator->remote_connector_, nullptr);
    EXPECT_EQ(coordinator->remote_connector_->init_params_->register_buffer_addr, sole_pool->getBaseAddress());
    EXPECT_EQ(coordinator->remote_connector_->init_params_->register_buffer_size, sole_pool->getTotalSizeBytes());
    EXPECT_NE(
        dynamic_cast<remote_connector::FullLayerGroupPolicy*>(coordinator->remote_connector_->group_policy_.get()),
        nullptr);

    coordinator->update_thread_->stop();
    coordinator->update_thread_.reset();
    coordinator.reset();
}

TEST_F(RemoteConnectorMockOnlyFullTest, ManagerRegistersOrdinarySingleFullCoordinatorWithRemoteConnector) {
    ASSERT_EQ(cache_config_.groupNums(), 1);

    auto meta_client = std::make_unique<kv_cache_manager::MockMetaClient>();
    meta_clients_.push_back(meta_client.get());
    EXPECT_CALL(*mock_client_factory_, CreateMetaClient(_, _))
        .WillOnce(
            Invoke([&](const std::string&, const kv_cache_manager::InitParams&) { return std::move(meta_client); }));

    KVCacheConfig manager_kv_config       = kv_cache_config_;
    manager_kv_config.reuse_cache         = true;
    manager_kv_config.enable_remote_cache = true;
    auto manager                          = std::make_shared<KVCacheManager>(std::move(cache_config_),
                                                    /*warmup=*/false,
                                                    /*metrics_reporter=*/nullptr,
                                                    manager_kv_config,
                                                    parallelism_config_,
                                                    runtime_config_,
                                                    sp_config_);

    ASSERT_TRUE(manager->init());
    auto coordinator_cache_manager = manager->coordinator_cache_manager_;
    ASSERT_NE(coordinator_cache_manager, nullptr);
    const auto sole_pool = coordinator_cache_manager->blockPool("default");
    ASSERT_NE(sole_pool, nullptr);

    auto coordinator = manager->connectorCoordinator();
    ASSERT_NE(coordinator, nullptr);
    ASSERT_NE(coordinator->remote_connector_, nullptr);
    EXPECT_EQ(coordinator->remote_connector_->init_params_->register_buffer_addr, sole_pool->getBaseAddress());
    EXPECT_EQ(coordinator->remote_connector_->init_params_->register_buffer_size, sole_pool->getTotalSizeBytes());
    EXPECT_NE(
        dynamic_cast<remote_connector::FullLayerGroupPolicy*>(coordinator->remote_connector_->group_policy_.get()),
        nullptr);
}
#endif

// 初始reuse_len = 0
TEST_F(RemoteConnectorMockOnlyFullTest, test_async_match_and_async_read_with_gpu_reuse_len_zero) {
    // match
    auto kv_cache_resouce = std::make_shared<KVCacheResource>();
    kv_cache_resouce->setCacheKeys({1, 2, 3, 4});
    initializeResourceTopology(*kv_cache_resouce, cache_config_, {1, 2, 3, 4});
    auto      meta               = std::make_shared<MetaImpl>(false, true, "trace_1");
    size_t    tp_rank            = 0;
    Locations expected_locations = genFullotherLocations({1, 2, 3});
    EXPECT_CALL(*meta_clients_[tp_rank],
                MatchLocation(Eq("match_trace_1"),                    // trace_id
                              _,                                      // query_type
                              std::vector<int64_t>({1, 2, 3}),        // keys
                              _,                                      // tokens
                              Eq(BlockMask(static_cast<size_t>(0))),  // block_mask
                              _,                                      // sw_size
                              _                                       // location_spec_names
                              ))
        .WillOnce(Return(MatchLocationReturnType({ClientErrorCode::ER_OK, expected_locations})));
    auto match_context = remote_connectors_[tp_rank]->asyncMatch(kv_cache_resouce, meta);
    waitAsyncContextDone(match_context);
    ASSERT_TRUE(match_context->success());
    ASSERT_EQ(match_context->matchedBlockCount(), 3);

    // read
    {
        // 没有其他connector
        UriStrVec          expected_uris        = genUris({1, 2, 3});
        BlockBuffersExpect block_buffers_expect = {
            3, kFakeLayerNum, cache_config_.group("default").spec->block_size_bytes()};
        std::vector<std::string> expect_block_ids({"1", "2", "3"});
        EXPECT_CALL(*transfer_client_,
                    LoadKvCaches(Eq(expected_uris),
                                 BlockBuffersMatcher(block_buffers_expect),
                                 TransferTraceInfoMatcher(expect_block_ids)))
            .WillOnce(Return(ClientErrorCode::ER_OK));

        const int gpu_reuse_num = static_cast<int>(kv_cache_resouce->reuseBlockNum());   // 0
        const int matched_num   = static_cast<int>(match_context->matchedBlockCount());  // 3
        // auto      meta     = std::make_shared<TestReadMeta>(gpu_reuse_num, matched_num - gpu_reuse_num);
        int  start_read_block_index = gpu_reuse_num;
        int  read_block_num         = matched_num - gpu_reuse_num;
        auto read_context           = remote_connectors_[tp_rank]->asyncRead(
            kv_cache_resouce, meta, match_context, start_read_block_index, read_block_num);
        waitAsyncContextDone(read_context);
        ASSERT_TRUE(read_context->success());
        ASSERT_EQ(kv_cache_resouce->remoteReuseBlockNum(), 3);
        ASSERT_EQ(kv_cache_resouce->reuseBlockNum(), 3);

        kv_cache_resouce->setRemoteReuseBlockNum(0);
    }

    {
        // 其他connector也命中了部分
        UriStrVec          expected_uris        = genUris({2, 3});
        BlockBuffersExpect block_buffers_expect = {
            2, kFakeLayerNum, cache_config_.group("default").spec->block_size_bytes()};
        std::vector<std::string> expect_block_ids({"2", "3"});
        EXPECT_CALL(*transfer_client_,
                    LoadKvCaches(Eq(expected_uris),
                                 BlockBuffersMatcher(block_buffers_expect),
                                 TransferTraceInfoMatcher(expect_block_ids)))
            .WillOnce(Return(ClientErrorCode::ER_OK));

        const int gpu_reuse_num = static_cast<int>(kv_cache_resouce->reuseBlockNum());   // 0
        const int matched_num   = static_cast<int>(match_context->matchedBlockCount());  // 3
        // auto      meta     = std::make_shared<TestReadMeta>(gpu_reuse_num + 1, matched_num - gpu_reuse_num - 1);
        int  start_read_block_index = gpu_reuse_num + 1;
        int  read_block_num         = matched_num - gpu_reuse_num - 1;
        auto read_context           = remote_connectors_[tp_rank]->asyncRead(
            kv_cache_resouce, meta, match_context, start_read_block_index, read_block_num);
        waitAsyncContextDone(read_context);
        ASSERT_TRUE(read_context->success());
        ASSERT_EQ(kv_cache_resouce->remoteReuseBlockNum(), 2);
        kv_cache_resouce->setRemoteReuseBlockNum(0);
    }

    {
        // 其他connector也命中了部分,超出了remote
        const int gpu_reuse_num = static_cast<int>(kv_cache_resouce->reuseBlockNum());   // 0
        const int matched_num   = static_cast<int>(match_context->matchedBlockCount());  // 3
        // auto      meta     = std::make_shared<TestReadMeta>(gpu_reuse_num + 4, matched_num - gpu_reuse_num - 4);
        int  start_read_block_index = gpu_reuse_num + 4;
        int  read_block_num         = matched_num - gpu_reuse_num - 4;
        auto read_context           = remote_connectors_[tp_rank]->asyncRead(
            kv_cache_resouce, meta, match_context, start_read_block_index, read_block_num);
        waitAsyncContextDone(read_context);
        ASSERT_TRUE(read_context->success());
        ASSERT_EQ(kv_cache_resouce->remoteReuseBlockNum(), 0);
    }
}

// 初始reuse_len = 1
TEST_F(RemoteConnectorMockOnlyFullTest, test_async_match_and_async_read_with_gpu_reuse_len_not_zero) {
    // match
    auto kv_cache_resouce = std::make_shared<KVCacheResource>();
    kv_cache_resouce->setCacheKeys({1, 2, 3, 4});
    initializeResourceTopology(*kv_cache_resouce, cache_config_, {1, 2, 3, 4});
    kv_cache_resouce->setDeviceReuseBlockNum(1);
    auto      meta               = std::make_shared<MetaImpl>(false, true, "trace_1");
    size_t    tp_rank            = 0;
    Locations expected_locations = genFullotherLocations({2, 3});
    EXPECT_CALL(*meta_clients_[tp_rank],
                MatchLocation(Eq("match_trace_1"),                    // trace_id
                              _,                                      // query_type
                              std::vector<int64_t>({1, 2, 3}),        // keys
                              _,                                      // tokens
                              Eq(BlockMask(static_cast<size_t>(1))),  // block_mask
                              _,                                      // sw_size
                              _                                       // location_spec_names
                              ))
        .WillOnce(Return(MatchLocationReturnType({ClientErrorCode::ER_OK, expected_locations})));
    auto match_context = remote_connectors_[tp_rank]->asyncMatch(kv_cache_resouce, meta);
    waitAsyncContextDone(match_context);
    ASSERT_TRUE(match_context->success());
    ASSERT_EQ(match_context->matchedBlockCount(), 3);

    // read
    {
        // 没有其他connector
        UriStrVec          expected_uris        = genUris({2, 3});
        BlockBuffersExpect block_buffers_expect = {
            2, kFakeLayerNum, cache_config_.group("default").spec->block_size_bytes()};
        std::vector<std::string> expect_block_ids({"2", "3"});
        EXPECT_CALL(*transfer_client_,
                    LoadKvCaches(Eq(expected_uris),
                                 BlockBuffersMatcher(block_buffers_expect),
                                 TransferTraceInfoMatcher(expect_block_ids)))
            .WillOnce(Return(ClientErrorCode::ER_OK));

        const int gpu_reuse_num = static_cast<int>(kv_cache_resouce->reuseBlockNum());   // 1
        const int matched_num   = static_cast<int>(match_context->matchedBlockCount());  // 3
        // auto      meta     = std::make_shared<TestReadMeta>(gpu_reuse_num, matched_num - gpu_reuse_num);
        int  start_read_block_index = gpu_reuse_num;
        int  read_block_num         = matched_num - gpu_reuse_num;
        auto read_context           = remote_connectors_[tp_rank]->asyncRead(
            kv_cache_resouce, meta, match_context, start_read_block_index, read_block_num);
        waitAsyncContextDone(read_context);
        ASSERT_TRUE(read_context->success());
        ASSERT_EQ(kv_cache_resouce->remoteReuseBlockNum(), 2);
        kv_cache_resouce->setRemoteReuseBlockNum(0);
    }
    {
        // 有其他connector
        UriStrVec          expected_uris        = genUris({3});
        BlockBuffersExpect block_buffers_expect = {
            1, kFakeLayerNum, cache_config_.group("default").spec->block_size_bytes()};
        std::vector<std::string> expect_block_ids({"3"});
        EXPECT_CALL(*transfer_client_,
                    LoadKvCaches(Eq(expected_uris),
                                 BlockBuffersMatcher(block_buffers_expect),
                                 TransferTraceInfoMatcher(expect_block_ids)))
            .WillOnce(Return(ClientErrorCode::ER_OK));

        const int gpu_reuse_num   = static_cast<int>(kv_cache_resouce->reuseBlockNum());   // 1
        const int other_reuse_num = 1;                                                     // other connector
        const int matched_num     = static_cast<int>(match_context->matchedBlockCount());  // 3
        // auto      meta       = std::make_shared<TestReadMeta>(gpu_reuse_num + other_reuse_num,
        //                                                 matched_num - gpu_reuse_num - other_reuse_num);
        int  start_read_block_index = gpu_reuse_num + other_reuse_num;
        int  read_block_num         = matched_num - gpu_reuse_num - other_reuse_num;
        auto read_context           = remote_connectors_[tp_rank]->asyncRead(
            kv_cache_resouce, meta, match_context, start_read_block_index, read_block_num);
        waitAsyncContextDone(read_context);
        ASSERT_TRUE(read_context->success());
        ASSERT_EQ(kv_cache_resouce->remoteReuseBlockNum(), 1);
        kv_cache_resouce->setRemoteReuseBlockNum(0);
    }
    {
        // 有其他connector,覆盖了
        const int gpu_reuse_num   = static_cast<int>(kv_cache_resouce->reuseBlockNum());   // 1
        const int other_reuse_num = 2;                                                     // other connector
        const int matched_num     = static_cast<int>(match_context->matchedBlockCount());  // 3
        // auto      meta       = std::make_shared<TestReadMeta>(gpu_reuse_num + other_reuse_num,
        //                                                 matched_num - gpu_reuse_num - other_reuse_num);
        int  start_read_block_index = gpu_reuse_num + other_reuse_num;
        int  read_block_num         = matched_num - gpu_reuse_num - other_reuse_num;
        auto read_context           = remote_connectors_[tp_rank]->asyncRead(
            kv_cache_resouce, meta, match_context, start_read_block_index, read_block_num);
        waitAsyncContextDone(read_context);
        ASSERT_TRUE(read_context->success());
        ASSERT_EQ(kv_cache_resouce->remoteReuseBlockNum(), 0);
    }
}

TEST_F(RemoteConnectorMockOnlyFullTest, test_write_success_broadcast_success_actual_locations_different) {
    auto kv_cache_resouce = std::make_shared<KVCacheResource>();
    kv_cache_resouce->setLastBlockAligned(true);
    kv_cache_resouce->setCacheKeys({1, 2, 3});
    initializeResourceTopology(*kv_cache_resouce, cache_config_, {1, 2, 3});
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
        3, kFakeLayerNum, cache_config_.group("default").spec->block_size_bytes()};
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
    kv_cache_resouce->setLastBlockAligned(true);
    kv_cache_resouce->setCacheKeys({1, 2, 3});
    initializeResourceTopology(*kv_cache_resouce, cache_config_, {1, 2, 3});

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
        2, kFakeLayerNum, cache_config_.group("default").spec->block_size_bytes()};
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
    kv_cache_resouce->setLastBlockAligned(true);
    kv_cache_resouce->setCacheKeys({1, 2, 3, 4});
    initializeResourceTopology(*kv_cache_resouce, cache_config_, {1, 2, 3, 4});

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
        2, kFakeLayerNum, cache_config_.group("default").spec->block_size_bytes()};
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
    kv_cache_resouce->setLastBlockAligned(true);
    kv_cache_resouce->setCacheKeys({1, 2, 3});
    initializeResourceTopology(*kv_cache_resouce, cache_config_, {1, 2, 3});

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
    kv_cache_resouce->setLastBlockAligned(true);
    kv_cache_resouce->setCacheKeys({1, 2, 3});
    initializeResourceTopology(*kv_cache_resouce, cache_config_, {1, 2, 3});
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
        3, kFakeLayerNum, cache_config_.group("default").spec->block_size_bytes()};
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

TEST_F(RemoteConnectorMockOnlyFullTest, test_write_last_block_not_aligned) {
    auto kv_cache_resource = std::make_shared<KVCacheResource>();
    kv_cache_resource->setLastBlockAligned(false);
    kv_cache_resource->setCacheKeys({1, 2, 3});
    initializeResourceTopology(*kv_cache_resource, cache_config_, {1, 2, 3});

    auto          meta    = std::make_shared<MetaImpl>(false, true, "trace_1");
    size_t        tp_rank = 0;
    std::string   write_session_id("write_session_id_1");
    Locations     expected_write_locations = genFullotherLocations({2});
    WriteLocation write_location({write_session_id, static_cast<size_t>(1), expected_write_locations});
    EXPECT_CALL(
        *meta_clients_[tp_rank],
        StartWrite(Eq("start_write_trace_1"), std::vector<int64_t>({1, 2}), _, Eq(std::vector<std::string>()), _))
        .WillOnce(Return(StartWriteReturnType({ClientErrorCode::ER_OK, write_location})));

    const UriStrVec          expected_uris        = genUris({2});
    const UriStrVec          actual_uris          = genUris({2}, {}, "actual_");
    const BlockBuffersExpect block_buffers_expect = {
        1, kFakeLayerNum, cache_config_.group("default").spec->block_size_bytes()};
    EXPECT_CALL(*transfer_client_,
                SaveKvCaches(Eq(expected_uris),
                             BlockBuffersMatcher(block_buffers_expect),
                             TransferTraceInfoMatcher(std::vector<std::string>({"2"}))))
        .WillOnce(Return(SaveKvCachesReturnType({ClientErrorCode::ER_OK, actual_uris})));

    const Locations expected_actual_locations = genFullotherLocations({2}, {}, "actual_");
    EXPECT_CALL(*meta_clients_[tp_rank],
                FinishWrite(Eq("finish_write_trace_1"),
                            write_session_id,
                            Eq(BlockMask(static_cast<size_t>(1))),
                            Eq(expected_actual_locations)))
        .WillOnce(Return(ClientErrorCode::ER_OK));

    auto async_context = remote_connectors_[tp_rank]->asyncWrite(kv_cache_resource, meta);
    waitAsyncContextDone(async_context);
    ASSERT_TRUE(async_context->success());
}

TEST_F(RemoteConnectorMockOnlyFullTest, test_match_fail) {
    auto kv_cache_resource = std::make_shared<KVCacheResource>();
    kv_cache_resource->setCacheKeys({1, 2, 3, 4});
    initializeResourceTopology(*kv_cache_resource, cache_config_, {1, 2, 3, 4});

    auto   meta    = std::make_shared<MetaImpl>(false, true, "trace_1");
    size_t tp_rank = 0;
    EXPECT_CALL(
        *meta_clients_[tp_rank],
        MatchLocation(
            Eq("match_trace_1"), _, std::vector<int64_t>({1, 2, 3}), _, Eq(BlockMask(static_cast<size_t>(0))), _, _))
        .WillOnce(Return(MatchLocationReturnType({ClientErrorCode::ER_INVALID_GRPCSTATUS, {}})));
    EXPECT_CALL(*transfer_client_, LoadKvCaches(_, _, _)).Times(0);

    auto match_context = remote_connectors_[tp_rank]->asyncMatch(kv_cache_resource, meta);
    waitAsyncContextDone(match_context);
    ASSERT_FALSE(match_context->success());
    auto context = std::dynamic_pointer_cast<RemoteAsyncMatchContext>(match_context);
    ASSERT_NE(nullptr, context);
    ASSERT_EQ(RemoteConnectorState::State::RCS_READ_MATCH_ERROR, context->state());
}

TEST_F(RemoteConnectorMockOnlyFullTest, test_match_success_load_fail) {
    auto kv_cache_resource = std::make_shared<KVCacheResource>();
    kv_cache_resource->setCacheKeys({1, 2, 3, 4});
    initializeResourceTopology(*kv_cache_resource, cache_config_, {1, 2, 3, 4});

    auto            meta               = std::make_shared<MetaImpl>(false, true, "trace_1");
    size_t          tp_rank            = 0;
    const Locations expected_locations = genFullotherLocations({1, 2, 3});
    EXPECT_CALL(
        *meta_clients_[tp_rank],
        MatchLocation(
            Eq("match_trace_1"), _, std::vector<int64_t>({1, 2, 3}), _, Eq(BlockMask(static_cast<size_t>(0))), _, _))
        .WillOnce(Return(MatchLocationReturnType({ClientErrorCode::ER_OK, expected_locations})));

    const UriStrVec          expected_uris        = genUris({1, 2, 3});
    const BlockBuffersExpect block_buffers_expect = {
        3, kFakeLayerNum, cache_config_.group("default").spec->block_size_bytes()};
    EXPECT_CALL(*transfer_client_,
                LoadKvCaches(Eq(expected_uris),
                             BlockBuffersMatcher(block_buffers_expect),
                             TransferTraceInfoMatcher(std::vector<std::string>({"1", "2", "3"}))))
        .WillOnce(Return(ClientErrorCode::ER_SDK_TIMEOUT));

    auto match_context = remote_connectors_[tp_rank]->asyncMatch(kv_cache_resource, meta);
    waitAsyncContextDone(match_context);
    ASSERT_TRUE(match_context->success());
    ASSERT_EQ(match_context->matchedBlockCount(), 3);

    auto read_context = remote_connectors_[tp_rank]->asyncRead(
        kv_cache_resource, meta, match_context, 0, static_cast<int>(match_context->matchedBlockCount()));
    waitAsyncContextDone(read_context);
    ASSERT_FALSE(read_context->success());
    auto context = std::dynamic_pointer_cast<RemoteConnectorAsyncContext>(read_context);
    ASSERT_NE(nullptr, context);
    ASSERT_EQ(RemoteConnectorState::State::RCS_ERROR, context->state());
}

TEST_F(RemoteConnectorMockOnlyFullTest, test_start_write_fail) {
    auto kv_cache_resource = std::make_shared<KVCacheResource>();
    kv_cache_resource->setLastBlockAligned(true);
    kv_cache_resource->setCacheKeys({1, 2, 3});
    initializeResourceTopology(*kv_cache_resource, cache_config_, {1, 2, 3});

    auto   meta    = std::make_shared<MetaImpl>(false, true, "trace_1");
    size_t tp_rank = 0;
    EXPECT_CALL(
        *meta_clients_[tp_rank],
        StartWrite(Eq("start_write_trace_1"), std::vector<int64_t>({1, 2, 3}), _, Eq(std::vector<std::string>()), _))
        .WillOnce(Return(StartWriteReturnType({ClientErrorCode::ER_INVALID_GRPCSTATUS, {}})));
    EXPECT_CALL(*transfer_client_, SaveKvCaches(_, _, _)).Times(0);
    EXPECT_CALL(*meta_clients_[tp_rank], FinishWrite(_, _, _, _)).Times(0);

    auto async_context = remote_connectors_[tp_rank]->asyncWrite(kv_cache_resource, meta);
    waitAsyncContextDone(async_context);
    ASSERT_FALSE(async_context->success());
    auto context = std::dynamic_pointer_cast<RemoteConnectorAsyncContext>(async_context);
    ASSERT_NE(nullptr, context);
    ASSERT_EQ(RemoteConnectorState::State::RCS_ERROR, context->state());
}

TEST_F(RemoteConnectorMockOnlyFullTest, test_write_invalid_block_ids) {
    auto kv_cache_resource = std::make_shared<KVCacheResource>();
    kv_cache_resource->setLastBlockAligned(true);
    kv_cache_resource->setCacheKeys({1, 2, 3});
    initializeResourceTopology(*kv_cache_resource, cache_config_, {1, 99, 3});

    auto              meta               = std::make_shared<MetaImpl>(false, true, "trace_1");
    size_t            tp_rank            = 0;
    const Locations   expected_locations = genFullotherLocations({1, 2, 3});
    const std::string write_session_id("write_session_id_1");
    WriteLocation     write_location({write_session_id, static_cast<size_t>(0), expected_locations});
    EXPECT_CALL(
        *meta_clients_[tp_rank],
        StartWrite(Eq("start_write_trace_1"), std::vector<int64_t>({1, 2, 3}), _, Eq(std::vector<std::string>()), _))
        .WillOnce(Return(StartWriteReturnType({ClientErrorCode::ER_OK, write_location})));
    EXPECT_CALL(*transfer_client_, SaveKvCaches(_, _, _)).Times(0);
    EXPECT_CALL(
        *meta_clients_[tp_rank],
        FinishWrite(
            Eq("finish_write_trace_1"), write_session_id, Eq(BlockMask(static_cast<size_t>(0))), Eq(Locations({}))))
        .WillOnce(Return(ClientErrorCode::ER_OK));

    auto async_context = remote_connectors_[tp_rank]->asyncWrite(kv_cache_resource, meta);
    waitAsyncContextDone(async_context);
    ASSERT_FALSE(async_context->success());
    auto context = std::dynamic_pointer_cast<RemoteConnectorAsyncContext>(async_context);
    ASSERT_NE(nullptr, context);
    ASSERT_EQ(RemoteConnectorState::State::RCS_ERROR, context->state());
}

TEST_F(RemoteConnectorMockOnlyFullTest, test_start_write_success_broadcast_success_finish_write_fail) {
    auto kv_cache_resource = std::make_shared<KVCacheResource>();
    kv_cache_resource->setLastBlockAligned(true);
    kv_cache_resource->setCacheKeys({1, 2, 3});
    initializeResourceTopology(*kv_cache_resource, cache_config_, {1, 2, 3});

    auto              meta    = std::make_shared<MetaImpl>(false, true, "trace_2");
    size_t            tp_rank = 0;
    const std::string write_session_id("write_session_id_2");
    const Locations   expected_locations = genFullotherLocations({1, 2, 3});
    WriteLocation     write_location({write_session_id, static_cast<size_t>(0), expected_locations});
    EXPECT_CALL(
        *meta_clients_[tp_rank],
        StartWrite(Eq("start_write_trace_2"), std::vector<int64_t>({1, 2, 3}), _, Eq(std::vector<std::string>()), _))
        .WillOnce(Return(StartWriteReturnType({ClientErrorCode::ER_OK, write_location})));

    const UriStrVec          expected_uris        = genUris({1, 2, 3});
    const BlockBuffersExpect block_buffers_expect = {
        3, kFakeLayerNum, cache_config_.group("default").spec->block_size_bytes()};
    EXPECT_CALL(*transfer_client_,
                SaveKvCaches(Eq(expected_uris),
                             BlockBuffersMatcher(block_buffers_expect),
                             TransferTraceInfoMatcher(std::vector<std::string>({"1", "2", "3"}))))
        .WillOnce(Return(SaveKvCachesReturnType({ClientErrorCode::ER_OK, expected_uris})));
    EXPECT_CALL(
        *meta_clients_[tp_rank],
        FinishWrite(
            Eq("finish_write_trace_2"), write_session_id, Eq(BlockMask(static_cast<size_t>(3))), Eq(Locations({}))))
        .WillOnce(Return(ClientErrorCode::ER_INVALID_GRPCSTATUS));

    auto async_context = remote_connectors_[tp_rank]->asyncWrite(kv_cache_resource, meta);
    waitAsyncContextDone(async_context);
    ASSERT_FALSE(async_context->success());
    auto context = std::dynamic_pointer_cast<RemoteConnectorAsyncContext>(async_context);
    ASSERT_NE(nullptr, context);
    ASSERT_EQ(RemoteConnectorState::State::RCS_ERROR, context->state());
}

TEST_F(RemoteConnectorMockOnlyFullTest, test_start_write_success_broadcast_success_save_fail) {
    auto kv_cache_resource = std::make_shared<KVCacheResource>();
    kv_cache_resource->setLastBlockAligned(true);
    kv_cache_resource->setCacheKeys({1, 2, 3});
    initializeResourceTopology(*kv_cache_resource, cache_config_, {1, 2, 3});

    auto              meta    = std::make_shared<MetaImpl>(false, true, "trace_2");
    size_t            tp_rank = 0;
    const std::string write_session_id("write_session_id_2");
    const Locations   expected_locations = genFullotherLocations({1, 2, 3});
    WriteLocation     write_location({write_session_id, static_cast<size_t>(0), expected_locations});
    EXPECT_CALL(
        *meta_clients_[tp_rank],
        StartWrite(Eq("start_write_trace_2"), std::vector<int64_t>({1, 2, 3}), _, Eq(std::vector<std::string>()), _))
        .WillOnce(Return(StartWriteReturnType({ClientErrorCode::ER_OK, write_location})));

    const UriStrVec          expected_uris        = genUris({1, 2, 3});
    const BlockBuffersExpect block_buffers_expect = {
        3, kFakeLayerNum, cache_config_.group("default").spec->block_size_bytes()};
    EXPECT_CALL(*transfer_client_,
                SaveKvCaches(Eq(expected_uris),
                             BlockBuffersMatcher(block_buffers_expect),
                             TransferTraceInfoMatcher(std::vector<std::string>({"1", "2", "3"}))))
        .WillOnce(Return(SaveKvCachesReturnType({ClientErrorCode::ER_SDK_TIMEOUT, {}})));
    EXPECT_CALL(
        *meta_clients_[tp_rank],
        FinishWrite(
            Eq("finish_write_trace_2"), write_session_id, Eq(BlockMask(static_cast<size_t>(0))), Eq(Locations({}))))
        .WillOnce(Return(ClientErrorCode::ER_OK));

    auto async_context = remote_connectors_[tp_rank]->asyncWrite(kv_cache_resource, meta);
    waitAsyncContextDone(async_context);
    ASSERT_FALSE(async_context->success());
    auto context = std::dynamic_pointer_cast<RemoteConnectorAsyncContext>(async_context);
    ASSERT_NE(nullptr, context);
    ASSERT_EQ(RemoteConnectorState::State::RCS_ERROR, context->state());
}

TEST_F(RemoteConnectorMockOnlyFullTest, test_start_write_success_broadcast_grpc_fail) {
    auto kv_cache_resource = std::make_shared<KVCacheResource>();
    kv_cache_resource->setLastBlockAligned(true);
    kv_cache_resource->setCacheKeys({1, 2, 3});
    initializeResourceTopology(*kv_cache_resource, cache_config_, {1, 2, 3});

    auto              meta    = std::make_shared<MetaImpl>(false, true, "trace_2");
    size_t            tp_rank = 0;
    const std::string write_session_id("write_session_id_2");
    const Locations   expected_locations = genFullotherLocations({1, 2, 3});
    WriteLocation     write_location({write_session_id, static_cast<size_t>(0), expected_locations});
    EXPECT_CALL(
        *meta_clients_[tp_rank],
        StartWrite(Eq("start_write_trace_2"), std::vector<int64_t>({1, 2, 3}), _, Eq(std::vector<std::string>()), _))
        .WillOnce(Return(StartWriteReturnType({ClientErrorCode::ER_OK, write_location})));
    EXPECT_CALL(*transfer_client_, SaveKvCaches(_, _, _)).Times(0);
    EXPECT_CALL(
        *meta_clients_[tp_rank],
        FinishWrite(
            Eq("finish_write_trace_2"), write_session_id, Eq(BlockMask(static_cast<size_t>(0))), Eq(Locations({}))))
        .WillOnce(Return(ClientErrorCode::ER_OK));

    servers_[tp_rank]->hack_grpc_status(true);
    auto async_context = remote_connectors_[tp_rank]->asyncWrite(kv_cache_resource, meta);
    waitAsyncContextDone(async_context);
    ASSERT_FALSE(async_context->success());
    auto context = std::dynamic_pointer_cast<RemoteConnectorAsyncContext>(async_context);
    ASSERT_NE(nullptr, context);
    ASSERT_EQ(RemoteConnectorState::State::RCS_ERROR, context->state());
}

TEST_F(RemoteConnectorMockOnlyFullTest, test_threadpool_ec) {
    auto kv_cache_resource = std::make_shared<KVCacheResource>();
    kv_cache_resource->setCacheKeys({1, 2, 3, 4});
    initializeResourceTopology(*kv_cache_resource, cache_config_, {1, 2, 3, 4});

    auto   meta    = std::make_shared<MetaImpl>(false, true, "trace");
    size_t tp_rank = 0;
    remote_connectors_[tp_rank]->thread_pool_->stop();
    remote_connectors_[tp_rank]->thread_pool_->waitFinish();
    remote_connectors_[tp_rank]->thread_pool_ =
        std::make_unique<autil::ThreadPool>(1, 1, nullptr, "RECOThreadPool", true);

    const bool saved_core_dump_on_exception               = rtp_llm::StaticConfig::user_ft_core_dump_on_exception;
    rtp_llm::StaticConfig::user_ft_core_dump_on_exception = false;
    remote_connectors_[tp_rank]->thread_pool_->_push      = false;
    EXPECT_CALL(*meta_clients_[tp_rank], MatchLocation(_, _, _, _, _, _, _)).Times(0);
    EXPECT_THROW((void)remote_connectors_[tp_rank]->asyncMatch(kv_cache_resource, meta), rtp_llm::RTPException);
    rtp_llm::StaticConfig::user_ft_core_dump_on_exception = saved_core_dump_on_exception;

    ASSERT_TRUE(remote_connectors_[tp_rank]->thread_pool_->start());
    remote_connectors_[tp_rank]->thread_pool_->_queueSize = 0;
    EXPECT_CALL(*meta_clients_[tp_rank], MatchLocation(_, _, _, _, _, _, _)).Times(0);
    ASSERT_EQ(nullptr, remote_connectors_[tp_rank]->asyncMatch(kv_cache_resource, meta));

    EXPECT_CALL(*meta_clients_[tp_rank], StartWrite(_, _, _, _, _)).Times(0);
    EXPECT_CALL(*transfer_client_, SaveKvCaches(_, _, _)).Times(0);
    EXPECT_CALL(*meta_clients_[tp_rank], FinishWrite(_, _, _, _)).Times(0);
    kv_cache_resource->setLastBlockAligned(true);
    ASSERT_EQ(nullptr, remote_connectors_[tp_rank]->asyncWrite(kv_cache_resource, meta));
}

}  // namespace test
}  // namespace rtp_llm
