#include "rtp_llm/cpp/cache/SingleTypeKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/connector/remote_connector/test/RemoteConnectorMockTestBase.h"

using namespace kv_cache_manager;
using namespace ::testing;
using namespace rtp_llm;
using namespace rtp_llm::remote_connector;

namespace rtp_llm {
namespace test {
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

class TestReadMeta: public rtp_llm::KVCacheConnector::Meta {
public:
    TestReadMeta(int start_block_index, int size): start_block_index_(start_block_index), size_(size) {}
    ~TestReadMeta() override = default;

public:
    std::pair<int, int> blockRange() const override {
        return {start_block_index_, size_};
    }

private:
    int start_block_index_{0};
    int size_{0};
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
            auto allocator = std::make_shared<SingleTypeKVCacheAllocator>(cache_config_, device_);
            ASSERT_TRUE(allocator->init());
            remote_connectors_.push_back(
                std::make_shared<RemoteConnector>(cache_config_,
                                                  kv_cache_config_,
                                                  runtime_config_,
                                                  parallelism_config_,
                                                  sp_config_,
                                                  device_,
                                                  nullptr,
                                                  0,
                                                  allocator,
                                                  RemoteConnectorGroupMode::RCGM_ONLY_FULL_LAYER,
                                                  full_group_ids_));
            ASSERT_TRUE(remote_connectors_[i]->init());
            servers_[i]->set_remote_connector(remote_connectors_[i]);
        }
    }

    void initCacheConfig(int layer_num = 4, int block_num = 10, int seq_size_per_block = 8) {
        cache_config_.layer_num          = layer_num;
        cache_config_.block_num          = block_num;
        cache_config_.seq_size_per_block = seq_size_per_block;

        auto mha_spec       = std::make_shared<MHAKVCacheSpec>();
        mha_spec->layer_num = layer_num;
        // mha_spec->block_nums         = block_num;
        mha_spec->local_head_num_kv  = 8;
        mha_spec->size_per_head      = 128;
        mha_spec->seq_size_per_block = seq_size_per_block;
        mha_spec->dtype              = rtp_llm::DataType::TYPE_FP16;
        mha_spec->type               = KVCacheType::MultiHeadAttention;
        cache_config_.dtype          = rtp_llm::DataType::TYPE_FP16;
        cache_config_.cache_specs.push_back(mha_spec);
        // cache_config_.block_size = static_cast<int>(mha_spec->block_size() * mha_spec->layer_num);
        cache_config_.block_size_bytes = static_cast<size_t>(mha_spec->block_size_bytes() * mha_spec->layer_num);
        std::vector<int> layer_ids(layer_num);
        for (int i = 0; i < layer_num; ++i) {
            layer_ids[i] = i;
        }
        cache_config_.layer_ids.push_back(layer_ids);
        cache_config_.global_layer_ids.push_back(layer_ids);
    }
};

// 初始reuse_len = 0
TEST_F(RemoteConnectorMockOnlyFullTest, test_async_match_and_async_read_with_gpu_reuse_len_zero) {
    // match
    auto kv_cache_resouce        = std::make_shared<KVCacheResource>();
    kv_cache_resouce->cache_keys = {1, 2, 3};
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({1, 2, 3}));
    auto      meta               = std::make_shared<RemoteConnectorMeta>("", "trace_1");
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
        BlockBuffersExpect block_buffers_expect = {3, kFakeLayerNum, cache_config_.cache_specs[0]->block_size_bytes()};
        EXPECT_CALL(*transfer_client_, LoadKvCaches(Eq(expected_uris), BlockBuffersMatcher(block_buffers_expect)))
            .WillOnce(Return(ClientErrorCode::ER_OK));

        const int gpu_reuse_num = static_cast<int>(kv_cache_resouce->reuseBlocksNum());  // 0
        const int matched_num   = static_cast<int>(match_context->matchedBlockCount());  // 3
        auto      read_meta     = std::make_shared<TestReadMeta>(gpu_reuse_num, matched_num - gpu_reuse_num);

        auto read_context = remote_connectors_[tp_rank]->asyncRead(kv_cache_resouce, read_meta, match_context);
        waitAsyncContextDone(read_context);
        ASSERT_TRUE(read_context->success());
        kv_cache_resouce->setReuseBlocksNum(0);
    }

    {
        // 其他connector也命中了部分
        UriStrVec          expected_uris        = genUris({2, 3});
        BlockBuffersExpect block_buffers_expect = {2, kFakeLayerNum, cache_config_.cache_specs[0]->block_size_bytes()};
        EXPECT_CALL(*transfer_client_, LoadKvCaches(Eq(expected_uris), BlockBuffersMatcher(block_buffers_expect)))
            .WillOnce(Return(ClientErrorCode::ER_OK));

        const int gpu_reuse_num = static_cast<int>(kv_cache_resouce->reuseBlocksNum());  // 0
        const int matched_num   = static_cast<int>(match_context->matchedBlockCount());  // 3
        auto      read_meta     = std::make_shared<TestReadMeta>(gpu_reuse_num + 1, matched_num - gpu_reuse_num - 1);

        auto read_context = remote_connectors_[tp_rank]->asyncRead(kv_cache_resouce, read_meta, match_context);
        waitAsyncContextDone(read_context);
        ASSERT_TRUE(read_context->success());
        kv_cache_resouce->setReuseBlocksNum(0);
    }

    {
        // 其他connector也命中了部分,超出了remote
        const int gpu_reuse_num = static_cast<int>(kv_cache_resouce->reuseBlocksNum());  // 0
        const int matched_num   = static_cast<int>(match_context->matchedBlockCount());  // 3
        auto      read_meta     = std::make_shared<TestReadMeta>(gpu_reuse_num + 4, matched_num - gpu_reuse_num - 4);

        auto read_context = remote_connectors_[tp_rank]->asyncRead(kv_cache_resouce, read_meta, match_context);
        waitAsyncContextDone(read_context);
        ASSERT_TRUE(read_context->success());
    }
}

// 初始reuse_len = 1
TEST_F(RemoteConnectorMockOnlyFullTest, test_async_match_and_async_read_with_gpu_reuse_len_not_zero) {
    // match
    auto kv_cache_resouce        = std::make_shared<KVCacheResource>();
    kv_cache_resouce->cache_keys = {1, 2, 3};
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({1, 2, 3}));
    kv_cache_resouce->setReuseBlocksNum(1);
    auto      meta               = std::make_shared<RemoteConnectorMeta>("", "trace_1");
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
        BlockBuffersExpect block_buffers_expect = {2, kFakeLayerNum, cache_config_.cache_specs[0]->block_size_bytes()};
        EXPECT_CALL(*transfer_client_, LoadKvCaches(Eq(expected_uris), BlockBuffersMatcher(block_buffers_expect)))
            .WillOnce(Return(ClientErrorCode::ER_OK));

        const int gpu_reuse_num = static_cast<int>(kv_cache_resouce->reuseBlocksNum());  // 1
        const int matched_num   = static_cast<int>(match_context->matchedBlockCount());  // 3
        auto      read_meta     = std::make_shared<TestReadMeta>(gpu_reuse_num, matched_num - gpu_reuse_num);

        auto read_context = remote_connectors_[tp_rank]->asyncRead(kv_cache_resouce, read_meta, match_context);
        waitAsyncContextDone(read_context);
        ASSERT_TRUE(read_context->success());
        kv_cache_resouce->setReuseBlocksNum(1);
    }
    {
        // 有其他connector
        UriStrVec          expected_uris        = genUris({3});
        BlockBuffersExpect block_buffers_expect = {1, kFakeLayerNum, cache_config_.cache_specs[0]->block_size_bytes()};
        EXPECT_CALL(*transfer_client_, LoadKvCaches(Eq(expected_uris), BlockBuffersMatcher(block_buffers_expect)))
            .WillOnce(Return(ClientErrorCode::ER_OK));

        const int gpu_reuse_num   = static_cast<int>(kv_cache_resouce->reuseBlocksNum());  // 1
        const int other_reuse_num = 1;                                                     // other connector
        const int matched_num     = static_cast<int>(match_context->matchedBlockCount());  // 3
        auto      read_meta       = std::make_shared<TestReadMeta>(gpu_reuse_num + other_reuse_num,
                                                        matched_num - gpu_reuse_num - other_reuse_num);

        auto read_context = remote_connectors_[tp_rank]->asyncRead(kv_cache_resouce, read_meta, match_context);
        waitAsyncContextDone(read_context);
        ASSERT_TRUE(read_context->success());
        kv_cache_resouce->setReuseBlocksNum(1);
    }
    {
        // 有其他connector,覆盖了
        const int gpu_reuse_num   = static_cast<int>(kv_cache_resouce->reuseBlocksNum());  // 1
        const int other_reuse_num = 2;                                                     // other connector
        const int matched_num     = static_cast<int>(match_context->matchedBlockCount());  // 3
        auto      read_meta       = std::make_shared<TestReadMeta>(gpu_reuse_num + other_reuse_num,
                                                        matched_num - gpu_reuse_num - other_reuse_num);

        auto read_context = remote_connectors_[tp_rank]->asyncRead(kv_cache_resouce, read_meta, match_context);
        waitAsyncContextDone(read_context);
        ASSERT_TRUE(read_context->success());
    }
}

TEST_F(RemoteConnectorMockOnlyFullTest, test_write_success_broadcast_success_actual_locations_different) {
    auto kv_cache_resouce        = std::make_shared<KVCacheResource>();
    kv_cache_resouce->cache_keys = {1, 2, 3};
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({1, 2, 3}));
    auto          meta    = std::make_shared<RemoteConnectorMeta>("", "trace_1");
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

    BlockBuffersExpect block_buffers_expect = {3, kFakeLayerNum, cache_config_.cache_specs[0]->block_size_bytes()};
    EXPECT_CALL(*transfer_client_, SaveKvCaches(Eq(expected_uris), BlockBuffersMatcher(block_buffers_expect)))
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
    auto kv_cache_resouce        = std::make_shared<KVCacheResource>();
    kv_cache_resouce->cache_keys = {1, 2, 3};
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({1, 2, 3}));

    auto          meta    = std::make_shared<RemoteConnectorMeta>("", "trace_1");
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

    BlockBuffersExpect block_buffers_expect = {2, kFakeLayerNum, cache_config_.cache_specs[0]->block_size_bytes()};
    EXPECT_CALL(*transfer_client_, SaveKvCaches(Eq(expected_uris), BlockBuffersMatcher(block_buffers_expect)))
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
       test_write_success_broadcast_success_actual_locations_different_with_empty_write_locations) {
    auto kv_cache_resouce        = std::make_shared<KVCacheResource>();
    kv_cache_resouce->cache_keys = {1, 2, 3};
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({1, 2, 3}));

    auto          meta    = std::make_shared<RemoteConnectorMeta>("", "trace_1");
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

    EXPECT_CALL(*transfer_client_, SaveKvCaches(_, _)).Times(0);
    EXPECT_CALL(*meta_clients_[tp_rank], FinishWrite(_, _, _, _)).Times(0);

    auto async_context = remote_connectors_[tp_rank]->asyncWrite(kv_cache_resouce, meta);
    waitAsyncContextDone(async_context);
    ASSERT_TRUE(async_context->success());
}

TEST_F(RemoteConnectorMockOnlyFullTest, test_write_success_broadcast_success_actual_locations_same) {
    auto kv_cache_resouce        = std::make_shared<KVCacheResource>();
    kv_cache_resouce->cache_keys = {1, 2, 3};
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({1, 2, 3}));
    auto          meta    = std::make_shared<RemoteConnectorMeta>("", "trace_2");
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

    BlockBuffersExpect block_buffers_expect = {3, kFakeLayerNum, cache_config_.cache_specs[0]->block_size_bytes()};
    EXPECT_CALL(*transfer_client_, SaveKvCaches(Eq(expected_uris), BlockBuffersMatcher(block_buffers_expect)))
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

}  // namespace test
}  // namespace rtp_llm