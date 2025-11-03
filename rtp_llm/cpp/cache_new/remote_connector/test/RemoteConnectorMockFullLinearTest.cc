#include "rtp_llm/cpp/cache_new/HybridLayerKVCacheAllocator.h"
#include "rtp_llm/cpp/cache_new/remote_connector/test/RemoteConnectorMockTestBase.h"

using namespace kv_cache_manager;
using namespace ::testing;
using namespace rtp_llm;
using namespace rtp_llm::remote_connector;

namespace rtp_llm {
namespace test {

// TODO : remove this, use ture HybridLayerKVCacheAllocator
class FakeHybridLayerKVCacheAllocator: public HybridLayerKVCacheAllocator {
public:
    FakeHybridLayerKVCacheAllocator(const CacheConfig&          config,
                                    rtp_llm::DeviceBase*        device,
                                    const std::vector<int32_t>& full_group_ids,
                                    const std::vector<int32_t>& other_group_ids):
        HybridLayerKVCacheAllocator(config, device) {
        for (int32_t full_group_id : full_group_ids) {
            for (int i = 0; i < kFakeLayerNum; i++) {
                fake_layout_.layer_to_groups.push_back(full_group_id);
            }
        }
        for (int32_t other_group_id : other_group_ids) {
            for (int i = 0; i < kFakeLayerNum; i++) {
                fake_layout_.layer_to_groups.push_back(other_group_id);
            }
        }
    }
    CacheLayerLayout layerCacheBase() const override {
        return fake_layout_;
    }

    BlockBufferPtrInfo convertIndexToBuffer(int layer_id, int block_id) const override {
        return {BufferPtr(new Buffer(MemoryType::MEMORY_GPU,
                                     DataType::TYPE_INT8,
                                     {1, fake_buffer_.size()},
                                     static_cast<const void*>(fake_buffer_.data()),
                                     nullptr)),
                BufferPtr(new Buffer(MemoryType::MEMORY_GPU,
                                     DataType::TYPE_INT8,
                                     {1, fake_buffer_.size()},
                                     static_cast<const void*>(fake_buffer_.data()),
                                     nullptr))};
    }

private:
    CacheLayerLayout                        fake_layout_;
    constexpr static size_t                 fake_buffer_size_ = kFakeIovSize / sizeof(int8_t);
    inline static const std::vector<int8_t> fake_buffer_      = std::vector<int8_t>(fake_buffer_size_, 0);
};

class RemoteConnectorMockFullLinearTest: public RemoteConnectorMockTestBase {
public:
    void SetUp() override {
        other_group_ids_ = {1, 2};
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
        initHybridLayerCacheConfig(kFakeLayerNum, block_num, seq_size_per_block);
        for (int i = 0; i < tp_size_; i++) {
            auto meta_client = std::make_unique<kv_cache_manager::MockMetaClient>();
            meta_clients_.push_back(meta_client.get());
            EXPECT_CALL(*mock_client_factory_, CreateMetaClient(_, _)).WillOnce(Return(std::move(meta_client)));
            auto allocator = std::make_shared<FakeHybridLayerKVCacheAllocator>(
                cache_config_, device_, full_group_ids_, other_group_ids_);
            ASSERT_TRUE(allocator->init());
            remote_connectors_.push_back(
                std::make_shared<RemoteConnector>(cache_config_,
                                                  gpt_init_params_,
                                                  device_,
                                                  nullptr,
                                                  0,
                                                  allocator,
                                                  RemoteConnectorGroupMode::RCGM_FULL_LINEAR_LAYER,
                                                  full_group_ids_,
                                                  other_group_ids_,
                                                  nullptr,
                                                  1));
            ASSERT_TRUE(remote_connectors_[i]->init());
            servers_[i]->set_remote_connector(remote_connectors_[i]);
        }
    }

    void initHybridLayerCacheConfig(int layer_num = 4, int block_num = 10, int seq_size_per_block = 8) {
        cache_config_.layer_num          = layer_num;
        cache_config_.block_num          = block_num;
        cache_config_.seq_size_per_block = seq_size_per_block;

        auto mha_spec                = std::make_shared<MHAKVCacheSpec>();
        mha_spec->layer_num          = layer_num;
        mha_spec->block_nums         = block_num;
        mha_spec->local_head_num_kv  = 8;
        mha_spec->size_per_head      = 128;
        mha_spec->seq_size_per_block = seq_size_per_block;
        mha_spec->dtype              = rtp_llm::DataType::TYPE_FP16;
        mha_spec->type               = KVCacheType::MultiHeadAttention;

        cache_config_.cache_specs.push_back(mha_spec);
        cache_config_.block_size = static_cast<int>(mha_spec->block_size() * mha_spec->layer_num);

        std::vector<int> layer_ids(layer_num);
        for (int i = 0; i < layer_num; ++i) {
            layer_ids[i] = i;
        }
        cache_config_.layer_ids.push_back(layer_ids);
    }
};

TEST_F(RemoteConnectorMockFullLinearTest, test_read_success_broadcast_success_match_all) {
    auto kv_cache_resouce        = std::make_shared<KVCacheResourceV1>();
    kv_cache_resouce->cache_keys = {1, 2, 3};
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({1, 2, 3}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({11, 12, 13}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({21, 22, 23}));
    auto      meta               = std::make_shared<RemoteConnectorMeta>("", "trace_1");
    size_t    tp_rank            = 0;
    Locations expected_locations = genFullotherLocations(kv_cache_resouce->cache_keys, {0, 1, 2});
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
    UriStrVec          expected_uris        = genUris({1, 2, 3}, {2});
    BlockBuffersExpect block_buffers_expect = {5, kFakeLayerNum, kFakeIovSize};
    EXPECT_CALL(*transfer_client_, LoadKvCaches(Eq(expected_uris), BlockBuffersMatcher(block_buffers_expect)))
        .WillOnce(Return(ClientErrorCode::ER_OK));

    auto async_context = remote_connectors_[tp_rank]->asyncRead(kv_cache_resouce, meta);
    async_context->waitDone();
    ASSERT_TRUE(async_context->success());
    ASSERT_EQ(3, kv_cache_resouce->reuseBlocksNum());
}

TEST_F(RemoteConnectorMockFullLinearTest, test_read_success_broadcast_success_with_block_mask) {
    auto kv_cache_resouce        = std::make_shared<KVCacheResourceV1>();
    kv_cache_resouce->cache_keys = {1, 2, 3};
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({1, 2, 3}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({11, 12, 13}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({21, 22, 23}));
    kv_cache_resouce->setReuseBlocksNum(1);
    auto      meta               = std::make_shared<RemoteConnectorMeta>("", "trace_2");
    size_t    tp_rank            = 0;
    Locations expected_locations = genFullotherLocations({2, 3}, {0, 1});
    EXPECT_CALL(*meta_clients_[tp_rank],
                MatchLocation(Eq("match_trace_2"),                    // trace_id
                              _,                                      // query_type
                              std::vector<int64_t>({1, 2, 3}),        // keys
                              _,                                      // tokens
                              Eq(BlockMask(static_cast<size_t>(1))),  // block_mask
                              _,                                      // sw_size
                              _                                       // location_spec_names
                              ))
        .WillOnce(Return(MatchLocationReturnType({ClientErrorCode::ER_OK, expected_locations})));

    UriStrVec          expected_uris        = genUris({2, 3}, {1});
    BlockBuffersExpect block_buffers_expect = {4, kFakeLayerNum, kFakeIovSize};
    EXPECT_CALL(*transfer_client_, LoadKvCaches(Eq(expected_uris), BlockBuffersMatcher(block_buffers_expect)))
        .WillOnce(Return(ClientErrorCode::ER_OK));

    auto async_context = remote_connectors_[tp_rank]->asyncRead(kv_cache_resouce, meta);
    async_context->waitDone();
    ASSERT_TRUE(async_context->success());
    ASSERT_EQ(3, kv_cache_resouce->reuseBlocksNum());
}

TEST_F(RemoteConnectorMockFullLinearTest, test_read_success_broadcast_success_with_part_empty_linear) {
    auto kv_cache_resouce        = std::make_shared<KVCacheResourceV1>();
    kv_cache_resouce->cache_keys = {1, 2, 3, 4};
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({1, 2, 3, 4}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({11, 12, 13, 14}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({21, 22, 23, 24}));
    kv_cache_resouce->setReuseBlocksNum(1);
    auto      meta               = std::make_shared<RemoteConnectorMeta>("", "trace_2");
    size_t    tp_rank            = 0;
    Locations expected_locations = genFullotherLocations({2, 3, 4}, {1});
    EXPECT_CALL(*meta_clients_[tp_rank],
                MatchLocation(Eq("match_trace_2"),                    // trace_id
                              _,                                      // query_type
                              std::vector<int64_t>({1, 2, 3, 4}),     // keys
                              _,                                      // tokens
                              Eq(BlockMask(static_cast<size_t>(1))),  // block_mask
                              _,                                      // sw_size
                              _                                       // location_spec_names
                              ))
        .WillOnce(Return(MatchLocationReturnType({ClientErrorCode::ER_OK, expected_locations})));

    UriStrVec          expected_uris        = genUris({2, 3}, {1});
    BlockBuffersExpect block_buffers_expect = {4, kFakeLayerNum, kFakeIovSize};
    EXPECT_CALL(*transfer_client_, LoadKvCaches(Eq(expected_uris), BlockBuffersMatcher(block_buffers_expect)))
        .WillOnce(Return(ClientErrorCode::ER_OK));

    auto async_context = remote_connectors_[tp_rank]->asyncRead(kv_cache_resouce, meta);
    async_context->waitDone();
    ASSERT_TRUE(async_context->success());
    ASSERT_EQ(3, kv_cache_resouce->reuseBlocksNum());
}

TEST_F(RemoteConnectorMockFullLinearTest, test_read_success_broadcast_success_with_all_empty_linear) {
    auto kv_cache_resouce        = std::make_shared<KVCacheResourceV1>();
    kv_cache_resouce->cache_keys = {1, 2, 3, 4};
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({1, 2, 3, 4}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({11, 12, 13, 14}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({21, 22, 23, 24}));
    kv_cache_resouce->setReuseBlocksNum(1);
    auto      meta               = std::make_shared<RemoteConnectorMeta>("", "trace_2");
    size_t    tp_rank            = 0;
    Locations expected_locations = genFullotherLocations({2, 3, 4}, {});
    EXPECT_CALL(*meta_clients_[tp_rank],
                MatchLocation(Eq("match_trace_2"),                    // trace_id
                              _,                                      // query_type
                              std::vector<int64_t>({1, 2, 3, 4}),     // keys
                              _,                                      // tokens
                              Eq(BlockMask(static_cast<size_t>(1))),  // block_mask
                              _,                                      // sw_size
                              _                                       // location_spec_names
                              ))
        .WillOnce(Return(MatchLocationReturnType({ClientErrorCode::ER_OK, expected_locations})));

    EXPECT_CALL(*transfer_client_, LoadKvCaches(_, _)).Times(0);

    auto async_context = remote_connectors_[tp_rank]->asyncRead(kv_cache_resouce, meta);
    async_context->waitDone();
    ASSERT_TRUE(async_context->success());
    ASSERT_EQ(1, kv_cache_resouce->reuseBlocksNum());
}

TEST_F(RemoteConnectorMockFullLinearTest, test_write_success_broadcast_success_actual_locations_different) {
    auto kv_cache_resouce        = std::make_shared<KVCacheResourceV1>();
    kv_cache_resouce->cache_keys = {1, 2, 3};
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({1, 2, 3}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({11, 12, 13}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({21, 22, 23}));
    auto          meta    = std::make_shared<RemoteConnectorMeta>("", "trace_1");
    size_t        tp_rank = 0;
    std::string   write_session_id("write_session_id_1");
    Locations     expected_write_locations = genFullotherLocations({1, 2, 3}, {0, 1, 2});
    WriteLocation write_location({write_session_id, static_cast<size_t>(0), expected_write_locations});
    EXPECT_CALL(*meta_clients_[tp_rank],
                StartWrite(Eq("start_write_trace_1"),        // trace_id
                           std::vector<int64_t>({1, 2, 3}),  // keys
                           _,                                // tokens
                           Eq(std::vector<std::string>()),   // location_spec_group_names
                           _                                 // write_timeout_seconds
                           ))
        .WillOnce(Return(StartWriteReturnType({ClientErrorCode::ER_OK, write_location})));

    UriStrVec expected_uris = genUris({1, 2, 3}, {0, 1, 2});
    UriStrVec actual_uris   = genUris({1, 2, 3}, {0, 1, 2}, "actual_");

    BlockBuffersExpect block_buffers_expect = {9, kFakeLayerNum, kFakeIovSize};
    EXPECT_CALL(*transfer_client_, SaveKvCaches(Eq(expected_uris), BlockBuffersMatcher(block_buffers_expect)))
        .WillOnce(Return(SaveKvCachesReturnType({ClientErrorCode::ER_OK, actual_uris})));

    Locations expected_actual_locations = genFullotherLocations({1, 2, 3}, {0, 1, 2}, "actual_");
    EXPECT_CALL(*meta_clients_[tp_rank],
                FinishWrite(Eq("finish_write_trace_1"),             // trace_id
                            write_session_id,                       // write_session_id
                            Eq(BlockMask(static_cast<size_t>(3))),  // success_block
                            Eq(expected_actual_locations)           // locations
                            ))
        .WillOnce(Return(ClientErrorCode::ER_OK));

    auto async_context = remote_connectors_[tp_rank]->asyncWrite(kv_cache_resouce, meta);
    async_context->waitDone();
    ASSERT_TRUE(async_context->success());
}

TEST_F(RemoteConnectorMockFullLinearTest,
       test_write_success_broadcast_success_actual_locations_different_with_block_mask) {
    auto kv_cache_resouce        = std::make_shared<KVCacheResourceV1>();
    kv_cache_resouce->cache_keys = {1, 2, 3};
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({1, 2, 3}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({11, 12, 13}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({21, 22, 23}));

    auto          meta    = std::make_shared<RemoteConnectorMeta>("", "trace_1");
    size_t        tp_rank = 0;
    std::string   write_session_id("write_session_id_1");
    Locations     expected_write_locations = genFullotherLocations({2, 3}, {0, 1});
    WriteLocation write_location({write_session_id, static_cast<size_t>(1), expected_write_locations});
    EXPECT_CALL(*meta_clients_[tp_rank],
                StartWrite(Eq("start_write_trace_1"),        // trace_id
                           std::vector<int64_t>({1, 2, 3}),  // keys
                           _,                                // tokens
                           Eq(std::vector<std::string>()),   // location_spec_group_names
                           _                                 // write_timeout_seconds
                           ))
        .WillOnce(Return(StartWriteReturnType({ClientErrorCode::ER_OK, write_location})));

    UriStrVec expected_uris = genUris({2, 3}, {0, 1});
    UriStrVec actual_uris   = genUris({2, 3}, {0, 1}, "actual_");

    BlockBuffersExpect block_buffers_expect = {6, kFakeLayerNum, kFakeIovSize};
    EXPECT_CALL(*transfer_client_, SaveKvCaches(Eq(expected_uris), BlockBuffersMatcher(block_buffers_expect)))
        .WillOnce(Return(SaveKvCachesReturnType({ClientErrorCode::ER_OK, actual_uris})));

    Locations expected_actual_locations = genFullotherLocations({2, 3}, {0, 1}, "actual_");
    EXPECT_CALL(*meta_clients_[tp_rank],
                FinishWrite(Eq("finish_write_trace_1"),             // trace_id
                            write_session_id,                       // write_session_id
                            Eq(BlockMask(static_cast<size_t>(2))),  // success_block
                            Eq(expected_actual_locations)           // locations
                            ))
        .WillOnce(Return(ClientErrorCode::ER_OK));

    auto async_context = remote_connectors_[tp_rank]->asyncWrite(kv_cache_resouce, meta);
    async_context->waitDone();
    ASSERT_TRUE(async_context->success());
}

TEST_F(RemoteConnectorMockFullLinearTest,
       test_write_success_broadcast_success_actual_locations_different_with_empty_write_locations) {
    auto kv_cache_resouce        = std::make_shared<KVCacheResourceV1>();
    kv_cache_resouce->cache_keys = {1, 2, 3};
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({1, 2, 3}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({11, 12, 13}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({21, 22, 23}));

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
    async_context->waitDone();
    ASSERT_TRUE(async_context->success());
}

TEST_F(RemoteConnectorMockFullLinearTest, test_write_success_broadcast_success_with_part_empty_linear) {
    auto kv_cache_resouce        = std::make_shared<KVCacheResourceV1>();
    kv_cache_resouce->cache_keys = {1, 2, 3, 4};
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({1, 2, 3, 4}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({11, -1, 13, 14}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({21, -1, 23, 24}));

    auto          meta    = std::make_shared<RemoteConnectorMeta>("", "trace_1");
    size_t        tp_rank = 0;
    std::string   write_session_id("write_session_id_1");
    Locations     expected_write_locations = genFullotherLocations({2, 3, 4}, {1, 2});
    WriteLocation write_location({write_session_id, static_cast<size_t>(1), expected_write_locations});
    EXPECT_CALL(
        *meta_clients_[tp_rank],
        StartWrite(Eq("start_write_trace_1"),                                           // trace_id
                   std::vector<int64_t>({1, 2, 3, 4}),                                  // keys
                   _,                                                                   // tokens
                   Eq(std::vector<std::string>({"F0L1L2", "F0", "F0L1L2", "F0L1L2"})),  // location_spec_group_names
                   _                                                                    // write_timeout_seconds
                   ))
        .WillOnce(Return(StartWriteReturnType({ClientErrorCode::ER_OK, write_location})));

    UriStrVec expected_uris = genUris({2, 3, 4}, {1, 2});
    UriStrVec actual_uris   = genUris({2, 3, 4}, {1, 2}, "actual_");

    BlockBuffersExpect block_buffers_expect = {7, kFakeLayerNum, kFakeIovSize};
    EXPECT_CALL(*transfer_client_, SaveKvCaches(Eq(expected_uris), BlockBuffersMatcher(block_buffers_expect)))
        .WillOnce(Return(SaveKvCachesReturnType({ClientErrorCode::ER_OK, actual_uris})));

    Locations expected_actual_locations = genFullotherLocations({2, 3, 4}, {1, 2}, "actual_");
    EXPECT_CALL(*meta_clients_[tp_rank],
                FinishWrite(Eq("finish_write_trace_1"),             // trace_id
                            write_session_id,                       // write_session_id
                            Eq(BlockMask(static_cast<size_t>(3))),  // success_block
                            Eq(expected_actual_locations)           // locations
                            ))
        .WillOnce(Return(ClientErrorCode::ER_OK));

    auto async_context = remote_connectors_[tp_rank]->asyncWrite(kv_cache_resouce, meta);
    async_context->waitDone();
    ASSERT_TRUE(async_context->success());
}

// In fact, this situation should not occur
TEST_F(RemoteConnectorMockFullLinearTest, test_write_success_broadcast_success_with_all_empty_linear) {
    auto kv_cache_resouce        = std::make_shared<KVCacheResourceV1>();
    kv_cache_resouce->cache_keys = {1, 2, 3, 4};
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({1, 2, 3, 4}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({-1, -1, -1, -1}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({-1, -1, -1, -1}));

    auto          meta    = std::make_shared<RemoteConnectorMeta>("", "trace_1");
    size_t        tp_rank = 0;
    std::string   write_session_id("write_session_id_1");
    Locations     expected_write_locations = genFullotherLocations({2, 3, 4}, {});
    WriteLocation write_location({write_session_id, static_cast<size_t>(1), expected_write_locations});
    EXPECT_CALL(*meta_clients_[tp_rank],
                StartWrite(Eq("start_write_trace_1"),                               // trace_id
                           std::vector<int64_t>({1, 2, 3, 4}),                      // keys
                           _,                                                       // tokens
                           Eq(std::vector<std::string>({"F0", "F0", "F0", "F0"})),  // location_spec_group_names
                           _                                                        // write_timeout_seconds
                           ))
        .WillOnce(Return(StartWriteReturnType({ClientErrorCode::ER_OK, write_location})));

    UriStrVec expected_uris = genUris({2, 3, 4}, {});
    UriStrVec actual_uris   = genUris({2, 3, 4}, {}, "actual_");

    BlockBuffersExpect block_buffers_expect = {3, kFakeLayerNum, kFakeIovSize};
    EXPECT_CALL(*transfer_client_, SaveKvCaches(Eq(expected_uris), BlockBuffersMatcher(block_buffers_expect)))
        .WillOnce(Return(SaveKvCachesReturnType({ClientErrorCode::ER_OK, actual_uris})));

    Locations expected_actual_locations = genFullotherLocations({2, 3, 4}, {}, "actual_");
    EXPECT_CALL(*meta_clients_[tp_rank],
                FinishWrite(Eq("finish_write_trace_1"),             // trace_id
                            write_session_id,                       // write_session_id
                            Eq(BlockMask(static_cast<size_t>(3))),  // success_block
                            Eq(expected_actual_locations)           // locations
                            ))
        .WillOnce(Return(ClientErrorCode::ER_OK));

    auto async_context = remote_connectors_[tp_rank]->asyncWrite(kv_cache_resouce, meta);
    async_context->waitDone();
    ASSERT_TRUE(async_context->success());
}

TEST_F(RemoteConnectorMockFullLinearTest, test_write_success_broadcast_success_actual_locations_same) {
    auto kv_cache_resouce        = std::make_shared<KVCacheResourceV1>();
    kv_cache_resouce->cache_keys = {1, 2, 3};
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({1, 2, 3}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({11, 12, 13}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({21, 22, 23}));
    auto          meta    = std::make_shared<RemoteConnectorMeta>("", "trace_2");
    size_t        tp_rank = 0;
    std::string   write_session_id("write_session_id_2");
    Locations     expected_locations = genFullotherLocations(kv_cache_resouce->cache_keys, {0, 1, 2});
    WriteLocation write_location({write_session_id, static_cast<size_t>(0), expected_locations});
    EXPECT_CALL(*meta_clients_[tp_rank],
                StartWrite(Eq("start_write_trace_2"),        // trace_id
                           std::vector<int64_t>({1, 2, 3}),  // keys
                           _,                                // tokens
                           Eq(std::vector<std::string>()),   // location_spec_group_names
                           _                                 // write_timeout_seconds
                           ))
        .WillOnce(Return(StartWriteReturnType({ClientErrorCode::ER_OK, write_location})));

    UriStrVec expected_uris = genUris({1, 2, 3}, {0, 1, 2});

    BlockBuffersExpect block_buffers_expect = {9, kFakeLayerNum, kFakeIovSize};
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
    async_context->waitDone();
    ASSERT_TRUE(async_context->success());
}

TEST_F(RemoteConnectorMockFullLinearTest, test_match_fail) {
    auto kv_cache_resouce        = std::make_shared<KVCacheResourceV1>();
    kv_cache_resouce->cache_keys = {1, 2, 3};
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({1, 2, 3}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({11, 12, 13}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({21, 22, 23}));
    auto   meta    = std::make_shared<RemoteConnectorMeta>("", "trace_1");
    size_t tp_rank = 0;
    EXPECT_CALL(*meta_clients_[tp_rank],
                MatchLocation(Eq("match_trace_1"),                    // trace_id
                              _,                                      // query_type
                              std::vector<int64_t>({1, 2, 3}),        // keys
                              _,                                      // tokens
                              Eq(BlockMask(static_cast<size_t>(0))),  // block_mask
                              _,                                      // sw_size
                              _                                       // location_spec_names
                              ))
        .WillOnce(Return(MatchLocationReturnType({ClientErrorCode::ER_INVALID_GRPCSTATUS, {}})));

    EXPECT_CALL(*transfer_client_, LoadKvCaches(_, _)).Times(0);
    auto async_context = remote_connectors_[tp_rank]->asyncRead(kv_cache_resouce, meta);
    async_context->waitDone();
    ASSERT_TRUE(async_context->success());
    auto remote_async_context = std::dynamic_pointer_cast<RemoteConnectorAsyncContext>(async_context);
    ASSERT_NE(nullptr, remote_async_context);
    ASSERT_EQ(RemoteConnectorAsyncContext::State::RCS_READ_MATCH_ERROR, remote_async_context->state());
}

TEST_F(RemoteConnectorMockFullLinearTest, test_match_success_load_fail) {
    auto kv_cache_resouce        = std::make_shared<KVCacheResourceV1>();
    kv_cache_resouce->cache_keys = {1, 2, 3};
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({1, 2, 3}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({11, 12, 13}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({21, 22, 23}));
    auto      meta               = std::make_shared<RemoteConnectorMeta>("", "trace_1");
    size_t    tp_rank            = 0;
    Locations expected_locations = genFullotherLocations(kv_cache_resouce->cache_keys, {0, 1, 2});
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
    UriStrVec          expected_uris        = genUris({1, 2, 3}, {2});
    BlockBuffersExpect block_buffers_expect = {5, kFakeLayerNum, kFakeIovSize};
    EXPECT_CALL(*transfer_client_, LoadKvCaches(Eq(expected_uris), BlockBuffersMatcher(block_buffers_expect)))
        .WillOnce(Return(ClientErrorCode::ER_SDK_TIMEOUT));

    auto async_context = remote_connectors_[tp_rank]->asyncRead(kv_cache_resouce, meta);
    async_context->waitDone();
    ASSERT_FALSE(async_context->success());
    auto remote_async_context = std::dynamic_pointer_cast<RemoteConnectorAsyncContext>(async_context);
    ASSERT_NE(nullptr, remote_async_context);
    ASSERT_EQ(RemoteConnectorAsyncContext::State::RCS_ERROR, remote_async_context->state());
}

TEST_F(RemoteConnectorMockFullLinearTest, test_match_success_broadcast_grpc_fail) {
    auto kv_cache_resouce        = std::make_shared<KVCacheResourceV1>();
    kv_cache_resouce->cache_keys = {1, 2, 3};
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({1, 2, 3}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({11, 12, 13}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({21, 22, 23}));
    auto      meta               = std::make_shared<RemoteConnectorMeta>("", "trace_1");
    size_t    tp_rank            = 0;
    Locations expected_locations = genFullotherLocations(kv_cache_resouce->cache_keys, {0, 1, 2});
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
    EXPECT_CALL(*transfer_client_, LoadKvCaches(_, _)).Times(0);

    servers_[tp_rank]->hack_grpc_status(true);
    auto async_context = remote_connectors_[tp_rank]->asyncRead(kv_cache_resouce, meta);
    async_context->waitDone();
    ASSERT_FALSE(async_context->success());
    auto remote_async_context = std::dynamic_pointer_cast<RemoteConnectorAsyncContext>(async_context);
    ASSERT_NE(nullptr, remote_async_context);
    ASSERT_EQ(RemoteConnectorAsyncContext::State::RCS_ERROR, remote_async_context->state());
}

TEST_F(RemoteConnectorMockFullLinearTest, test_start_write_fail) {
    auto kv_cache_resouce        = std::make_shared<KVCacheResourceV1>();
    kv_cache_resouce->cache_keys = {1, 2, 3};
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({1, 2, 3}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({11, 12, 13}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({21, 22, 23}));
    auto          meta    = std::make_shared<RemoteConnectorMeta>("", "trace_1");
    size_t        tp_rank = 0;
    std::string   write_session_id("write_session_id_1");
    Locations     expected_write_locations = genFullotherLocations({1, 2, 3}, {0, 1, 2});
    WriteLocation write_location({write_session_id, static_cast<size_t>(0), expected_write_locations});
    EXPECT_CALL(*meta_clients_[tp_rank],
                StartWrite(Eq("start_write_trace_1"),        // trace_id
                           std::vector<int64_t>({1, 2, 3}),  // keys
                           _,                                // tokens
                           Eq(std::vector<std::string>()),   // location_spec_group_names
                           _                                 // write_timeout_seconds
                           ))
        .WillOnce(Return(StartWriteReturnType({ClientErrorCode::ER_INVALID_GRPCSTATUS, {}})));

    EXPECT_CALL(*transfer_client_, SaveKvCaches(_, _)).Times(0);
    EXPECT_CALL(*meta_clients_[tp_rank], FinishWrite(_, _, _, _)).Times(0);

    auto async_context = remote_connectors_[tp_rank]->asyncWrite(kv_cache_resouce, meta);
    async_context->waitDone();
    ASSERT_FALSE(async_context->success());
    auto remote_async_context = std::dynamic_pointer_cast<RemoteConnectorAsyncContext>(async_context);
    ASSERT_NE(nullptr, remote_async_context);
    ASSERT_EQ(RemoteConnectorAsyncContext::State::RCS_ERROR, remote_async_context->state());
}

TEST_F(RemoteConnectorMockFullLinearTest, test_write_invalid_block_ids) {
    auto kv_cache_resouce        = std::make_shared<KVCacheResourceV1>();
    kv_cache_resouce->cache_keys = {1, 2, 3};
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({1, 2, 3}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({11, -1, 13}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({21, 22, 23}));
    auto   meta    = std::make_shared<RemoteConnectorMeta>("", "trace_1");
    size_t tp_rank = 0;
    EXPECT_CALL(*meta_clients_[tp_rank], StartWrite(_, _, _, _, _)).Times(0);
    EXPECT_CALL(*transfer_client_, SaveKvCaches(_, _)).Times(0);
    EXPECT_CALL(*meta_clients_[tp_rank], FinishWrite(_, _, _, _)).Times(0);

    auto async_context = remote_connectors_[tp_rank]->asyncWrite(kv_cache_resouce, meta);
    async_context->waitDone();
    ASSERT_FALSE(async_context->success());
    auto remote_async_context = std::dynamic_pointer_cast<RemoteConnectorAsyncContext>(async_context);
    ASSERT_NE(nullptr, remote_async_context);
    ASSERT_EQ(RemoteConnectorAsyncContext::State::RCS_ERROR, remote_async_context->state());
}

TEST_F(RemoteConnectorMockFullLinearTest, test_start_write_success_broadcast_success_finish_write_fail) {
    auto kv_cache_resouce        = std::make_shared<KVCacheResourceV1>();
    kv_cache_resouce->cache_keys = {1, 2, 3};
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({1, 2, 3}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({11, 12, 13}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({21, 22, 23}));
    auto          meta    = std::make_shared<RemoteConnectorMeta>("", "trace_2");
    size_t        tp_rank = 0;
    std::string   write_session_id("write_session_id_2");
    Locations     expected_write_locations = genFullotherLocations({1, 2, 3}, {0, 1, 2});
    WriteLocation write_location({write_session_id, static_cast<size_t>(0), expected_write_locations});
    EXPECT_CALL(*meta_clients_[tp_rank],
                StartWrite(Eq("start_write_trace_2"),        // trace_id
                           std::vector<int64_t>({1, 2, 3}),  // keys
                           _,                                // tokens
                           Eq(std::vector<std::string>()),   // location_spec_group_names
                           _                                 // write_timeout_seconds
                           ))
        .WillOnce(Return(StartWriteReturnType({ClientErrorCode::ER_OK, write_location})));

    UriStrVec          expected_uris        = genUris({1, 2, 3}, {0, 1, 2});
    BlockBuffersExpect block_buffers_expect = {9, kFakeLayerNum, kFakeIovSize};
    EXPECT_CALL(*transfer_client_, SaveKvCaches(Eq(expected_uris), BlockBuffersMatcher(block_buffers_expect)))
        .WillOnce(Return(SaveKvCachesReturnType({ClientErrorCode::ER_OK, expected_uris})));

    EXPECT_CALL(*meta_clients_[tp_rank],
                FinishWrite(Eq("finish_write_trace_2"),             // trace_id
                            write_session_id,                       // write_session_id
                            Eq(BlockMask(static_cast<size_t>(3))),  // success_block
                            Eq(Locations({}))                       // locations
                            ))
        .WillOnce(Return(ClientErrorCode::ER_INVALID_GRPCSTATUS));

    auto async_context = remote_connectors_[tp_rank]->asyncWrite(kv_cache_resouce, meta);
    async_context->waitDone();
    ASSERT_FALSE(async_context->success());
    auto remote_async_context = std::dynamic_pointer_cast<RemoteConnectorAsyncContext>(async_context);
    ASSERT_NE(nullptr, remote_async_context);
    ASSERT_EQ(RemoteConnectorAsyncContext::State::RCS_ERROR, remote_async_context->state());
}

TEST_F(RemoteConnectorMockFullLinearTest, test_start_write_success_broadcast_success_save_fail) {
    auto kv_cache_resouce        = std::make_shared<KVCacheResourceV1>();
    kv_cache_resouce->cache_keys = {1, 2, 3};
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({1, 2, 3}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({11, 12, 13}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({21, 22, 23}));
    auto          meta    = std::make_shared<RemoteConnectorMeta>("", "trace_2");
    size_t        tp_rank = 0;
    std::string   write_session_id("write_session_id_2");
    Locations     expected_write_locations = genFullotherLocations({1, 2, 3}, {0, 1, 2});
    WriteLocation write_location({write_session_id, static_cast<size_t>(0), expected_write_locations});
    EXPECT_CALL(*meta_clients_[tp_rank],
                StartWrite(Eq("start_write_trace_2"),        // trace_id
                           std::vector<int64_t>({1, 2, 3}),  // keys
                           _,                                // tokens
                           Eq(std::vector<std::string>()),   // location_spec_group_names
                           _                                 // write_timeout_seconds
                           ))
        .WillOnce(Return(StartWriteReturnType({ClientErrorCode::ER_OK, write_location})));

    UriStrVec          expected_uris        = genUris({1, 2, 3}, {0, 1, 2});
    BlockBuffersExpect block_buffers_expect = {9, kFakeLayerNum, kFakeIovSize};
    EXPECT_CALL(*transfer_client_, SaveKvCaches(Eq(expected_uris), BlockBuffersMatcher(block_buffers_expect)))
        .WillOnce(Return(SaveKvCachesReturnType({ClientErrorCode::ER_SDK_TIMEOUT, UriStrVec({})})));

    EXPECT_CALL(*meta_clients_[tp_rank],
                FinishWrite(Eq("finish_write_trace_2"),             // trace_id
                            write_session_id,                       // write_session_id
                            Eq(BlockMask(static_cast<size_t>(0))),  // success_block
                            Eq(Locations({}))                       // locations
                            ))
        .WillOnce(Return(ClientErrorCode::ER_OK));

    auto async_context = remote_connectors_[tp_rank]->asyncWrite(kv_cache_resouce, meta);
    async_context->waitDone();
    ASSERT_FALSE(async_context->success());
    auto remote_async_context = std::dynamic_pointer_cast<RemoteConnectorAsyncContext>(async_context);
    ASSERT_NE(nullptr, remote_async_context);
    ASSERT_EQ(RemoteConnectorAsyncContext::State::RCS_ERROR, remote_async_context->state());
}

TEST_F(RemoteConnectorMockFullLinearTest, test_start_write_success_broadcast_grpc_fail) {
    auto kv_cache_resouce        = std::make_shared<KVCacheResourceV1>();
    kv_cache_resouce->cache_keys = {1, 2, 3};
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({1, 2, 3}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({11, 12, 13}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({21, 22, 23}));
    auto          meta    = std::make_shared<RemoteConnectorMeta>("", "trace_2");
    size_t        tp_rank = 0;
    std::string   write_session_id("write_session_id_2");
    Locations     expected_write_locations = genFullotherLocations({1, 2, 3}, {0, 1, 2});
    WriteLocation write_location({write_session_id, static_cast<size_t>(0), expected_write_locations});
    EXPECT_CALL(*meta_clients_[tp_rank],
                StartWrite(Eq("start_write_trace_2"),        // trace_id
                           std::vector<int64_t>({1, 2, 3}),  // keys
                           _,                                // tokens
                           Eq(std::vector<std::string>()),   // location_spec_group_names
                           _                                 // write_timeout_seconds
                           ))
        .WillOnce(Return(StartWriteReturnType({ClientErrorCode::ER_OK, write_location})));

    EXPECT_CALL(*transfer_client_, SaveKvCaches(_, _)).Times(0);

    EXPECT_CALL(*meta_clients_[tp_rank],
                FinishWrite(Eq("finish_write_trace_2"),             // trace_id
                            write_session_id,                       // write_session_id
                            Eq(BlockMask(static_cast<size_t>(0))),  // success_block
                            Eq(Locations({}))                       // locations
                            ))
        .WillOnce(Return(ClientErrorCode::ER_OK));

    servers_[tp_rank]->hack_grpc_status(true);
    auto async_context = remote_connectors_[tp_rank]->asyncWrite(kv_cache_resouce, meta);
    async_context->waitDone();
    ASSERT_FALSE(async_context->success());
    auto remote_async_context = std::dynamic_pointer_cast<RemoteConnectorAsyncContext>(async_context);
    ASSERT_NE(nullptr, remote_async_context);
    ASSERT_EQ(RemoteConnectorAsyncContext::State::RCS_ERROR, remote_async_context->state());
}

TEST_F(RemoteConnectorMockFullLinearTest, test_threadpool_full) {
    auto kv_cache_resouce        = std::make_shared<KVCacheResourceV1>();
    kv_cache_resouce->cache_keys = {1, 2, 3};
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({1, 2, 3}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({11, 12, 13}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({21, 22, 23}));
    auto   meta    = std::make_shared<RemoteConnectorMeta>("", "trace");
    size_t tp_rank = 0;
    remote_connectors_[tp_rank]->thread_pool_->stop();
    remote_connectors_[tp_rank]->thread_pool_->waitFinish();
    remote_connectors_[tp_rank]->thread_pool_.reset();
    remote_connectors_[tp_rank]->thread_pool_ =
        std::make_unique<autil::LockFreeThreadPool>(1, /* queueSize= */ 0, nullptr, "RECOThreadPool");

    EXPECT_CALL(*meta_clients_[tp_rank], MatchLocation(_, _, _, _, _, _, _)).Times(0);
    EXPECT_CALL(*transfer_client_, LoadKvCaches(_, _)).Times(0);
    ASSERT_EQ(nullptr, remote_connectors_[tp_rank]->asyncRead(kv_cache_resouce, meta));

    EXPECT_CALL(*meta_clients_[tp_rank], StartWrite(_, _, _, _, _)).Times(0);
    EXPECT_CALL(*transfer_client_, SaveKvCaches(_, _)).Times(0);
    EXPECT_CALL(*meta_clients_[tp_rank], FinishWrite(_, _, _, _)).Times(0);
    ASSERT_EQ(nullptr, remote_connectors_[tp_rank]->asyncWrite(kv_cache_resouce, meta));
}

}  // namespace test
}  // namespace rtp_llm