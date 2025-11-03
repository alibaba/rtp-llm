#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache/connector/remote_connector/test/RemoteConnectorMockTestBase.h"
#include "rtp_llm/cpp/cache/connector/Meta.h"

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
    std::string          unique_id_ = "";  // TODO : support lora (remote connector)
    std::vector<int64_t> tokens_;          // TODO : get tokens (remote connector)
};

// TODO : remove this, use ture HybridLayerKVCacheAllocator
class FakeHybridLayerKVCacheAllocator: public KVCacheAllocator {
public:
    FakeHybridLayerKVCacheAllocator(const CacheConfig&          config,
                                    rtp_llm::DeviceBase*        device,
                                    const std::vector<int32_t>& full_group_ids,
                                    const std::vector<int32_t>& other_group_ids):
        KVCacheAllocator(config, device) {
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
    CacheLayerLayout allLayerCacheBase() const override {
        return fake_layout_;
    }
    int singleBatchNeedBlocks(const BatchKVCacheResourcePtr& batch_kv_cache_resource, int seq_len) const override {
        return 0;
    }
    int getNeedBlocks(const MallocInfo& malloc_info) const override {
        return 0;
    }
    // not support scale now
    std::vector<BlockInfo> convertIndexToBuffer(int layer_id, int block_id) const override {
        return {BlockInfo{true, 0, 0, static_cast<void*>(fake_buffer_.data()), fake_buffer_size_}};
    }

    bool init() override {
        return true;
    }
    void          free(const FreeInfo& free_info) override {}
    void          insertIntoCache(const InsertInfo& insert_info) override {}
    BlockAddrInfo convertIndexToAddr(int layer_id, int block_id) const override {
        return {};
    }
    std::vector<BlockInfo>
    convertIndexToBuffer(int layer_id, int block_id, int partition_count, int partition_id) const override {
        return {};
    }
    std::shared_ptr<KVCacheResource> incrKVCacheRef(const KVCacheResource& kvcache_resource,
                                                    const CacheKeysType&   cache_keys,
                                                    bool                   is_connector = false) override {
        return {};
    }
    void decrKVCacheRef(const KVCacheResource& kvcache_resource, bool is_connector = false) override {
        return;
    }

    bool updateKVBlock(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                       const std::vector<int>&        block_src_batch,
                       bool                           copy_last_block,
                       std::vector<BlockIdPair>&      block_update_mapping) override {
        return true;
    }
    int seqSizePerBlock() const override {
        return 0;
    }

private:
    MallocResult incrMalloc(const MallocInfo& malloc_info) override {
        return {};
    }
    MallocResult initMallocForCommonLen(const MallocInfo& malloc_info) override {
        return {};
    }

    CacheLayerLayout                  fake_layout_;
    constexpr static size_t           fake_buffer_size_ = kFakeIovSize / sizeof(int8_t);
    inline static std::vector<int8_t> fake_buffer_      = std::vector<int8_t>(fake_buffer_size_, 0);
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
            EXPECT_CALL(*mock_client_factory_, CreateMetaClient(_, _))
                .WillOnce(Invoke(
                    [&](const std::string&, const kv_cache_manager::InitParams&) { return std::move(meta_client); }));
            auto allocator = std::make_shared<FakeHybridLayerKVCacheAllocator>(
                cache_config_, device_, full_group_ids_, other_group_ids_);
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
        cache_config_.layer_all_num      = layer_num;
        cache_config_.block_num          = block_num;
        cache_config_.seq_size_per_block = seq_size_per_block;

        auto mha_spec                = std::make_shared<MHAKVCacheSpec>();
        mha_spec->layer_num          = layer_num;
        mha_spec->local_head_num_kv  = 8;
        mha_spec->size_per_head      = 128;
        mha_spec->seq_size_per_block = seq_size_per_block;
        mha_spec->dtype              = rtp_llm::DataType::TYPE_FP16;
        mha_spec->type               = KVCacheType::MultiHeadAttention;

        cache_config_.cache_specs.push_back(mha_spec);
        cache_config_.block_size_bytes = static_cast<size_t>(mha_spec->block_size_bytes() * mha_spec->layer_num);
        cache_config_.dtype            = rtp_llm::DataType::TYPE_FP16;
        std::vector<int> layer_ids(layer_num);
        for (int i = 0; i < layer_num; ++i) {
            layer_ids[i] = i;
        }
        cache_config_.layer_ids.push_back(layer_ids);
    }
};

TEST_F(RemoteConnectorMockFullLinearTest, test_async_match_and_async_read_with_gpu_reuse_len_zero) {
    // match
    auto kv_cache_resouce        = std::make_shared<KVCacheResource>();
    kv_cache_resouce->cache_keys = {1, 2, 3, 4};
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({1, 2, 3, 4}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({11, 12, 13, 14}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({21, 22, 23, 24}));
    auto      meta               = std::make_shared<MetaImpl>(false, true, "trace_1");
    size_t    tp_rank            = 0;
    Locations expected_locations = genFullotherLocations({1, 2, 3}, {0, 1, 2});

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
        UriStrVec          expected_uris        = genUris({1, 2, 3}, {2});
        BlockBuffersExpect block_buffers_expect = {5, kFakeLayerNum, kFakeIovSize};
        EXPECT_CALL(*transfer_client_, LoadKvCaches(Eq(expected_uris), BlockBuffersMatcher(block_buffers_expect)))
            .WillOnce(Return(ClientErrorCode::ER_OK));

        const int gpu_reuse_num = static_cast<int>(kv_cache_resouce->reuseBlockNum());   // 0
        const int matched_num   = static_cast<int>(match_context->matchedBlockCount());  // 3

        int  start_read_block_index = gpu_reuse_num;
        int  read_block_num         = matched_num - gpu_reuse_num;
        auto read_context           = remote_connectors_[tp_rank]->asyncRead(
            kv_cache_resouce, meta, match_context, start_read_block_index, read_block_num);
        waitAsyncContextDone(read_context);
        ASSERT_TRUE(read_context->success());
        ASSERT_EQ(kv_cache_resouce->remoteReuseBlockNum(), 3);
        kv_cache_resouce->setRemoteReuseBlockNum(0);
    }
    {
        UriStrVec          expected_uris        = genUris({2, 3}, {1});
        BlockBuffersExpect block_buffers_expect = {4, kFakeLayerNum, kFakeIovSize};
        EXPECT_CALL(*transfer_client_, LoadKvCaches(Eq(expected_uris), BlockBuffersMatcher(block_buffers_expect)))
            .WillOnce(Return(ClientErrorCode::ER_OK));

        const int gpu_reuse_num          = static_cast<int>(kv_cache_resouce->reuseBlockNum());   // 0
        const int matched_num            = static_cast<int>(match_context->matchedBlockCount());  // 3
        int       start_read_block_index = gpu_reuse_num + 1;
        int       read_block_num         = matched_num - gpu_reuse_num - 1;
        auto      read_context           = remote_connectors_[tp_rank]->asyncRead(
            kv_cache_resouce, meta, match_context, start_read_block_index, read_block_num);
        waitAsyncContextDone(read_context);
        ASSERT_TRUE(read_context->success());
        ASSERT_EQ(kv_cache_resouce->remoteReuseBlockNum(), 2);
        kv_cache_resouce->setRemoteReuseBlockNum(0);
    }
    {
        const int gpu_reuse_num          = static_cast<int>(kv_cache_resouce->reuseBlockNum());   // 0
        const int matched_num            = static_cast<int>(match_context->matchedBlockCount());  // 3
        int       start_read_block_index = gpu_reuse_num + 4;
        int       read_block_num         = matched_num - gpu_reuse_num - 4;
        auto      read_context           = remote_connectors_[tp_rank]->asyncRead(
            kv_cache_resouce, meta, match_context, start_read_block_index, read_block_num);
        waitAsyncContextDone(read_context);
        ASSERT_TRUE(read_context->success());
        ASSERT_EQ(kv_cache_resouce->remoteReuseBlockNum(), 0);
        kv_cache_resouce->setRemoteReuseBlockNum(0);
    }
}

TEST_F(RemoteConnectorMockFullLinearTest, test_async_match_and_async_read_with_gpu_reuse_len_not_zero) {
    // match
    auto kv_cache_resouce        = std::make_shared<KVCacheResource>();
    kv_cache_resouce->cache_keys = {1, 2, 3, 4};
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({1, 2, 3, 4}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({11, 12, 13, 14}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({21, 22, 23, 24}));
    kv_cache_resouce->setDeviceReuseBlockNum(1);
    auto      meta               = std::make_shared<MetaImpl>(false, true, "trace_2");
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
    auto match_context = remote_connectors_[tp_rank]->asyncMatch(kv_cache_resouce, meta);
    waitAsyncContextDone(match_context);
    ASSERT_TRUE(match_context->success());
    ASSERT_EQ(match_context->matchedBlockCount(), 3);
    {
        UriStrVec          expected_uris        = genUris({2, 3}, {1});
        BlockBuffersExpect block_buffers_expect = {4, kFakeLayerNum, kFakeIovSize};
        EXPECT_CALL(*transfer_client_, LoadKvCaches(Eq(expected_uris), BlockBuffersMatcher(block_buffers_expect)))
            .WillOnce(Return(ClientErrorCode::ER_OK));

        const int gpu_reuse_num          = static_cast<int>(kv_cache_resouce->reuseBlockNum());   // 1
        const int matched_num            = static_cast<int>(match_context->matchedBlockCount());  // 3
        int       start_read_block_index = gpu_reuse_num;
        int       read_block_num         = matched_num - gpu_reuse_num;
        auto      read_context           = remote_connectors_[tp_rank]->asyncRead(
            kv_cache_resouce, meta, match_context, start_read_block_index, read_block_num);
        waitAsyncContextDone(read_context);
        ASSERT_TRUE(read_context->success());
        ASSERT_EQ(kv_cache_resouce->remoteReuseBlockNum(), 2);
        kv_cache_resouce->setRemoteReuseBlockNum(0);
    }
    {
        UriStrVec          expected_uris        = genUris({3}, {0});
        BlockBuffersExpect block_buffers_expect = {3, kFakeLayerNum, kFakeIovSize};
        EXPECT_CALL(*transfer_client_, LoadKvCaches(Eq(expected_uris), BlockBuffersMatcher(block_buffers_expect)))
            .WillOnce(Return(ClientErrorCode::ER_OK));

        const int gpu_reuse_num          = static_cast<int>(kv_cache_resouce->reuseBlockNum());   // 1
        const int matched_num            = static_cast<int>(match_context->matchedBlockCount());  // 3
        int       start_read_block_index = gpu_reuse_num + 1;
        int       read_block_num         = matched_num - gpu_reuse_num - 1;
        auto      read_context           = remote_connectors_[tp_rank]->asyncRead(
            kv_cache_resouce, meta, match_context, start_read_block_index, read_block_num);
        waitAsyncContextDone(read_context);
        ASSERT_TRUE(read_context->success());
        ASSERT_EQ(kv_cache_resouce->remoteReuseBlockNum(), 1);
        kv_cache_resouce->setRemoteReuseBlockNum(0);
    }
    {
        const int gpu_reuse_num          = static_cast<int>(kv_cache_resouce->reuseBlockNum());   // 1
        const int matched_num            = static_cast<int>(match_context->matchedBlockCount());  // 3
        int       start_read_block_index = gpu_reuse_num + 3;
        int       read_block_num         = matched_num - gpu_reuse_num - 3;
        auto      read_context           = remote_connectors_[tp_rank]->asyncRead(
            kv_cache_resouce, meta, match_context, start_read_block_index, read_block_num);
        waitAsyncContextDone(read_context);
        ASSERT_TRUE(read_context->success());
        ASSERT_EQ(kv_cache_resouce->remoteReuseBlockNum(), 0);
        kv_cache_resouce->setRemoteReuseBlockNum(0);
    }
}

TEST_F(RemoteConnectorMockFullLinearTest, test_read_success_broadcast_success_with_part_empty_linear) {
    auto kv_cache_resouce        = std::make_shared<KVCacheResource>();
    kv_cache_resouce->cache_keys = {1, 2, 3, 4, 5};
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({1, 2, 3, 4, 5}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({11, 12, 13, 14, 15}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({21, 22, 23, 24, 25}));
    kv_cache_resouce->setDeviceReuseBlockNum(1);
    auto      meta               = std::make_shared<MetaImpl>(false, true, "trace_2");
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

    auto match_context = remote_connectors_[tp_rank]->asyncMatch(kv_cache_resouce, meta);
    waitAsyncContextDone(match_context);
    ASSERT_TRUE(match_context->success());
    ASSERT_EQ(match_context->matchedBlockCount(), 4);

    const int gpu_reuse_num          = static_cast<int>(kv_cache_resouce->reuseBlockNum());   // 1
    const int matched_num            = static_cast<int>(match_context->matchedBlockCount());  // 4
    int       start_read_block_index = gpu_reuse_num;
    int       read_block_num         = matched_num - gpu_reuse_num;
    auto      read_context           = remote_connectors_[tp_rank]->asyncRead(
        kv_cache_resouce, meta, match_context, start_read_block_index, read_block_num);
    waitAsyncContextDone(read_context);
    ASSERT_TRUE(read_context->success());
}

TEST_F(RemoteConnectorMockFullLinearTest, test_read_success_broadcast_success_with_all_empty_linear) {
    auto kv_cache_resouce        = std::make_shared<KVCacheResource>();
    kv_cache_resouce->cache_keys = {1, 2, 3, 4, 5};
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({1, 2, 3, 4, 5}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({11, 12, 13, 14, 15}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({21, 22, 23, 24, 25}));
    kv_cache_resouce->setDeviceReuseBlockNum(1);
    auto      meta               = std::make_shared<MetaImpl>(false, true, "trace_2");
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

    auto match_context = remote_connectors_[tp_rank]->asyncMatch(kv_cache_resouce, meta);
    waitAsyncContextDone(match_context);
    ASSERT_TRUE(match_context->success());
    ASSERT_EQ(match_context->matchedBlockCount(), 4);

    const int gpu_reuse_num          = static_cast<int>(kv_cache_resouce->reuseBlockNum());   // 1
    const int matched_num            = static_cast<int>(match_context->matchedBlockCount());  // 4
    int       start_read_block_index = gpu_reuse_num;
    int       read_block_num         = matched_num - gpu_reuse_num;
    auto      read_context           = remote_connectors_[tp_rank]->asyncRead(
        kv_cache_resouce, meta, match_context, start_read_block_index, read_block_num);
    waitAsyncContextDone(read_context);
    ASSERT_TRUE(read_context->success());
}

TEST_F(RemoteConnectorMockFullLinearTest, test_write_success_broadcast_success_actual_locations_different) {
    auto kv_cache_resouce = std::make_shared<KVCacheResource>();
    kv_cache_resouce->setLastBlockAligned(true);
    kv_cache_resouce->cache_keys = {1, 2, 3};
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({1, 2, 3}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({11, 12, 13}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({21, 22, 23}));
    auto          meta    = std::make_shared<MetaImpl>(false, true, "trace_1");
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
    waitAsyncContextDone(async_context);
    ASSERT_TRUE(async_context->success());
}

TEST_F(RemoteConnectorMockFullLinearTest,
       test_write_success_broadcast_success_actual_locations_different_with_block_mask) {
    auto kv_cache_resouce = std::make_shared<KVCacheResource>();
    kv_cache_resouce->setLastBlockAligned(true);
    kv_cache_resouce->cache_keys = {1, 2, 3};
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({1, 2, 3}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({11, 12, 13}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({21, 22, 23}));

    auto          meta    = std::make_shared<MetaImpl>(false, true, "trace_1");
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
    waitAsyncContextDone(async_context);
    ASSERT_TRUE(async_context->success());
}

TEST_F(RemoteConnectorMockFullLinearTest,
       test_write_success_broadcast_success_actual_locations_different_with_empty_write_locations) {
    auto kv_cache_resouce = std::make_shared<KVCacheResource>();
    kv_cache_resouce->setLastBlockAligned(true);
    kv_cache_resouce->cache_keys = {1, 2, 3};
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({1, 2, 3}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({11, 12, 13}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({21, 22, 23}));

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

    EXPECT_CALL(*transfer_client_, SaveKvCaches(_, _)).Times(0);
    EXPECT_CALL(*meta_clients_[tp_rank], FinishWrite(_, _, _, _)).Times(0);

    auto async_context = remote_connectors_[tp_rank]->asyncWrite(kv_cache_resouce, meta);
    waitAsyncContextDone(async_context);
    ASSERT_TRUE(async_context->success());
}

TEST_F(RemoteConnectorMockFullLinearTest, test_write_success_broadcast_success_with_part_empty_linear) {
    auto kv_cache_resouce = std::make_shared<KVCacheResource>();
    kv_cache_resouce->setLastBlockAligned(true);
    kv_cache_resouce->cache_keys = {1, 2, 3, 4};
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({1, 2, 3, 4}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({11, -1, 13, 14}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({21, -1, 23, 24}));

    auto          meta    = std::make_shared<MetaImpl>(false, true, "trace_1");
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
    waitAsyncContextDone(async_context);
    ASSERT_TRUE(async_context->success());
}

// In fact, this situation should not occur
TEST_F(RemoteConnectorMockFullLinearTest, test_write_success_broadcast_success_with_all_empty_linear) {
    auto kv_cache_resouce = std::make_shared<KVCacheResource>();
    kv_cache_resouce->setLastBlockAligned(true);
    kv_cache_resouce->cache_keys = {1, 2, 3, 4};
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({1, 2, 3, 4}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({-1, -1, -1, -1}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({-1, -1, -1, -1}));

    auto          meta    = std::make_shared<MetaImpl>(false, true, "trace_1");
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
    waitAsyncContextDone(async_context);
    ASSERT_TRUE(async_context->success());
}

TEST_F(RemoteConnectorMockFullLinearTest, test_write_success_broadcast_success_actual_locations_same) {
    auto kv_cache_resouce = std::make_shared<KVCacheResource>();
    kv_cache_resouce->setLastBlockAligned(true);
    kv_cache_resouce->cache_keys = {1, 2, 3};
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({1, 2, 3}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({11, 12, 13}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({21, 22, 23}));
    auto          meta    = std::make_shared<MetaImpl>(false, true, "trace_2");
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
    waitAsyncContextDone(async_context);
    ASSERT_TRUE(async_context->success());
}

TEST_F(RemoteConnectorMockFullLinearTest, test_match_fail) {
    auto kv_cache_resouce        = std::make_shared<KVCacheResource>();
    kv_cache_resouce->cache_keys = {1, 2, 3, 4};
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({1, 2, 3, 4}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({11, 12, 13, 14}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({21, 22, 23, 24}));
    auto   meta    = std::make_shared<MetaImpl>(false, true, "trace_1");
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

    auto match_context = remote_connectors_[tp_rank]->asyncMatch(kv_cache_resouce, meta);
    waitAsyncContextDone(match_context);
    ASSERT_TRUE(match_context->success());
    auto context = std::dynamic_pointer_cast<RemoteAsyncMatchContext>(match_context);
    ASSERT_NE(nullptr, context);
    ASSERT_EQ(RemoteConnectorAsyncContext::State::RCS_READ_MATCH_ERROR, context->state());
}

TEST_F(RemoteConnectorMockFullLinearTest, test_match_success_load_fail) {
    auto kv_cache_resouce        = std::make_shared<KVCacheResource>();
    kv_cache_resouce->cache_keys = {1, 2, 3, 4};
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({1, 2, 3, 4}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({11, 12, 13, 14}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({21, 22, 23, 24}));
    auto      meta               = std::make_shared<MetaImpl>(false, true, "trace_1");
    size_t    tp_rank            = 0;
    Locations expected_locations = genFullotherLocations({1, 2, 3}, {0, 1, 2});
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

    auto match_context = remote_connectors_[tp_rank]->asyncMatch(kv_cache_resouce, meta);
    waitAsyncContextDone(match_context);
    ASSERT_TRUE(match_context->success());
    ASSERT_EQ(match_context->matchedBlockCount(), 3);

    const int gpu_reuse_num          = static_cast<int>(kv_cache_resouce->reuseBlockNum());   // 0
    const int matched_num            = static_cast<int>(match_context->matchedBlockCount());  // 3
    int       start_read_block_index = gpu_reuse_num;
    int       read_block_num         = matched_num - gpu_reuse_num;
    auto      read_context           = remote_connectors_[tp_rank]->asyncRead(
        kv_cache_resouce, meta, match_context, start_read_block_index, read_block_num);
    waitAsyncContextDone(read_context);
    ASSERT_FALSE(read_context->success());
    auto context = std::dynamic_pointer_cast<RemoteConnectorAsyncContext>(read_context);
    ASSERT_NE(nullptr, context);
    ASSERT_EQ(RemoteConnectorAsyncContext::State::RCS_ERROR, context->state());
}

// TEST_F(RemoteConnectorMockFullLinearTest, test_match_success_broadcast_grpc_fail) {
//     auto kv_cache_resouce        = std::make_shared<KVCacheResource>();
//     kv_cache_resouce->cache_keys = {1, 2, 3, 4};
//     kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({1, 2, 3, 4}));
//     kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({11, 12, 13, 14}));
//     kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({21, 22, 23, 24}));
//     auto      meta               = std::make_shared<MetaImpl>(false, true, "trace_1");
//     size_t    tp_rank            = 0;
//     Locations expected_locations = genFullotherLocations({1, 2, 3}, {0, 1, 2});
//     EXPECT_CALL(*meta_clients_[tp_rank],
//                 MatchLocation(Eq("match_trace_1"),                    // trace_id
//                               _,                                      // query_type
//                               std::vector<int64_t>({1, 2, 3}),        // keys
//                               _,                                      // tokens
//                               Eq(BlockMask(static_cast<size_t>(0))),  // block_mask
//                               _,                                      // sw_size
//                               _                                       // location_spec_names
//                               ))
//         .WillOnce(Return(MatchLocationReturnType({ClientErrorCode::ER_OK, expected_locations})));
//     EXPECT_CALL(*transfer_client_, LoadKvCaches(_, _)).Times(0);

//     servers_[tp_rank]->hack_grpc_status(true);
//     auto match_context = remote_connectors_[tp_rank]->asyncMatch(kv_cache_resouce, meta);
//     waitAsyncContextDone(match_context);
//     ASSERT_TRUE(match_context->success());
//     ASSERT_EQ(match_context->matchedBlockCount(), 3);
//     const int gpu_reuse_num = static_cast<int>(kv_cache_resouce->reuseBlockNum());   // 0
//     const int matched_num   = static_cast<int>(match_context->matchedBlockCount());  // 3
//     // auto      read_meta     = std::make_shared<TestReadMeta>(gpu_reuse_num, matched_num - gpu_reuse_num);
//     int  start_read_block_index = gpu_reuse_num;
//     int  read_block_num         = matched_num - gpu_reuse_num;
//     auto read_context           = remote_connectors_[tp_rank]->asyncRead(
//         kv_cache_resouce, meta, match_context, start_read_block_index, read_block_num);
//     ASSERT_FALSE(read_context->success());
//     auto context = std::dynamic_pointer_cast<RemoteConnectorAsyncContext>(read_context);
//     ASSERT_NE(nullptr, context);
//     ASSERT_EQ(RemoteConnectorAsyncContext::State::RCS_ERROR, context->state());
// }

TEST_F(RemoteConnectorMockFullLinearTest, test_start_write_fail) {
    auto kv_cache_resouce = std::make_shared<KVCacheResource>();
    kv_cache_resouce->setLastBlockAligned(true);
    kv_cache_resouce->cache_keys = {1, 2, 3};
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({1, 2, 3}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({11, 12, 13}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({21, 22, 23}));
    auto          meta    = std::make_shared<MetaImpl>(false, true, "trace_1");
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
    waitAsyncContextDone(async_context);
    ASSERT_FALSE(async_context->success());
    auto remote_async_context = std::dynamic_pointer_cast<RemoteConnectorAsyncContext>(async_context);
    ASSERT_NE(nullptr, remote_async_context);
    ASSERT_EQ(RemoteConnectorAsyncContext::State::RCS_ERROR, remote_async_context->state());
}

TEST_F(RemoteConnectorMockFullLinearTest, test_write_invalid_block_ids) {
    auto kv_cache_resouce = std::make_shared<KVCacheResource>();
    kv_cache_resouce->setLastBlockAligned(true);
    kv_cache_resouce->cache_keys = {1, 2, 3};
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({1, 2, 3}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({11, -1, 13}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({21, 22, 23}));
    auto   meta    = std::make_shared<MetaImpl>(false, true, "trace_1");
    size_t tp_rank = 0;
    EXPECT_CALL(*meta_clients_[tp_rank], StartWrite(_, _, _, _, _)).Times(0);
    EXPECT_CALL(*transfer_client_, SaveKvCaches(_, _)).Times(0);
    EXPECT_CALL(*meta_clients_[tp_rank], FinishWrite(_, _, _, _)).Times(0);

    auto async_context = remote_connectors_[tp_rank]->asyncWrite(kv_cache_resouce, meta);
    waitAsyncContextDone(async_context);
    ASSERT_FALSE(async_context->success());
    auto remote_async_context = std::dynamic_pointer_cast<RemoteConnectorAsyncContext>(async_context);
    ASSERT_NE(nullptr, remote_async_context);
    ASSERT_EQ(RemoteConnectorAsyncContext::State::RCS_ERROR, remote_async_context->state());
}

TEST_F(RemoteConnectorMockFullLinearTest, test_start_write_success_broadcast_success_finish_write_fail) {
    auto kv_cache_resouce = std::make_shared<KVCacheResource>();
    kv_cache_resouce->setLastBlockAligned(true);
    kv_cache_resouce->cache_keys = {1, 2, 3};
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({1, 2, 3}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({11, 12, 13}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({21, 22, 23}));
    auto          meta    = std::make_shared<MetaImpl>(false, true, "trace_2");
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
    waitAsyncContextDone(async_context);
    ASSERT_FALSE(async_context->success());
    auto remote_async_context = std::dynamic_pointer_cast<RemoteConnectorAsyncContext>(async_context);
    ASSERT_NE(nullptr, remote_async_context);
    ASSERT_EQ(RemoteConnectorAsyncContext::State::RCS_ERROR, remote_async_context->state());
}

TEST_F(RemoteConnectorMockFullLinearTest, test_start_write_success_broadcast_success_save_fail) {
    auto kv_cache_resouce = std::make_shared<KVCacheResource>();
    kv_cache_resouce->setLastBlockAligned(true);
    kv_cache_resouce->cache_keys = {1, 2, 3};
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({1, 2, 3}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({11, 12, 13}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({21, 22, 23}));
    auto          meta    = std::make_shared<MetaImpl>(false, true, "trace_2");
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
    waitAsyncContextDone(async_context);
    ASSERT_FALSE(async_context->success());
    auto remote_async_context = std::dynamic_pointer_cast<RemoteConnectorAsyncContext>(async_context);
    ASSERT_NE(nullptr, remote_async_context);
    ASSERT_EQ(RemoteConnectorAsyncContext::State::RCS_ERROR, remote_async_context->state());
}

TEST_F(RemoteConnectorMockFullLinearTest, test_start_write_success_broadcast_grpc_fail) {
    auto kv_cache_resouce = std::make_shared<KVCacheResource>();
    kv_cache_resouce->setLastBlockAligned(true);
    kv_cache_resouce->cache_keys = {1, 2, 3};
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({1, 2, 3}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({11, 12, 13}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({21, 22, 23}));
    auto          meta    = std::make_shared<MetaImpl>(false, true, "trace_2");
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
    waitAsyncContextDone(async_context);
    ASSERT_FALSE(async_context->success());
    auto remote_async_context = std::dynamic_pointer_cast<RemoteConnectorAsyncContext>(async_context);
    ASSERT_NE(nullptr, remote_async_context);
    ASSERT_EQ(RemoteConnectorAsyncContext::State::RCS_ERROR, remote_async_context->state());
}

TEST_F(RemoteConnectorMockFullLinearTest, test_threadpool_full) {
    auto kv_cache_resouce        = std::make_shared<KVCacheResource>();
    kv_cache_resouce->cache_keys = {1, 2, 3, 4};
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({1, 2, 3, 4}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({11, 12, 13, 14}));
    kv_cache_resouce->group_block_ids.push_back(makeGroupBlockIds({21, 22, 23, 24}));
    auto   meta    = std::make_shared<MetaImpl>(false, true, "trace");
    size_t tp_rank = 0;
    remote_connectors_[tp_rank]->thread_pool_->stop();
    remote_connectors_[tp_rank]->thread_pool_->waitFinish();
    remote_connectors_[tp_rank]->thread_pool_.reset();
    remote_connectors_[tp_rank]->thread_pool_ =
        std::make_unique<autil::LockFreeThreadPool>(1, /* queueSize= */ 0, nullptr, "RECOThreadPool");

    EXPECT_CALL(*meta_clients_[tp_rank], MatchLocation(_, _, _, _, _, _, _)).Times(0);
    EXPECT_CALL(*transfer_client_, LoadKvCaches(_, _)).Times(0);
    ASSERT_EQ(nullptr, remote_connectors_[tp_rank]->asyncMatch(kv_cache_resouce, meta));

    EXPECT_CALL(*meta_clients_[tp_rank], StartWrite(_, _, _, _, _)).Times(0);
    EXPECT_CALL(*transfer_client_, SaveKvCaches(_, _)).Times(0);
    EXPECT_CALL(*meta_clients_[tp_rank], FinishWrite(_, _, _, _)).Times(0);
    kv_cache_resouce->setLastBlockAligned(true);
    ASSERT_EQ(nullptr, remote_connectors_[tp_rank]->asyncWrite(kv_cache_resouce, meta));
}

}  // namespace test
}  // namespace rtp_llm