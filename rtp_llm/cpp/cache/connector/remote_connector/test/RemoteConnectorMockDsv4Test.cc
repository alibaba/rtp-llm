#include <chrono>
#include <thread>

#include "rtp_llm/cpp/cache/connector/AsyncContext.h"
#include "rtp_llm/cpp/cache/DSV4ConfigCreator.h"
#include "rtp_llm/cpp/cache/DSV4CacheConfig.h"
#include "rtp_llm/cpp/cache/HybridPoolKVCacheAllocator.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/cpp/cache/connector/remote_connector/RemoteConnector.h"
#include "rtp_llm/cpp/cache/connector/remote_connector/test/RemoteConnectorMockTestBase.h"
#include "rtp_llm/cpp/cache/connector/Meta.h"

using namespace kv_cache_manager;
using namespace ::testing;
using namespace rtp_llm;
using namespace rtp_llm::remote_connector;

namespace rtp_llm {
namespace test {

void waitDsv4AsyncContextDone(const std::shared_ptr<rtp_llm::AsyncContext>& ctx) {
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

namespace {

static ModelConfig makeMinimalDsv4Model() {
    ModelConfig mc;
    mc.num_layers                        = 2;
    mc.hidden_size                       = 4096;
    mc.attn_config.head_num              = 64;
    mc.attn_config.kv_head_num           = 1;
    mc.attn_config.size_per_head         = 512;
    mc.attn_config.rope_head_dim         = 64;
    mc.attn_config.sliding_window        = 128;
    mc.attn_config.indexer_head_dim      = 128;
    mc.attn_config.indexer_head_num      = 64;
    mc.attn_config.indexer_topk          = 512;
    mc.attn_config.o_groups              = 8;
    mc.attn_config.o_lora_rank           = 1024;
    mc.attn_config.layer_compress_ratios = {4, 128};
    return mc;
}

static CacheConfig makeMinimalHybridPoolDsv4Config() {
    ParallelismConfig pc;
    auto              mc               = makeMinimalDsv4Model();
    CacheConfig       config           = DSV4ConfigCreator::createConfig(mc, pc, false);
    config.use_independent_block_pools = true;
    config.block_num                   = 200;
    return config;
}

}  // namespace

class RemoteConnectorMockDsv4Test: public RemoteConnectorMockTestBase {
public:
    void SetUp() override {
        RemoteConnectorMockTestBase::SetUp();
        initDsv4Connector();
    }

    void TearDown() override {
        RemoteConnectorMockTestBase::TearDown();
    }

protected:
    void initDsv4Connector() {
        cache_config_ = makeMinimalHybridPoolDsv4Config();
        ASSERT_TRUE(cache_config_.dsv4_config.has_value());
        for (int i = 0; i < tp_size_; i++) {
            auto meta_client = std::make_unique<kv_cache_manager::MockMetaClient>();
            meta_clients_.push_back(meta_client.get());
            EXPECT_CALL(*mock_client_factory_, CreateMetaClient(_, _))
                .WillOnce(Invoke(
                    [&](const std::string&, const kv_cache_manager::InitParams&) { return std::move(meta_client); }));
            auto allocator = std::make_shared<HybridPoolKVCacheAllocator>(cache_config_);
            ASSERT_TRUE(allocator->init());
            auto layout = allocator->allLayerCacheBase();
            ASSERT_FALSE(layout.layer_to_group_ids.empty());

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

    void fillDsv4GroupBlocks(const std::shared_ptr<KVCacheResource>& kv_res,
                             const CacheKeysType&                    keys = {101, 102, 103, 104}) {
        kv_res->group_block_ids.clear();
        int n = static_cast<int>(keys.size());
        for (int g = 0; g < DSV4_NUM_POOLS; g++) {
            BlockIndicesType indices;
            for (int i = 0; i < n; i++) {
                indices.push_back(static_cast<BlockIdxType>((i + 1) + g * 10));
            }
            kv_res->group_block_ids.push_back(makeGroupBlockIds(indices));
        }
        kv_res->cache_keys = keys;
    }

    // Generate URIs matching genDsv4FullSwaLocations (F0..F2 for all, L3..L6 at other_pos_vec).
    UriStrVec genDsv4Uris(const CacheKeysType&       cache_keys,
                          const std::vector<size_t>& other_pos_vec = {},
                          const std::string&         uri_prefix    = "") {
        UriStrVec res;
        size_t    pos_idx = 0;
        for (size_t i = 0; i < cache_keys.size(); i++) {
            for (int g = 0; g < DSV4_REMOTE_FULL_POOL_NUM; g++) {
                std::string full_group_name = "F" + std::to_string(g);
                for (int r = 0; r < tp_size_; r++) {
                    res.push_back(uri_prefix + "uri_" + full_group_name + "_" + std::to_string(r) + "_"
                                  + std::to_string(cache_keys[i]));
                }
            }
            if (!other_pos_vec.empty() && pos_idx < other_pos_vec.size() && i == other_pos_vec[pos_idx]) {
                for (int g = DSV4_REMOTE_FULL_POOL_NUM; g < DSV4_NUM_POOLS; g++) {
                    std::string other_group_name = "L" + std::to_string(g);
                    for (int r = 0; r < tp_size_; r++) {
                        res.push_back(uri_prefix + "uri_" + other_group_name + "_" + std::to_string(r) + "_"
                                      + std::to_string(cache_keys[i]));
                    }
                }
                pos_idx++;
            }
        }
        return res;
    }

    kv_cache_manager::Locations genDsv4FullSwaLocations(const CacheKeysType&       cache_keys,
                                                        const std::vector<size_t>& other_pos_vec = {},
                                                        const std::string&         uri_prefix    = "") {
        kv_cache_manager::Locations locations;
        locations.resize(cache_keys.size(), {});
        for (size_t i = 0; i < cache_keys.size(); i++) {
            for (int g = 0; g < DSV4_REMOTE_FULL_POOL_NUM; g++) {
                std::string full_group_name = "F" + std::to_string(g);
                for (int r = 0; r < tp_size_; r++) {
                    std::string uri = uri_prefix + "uri_" + full_group_name + "_" + std::to_string(r) + "_"
                                      + std::to_string(cache_keys[i]);
                    locations[i].push_back(
                        kv_cache_manager::LocationSpecUnit({genLocationSpecName(r, full_group_name), uri}));
                }
            }
        }
        for (auto pos : other_pos_vec) {
            for (int g = DSV4_REMOTE_FULL_POOL_NUM; g < DSV4_NUM_POOLS; g++) {
                std::string other_group_name = "L" + std::to_string(g);
                for (int r = 0; r < tp_size_; r++) {
                    std::string uri = uri_prefix + "uri_" + other_group_name + "_" + std::to_string(r) + "_"
                                      + std::to_string(cache_keys[pos]);
                    locations[pos].push_back(
                        kv_cache_manager::LocationSpecUnit({genLocationSpecName(r, other_group_name), uri}));
                }
            }
        }
        return locations;
    }
};

// Minimal Meta for remote connector
class Dsv4Meta: public Meta {
public:
    explicit Dsv4Meta(std::string trace_id): trace_id_(std::move(trace_id)) {}
    bool enableMemoryCache() const override {
        return false;
    }
    bool enableRemoteCache() const override {
        return true;
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
    std::string          trace_id_;
    std::string          unique_id_;
    std::vector<int64_t> tokens_;
};

TEST_F(RemoteConnectorMockDsv4Test, dsv4_async_match_three_keys) {
    auto kv_res = std::make_shared<KVCacheResource>();
    fillDsv4GroupBlocks(kv_res);

    const size_t tp_rank            = 0;
    Locations    expected_locations = genDsv4FullSwaLocations({101, 102, 103}, {0, 1, 2});
    auto         meta               = std::make_shared<Dsv4Meta>("ut_dsv4_1");

    EXPECT_CALL(
        *meta_clients_[tp_rank],
        MatchLocation(Eq("match_ut_dsv4_1"), _, std::vector<int64_t>({101, 102, 103}), _, Eq(BlockMask(0U)), _, _))
        .WillOnce(Return(MatchLocationReturnType({ClientErrorCode::ER_OK, expected_locations})));

    std::shared_ptr<AsyncMatchContext> match_ctx = remote_connectors_[tp_rank]->asyncMatch(kv_res, meta);
    waitDsv4AsyncContextDone(std::static_pointer_cast<AsyncContext>(match_ctx));
    ASSERT_TRUE(match_ctx->success());
    EXPECT_EQ(match_ctx->matchedBlockCount(), 3u);
}

// Same pattern as RemoteConnectorMockFullLinearTest::test_match_fail
TEST_F(RemoteConnectorMockDsv4Test, dsv4_test_match_fail) {
    auto kv_res = std::make_shared<KVCacheResource>();
    fillDsv4GroupBlocks(kv_res);
    const size_t tp_rank = 0;
    auto         meta    = std::make_shared<Dsv4Meta>("ut_dsv4_match_fail");

    EXPECT_CALL(*meta_clients_[tp_rank],
                MatchLocation(Eq("match_ut_dsv4_match_fail"),
                              _,
                              std::vector<int64_t>({101, 102, 103}),
                              _,
                              Eq(BlockMask(static_cast<size_t>(0))),
                              _,
                              _))
        .WillOnce(Return(MatchLocationReturnType({ClientErrorCode::ER_INVALID_GRPCSTATUS, {}})));
    EXPECT_CALL(*transfer_client_, LoadKvCaches(_, _, _)).Times(0);

    std::shared_ptr<AsyncMatchContext> match_context = remote_connectors_[tp_rank]->asyncMatch(kv_res, meta);
    waitDsv4AsyncContextDone(std::static_pointer_cast<AsyncContext>(match_context));
    ASSERT_FALSE(match_context->success());
    auto rctx = std::dynamic_pointer_cast<RemoteAsyncMatchContext>(match_context);
    ASSERT_NE(nullptr, rctx);
    ASSERT_EQ(RemoteConnectorState::State::RCS_READ_MATCH_ERROR, rctx->state());
}

// Same pattern as RemoteConnectorMockFullLinearTest::test_match_success_load_fail (LoadKvCaches timeout on read)
TEST_F(RemoteConnectorMockDsv4Test, dsv4_test_match_success_load_fail) {
    auto kv_res = std::make_shared<KVCacheResource>();
    fillDsv4GroupBlocks(kv_res);
    const size_t tp_rank            = 0;
    Locations    expected_locations = genDsv4FullSwaLocations({101, 102, 103}, {0, 1, 2});
    auto         meta               = std::make_shared<Dsv4Meta>("ut_dsv4_load_fail");

    EXPECT_CALL(*meta_clients_[tp_rank],
                MatchLocation(Eq("match_ut_dsv4_load_fail"),
                              _,
                              std::vector<int64_t>({101, 102, 103}),
                              _,
                              Eq(BlockMask(static_cast<size_t>(0))),
                              _,
                              _))
        .WillOnce(Return(MatchLocationReturnType({ClientErrorCode::ER_OK, expected_locations})));
    EXPECT_CALL(*transfer_client_, LoadKvCaches(_, _, _)).WillOnce(Return(ClientErrorCode::ER_SDK_TIMEOUT));

    std::shared_ptr<AsyncMatchContext> match_context = remote_connectors_[tp_rank]->asyncMatch(kv_res, meta);
    waitDsv4AsyncContextDone(std::static_pointer_cast<AsyncContext>(match_context));
    ASSERT_TRUE(match_context->success());
    ASSERT_EQ(match_context->matchedBlockCount(), 3u);

    const int gpu_reuse_num          = static_cast<int>(kv_res->reuseBlockNum());
    const int matched_num            = static_cast<int>(match_context->matchedBlockCount());
    int       start_read_block_index = gpu_reuse_num;
    int       read_block_num         = matched_num - gpu_reuse_num;
    auto      read_context =
        remote_connectors_[tp_rank]->asyncRead(kv_res, meta, match_context, start_read_block_index, read_block_num);
    waitDsv4AsyncContextDone(std::static_pointer_cast<AsyncContext>(read_context));
    ASSERT_FALSE(read_context->success());
    auto rctx = std::dynamic_pointer_cast<RemoteConnectorAsyncContext>(read_context);
    ASSERT_NE(nullptr, rctx);
    ASSERT_EQ(RemoteConnectorState::State::RCS_ERROR, rctx->state());
}

// ==================== Read tests ====================

TEST_F(RemoteConnectorMockDsv4Test, dsv4_async_match_and_read_gpu_reuse_zero) {
    auto kv_res = std::make_shared<KVCacheResource>();
    fillDsv4GroupBlocks(kv_res);

    const size_t tp_rank            = 0;
    Locations    expected_locations = genDsv4FullSwaLocations({101, 102, 103}, {0, 1, 2});
    auto         meta               = std::make_shared<Dsv4Meta>("ut_dsv4_read_reuse0");

    EXPECT_CALL(*meta_clients_[tp_rank],
                MatchLocation(Eq("match_ut_dsv4_read_reuse0"),
                              _,
                              std::vector<int64_t>({101, 102, 103}),
                              _,
                              Eq(BlockMask(static_cast<size_t>(0))),
                              _,
                              _))
        .WillOnce(Return(MatchLocationReturnType({ClientErrorCode::ER_OK, expected_locations})));

    auto match_ctx = remote_connectors_[tp_rank]->asyncMatch(kv_res, meta);
    waitDsv4AsyncContextDone(std::static_pointer_cast<AsyncContext>(match_ctx));
    ASSERT_TRUE(match_ctx->success());
    ASSERT_EQ(match_ctx->matchedBlockCount(), 3u);

    // Read all 3 blocks: full groups at all positions, linear groups only at last block
    {
        UriStrVec                expected_uris = genDsv4Uris({101, 102, 103}, {2});
        std::vector<std::string> expect_block_ids(
            {"1", "11", "21", "2", "12", "22", "3", "13", "23", "33", "43", "53", "63"});
        EXPECT_CALL(*transfer_client_, LoadKvCaches(Eq(expected_uris), _, TransferTraceInfoMatcher(expect_block_ids)))
            .WillOnce(Return(ClientErrorCode::ER_OK));

        const int gpu_reuse_num          = static_cast<int>(kv_res->reuseBlockNum());
        const int matched_num            = static_cast<int>(match_ctx->matchedBlockCount());
        int       start_read_block_index = gpu_reuse_num;
        int       read_block_num         = matched_num - gpu_reuse_num;
        auto      read_context =
            remote_connectors_[tp_rank]->asyncRead(kv_res, meta, match_ctx, start_read_block_index, read_block_num);
        waitDsv4AsyncContextDone(std::static_pointer_cast<AsyncContext>(read_context));
        ASSERT_TRUE(read_context->success());
        ASSERT_EQ(kv_res->remoteReuseBlockNum(), 3);
        kv_res->setRemoteReuseBlockNum(0);
    }
    // Read last 2 blocks (skip first)
    {
        UriStrVec                expected_uris = genDsv4Uris({102, 103}, {1});
        std::vector<std::string> expect_block_ids({"2", "12", "22", "3", "13", "23", "33", "43", "53", "63"});
        EXPECT_CALL(*transfer_client_, LoadKvCaches(Eq(expected_uris), _, TransferTraceInfoMatcher(expect_block_ids)))
            .WillOnce(Return(ClientErrorCode::ER_OK));

        const int gpu_reuse_num          = static_cast<int>(kv_res->reuseBlockNum());
        const int matched_num            = static_cast<int>(match_ctx->matchedBlockCount());
        int       start_read_block_index = gpu_reuse_num + 1;
        int       read_block_num         = matched_num - gpu_reuse_num - 1;
        auto      read_context =
            remote_connectors_[tp_rank]->asyncRead(kv_res, meta, match_ctx, start_read_block_index, read_block_num);
        waitDsv4AsyncContextDone(std::static_pointer_cast<AsyncContext>(read_context));
        ASSERT_TRUE(read_context->success());
        ASSERT_EQ(kv_res->remoteReuseBlockNum(), 2);
        kv_res->setRemoteReuseBlockNum(0);
    }
    // Negative read (start beyond matched range) => no transfer, remoteReuse=0
    {
        const int gpu_reuse_num          = static_cast<int>(kv_res->reuseBlockNum());
        const int matched_num            = static_cast<int>(match_ctx->matchedBlockCount());
        int       start_read_block_index = gpu_reuse_num + 4;
        int       read_block_num         = matched_num - gpu_reuse_num - 4;
        auto      read_context =
            remote_connectors_[tp_rank]->asyncRead(kv_res, meta, match_ctx, start_read_block_index, read_block_num);
        waitDsv4AsyncContextDone(std::static_pointer_cast<AsyncContext>(read_context));
        ASSERT_TRUE(read_context->success());
        ASSERT_EQ(kv_res->remoteReuseBlockNum(), 0);
        kv_res->setRemoteReuseBlockNum(0);
    }
}

TEST_F(RemoteConnectorMockDsv4Test, dsv4_async_match_and_read_gpu_reuse_not_zero) {
    auto kv_res = std::make_shared<KVCacheResource>();
    fillDsv4GroupBlocks(kv_res);
    kv_res->setDeviceReuseBlockNum(1);

    const size_t tp_rank            = 0;
    Locations    expected_locations = genDsv4FullSwaLocations({102, 103}, {0, 1});
    auto         meta               = std::make_shared<Dsv4Meta>("ut_dsv4_read_reuse1");

    EXPECT_CALL(*meta_clients_[tp_rank],
                MatchLocation(Eq("match_ut_dsv4_read_reuse1"),
                              _,
                              std::vector<int64_t>({101, 102, 103}),
                              _,
                              Eq(BlockMask(static_cast<size_t>(1))),
                              _,
                              _))
        .WillOnce(Return(MatchLocationReturnType({ClientErrorCode::ER_OK, expected_locations})));

    auto match_ctx = remote_connectors_[tp_rank]->asyncMatch(kv_res, meta);
    waitDsv4AsyncContextDone(std::static_pointer_cast<AsyncContext>(match_ctx));
    ASSERT_TRUE(match_ctx->success());
    ASSERT_EQ(match_ctx->matchedBlockCount(), 3u);

    // Read 2 blocks (after gpu_reuse=1): linear at last block only
    {
        UriStrVec                expected_uris = genDsv4Uris({102, 103}, {1});
        std::vector<std::string> expect_block_ids({"2", "12", "22", "3", "13", "23", "33", "43", "53", "63"});
        EXPECT_CALL(*transfer_client_, LoadKvCaches(Eq(expected_uris), _, TransferTraceInfoMatcher(expect_block_ids)))
            .WillOnce(Return(ClientErrorCode::ER_OK));

        const int gpu_reuse_num          = static_cast<int>(kv_res->reuseBlockNum());
        const int matched_num            = static_cast<int>(match_ctx->matchedBlockCount());
        int       start_read_block_index = gpu_reuse_num;
        int       read_block_num         = matched_num - gpu_reuse_num;
        auto      read_context =
            remote_connectors_[tp_rank]->asyncRead(kv_res, meta, match_ctx, start_read_block_index, read_block_num);
        waitDsv4AsyncContextDone(std::static_pointer_cast<AsyncContext>(read_context));
        ASSERT_TRUE(read_context->success());
        ASSERT_EQ(kv_res->remoteReuseBlockNum(), 2);
        kv_res->setRemoteReuseBlockNum(0);
    }
    // Read last 1 block only
    {
        UriStrVec                expected_uris = genDsv4Uris({103}, {0});
        std::vector<std::string> expect_block_ids({"3", "13", "23", "33", "43", "53", "63"});
        EXPECT_CALL(*transfer_client_, LoadKvCaches(Eq(expected_uris), _, TransferTraceInfoMatcher(expect_block_ids)))
            .WillOnce(Return(ClientErrorCode::ER_OK));

        const int gpu_reuse_num          = static_cast<int>(kv_res->reuseBlockNum());
        const int matched_num            = static_cast<int>(match_ctx->matchedBlockCount());
        int       start_read_block_index = gpu_reuse_num + 1;
        int       read_block_num         = matched_num - gpu_reuse_num - 1;
        auto      read_context =
            remote_connectors_[tp_rank]->asyncRead(kv_res, meta, match_ctx, start_read_block_index, read_block_num);
        waitDsv4AsyncContextDone(std::static_pointer_cast<AsyncContext>(read_context));
        ASSERT_TRUE(read_context->success());
        ASSERT_EQ(kv_res->remoteReuseBlockNum(), 1);
        kv_res->setRemoteReuseBlockNum(0);
    }
}

TEST_F(RemoteConnectorMockDsv4Test, dsv4_read_last_block_full_only_skips_load) {
    auto kv_res = std::make_shared<KVCacheResource>();
    fillDsv4GroupBlocks(kv_res);

    const size_t tp_rank = 0;
    // 101, 102 have full+linear; 103 has full only (no linear)
    Locations expected_locations = genDsv4FullSwaLocations({101, 102, 103}, {0, 1});
    auto      meta               = std::make_shared<Dsv4Meta>("ut_dsv4_read_last_full_only");

    EXPECT_CALL(*meta_clients_[tp_rank],
                MatchLocation(Eq("match_ut_dsv4_read_last_full_only"),
                              _,
                              std::vector<int64_t>({101, 102, 103}),
                              _,
                              Eq(BlockMask(static_cast<size_t>(0))),
                              _,
                              _))
        .WillOnce(Return(MatchLocationReturnType({ClientErrorCode::ER_OK, expected_locations})));

    auto match_ctx = remote_connectors_[tp_rank]->asyncMatch(kv_res, meta);
    waitDsv4AsyncContextDone(std::static_pointer_cast<AsyncContext>(match_ctx));
    ASSERT_TRUE(match_ctx->success());
    ASSERT_EQ(match_ctx->matchedBlockCount(), 3u);

    // filterNeedLoadLocations (back→front): 103 full-only validated but not in view,
    // 102 is first full+linear → kept with linear, 101 full+linear → kept full-only.
    // Result: load 101 (full) + 102 (full+linear). 103 not loaded.
    {
        UriStrVec                expected_uris = genDsv4Uris({101, 102}, {1});
        std::vector<std::string> expect_block_ids({"1", "11", "21", "2", "12", "22", "32", "42", "52", "62"});
        EXPECT_CALL(*transfer_client_, LoadKvCaches(Eq(expected_uris), _, TransferTraceInfoMatcher(expect_block_ids)))
            .WillOnce(Return(ClientErrorCode::ER_OK));

        const int gpu_reuse_num          = static_cast<int>(kv_res->reuseBlockNum());
        const int matched_num            = static_cast<int>(match_ctx->matchedBlockCount());
        int       start_read_block_index = gpu_reuse_num;
        int       read_block_num         = matched_num - gpu_reuse_num;
        auto      read_context =
            remote_connectors_[tp_rank]->asyncRead(kv_res, meta, match_ctx, start_read_block_index, read_block_num);
        waitDsv4AsyncContextDone(std::static_pointer_cast<AsyncContext>(read_context));
        ASSERT_TRUE(read_context->success());
        ASSERT_EQ(kv_res->remoteReuseBlockNum(), 2);
    }
}

TEST_F(RemoteConnectorMockDsv4Test, dsv4_read_partial_linear_at_last_block_fails) {
    auto kv_res = std::make_shared<KVCacheResource>();
    fillDsv4GroupBlocks(kv_res);

    const size_t tp_rank = 0;
    // Build locations: 101/102 complete full+linear, 103 has full + partial linear (only L3, L4)
    Locations expected_locations = genDsv4FullSwaLocations({101, 102, 103}, {0, 1});
    // Append partial linear specs (L3, L4 only) to position 2 (key=103)
    for (int g = DSV4_REMOTE_FULL_POOL_NUM; g < DSV4_REMOTE_FULL_POOL_NUM + 2; g++) {
        std::string other_group_name = "L" + std::to_string(g);
        for (int r = 0; r < tp_size_; r++) {
            std::string uri = "uri_" + other_group_name + "_" + std::to_string(r) + "_103";
            expected_locations[2].push_back(
                kv_cache_manager::LocationSpecUnit({genLocationSpecName(r, other_group_name), uri}));
        }
    }
    // Now location[2].size() = 3 (full) + 2 (partial linear) = 5, which is neither 3 nor 7
    auto meta = std::make_shared<Dsv4Meta>("ut_dsv4_read_partial_linear");

    EXPECT_CALL(*meta_clients_[tp_rank],
                MatchLocation(Eq("match_ut_dsv4_read_partial_linear"),
                              _,
                              std::vector<int64_t>({101, 102, 103}),
                              _,
                              Eq(BlockMask(static_cast<size_t>(0))),
                              _,
                              _))
        .WillOnce(Return(MatchLocationReturnType({ClientErrorCode::ER_OK, expected_locations})));

    // filterNeedLoadLocations sees location[2].size()=5, doesn't match full(3) or full_other(7) → return false
    EXPECT_CALL(*transfer_client_, LoadKvCaches(_, _, _)).Times(0);

    auto match_ctx = remote_connectors_[tp_rank]->asyncMatch(kv_res, meta);
    waitDsv4AsyncContextDone(std::static_pointer_cast<AsyncContext>(match_ctx));
    ASSERT_TRUE(match_ctx->success());
    ASSERT_EQ(match_ctx->matchedBlockCount(), 3u);

    const int gpu_reuse_num          = static_cast<int>(kv_res->reuseBlockNum());
    const int matched_num            = static_cast<int>(match_ctx->matchedBlockCount());
    int       start_read_block_index = gpu_reuse_num;
    int       read_block_num         = matched_num - gpu_reuse_num;
    auto      read_context =
        remote_connectors_[tp_rank]->asyncRead(kv_res, meta, match_ctx, start_read_block_index, read_block_num);
    waitDsv4AsyncContextDone(std::static_pointer_cast<AsyncContext>(read_context));
    ASSERT_FALSE(read_context->success());
    auto rctx = std::dynamic_pointer_cast<RemoteConnectorAsyncContext>(read_context);
    ASSERT_NE(nullptr, rctx);
    ASSERT_EQ(RemoteConnectorState::State::RCS_ERROR, rctx->state());
}

// ==================== Write tests ====================

TEST_F(RemoteConnectorMockDsv4Test, dsv4_write_success) {
    auto kv_res = std::make_shared<KVCacheResource>();
    fillDsv4GroupBlocks(kv_res, {101, 102, 103});
    kv_res->setLastBlockAligned(true);

    const size_t  tp_rank = 0;
    auto          meta    = std::make_shared<Dsv4Meta>("ut_dsv4_write");
    std::string   write_session_id("dsv4_write_session_1");
    Locations     expected_write_locations = genDsv4FullSwaLocations({101, 102, 103}, {0, 1, 2});
    WriteLocation write_location({write_session_id, static_cast<size_t>(0), expected_write_locations});

    EXPECT_CALL(*meta_clients_[tp_rank],
                StartWrite(Eq("start_write_ut_dsv4_write"),
                           std::vector<int64_t>({101, 102, 103}),
                           _,
                           Eq(std::vector<std::string>()),
                           _))
        .WillOnce(Return(StartWriteReturnType({ClientErrorCode::ER_OK, write_location})));

    UriStrVec                expected_uris = genDsv4Uris({101, 102, 103}, {0, 1, 2});
    UriStrVec                actual_uris   = genDsv4Uris({101, 102, 103}, {0, 1, 2}, "actual_");
    std::vector<std::string> expect_block_ids({"1",  "11", "21", "31", "41", "51", "61", "2",  "12", "22", "32",
                                               "42", "52", "62", "3",  "13", "23", "33", "43", "53", "63"});
    EXPECT_CALL(*transfer_client_, SaveKvCaches(Eq(expected_uris), _, TransferTraceInfoMatcher(expect_block_ids)))
        .WillOnce(Return(SaveKvCachesReturnType({ClientErrorCode::ER_OK, actual_uris})));

    Locations expected_actual_locations = genDsv4FullSwaLocations({101, 102, 103}, {0, 1, 2}, "actual_");
    EXPECT_CALL(*meta_clients_[tp_rank],
                FinishWrite(Eq("finish_write_ut_dsv4_write"),
                            write_session_id,
                            Eq(BlockMask(static_cast<size_t>(3))),
                            Eq(expected_actual_locations)))
        .WillOnce(Return(ClientErrorCode::ER_OK));

    auto ctx = remote_connectors_[tp_rank]->asyncWrite(kv_res, meta);
    waitDsv4AsyncContextDone(std::static_pointer_cast<AsyncContext>(ctx));
    ASSERT_TRUE(ctx->success());
}

TEST_F(RemoteConnectorMockDsv4Test, dsv4_write_with_block_mask) {
    auto kv_res = std::make_shared<KVCacheResource>();
    fillDsv4GroupBlocks(kv_res, {101, 102, 103});
    kv_res->setLastBlockAligned(true);

    const size_t  tp_rank = 0;
    auto          meta    = std::make_shared<Dsv4Meta>("ut_dsv4_write_mask");
    std::string   write_session_id("dsv4_write_session_mask");
    Locations     expected_write_locations = genDsv4FullSwaLocations({102, 103}, {0, 1});
    WriteLocation write_location({write_session_id, static_cast<size_t>(1), expected_write_locations});

    EXPECT_CALL(*meta_clients_[tp_rank],
                StartWrite(Eq("start_write_ut_dsv4_write_mask"),
                           std::vector<int64_t>({101, 102, 103}),
                           _,
                           Eq(std::vector<std::string>()),
                           _))
        .WillOnce(Return(StartWriteReturnType({ClientErrorCode::ER_OK, write_location})));

    UriStrVec                expected_uris = genDsv4Uris({102, 103}, {0, 1});
    UriStrVec                actual_uris   = genDsv4Uris({102, 103}, {0, 1}, "actual_");
    std::vector<std::string> expect_block_ids(
        {"2", "12", "22", "32", "42", "52", "62", "3", "13", "23", "33", "43", "53", "63"});
    EXPECT_CALL(*transfer_client_, SaveKvCaches(Eq(expected_uris), _, TransferTraceInfoMatcher(expect_block_ids)))
        .WillOnce(Return(SaveKvCachesReturnType({ClientErrorCode::ER_OK, actual_uris})));

    Locations expected_actual_locations = genDsv4FullSwaLocations({102, 103}, {0, 1}, "actual_");
    EXPECT_CALL(*meta_clients_[tp_rank],
                FinishWrite(Eq("finish_write_ut_dsv4_write_mask"),
                            write_session_id,
                            Eq(BlockMask(static_cast<size_t>(2))),
                            Eq(expected_actual_locations)))
        .WillOnce(Return(ClientErrorCode::ER_OK));

    auto ctx = remote_connectors_[tp_rank]->asyncWrite(kv_res, meta);
    waitDsv4AsyncContextDone(std::static_pointer_cast<AsyncContext>(ctx));
    ASSERT_TRUE(ctx->success());
}

TEST_F(RemoteConnectorMockDsv4Test, dsv4_write_last_block_not_aligned) {
    auto kv_res = std::make_shared<KVCacheResource>();
    fillDsv4GroupBlocks(kv_res, {101, 102, 103});
    kv_res->setLastBlockAligned(false);

    const size_t  tp_rank = 0;
    auto          meta    = std::make_shared<Dsv4Meta>("ut_dsv4_write_unaligned");
    std::string   write_session_id("dsv4_write_session_unaligned");
    Locations     expected_write_locations = genDsv4FullSwaLocations({102}, {0});
    WriteLocation write_location({write_session_id, static_cast<size_t>(1), expected_write_locations});

    EXPECT_CALL(*meta_clients_[tp_rank],
                StartWrite(Eq("start_write_ut_dsv4_write_unaligned"),
                           std::vector<int64_t>({101, 102}),
                           _,
                           Eq(std::vector<std::string>()),
                           _))
        .WillOnce(Return(StartWriteReturnType({ClientErrorCode::ER_OK, write_location})));

    UriStrVec                expected_uris = genDsv4Uris({102}, {0});
    UriStrVec                actual_uris   = genDsv4Uris({102}, {0}, "actual_");
    std::vector<std::string> expect_block_ids({"2", "12", "22", "32", "42", "52", "62"});
    EXPECT_CALL(*transfer_client_, SaveKvCaches(Eq(expected_uris), _, TransferTraceInfoMatcher(expect_block_ids)))
        .WillOnce(Return(SaveKvCachesReturnType({ClientErrorCode::ER_OK, actual_uris})));

    Locations expected_actual_locations = genDsv4FullSwaLocations({102}, {0}, "actual_");
    EXPECT_CALL(*meta_clients_[tp_rank],
                FinishWrite(Eq("finish_write_ut_dsv4_write_unaligned"),
                            write_session_id,
                            Eq(BlockMask(static_cast<size_t>(1))),
                            Eq(expected_actual_locations)))
        .WillOnce(Return(ClientErrorCode::ER_OK));

    auto ctx = remote_connectors_[tp_rank]->asyncWrite(kv_res, meta);
    waitDsv4AsyncContextDone(std::static_pointer_cast<AsyncContext>(ctx));
    ASSERT_TRUE(ctx->success());
}

TEST_F(RemoteConnectorMockDsv4Test, dsv4_write_start_write_fail) {
    auto kv_res = std::make_shared<KVCacheResource>();
    fillDsv4GroupBlocks(kv_res, {101, 102, 103});
    kv_res->setLastBlockAligned(true);

    const size_t tp_rank = 0;
    auto         meta    = std::make_shared<Dsv4Meta>("ut_dsv4_start_fail");

    EXPECT_CALL(*meta_clients_[tp_rank],
                StartWrite(Eq("start_write_ut_dsv4_start_fail"),
                           std::vector<int64_t>({101, 102, 103}),
                           _,
                           Eq(std::vector<std::string>()),
                           _))
        .WillOnce(Return(StartWriteReturnType({ClientErrorCode::ER_INVALID_GRPCSTATUS, {}})));

    EXPECT_CALL(*transfer_client_, SaveKvCaches(_, _, _)).Times(0);
    EXPECT_CALL(*meta_clients_[tp_rank], FinishWrite(_, _, _, _)).Times(0);

    auto ctx = remote_connectors_[tp_rank]->asyncWrite(kv_res, meta);
    waitDsv4AsyncContextDone(std::static_pointer_cast<AsyncContext>(ctx));
    ASSERT_FALSE(ctx->success());
    auto rctx = std::dynamic_pointer_cast<RemoteConnectorAsyncContext>(ctx);
    ASSERT_NE(nullptr, rctx);
    ASSERT_EQ(RemoteConnectorState::State::RCS_ERROR, rctx->state());
}

TEST_F(RemoteConnectorMockDsv4Test, dsv4_write_save_fail) {
    auto kv_res = std::make_shared<KVCacheResource>();
    fillDsv4GroupBlocks(kv_res, {101, 102, 103});
    kv_res->setLastBlockAligned(true);

    const size_t  tp_rank = 0;
    auto          meta    = std::make_shared<Dsv4Meta>("ut_dsv4_save_fail");
    std::string   write_session_id("dsv4_write_session_save_fail");
    Locations     expected_write_locations = genDsv4FullSwaLocations({101, 102, 103}, {0, 1, 2});
    WriteLocation write_location({write_session_id, static_cast<size_t>(0), expected_write_locations});

    EXPECT_CALL(*meta_clients_[tp_rank],
                StartWrite(Eq("start_write_ut_dsv4_save_fail"),
                           std::vector<int64_t>({101, 102, 103}),
                           _,
                           Eq(std::vector<std::string>()),
                           _))
        .WillOnce(Return(StartWriteReturnType({ClientErrorCode::ER_OK, write_location})));

    UriStrVec                expected_uris = genDsv4Uris({101, 102, 103}, {0, 1, 2});
    std::vector<std::string> expect_block_ids({"1",  "11", "21", "31", "41", "51", "61", "2",  "12", "22", "32",
                                               "42", "52", "62", "3",  "13", "23", "33", "43", "53", "63"});
    EXPECT_CALL(*transfer_client_, SaveKvCaches(Eq(expected_uris), _, TransferTraceInfoMatcher(expect_block_ids)))
        .WillOnce(Return(SaveKvCachesReturnType({ClientErrorCode::ER_SDK_TIMEOUT, UriStrVec({})})));

    EXPECT_CALL(*meta_clients_[tp_rank],
                FinishWrite(Eq("finish_write_ut_dsv4_save_fail"),
                            write_session_id,
                            Eq(BlockMask(static_cast<size_t>(0))),
                            Eq(Locations({}))))
        .WillOnce(Return(ClientErrorCode::ER_OK));

    auto ctx = remote_connectors_[tp_rank]->asyncWrite(kv_res, meta);
    waitDsv4AsyncContextDone(std::static_pointer_cast<AsyncContext>(ctx));
    ASSERT_FALSE(ctx->success());
    auto rctx = std::dynamic_pointer_cast<RemoteConnectorAsyncContext>(ctx);
    ASSERT_NE(nullptr, rctx);
    ASSERT_EQ(RemoteConnectorState::State::RCS_ERROR, rctx->state());
}

}  // namespace test
}  // namespace rtp_llm
