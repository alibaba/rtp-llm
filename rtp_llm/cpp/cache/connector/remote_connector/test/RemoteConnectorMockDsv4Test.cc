#include <algorithm>
#include <chrono>
#include <optional>
#include <thread>

#include "rtp_llm/cpp/cache/connector/AsyncContext.h"
#include "rtp_llm/cpp/cache/HybridPoolConfigCreator.h"
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

// DSV4 architectural ground-truth used to cross-check cache layout in this test fixture.
// Mirrors DSV4CacheConfigHelper's pool layout: 3 paged FULL pools (CSA_KV / HCA_KV / INDEXER_KV)
// followed by 4 fixed-size SWA pools (INDEXER_STATE / CSA_STATE / HCA_STATE / SWA_KV).
constexpr int kDsv4PoolNum     = 7;
constexpr int kDsv4FullPoolNum = 3;

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
    KVCacheConfig     kv_cache_config;
    kv_cache_config.seq_size_per_block     = 128;
    kv_cache_config.dsv4_fixed_pool_blocks = 256;
    CacheConfig       config           = HybridPoolConfigCreator::createConfig(mc, pc, kv_cache_config, false, 0);
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
        ASSERT_EQ(cache_config_.cache_specs.size(), static_cast<size_t>(kDsv4PoolNum));
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

    // swa_valid_positions semantics:
    //   - std::nullopt (default): SWA groups (gid 3..6) valid at every position — back-compat
    //                             for tests that pre-date linear_step support.
    //   - empty vector {}: SWA all NULL — exercises the "no SWA data anywhere" path.
    //   - {i, j, ...}: SWA valid only at those cache_key positions — exercises the sparse
    //                  layout produced by SWAKVCacheGroup when linear_step > 1 or
    //                  enable_reuse_cache=false (only step-hit + active-tail allocated).
    void fillDsv4GroupBlocks(const std::shared_ptr<KVCacheResource>&   kv_res,
                             const CacheKeysType&                      keys                = {101, 102, 103, 104},
                             const std::optional<std::vector<size_t>>& swa_valid_positions = std::nullopt) {
        kv_res->group_block_ids.clear();
        int n = static_cast<int>(keys.size());
        for (int g = 0; g < kDsv4PoolNum; g++) {
            const bool       is_swa_group = (g >= kDsv4FullPoolNum);
            BlockIndicesType indices;
            for (int i = 0; i < n; i++) {
                bool valid = true;
                if (is_swa_group && swa_valid_positions.has_value()) {
                    valid = std::find(swa_valid_positions->begin(),
                                      swa_valid_positions->end(),
                                      static_cast<size_t>(i))
                            != swa_valid_positions->end();
                }
                indices.push_back(valid ? static_cast<BlockIdxType>((i + 1) + g * 10) : NULL_BLOCK_IDX);
            }
            kv_res->group_block_ids.push_back(makeGroupBlockIds(indices));
        }
        kv_res->cache_keys = keys;
    }

    // Generate URIs matching genDsv4FullSwaLocations (F0..F2 for all, L3..L6 at linear_pos_vec).
    UriStrVec genDsv4Uris(const CacheKeysType&       cache_keys,
                          const std::vector<size_t>& linear_pos_vec = {},
                          const std::string&         uri_prefix    = "") {
        UriStrVec res;
        size_t    pos_idx = 0;
        for (size_t i = 0; i < cache_keys.size(); i++) {
            for (int g = 0; g < kDsv4FullPoolNum; g++) {
                std::string full_group_name = "F" + std::to_string(g);
                for (int r = 0; r < tp_size_; r++) {
                    res.push_back(uri_prefix + "uri_" + full_group_name + "_" + std::to_string(r) + "_"
                                  + std::to_string(cache_keys[i]));
                }
            }
            if (!linear_pos_vec.empty() && pos_idx < linear_pos_vec.size() && i == linear_pos_vec[pos_idx]) {
                for (int g = kDsv4FullPoolNum; g < kDsv4PoolNum; g++) {
                    std::string linear_group_name = "L" + std::to_string(g);
                    for (int r = 0; r < tp_size_; r++) {
                        res.push_back(uri_prefix + "uri_" + linear_group_name + "_" + std::to_string(r) + "_"
                                      + std::to_string(cache_keys[i]));
                    }
                }
                pos_idx++;
            }
        }
        return res;
    }

    kv_cache_manager::Locations genDsv4FullSwaLocations(const CacheKeysType&       cache_keys,
                                                        const std::vector<size_t>& linear_pos_vec = {},
                                                        const std::string&         uri_prefix    = "") {
        kv_cache_manager::Locations locations;
        locations.resize(cache_keys.size(), {});
        for (size_t i = 0; i < cache_keys.size(); i++) {
            for (int g = 0; g < kDsv4FullPoolNum; g++) {
                std::string full_group_name = "F" + std::to_string(g);
                for (int r = 0; r < tp_size_; r++) {
                    std::string uri = uri_prefix + "uri_" + full_group_name + "_" + std::to_string(r) + "_"
                                      + std::to_string(cache_keys[i]);
                    locations[i].push_back(
                        kv_cache_manager::LocationSpecUnit({genLocationSpecName(r, full_group_name), uri}));
                }
            }
        }
        for (auto pos : linear_pos_vec) {
            for (int g = kDsv4FullPoolNum; g < kDsv4PoolNum; g++) {
                std::string linear_group_name = "L" + std::to_string(g);
                for (int r = 0; r < tp_size_; r++) {
                    std::string uri = uri_prefix + "uri_" + linear_group_name + "_" + std::to_string(r) + "_"
                                      + std::to_string(cache_keys[pos]);
                    locations[pos].push_back(
                        kv_cache_manager::LocationSpecUnit({genLocationSpecName(r, linear_group_name), uri}));
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
    for (int g = kDsv4FullPoolNum; g < kDsv4FullPoolNum + 2; g++) {
        std::string linear_group_name = "L" + std::to_string(g);
        for (int r = 0; r < tp_size_; r++) {
            std::string uri = "uri_" + linear_group_name + "_" + std::to_string(r) + "_103";
            expected_locations[2].push_back(
                kv_cache_manager::LocationSpecUnit({genLocationSpecName(r, linear_group_name), uri}));
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

    // filterNeedLoadLocations sees location[2].size()=5, doesn't match full(3) or full_linear(7) → return false
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

// With null checks inside each helper, CheckInvalidFullLinearLocationAndSetView
// rejects a full_linear anchor whose L specs have NULL local blocks. The backward
// iter then falls back to a lower key where L specs are valid and anchors there.
//
// Scenario: SDK returns full_linear at positions 0 and 1, full_only at position 2.
// SWA valid only at position 0 → pos 1 anchor rejected → pos 0 becomes anchor.
TEST_F(RemoteConnectorMockDsv4Test, dsv4_read_drops_l_specs_when_local_swa_is_null) {
    auto kv_res = std::make_shared<KVCacheResource>();
    // SWA valid only at position 0; positions 1..3 carry NULL_BLOCK_IDX.
    fillDsv4GroupBlocks(kv_res, {101, 102, 103, 104}, std::optional<std::vector<size_t>>(std::vector<size_t>{0}));

    const size_t tp_rank = 0;
    // SDK sees L data at pos 0 and pos 1.  Backward iter:
    //   i=2: full-only → IsValidFullLocation, OK.
    //   i=1: full_linear, SWA NULL → anchor rejected (view cleared, exist_linear stays false).
    //   i=0: full_linear, SWA valid → anchor accepted (view[0] = all 7 specs).
    Locations expected_locations = genDsv4FullSwaLocations({101, 102, 103}, {0, 1});
    auto      meta               = std::make_shared<Dsv4Meta>("ut_dsv4_read_swa_null");

    EXPECT_CALL(*meta_clients_[tp_rank],
                MatchLocation(Eq("match_ut_dsv4_read_swa_null"),
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

    // Only pos 0 loaded (all 7 groups); pos 1 anchor rejected, pos 2 full-only not in view.
    UriStrVec                expected_uris = genDsv4Uris({101}, {0});
    std::vector<std::string> expect_block_ids({"1", "11", "21", "31", "41", "51", "61"});
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
    ASSERT_EQ(kv_res->remoteReuseBlockNum(), 1);
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

// =====================================================================================
// linear_step > 1 sparse SWA write coverage.
//
// SWAKVCacheGroup (post 60004e2fa) keeps block_indices.size() == valid_keys_size with
// NULL_BLOCK_IDX at non-step-hit positions. Each cache_key that has SWA blocks valid is
// written as full_linear (all 7 specs); each cache_key whose SWA slots are NULL is
// written as full_only (just F0/F1/F2). The mocks below pin the exact spec-group-name
// vector handed to StartWrite, the URI + block_id transfer order for SaveKvCaches, and
// the FinishWrite block_mask. These exercise the path the prior tests missed because
// fillDsv4GroupBlocks used to mark every position SWA-valid (which collapsed to the
// is_all_full_linear → clear() shortcut and matched Eq(vector<string>())).
// =====================================================================================

TEST_F(RemoteConnectorMockDsv4Test, dsv4_write_sparse_swa_full_only_at_swa_null_positions) {
    auto kv_res = std::make_shared<KVCacheResource>();
    // SWA valid only at positions 1 and 3; positions 0 and 2 have NULL SWA → full_only.
    fillDsv4GroupBlocks(kv_res, {101, 102, 103, 104}, std::optional<std::vector<size_t>>({1, 3}));
    kv_res->setLastBlockAligned(true);

    const size_t  tp_rank = 0;
    auto          meta    = std::make_shared<Dsv4Meta>("ut_dsv4_write_sparse");
    std::string   write_session_id("dsv4_write_session_sparse");
    Locations     expected_write_locations = genDsv4FullSwaLocations({101, 102, 103, 104}, {1, 3});
    WriteLocation write_location({write_session_id, static_cast<size_t>(0), expected_write_locations});

    // FullLinearLayerGroupPolicy::getNeedWriteGroups emits per-key names; not cleared
    // because is_all_full_linear == false (positions 0 and 2 are full_only).
    std::vector<std::string> expected_spec_group_names = {
        "F0F1F2", "F0F1F2L3L4L5L6", "F0F1F2", "F0F1F2L3L4L5L6"};

    EXPECT_CALL(*meta_clients_[tp_rank],
                StartWrite(Eq("start_write_ut_dsv4_write_sparse"),
                           std::vector<int64_t>({101, 102, 103, 104}),
                           _,
                           Eq(expected_spec_group_names),
                           _))
        .WillOnce(Return(StartWriteReturnType({ClientErrorCode::ER_OK, write_location})));

    UriStrVec                expected_uris = genDsv4Uris({101, 102, 103, 104}, {1, 3});
    UriStrVec                actual_uris   = genDsv4Uris({101, 102, 103, 104}, {1, 3}, "actual_");
    // Transfer order is forward by position; per position iterate the spec list.
    // pos 0: F0/F1/F2 → 1,11,21    (3 ids)
    // pos 1: F0/F1/F2/L3/L4/L5/L6 → 2,12,22,32,42,52,62  (7 ids)
    // pos 2: F0/F1/F2 → 3,13,23    (3 ids)
    // pos 3: F0/F1/F2/L3/L4/L5/L6 → 4,14,24,34,44,54,64  (7 ids)
    std::vector<std::string> expect_block_ids({"1",  "11", "21", "2",  "12", "22", "32", "42", "52", "62",
                                                "3",  "13", "23", "4",  "14", "24", "34", "44", "54", "64"});
    EXPECT_CALL(*transfer_client_, SaveKvCaches(Eq(expected_uris), _, TransferTraceInfoMatcher(expect_block_ids)))
        .WillOnce(Return(SaveKvCachesReturnType({ClientErrorCode::ER_OK, actual_uris})));

    Locations expected_actual_locations = genDsv4FullSwaLocations({101, 102, 103, 104}, {1, 3}, "actual_");
    EXPECT_CALL(*meta_clients_[tp_rank],
                FinishWrite(Eq("finish_write_ut_dsv4_write_sparse"),
                            write_session_id,
                            Eq(BlockMask(static_cast<size_t>(4))),
                            Eq(expected_actual_locations)))
        .WillOnce(Return(ClientErrorCode::ER_OK));

    auto ctx = remote_connectors_[tp_rank]->asyncWrite(kv_res, meta);
    waitDsv4AsyncContextDone(std::static_pointer_cast<AsyncContext>(ctx));
    ASSERT_TRUE(ctx->success());
}

TEST_F(RemoteConnectorMockDsv4Test, dsv4_write_all_swa_null_writes_full_only_everywhere) {
    auto kv_res = std::make_shared<KVCacheResource>();
    // All SWA slots NULL → every key is full_only.
    // Use std::in_place to disambiguate "optional containing an empty vector" from
    // "default-constructed optional (== nullopt)", which would otherwise wrongly mark
    // every SWA position valid via the fixture's nullopt back-compat path.
    fillDsv4GroupBlocks(kv_res, {101, 102, 103}, std::optional<std::vector<size_t>>(std::in_place));
    kv_res->setLastBlockAligned(true);

    const size_t  tp_rank = 0;
    auto          meta    = std::make_shared<Dsv4Meta>("ut_dsv4_write_all_null");
    std::string   write_session_id("dsv4_write_session_all_null");
    // Empty linear_pos_vec → no L specs at any position.
    Locations     expected_write_locations = genDsv4FullSwaLocations({101, 102, 103}, {});
    WriteLocation write_location({write_session_id, static_cast<size_t>(0), expected_write_locations});

    std::vector<std::string> expected_spec_group_names = {"F0F1F2", "F0F1F2", "F0F1F2"};

    EXPECT_CALL(*meta_clients_[tp_rank],
                StartWrite(Eq("start_write_ut_dsv4_write_all_null"),
                           std::vector<int64_t>({101, 102, 103}),
                           _,
                           Eq(expected_spec_group_names),
                           _))
        .WillOnce(Return(StartWriteReturnType({ClientErrorCode::ER_OK, write_location})));

    UriStrVec                expected_uris = genDsv4Uris({101, 102, 103}, {});
    UriStrVec                actual_uris   = genDsv4Uris({101, 102, 103}, {}, "actual_");
    // Per position only F0/F1/F2 → 3 ids per position.
    std::vector<std::string> expect_block_ids({"1", "11", "21", "2", "12", "22", "3", "13", "23"});
    EXPECT_CALL(*transfer_client_, SaveKvCaches(Eq(expected_uris), _, TransferTraceInfoMatcher(expect_block_ids)))
        .WillOnce(Return(SaveKvCachesReturnType({ClientErrorCode::ER_OK, actual_uris})));

    Locations expected_actual_locations = genDsv4FullSwaLocations({101, 102, 103}, {}, "actual_");
    EXPECT_CALL(*meta_clients_[tp_rank],
                FinishWrite(Eq("finish_write_ut_dsv4_write_all_null"),
                            write_session_id,
                            Eq(BlockMask(static_cast<size_t>(3))),
                            Eq(expected_actual_locations)))
        .WillOnce(Return(ClientErrorCode::ER_OK));

    auto ctx = remote_connectors_[tp_rank]->asyncWrite(kv_res, meta);
    waitDsv4AsyncContextDone(std::static_pointer_cast<AsyncContext>(ctx));
    ASSERT_TRUE(ctx->success());
}

TEST_F(RemoteConnectorMockDsv4Test, dsv4_write_sparse_swa_at_active_tail_only) {
    auto kv_res = std::make_shared<KVCacheResource>();
    // Mimics SWAKVCacheGroup with reuse_cache=false: only the last kSwaActiveTailBlocks=2
    // positions hold SWA blocks; earlier positions are NULL_BLOCK_IDX.
    fillDsv4GroupBlocks(kv_res, {101, 102, 103, 104}, std::optional<std::vector<size_t>>({2, 3}));
    kv_res->setLastBlockAligned(true);

    const size_t  tp_rank = 0;
    auto          meta    = std::make_shared<Dsv4Meta>("ut_dsv4_write_tail_only");
    std::string   write_session_id("dsv4_write_session_tail_only");
    Locations     expected_write_locations = genDsv4FullSwaLocations({101, 102, 103, 104}, {2, 3});
    WriteLocation write_location({write_session_id, static_cast<size_t>(0), expected_write_locations});

    // Per-key (forward) with the simplified write logic:
    //   i=0: SWA NULL → full_only.
    //   i=1: SWA NULL → full_only.
    //   i=2: SWA valid → full_linear.
    //   i=3: SWA valid → full_linear.
    // is_all_full_linear = false → vector kept.
    std::vector<std::string> expected_spec_group_names = {
        "F0F1F2", "F0F1F2", "F0F1F2L3L4L5L6", "F0F1F2L3L4L5L6"};

    EXPECT_CALL(*meta_clients_[tp_rank],
                StartWrite(Eq("start_write_ut_dsv4_write_tail_only"),
                           std::vector<int64_t>({101, 102, 103, 104}),
                           _,
                           Eq(expected_spec_group_names),
                           _))
        .WillOnce(Return(StartWriteReturnType({ClientErrorCode::ER_OK, write_location})));

    UriStrVec                expected_uris = genDsv4Uris({101, 102, 103, 104}, {2, 3});
    UriStrVec                actual_uris   = genDsv4Uris({101, 102, 103, 104}, {2, 3}, "actual_");
    std::vector<std::string> expect_block_ids({"1",  "11", "21", "2",  "12", "22", "3",  "13", "23", "33",
                                                "43", "53", "63", "4",  "14", "24", "34", "44", "54", "64"});
    EXPECT_CALL(*transfer_client_, SaveKvCaches(Eq(expected_uris), _, TransferTraceInfoMatcher(expect_block_ids)))
        .WillOnce(Return(SaveKvCachesReturnType({ClientErrorCode::ER_OK, actual_uris})));

    Locations expected_actual_locations = genDsv4FullSwaLocations({101, 102, 103, 104}, {2, 3}, "actual_");
    EXPECT_CALL(*meta_clients_[tp_rank],
                FinishWrite(Eq("finish_write_ut_dsv4_write_tail_only"),
                            write_session_id,
                            Eq(BlockMask(static_cast<size_t>(4))),
                            Eq(expected_actual_locations)))
        .WillOnce(Return(ClientErrorCode::ER_OK));

    auto ctx = remote_connectors_[tp_rank]->asyncWrite(kv_res, meta);
    waitDsv4AsyncContextDone(std::static_pointer_cast<AsyncContext>(ctx));
    ASSERT_TRUE(ctx->success());
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
