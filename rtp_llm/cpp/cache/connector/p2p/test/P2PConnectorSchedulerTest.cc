#include <algorithm>
#include <future>
#include <set>
#include <map>
#include <chrono>
#include <thread>
#include <gtest/gtest.h>
#include "grpc++/grpc++.h"

#include "autil/NetUtil.h"
#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/cache/CoordinatorCacheManager.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PBroadcastClient.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PSchedulerDecodeRead.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PSchedulerPrefillRead.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBufferUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/plan/PipelineLayoutCodec.h"
#include "rtp_llm/cpp/cache/connector/p2p/plan/ShardLayoutFactory.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/utils/Exception.h"
#include "rtp_llm/cpp/cache/connector/p2p/test/TestRpcServer.h"
#include "rtp_llm/cpp/cache/connector/p2p/test/MockGenerateStream.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorAsyncContext.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

namespace rtp_llm {

class P2PConnectorSchedulerTest: public ::testing::Test {
protected:
    struct AllocatedConnectorResource {
        CacheConfig                                 config;
        std::shared_ptr<CoordinatorCacheManager> allocator;
        KVCacheResourcePtr                          resource;
        BlockIdList                                 blocks;
        size_t                                      free_blocks_before{0};
        size_t                                      held_blocks{0};
    };

    ErrorInfo sendPrefillKV(const std::string&                                   unique_key,
                            int64_t                                              request_id,
                            const std::vector<std::pair<std::string, uint32_t>>& workers,
                            int64_t                                              deadline_ms,
                            std::function<bool()>                                is_cancelled,
                            int64_t                                              request_deadline_ms) {
        const int  tp_size = static_cast<int>(workers.size());
        const auto plan    = prefill_scheduler_->planFor(tp_size, unique_key);
        if (!plan->ok()) {
            return plan->error;
        }
        const auto prepared =
            prefill_scheduler_->preparePlan(tp_size, plan->plan.digest(), unique_key, workers.size(), {});
        if (!prepared.ok()) {
            return prepared.status();
        }
        return prefill_scheduler_->sendKVCache(unique_key,
                                               request_id,
                                               workers,
                                               deadline_ms,
                                               &prepared.value(),
                                               std::move(is_cancelled),
                                               request_deadline_ms);
    }

    void SetUp() override {
        // 创建测试用的 RPC 服务器（用于 P2PBroadcastClient）
        for (int i = 0; i < 2; ++i) {
            auto service = std::make_unique<TestRpcService>();
            auto server  = std::make_unique<TestRpcServer>(std::move(service));
            ASSERT_TRUE(server->start());
            tp_broadcast_servers_.push_back(std::move(server));
            tp_broadcast_addrs_.push_back("127.0.0.1:" + std::to_string(tp_broadcast_servers_.back()->listenPort()));
        }

        // 创建测试用的 RPC 服务器（用于 DecodeLoadHelper）
        auto prefill_service = std::make_unique<TestRpcService>();
        prefill_server_      = std::make_unique<TestRpcServer>(std::move(prefill_service));
        ASSERT_TRUE(prefill_server_->start());
        prefill_addr_ = "127.0.0.1:" + std::to_string(prefill_server_->listenPort());

        P2PConnectorSchedulerConfig scheduler_config;
        scheduler_config.worker_grpc_addrs = tp_broadcast_addrs_;
        scheduler_config.worker_addrs.push_back("127.0.0.1:12345:" + std::to_string(prefill_server_->listenPort()));
        scheduler_config.topology = test::makeTestCacheTopology(/*group_num=*/2, /*layer_num=*/2, {{0}, {1}});

        rebuildSchedulers(std::move(scheduler_config));
    }

    void TearDown() override {
        decode_scheduler_.reset();
        prefill_scheduler_.reset();
        tp_broadcast_client_.reset();
        tp_broadcast_servers_.clear();
        prefill_server_.reset();
    }

    // Create a resource with one tagged group per layer.
    KVCacheResourcePtr createValidKVCacheResource(int num_layers = 2, int blocks_per_layer = 2) {
        auto             resource = std::make_shared<KVCacheResource>();
        std::vector<std::vector<int>> layer_group_ids;
        layer_group_ids.reserve(static_cast<size_t>(num_layers));
        for (int i = 0; i < num_layers; ++i) {
            layer_group_ids.push_back({i});
        }
        resource->initGroups(test::makeTestCacheTopology(num_layers, num_layers, layer_group_ids));

        for (int layer_id = 0; layer_id < num_layers; ++layer_id) {
            for (int i = 0; i < blocks_per_layer; ++i) {
                resource->mutableBlockIds("group" + std::to_string(layer_id)).add({i});
            }
        }

        for (int i = 0; i < blocks_per_layer; ++i) {
            resource->cacheKeys().push_back(1000 + i);
        }

        return resource;
    }

    std::shared_ptr<MockMeta> createMockMeta(int64_t request_id, const std::string& unique_key, int64_t deadline_ms) {
        auto meta = std::make_shared<MockMeta>();
        meta->setRequestId(request_id);
        meta->setUniqueKey(unique_key);
        meta->setDeadlineMs(deadline_ms);
        meta->setPrefillAddr("127.0.0.1", static_cast<uint32_t>(prefill_server_->listenPort()));
        meta->setPrefillTpSize(1);
        meta->setPrefillCpSize(1);
        return meta;
    }

    KVCacheResourcePtr createInvalidKVCacheResource() {
        auto resource = std::make_shared<KVCacheResource>();
        return resource;
    }

    AllocatedConnectorResource createAllocatedConnectorResource() {
        AllocatedConnectorResource result;
        result.config = test::makeSimpleMhaCacheConfig(
            /*layer_num=*/2, /*block_num=*/8, /*tokens_per_block=*/1, DataType::TYPE_FP16);
        result.allocator = std::make_shared<CoordinatorCacheManager>(result.config, AllocationType::HOST);
        EXPECT_TRUE(result.allocator->init());

        auto block_pool = result.allocator->groupBlockPools().front();
        EXPECT_NE(block_pool, nullptr);
        result.free_blocks_before = result.allocator->freeBlocksNum();
        result.blocks             = block_pool->malloc(2).value();
        result.held_blocks        = result.blocks.size();
        block_pool->incRef(result.blocks);

        KVCacheResource source;
        source.initGroups(result.config.topologyPtr());
        source.cacheKeys() = {101, 102};
        source.mutableBlockIds(source.groupTags().at(0)).assign(result.blocks);
        result.resource = result.allocator->incrKVCacheRef(source, source.cacheKeys(), /*is_connector=*/true);
        block_pool->decRef(result.blocks);
        EXPECT_NE(result.resource, nullptr);
        EXPECT_EQ(result.allocator->freeBlocksNum(), result.free_blocks_before - result.held_blocks);
        for (const auto block : result.blocks) {
            EXPECT_EQ(block_pool->refCount(block), 1);
        }
        return result;
    }

    bool allBlocksHeldByLease(const AllocatedConnectorResource& allocated) const {
        const auto block_pool = allocated.allocator->groupBlockPools().front();
        return std::all_of(allocated.blocks.begin(), allocated.blocks.end(), [&](BlockIdxType block) {
            return block_pool->isAllocated(block) && block_pool->refCount(block) == 1;
        });
    }

    bool allBlocksReclaimed(const AllocatedConnectorResource& allocated) const {
        const auto block_pool = allocated.allocator->groupBlockPools().front();
        return std::none_of(allocated.blocks.begin(), allocated.blocks.end(), [&](BlockIdxType block) {
            return block_pool->isAllocated(block);
        });
    }

    void rebuildSchedulerForDeadlineTest(const std::shared_ptr<const CacheTopology>& topology) {
        P2PConnectorSchedulerConfig cfg;
        cfg.worker_grpc_addrs                  = tp_broadcast_addrs_;
        cfg.worker_addrs                       = {
            "127.0.0.1:12345:" + std::to_string(prefill_server_->listenPort())};
        cfg.topology                           = topology;
        cfg.load_cache_timeout_ms              = 80;
        cfg.p2p_cancel_broadcast_timeout_ms    = 50;
        cfg.p2p_lease_query_timeout_ms         = 10000;
        rebuildSchedulers(std::move(cfg));
    }

    template<typename Predicate>
    bool waitUntil(Predicate predicate, int timeout_ms = 2000) {
        const int64_t deadline_ms = currentTimeMs() + timeout_ms;
        while (!predicate() && currentTimeMs() < deadline_ms) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        return predicate();
    }

    // 等待 async context 完成，调用 checkDone() 以便异常在测试线程中抛出
    // 注意：如果需要在测试中捕获超时异常，请先停止 Decode scheduler 的后台线程。
    void waitAsyncContextDone(std::shared_ptr<P2PConnectorAsyncReadContext>& context,
                              int                                            timeout_ms = 5000,
                              bool                                           check_done = false) {
        int waited_ms = 0;
        while (!context->done() && waited_ms < timeout_ms) {
            if (check_done) {
                context->checkDone();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            waited_ms += 10;
        }
    }

    void rebuildSchedulerWithLeaseQueryTimeoutMs(int64_t timeout_ms) {
        P2PConnectorSchedulerConfig cfg;
        cfg.worker_grpc_addrs = tp_broadcast_addrs_;
        cfg.worker_addrs.push_back("127.0.0.1:12345:" + std::to_string(prefill_server_->listenPort()));
        cfg.topology = test::makeTestCacheTopology(/*group_num=*/2, /*layer_num=*/2, {{0}, {1}});
        cfg.p2p_lease_query_timeout_ms = timeout_ms;
        rebuildSchedulers(std::move(cfg));
    }

    std::shared_ptr<const CacheTopology> makeSingleMlaTopology() const {
        auto spec = test::makeResolvedMlaSpec(DataType::TYPE_FP16,
                                              /*kv_lora_rank=*/1,
                                              /*rope_head_dim=*/1,
                                              /*seq_size_per_block=*/1,
                                              "group0");
        GroupBase group;
        group.tag                       = "group0";
        group.spec                      = spec;
        group.policy                    = defaultCacheGroupPolicy(CacheGroupType::FULL);
        group.block_num                 = 16;
        return CacheTopology::create({std::move(group)}, {{0, {"group0"}}});
    }

    void rebuildSchedulerWithLayerAttnTypes(const std::vector<CacheGroupType>& layer_attn_types, int cp_size = 1) {
        P2PConnectorSchedulerConfig cfg;
        cfg.worker_grpc_addrs = tp_broadcast_addrs_;
        cfg.worker_addrs.push_back("127.0.0.1:12345:" + std::to_string(prefill_server_->listenPort()));
        std::vector<std::vector<int>> layer_group_ids;
        for (size_t i = 0; i < layer_attn_types.size(); ++i) {
            layer_group_ids.push_back({static_cast<int>(i)});
        }
        cfg.topology = cp_size > 1 ?
                           makeSingleMlaTopology() :
                           test::makeTestCacheTopology(static_cast<int>(layer_attn_types.size()),
                                                       static_cast<int>(layer_attn_types.size()),
                                                       layer_group_ids,
                                                       /*kernel_blocks_per_kv_block=*/1,
                                                       layer_attn_types);
        cfg.cp_size = cp_size;
        if (cp_size > 1) {
            // Keep the planner-facing ParallelismConfig consistent with the
            // legacy execution projection above. Production config populates
            // both from the same Prefill CP settings in KVCacheManager.
            cfg.parallelism_config.tp_size                                = cp_size;
            cfg.parallelism_config.prefill_cp_config.method               = CPRotateMethod::PREFILL_CP;
            cfg.parallelism_config.prefill_cp_config.kv_cache_sharded     = true;
            cfg.parallelism_config.prefill_cp_config.prefill_cp_size      = cp_size;
        }
        rebuildSchedulers(std::move(cfg));
    }

    void rebuildSchedulers(P2PConnectorSchedulerConfig scheduler_config) {
        decode_scheduler_.reset();
        prefill_scheduler_.reset();
        tp_broadcast_client_ = std::make_shared<P2PBroadcastClient>(
            scheduler_config.worker_grpc_addrs, scheduler_config.p2p_cancel_broadcast_timeout_ms);
        ASSERT_TRUE(tp_broadcast_client_->init());
        auto prefill_config = scheduler_config;
        prefill_config.parallelism_config.tp_size = prefill_config.worker_grpc_addrs.size();
        prefill_scheduler_ =
            std::make_unique<P2PSchedulerPrefillRead>(std::move(prefill_config), nullptr, tp_broadcast_client_);
        decode_scheduler_ =
            std::make_unique<P2PSchedulerDecodeRead>(std::move(scheduler_config), nullptr, tp_broadcast_client_);
        ASSERT_TRUE(decode_scheduler_->init("p2p_connector_scheduler_test"));
    }

protected:
    std::vector<std::unique_ptr<TestRpcServer>> tp_broadcast_servers_;
    std::vector<std::string>                    tp_broadcast_addrs_;
    std::unique_ptr<TestRpcServer>              prefill_server_;
    std::string                                 prefill_addr_;
    std::shared_ptr<P2PBroadcastClient>             tp_broadcast_client_;
    std::unique_ptr<P2PSchedulerPrefillRead>   prefill_scheduler_;
    std::unique_ptr<P2PSchedulerDecodeRead>    decode_scheduler_;
};

TEST_F(P2PConnectorSchedulerTest, AsyncReadUsesConfiguredLoadBudgetAndRequestDeadline) {
    P2PConnectorSchedulerConfig config;
    config.worker_grpc_addrs = tp_broadcast_addrs_;
    config.worker_addrs.push_back("127.0.0.1:12345:" + std::to_string(prefill_server_->listenPort()));
    config.topology              = test::makeTestCacheTopology(2, 2, {{0}, {1}});
    config.load_cache_timeout_ms = 1000;
    rebuildSchedulers(config);
    const int64_t before = currentTimeMs();
    const int64_t request_deadline_ms = before + 60000;
    auto meta = createMockMeta(9010, "configured_load_budget", request_deadline_ms);
    auto result = decode_scheduler_->asyncRead(createValidKVCacheResource(), meta, {2, 0}, true);
    ASSERT_TRUE(result.ok());
    waitAsyncContextDone(result.context);
    ASSERT_TRUE(result.context->success());
    auto request = prefill_server_->service()->getLastStartLoadRequest();
    EXPECT_EQ(request.timeout_ms(), 1000);
    EXPECT_GE(request.handoff_timeout_ms(), request_deadline_ms - currentTimeMs());
}

TEST_F(P2PConnectorSchedulerTest, AsyncReadKeepsRequestDeadlineSeparateFromLoadBudget) {
    const int64_t request_deadline_ms = currentTimeMs() + 2000;
    auto meta = createMockMeta(9011, "request_bounds_load", request_deadline_ms);
    auto result = decode_scheduler_->asyncRead(createValidKVCacheResource(), meta, {2, 0}, true);
    ASSERT_TRUE(result.ok());
    waitAsyncContextDone(result.context);
    ASSERT_TRUE(result.context->success());
    auto request = prefill_server_->service()->getLastStartLoadRequest();
    EXPECT_EQ(request.timeout_ms(), PDSepConfig{}.load_cache_timeout_ms);
    EXPECT_LE(request.handoff_timeout_ms(), 2000);
    EXPECT_GE(request.handoff_timeout_ms(), request_deadline_ms - currentTimeMs());
    EXPECT_EQ(result.context->request_deadline_ms_, request_deadline_ms);
    EXPECT_EQ(result.context->transfer_deadline_ms_, request_deadline_ms);
}

TEST_F(P2PConnectorSchedulerTest, PeerCpPlanUsesAdvertisedSizeInsteadOfDecodeConfiguration) {
    for (bool local_sharded : {false, true}) {
        P2PConnectorSchedulerConfig config;
        config.topology                   = makeSingleMlaTopology();
        config.parallelism_config.tp_size = 2;
        auto& cp                          = config.parallelism_config.prefill_cp_config;
        cp.method                         = local_sharded ? CPRotateMethod::PREFILL_CP : CPRotateMethod::DISABLED;
        cp.kv_cache_sharded               = local_sharded;
        cp.prefill_cp_size                = 8;  // Deliberately differs from the selected peer.
        P2PSchedulerDecodeRead scheduler(config, nullptr, nullptr);
        for (int peer_cp_size : {1, 2}) {
            SCOPED_TRACE(::testing::Message() << "local_sharded=" << local_sharded << " peer_cp_size=" << peer_cp_size);
            const auto plan = scheduler.planFor(2, peer_cp_size, "advertised-cp-layout");
            ASSERT_TRUE(plan->ok()) << plan->error.ToString();
            EXPECT_FALSE(plan->plan.routes.empty());
        }
    }
}

TEST_F(P2PConnectorSchedulerTest, SingleStagePlanUsesFullPeerLayoutWithLocalCpDisabled) {
    const auto topology      = makeSingleMlaTopology();
    const auto encode_layout = [&](ParallelismConfig pc, RoleType role) {
        P2PPipelineLayoutPB layout;
        layout.set_worker_count(pc.tp_size);
        auto* stage = layout.add_stages();
        stage->set_main_layer_begin(0);
        stage->set_main_layer_count(1);
        stage->add_block_table_tags("group0");
        for (int rank = 0; rank < pc.tp_size; ++rank) {
            stage->add_worker_indices(rank);
        }
        encodeP2PShardLayout(ShardLayoutFactory::fromTopology(*topology, pc, role, 1), stage->mutable_main_shards());
        stage->add_layers()->add_tags("group0");
        return layout.SerializeAsString();
    };
    P2PConnectorSchedulerConfig config;
    config.topology                   = topology;
    config.parallelism_config.tp_size = 2;
    config.pipeline_layout            = encode_layout(config.parallelism_config, RoleType::DECODE);
    P2PSchedulerDecodeRead scheduler(config, nullptr, nullptr);
    ParallelismConfig      peer;
    peer.tp_size                  = 2;
    peer.prefill_cp_config.method = CPRotateMethod::ALL_GATHER;
    for (bool peer_sharded : {false, true}) {
        peer.prefill_cp_config.kv_cache_sharded = peer_sharded;
        const auto advertised_layout            = encode_layout(peer, RoleType::PREFILL);
        const auto plan = scheduler.planFor(2, peer_sharded ? 2 : 1, "single-stage-peer-layout", advertised_layout);
        ASSERT_TRUE(plan->ok()) << plan->error.ToString();
        EXPECT_FALSE(plan->plan.routes.empty());
    }
}

TEST_F(P2PConnectorSchedulerTest, InvalidLoadConfigurationUsesBoundedSharedFallback) {
    for (const int64_t timeout : {int64_t{0}, int64_t{-1}, int64_t{2147483648LL}}) {
        P2PConnectorSchedulerConfig config;
        config.worker_grpc_addrs = tp_broadcast_addrs_;
        config.worker_addrs.push_back("127.0.0.1:12345:" + std::to_string(prefill_server_->listenPort()));
        config.topology = test::makeTestCacheTopology(2, 2, {{0}, {1}});
        config.load_cache_timeout_ms = timeout;
        rebuildSchedulers(config);
        const int64_t expected = timeout <= 0 ? PDSepConfig{}.load_cache_timeout_ms : 2147483647LL;
        EXPECT_EQ(decode_scheduler_->config_.load_cache_timeout_ms, expected);
        const int64_t request_deadline_ms = currentTimeMs() + 60000;
        auto meta = createMockMeta(9012, "normalized_load_budget", request_deadline_ms);
        auto result = decode_scheduler_->asyncRead(createValidKVCacheResource(), meta, {2, 0}, true);
        ASSERT_TRUE(result.ok());
        waitAsyncContextDone(result.context);
        ASSERT_TRUE(result.context->success());
        const int64_t wire_timeout = prefill_server_->service()->getLastStartLoadRequest().timeout_ms();
        EXPECT_EQ(wire_timeout, expected);
    }
}

TEST(P2PConnectorConfigTest, ProductionConfigNormalizesLoadTimeoutBeforeInit) {
    RuntimeConfig     runtime;
    CacheStoreConfig  cache_store;
    ParallelismConfig parallelism;
    for (auto role : {RoleType::PREFILL, RoleType::DECODE}) {
        for (int64_t timeout : {int64_t{-1}, int64_t{0}, int64_t{900000}, int64_t{2147483648LL}}) {
            PDSepConfig pd_sep;
            pd_sep.role_type             = role;
            pd_sep.load_cache_timeout_ms = timeout;
            const auto config            = P2PConnectorConfig::create(runtime, cache_store, parallelism, pd_sep, 1, 1);
            const auto expected          = timeout <= 0 ? PDSepConfig{}.load_cache_timeout_ms :
                                                          std::min<int64_t>(timeout, std::numeric_limits<int32_t>::max());
            EXPECT_EQ(config.scheduler_config.load_cache_timeout_ms, expected);
            EXPECT_EQ(config.worker_config.load_cache_timeout_ms, expected);
        }
    }
}

TEST(P2PConnectorConfigTest, LoadTimeoutComesFromSharedPDSepConfig) {
    RuntimeConfig runtime;
    CacheStoreConfig cache_store;
    ParallelismConfig parallelism;
    PDSepConfig pd_sep;
    EXPECT_EQ(pd_sep.load_cache_timeout_ms, 5000);
    pd_sep.load_cache_timeout_ms = 900000;
    auto config = P2PConnectorSchedulerConfig::create(runtime, cache_store, parallelism, pd_sep);
    EXPECT_EQ(config.load_cache_timeout_ms, 900000);
}

// ==================== sendKVCache 测试 (Prefill 端功能) ====================

// 测试：broadcast 成功
TEST_F(P2PConnectorSchedulerTest, HandleRead_ReturnOK_BroadcastSuccess) {
    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});
    decode_transfer_servers.push_back({"127.0.0.1", 12346});

    auto deadline_ms = currentTimeMs() + 1000;

    ErrorInfo error_info =
        sendPrefillKV("test_broadcast_success", 1001, decode_transfer_servers, deadline_ms, nullptr, deadline_ms);

    EXPECT_TRUE(error_info.ok());

    // 验证 BroadcastTp 被调用
    for (size_t i = 0; i < tp_broadcast_servers_.size(); ++i) {
        EXPECT_EQ(tp_broadcast_servers_[i]->service()->getBroadcastTpCallCount(), 1);
        EXPECT_EQ(tp_broadcast_servers_[i]->service()->getBroadcastTpCancelCallCount(), 0);
    }
}

TEST_F(P2PConnectorSchedulerTest, HandleReadReusesPreparedPlanAndBroadcastsOnlyActiveRoutes) {
    const std::string                             unique_key              = "test_active_route_filter";
    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers = {{"127.0.0.1", 12345},
                                                                             {"127.0.0.1", 12346}};
    auto plan = prefill_scheduler_->planFor(static_cast<int>(decode_transfer_servers.size()), unique_key);
    ASSERT_TRUE(plan->ok()) << plan->error.ToString();
    ASSERT_GT(plan->plan.routes.size(), 1u);

    const int  selected_route_id = plan->plan.routes.back().route_id;
    const auto prepared = prefill_scheduler_->preparePlan(2, plan->plan.digest(), unique_key, 2, {selected_route_id});
    ASSERT_TRUE(prepared.ok()) << prepared.status().ToString();
    plan.reset();
    prefill_scheduler_->plan_cache_.clear();
    const auto deadline_ms = currentTimeMs() + 1000;
    const auto error_info  = prefill_scheduler_->sendKVCache(
        unique_key, 1011, decode_transfer_servers, deadline_ms, &prepared.value(), nullptr, deadline_ms);
    ASSERT_TRUE(error_info.ok()) << error_info.ToString();
    EXPECT_TRUE(prefill_scheduler_->plan_cache_.empty());

    size_t broadcast_route_count = 0;
    for (const auto& server : tp_broadcast_servers_) {
        const auto request = server->service()->getLastBroadcastTpRequest();
        EXPECT_EQ(request.plan_digest(), prepared.value().digest);
        for (const auto& route : request.routes()) {
            EXPECT_EQ(route.route_id(), selected_route_id);
            ++broadcast_route_count;
        }
    }
    EXPECT_EQ(broadcast_route_count, 1u);
}

TEST_F(P2PConnectorSchedulerTest, HandleRead_FiltersLinearLayersByAttentionType) {
    rebuildSchedulerWithLayerAttnTypes({CacheGroupType::FULL, CacheGroupType::LINEAR});

    auto topology = test::makeTestCacheTopology(
        2, 2, {{0}, {1}}, /*kernel_blocks_per_kv_block=*/1, {CacheGroupType::FULL, CacheGroupType::LINEAR});
    auto resource = std::make_shared<KVCacheResource>();
    resource->initGroups(topology);
    resource->mutableBlockIds(resource->groupTags().at(0)).assign({10, 11, 12, 13});
    resource->mutableBlockIds(resource->groupTags().at(1)).assign({NULL_BLOCK_IDX, 21, NULL_BLOCK_IDX, 25});
    resource->cacheKeys() = {1000, 1001, 1002, 1003};

    auto converted = LayerCacheBufferUtil::convert(*resource, *topology);
    ASSERT_TRUE(converted.ok()) << converted.status().ToString();
    const auto& layer_buffers = converted.value();
    ASSERT_EQ(layer_buffers.size(), 2);
    EXPECT_EQ(layer_buffers[0]->blockIdMap().size(), 4);
    ASSERT_EQ(layer_buffers[1]->blockIdMap().size(), 1);
    EXPECT_EQ(layer_buffers[1]->getBlockId(1003), 25);

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});

    ErrorInfo error_info = sendPrefillKV(
        "test_linear_filter", 1009, decode_transfer_servers, currentTimeMs() + 1000, nullptr, currentTimeMs() + 1000);

    ASSERT_TRUE(error_info.ok());
    const auto rank0_request = tp_broadcast_servers_[0]->service()->getLastBroadcastTpRequest();
    const auto rank1_request = tp_broadcast_servers_[1]->service()->getLastBroadcastTpRequest();
    // Plan-driven Prefill broadcasts routes only. Each worker has already
    // produced its local LayerCacheBuffer through the per-layer path.
    EXPECT_EQ(rank0_request.layer_blocks_size(), 0);
    EXPECT_EQ(rank1_request.layer_blocks_size(), 0);
    const auto& active = rank0_request.routes_size() ? rank0_request : rank1_request;
    ASSERT_EQ(rank0_request.routes_size() + rank1_request.routes_size(), 2);
    ASSERT_EQ(active.routes_size(), 2);
    EXPECT_EQ(active.routes(0).cache_tag(), "group0");
    EXPECT_EQ(active.routes(1).cache_tag(), "group1");
}

// 测试：broadcast 返回失败（所有响应失败）
TEST_F(P2PConnectorSchedulerTest, HandleRead_ReturnError_BroadcastPartialFailed) {
    for (auto& server : tp_broadcast_servers_) {
        server->service()->setP2PResponseSuccess(false);
        break;
    }

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});

    auto deadline_ms = currentTimeMs() + 1000;

    ErrorInfo error_info =
        sendPrefillKV("test_broadcast_all_fail", 1003, decode_transfer_servers, deadline_ms, nullptr, deadline_ms);

    EXPECT_TRUE(error_info.hasError());

    // 验证 BroadcastTp 被调用
    for (size_t i = 0; i < tp_broadcast_servers_.size(); ++i) {
        EXPECT_EQ(tp_broadcast_servers_[i]->service()->getBroadcastTpCallCount(), 1);
    }
}

// 测试: broadcast worker 慢于 gRPC deadline，sendKVCache 返回超时错误
TEST_F(P2PConnectorSchedulerTest, HandleRead_ReturnError_BroadcastTimeout) {
    for (auto& server : tp_broadcast_servers_) {
        server->service()->setSleepMillis(500);  // 延迟 500ms
        break;
    }

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});

    auto deadline_ms = currentTimeMs() + 50;

    ErrorInfo error_info =
        sendPrefillKV("test_broadcast_timeout", 1004, decode_transfer_servers, deadline_ms, nullptr, deadline_ms);

    EXPECT_TRUE(error_info.hasError());
    EXPECT_EQ(error_info.code(), ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_TIMEOUT);
}

TEST_F(P2PConnectorSchedulerTest, HandleRead_DeadlineCrossingInCancellationCheckRemainsTimeout) {
    auto release = std::make_shared<std::promise<void>>();
    auto gate    = release->get_future().share();
    auto cleanup = std::shared_ptr<void>(nullptr, [release](void*) { release->set_value(); });
    for (auto& server : tp_broadcast_servers_) {
        server->service()->setP2PRequestGate(P2PConnectorBroadcastType::HANDLE_READ, [gate] { gate.wait(); });
    }
    const auto deadline_ms          = currentTimeMs() + 300;
    const auto deadline             = std::chrono::system_clock::time_point(std::chrono::milliseconds(deadline_ms));
    bool       cancellation_checked = false;
    const auto error                = sendPrefillKV(
        "deadline-crosses-cancel-check",
        1005,
        {{"127.0.0.1", 12345}},
        deadline_ms,
        [&] {
            cancellation_checked = true;
            // Deterministically model resource-store expiry being observed by
            // the cancellation callback at the transfer deadline.
            gate.wait_until(deadline);
            return true;
        },
        deadline_ms + 5000);
    EXPECT_TRUE(cancellation_checked);
    EXPECT_EQ(error.code(), ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_TIMEOUT);
}

// Cancel completion must release the caller without waiting for blocked HANDLE_READ workers.
TEST_F(P2PConnectorSchedulerTest, HandleRead_ExitsImmediately_AfterCancelDoneEvenIfBroadcastSlow) {
    auto release = std::make_shared<std::promise<void>>();
    auto gate = release->get_future().share();
    auto exited = std::make_shared<std::atomic<int>>(0);
    auto cleanup = std::shared_ptr<void>(nullptr, [release](void*) { release->set_value(); });
    std::vector<std::future<void>> entered;
    for (auto& server : tp_broadcast_servers_) {
        auto signal = std::make_shared<std::promise<void>>();
        entered.push_back(signal->get_future());
        server->service()->setP2PRequestGate(P2PConnectorBroadcastType::HANDLE_READ, [signal, gate, exited] {
            signal->set_value();
            gate.wait();
            exited->fetch_add(1);
        });
    }
    const auto deadline = currentTimeMs() + 30000;
    std::atomic<bool> cancelled{false};
    auto              sending = std::async(std::launch::async, [&] {
        return sendPrefillKV(
            "test_exit_after_cancel_done",
            4007,
            {{"127.0.0.1", 12345}},
            deadline,
            [&] { return cancelled.load(); },
            deadline);
    });
    for (auto& ready : entered) {
        const auto state = ready.wait_for(std::chrono::seconds(5));
        if (state != std::future_status::ready) cleanup.reset();
        ASSERT_EQ(state, std::future_status::ready);
    }
    cancelled.store(true);
    const auto state = sending.wait_for(std::chrono::seconds(5));
    // Cancellation must finish while every original HANDLE_READ remains blocked.
    EXPECT_EQ(state, std::future_status::ready);
    // sendKVCache submits best-effort cancellation without waiting for its RPC.
    // Observe receipt before checking exact counts, with HANDLE_READ still gated.
    EXPECT_TRUE(waitUntil([&] {
        return std::all_of(tp_broadcast_servers_.begin(), tp_broadcast_servers_.end(), [](const auto& server) {
            return server->service()->getBroadcastTpCancelCallCount() == 1;
        });
    }));
    EXPECT_EQ(exited->load(), 0);
    cleanup.reset();
    const auto error = sending.get();
    EXPECT_EQ(error.code(), ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_CANCELLED);
    for (const auto& server : tp_broadcast_servers_) {
        EXPECT_EQ(server->service()->getBroadcastTpCancelCallCount(), 1);
    }
}

// 测试: handleRead 被 client 取消, 返回失败
TEST_F(P2PConnectorSchedulerTest, HandleRead_ReturnFalse_BroadcastCancelled) {
    // 设置 TP worker 延迟响应，以便有足够的时间触发取消
    for (auto& server : tp_broadcast_servers_) {
        server->service()->setSleepMillis(200);
    }

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});

    auto deadline_ms = currentTimeMs() + 5000;
    auto request_deadline_ms = deadline_ms + 5000;

    // 使用 atomic 来控制取消状态
    std::atomic<bool> cancelled{false};
    auto              is_cancelled = [&cancelled]() { return cancelled.load(); };

    // 在另一个线程中延迟设置取消标志
    std::thread cancel_thread([&cancelled]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));  // 50ms 后取消
        cancelled = true;
    });

    ErrorInfo error_info = sendPrefillKV(
        "test_broadcast_cancelled", 1005, decode_transfer_servers, deadline_ms, is_cancelled, request_deadline_ms);

    cancel_thread.join();

    // 由于被取消，handleRead 应该返回错误码
    EXPECT_TRUE(error_info.hasError());
    EXPECT_EQ(error_info.code(), ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_CANCELLED);

    // 验证 BroadcastTp 被调用，且 CANCEL_HANDLE_READ 也被发送
    for (size_t i = 0; i < tp_broadcast_servers_.size(); ++i) {
        EXPECT_EQ(tp_broadcast_servers_[i]->service()->getP2PRequestCallCount(P2PConnectorBroadcastType::HANDLE_READ), 1);
        const auto cancel_check_deadline = currentTimeMs() + 1000;
        while (tp_broadcast_servers_[i]->service()->getBroadcastTpCancelCallCount() == 0
               && currentTimeMs() < cancel_check_deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        EXPECT_EQ(tp_broadcast_servers_[i]->service()->getBroadcastTpCancelCallCount(), 1);
        const auto cancel_request = tp_broadcast_servers_[i]->service()->getLastBroadcastTpRequest();
        EXPECT_EQ(cancel_request.request_id(), 1005);
        EXPECT_EQ(cancel_request.deadline_ms(), deadline_ms);
        EXPECT_EQ(cancel_request.request_deadline_ms(), request_deadline_ms);
    }
}

TEST_F(P2PConnectorSchedulerTest, AsyncRead_UnconfirmedLeaseKeepsPollingAndReleasesAfterLateStop) {
    auto tp_ctx = std::make_shared<P2PBroadcastClient::TpBroadcastResult::WorkerRpcContext>();
    tp_ctx->response.mutable_p2p_response()->set_error_code(
        transErrorCodeToRPC(ErrorCode::P2P_CONNECTOR_WORKER_READ_CANCELLED));
    tp_ctx->response.mutable_p2p_response()->set_error_message("worker cancelled");

    auto tp_result = std::make_shared<P2PBroadcastClient::TpBroadcastResult>(
        std::vector<std::shared_ptr<P2PBroadcastClient::TpBroadcastResult::WorkerRpcContext>>{tp_ctx});
    tp_result->finishRank(0, true);

    auto broadcast_result = std::make_shared<P2PBroadcastClient::Result>("cancel-hold", tp_result);

    auto server_result          = std::make_shared<DecodeLoadHelper::Result>();
    server_result->done_        = true;
    server_result->success_     = true;
    server_result->error_code   = ErrorCode::NONE_ERROR;
    server_result->error_message.clear();

    auto resource = createValidKVCacheResource(1, 1);
    auto collector = std::make_shared<P2PConnectorMetricsCollector>();
    auto context = std::make_shared<P2PConnectorAsyncReadContext>(
        resource, broadcast_result, server_result, collector, /*lease_query_timeout_ms=*/80);
    std::weak_ptr<KVCacheResource> weak_resource = resource;
    resource.reset();
    for (auto& server : tp_broadcast_servers_) {
        server->service()->setLeaseStatus(true, 1, 0, false);
    }

    context->checkDone();

    EXPECT_TRUE(context->done());
    EXPECT_FALSE(context->success());
    EXPECT_TRUE(context->needCancel());
    EXPECT_TRUE(context->resourceHoldPending());
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::P2P_CONNECTOR_WORKER_READ_CANCELLED);

    std::this_thread::sleep_for(std::chrono::milliseconds(120));
    context->cancel(tp_broadcast_client_);
    ASSERT_TRUE(waitUntil([&] { context->checkCancelDone(); return !context->needCancel(); }));
    context->pollLeaseIfNeeded(tp_broadcast_client_);
    EXPECT_TRUE(context->resourceHoldPending());
    EXPECT_FALSE(weak_resource.expired());
    for (auto& server : tp_broadcast_servers_) {
        server->service()->setLeaseStatus(true, 1, 1, true);
    }
    ASSERT_TRUE(waitUntil([&] {
        context->pollLeaseIfNeeded(tp_broadcast_client_);
        return !context->resourceHoldPending();
    }));
    // Holding the terminal context must not retain KV after physical completion.
    EXPECT_TRUE(weak_resource.expired());
}

TEST_F(P2PConnectorSchedulerTest, AsyncRead_CompletedBeforeDeadlineObservationRemainsSuccessful) {
    auto tp_ctx = std::make_shared<P2PBroadcastClient::TpBroadcastResult::WorkerRpcContext>();
    tp_ctx->response.mutable_p2p_response()->set_error_code(ErrorCodePB::NONE_ERROR);

    auto tp_result = std::make_shared<P2PBroadcastClient::TpBroadcastResult>(
        std::vector<std::shared_ptr<P2PBroadcastClient::TpBroadcastResult::WorkerRpcContext>>{tp_ctx});
    tp_result->finishRank(0, true);

    auto broadcast_result = std::make_shared<P2PBroadcastClient::Result>("completed-before-observation", tp_result);
    auto server_result     = std::make_shared<DecodeLoadHelper::Result>();
    server_result->done_    = true;
    server_result->success_ = true;
    auto collector          = std::make_shared<P2PConnectorMetricsCollector>();

    auto context = std::make_shared<P2PConnectorAsyncReadContext>(createValidKVCacheResource(1, 1),
                                                                   broadcast_result,
                                                                   server_result,
                                                                   collector,
                                                                   /*lease_query_timeout_ms=*/1000,
                                                                   /*no_transfer=*/false,
                                                                   /*request_deadline_ms=*/currentTimeMs() + 1000,
                                                                   /*transfer_deadline_ms=*/currentTimeMs() - 1);

    context->checkDone();
    EXPECT_FALSE(context->expireTransferDeadlineIfNeeded());
    EXPECT_TRUE(context->done());
    EXPECT_TRUE(context->success());
}

TEST_F(P2PConnectorSchedulerTest, AsyncRead_DeadlineBroadcastsCancelWhenReadAlreadySucceeded) {
    auto tp_ctx = std::make_shared<P2PBroadcastClient::TpBroadcastResult::WorkerRpcContext>();
    tp_ctx->response.mutable_p2p_response()->set_error_code(ErrorCodePB::NONE_ERROR);

    auto tp_result = std::make_shared<P2PBroadcastClient::TpBroadcastResult>(
        std::vector<std::shared_ptr<P2PBroadcastClient::TpBroadcastResult::WorkerRpcContext>>{tp_ctx});
    tp_result->finishRank(0, true);

    auto collector = std::make_shared<P2PConnectorMetricsCollector>();
    auto context = std::make_shared<P2PConnectorAsyncReadContext>(createValidKVCacheResource(1, 1),
                                                                   std::make_shared<P2PBroadcastClient::Result>(
                                                                       "read-success-at-deadline", tp_result),
                                                                   std::make_shared<DecodeLoadHelper::Result>(),
                                                                   collector,
                                                                   /*lease_query_timeout_ms=*/1000,
                                                                   /*no_transfer=*/false,
                                                                   /*request_deadline_ms=*/currentTimeMs() + 1000,
                                                                   /*transfer_deadline_ms=*/currentTimeMs() - 1);

    ASSERT_TRUE(context->expireTransferDeadlineIfNeeded());
    EXPECT_TRUE(context->needCancel());
    context->cancel(tp_broadcast_client_);
    for (int i = 0; i < 1000 && context->needCancel(); ++i) {
        context->checkCancelDone();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_FALSE(context->needCancel());
    for (const auto& server : tp_broadcast_servers_) {
        EXPECT_EQ(server->service()->getBroadcastTpCancelCallCount(), 1);
    }
}

TEST_F(P2PConnectorSchedulerTest, TransferWakeupUsesLocalSteadyBudgetAndPreservesWireDeadline) {
    const auto wire_deadline = currentTimeMs() + 60000;
    const auto before = P2PNotification::nowMs();
    auto context = std::make_shared<P2PConnectorAsyncReadContext>(
        createValidKVCacheResource(1, 1), "steady-transfer-budget",
        std::make_shared<P2PConnectorMetricsCollector>(), 10000, false,
        wire_deadline + 1000, wire_deadline);
    const auto wakeup = context->nextWakeupMs(false);
    EXPECT_EQ(context->transfer_deadline_ms_, wire_deadline);
    EXPECT_GT(wakeup, before);
    EXPECT_LE(wakeup, P2PNotification::nowMs() + 60000);
    EXPECT_FALSE(context->expireTransferDeadlineIfNeeded());
}

TEST_F(P2PConnectorSchedulerTest, ExpiredWireBudgetAnchorsImmediateLocalDeadline) {
    const auto wire_deadline = currentTimeMs() - 1;
    auto context = std::make_shared<P2PConnectorAsyncReadContext>(
        createValidKVCacheResource(1, 1), "steady-expired-budget",
        std::make_shared<P2PConnectorMetricsCollector>(), 10000, false,
        currentTimeMs() + 10000, wire_deadline);
    EXPECT_EQ(context->transfer_deadline_ms_, wire_deadline);
    EXPECT_LE(context->nextWakeupMs(false), P2PNotification::nowMs());
    EXPECT_TRUE(context->expireTransferDeadlineIfNeeded());
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::GENERATE_TIMEOUT);
}

TEST_F(P2PConnectorSchedulerTest, AsyncReadPendingContextWaitsForKickoff) {
    auto resource  = createValidKVCacheResource(1, 1);
    auto collector = std::make_shared<P2PConnectorMetricsCollector>();
    auto context   = std::make_shared<P2PConnectorAsyncReadContext>(
        resource, "pending-kickoff", collector, /*lease_query_timeout_ms=*/0);

    context->checkDone();

    EXPECT_FALSE(context->done());
    EXPECT_FALSE(context->success());
    EXPECT_FALSE(context->needCancel());
    EXPECT_EQ(context->uniqueKey(), "pending-kickoff");
}

TEST_F(P2PConnectorSchedulerTest, KickoffExceptionRetainsLeaseAndCanCancelWithoutCallResults) {
    auto context = std::make_shared<P2PConnectorAsyncReadContext>(
        createValidKVCacheResource(1, 1), "kickoff-exception",
        std::make_shared<P2PConnectorMetricsCollector>(), 10000, false, currentTimeMs() + 10000);
    ASSERT_TRUE(context->beginKickoff());
    context->markKickoffUnconfirmed({ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED, "dispatch exception"});
    EXPECT_TRUE(context->done());
    EXPECT_FALSE(context->success());
    EXPECT_TRUE(context->resourceHoldPending());
    EXPECT_NO_THROW(context->checkDone());
    EXPECT_TRUE(context->needCancel());
    EXPECT_NO_THROW(context->cancel(tp_broadcast_client_));
    for (int i = 0; i < 1000 && context->needCancel(); ++i) {
        context->checkCancelDone();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_FALSE(context->needCancel());
    // A cancel ACK alone is insufficient to release a possibly targeted KV block.
    EXPECT_TRUE(context->resourceHoldPending());
    context->markKickoffUnconfirmed({ErrorCode::GENERATE_TIMEOUT, "later timeout"});
    EXPECT_NE(context->errorInfo().ToString().find("dispatch exception"), std::string::npos);
    for (const auto& server : tp_broadcast_servers_) {
        EXPECT_EQ(server->service()->getBroadcastTpCancelCallCount(), 1);
    }
}

TEST_F(P2PConnectorSchedulerTest, StartingKickoffCancelsRanksButCannotReleaseBeforeCallsArePublished) {
    auto resource = createValidKVCacheResource(1, 1);
    std::weak_ptr<KVCacheResource> weak_resource = resource;
    auto context = std::make_shared<P2PConnectorAsyncReadContext>(
        resource, "cancel-starting-before-publication",
        std::make_shared<P2PConnectorMetricsCollector>(), 10000, false,
        currentTimeMs() + 10000, currentTimeMs() - 1);
    resource.reset();
    ASSERT_TRUE(context->beginKickoff());
    ASSERT_TRUE(context->expireTransferDeadlineIfNeeded());
    ASSERT_TRUE(context->needCancel());
    context->cancel(tp_broadcast_client_);
    ASSERT_TRUE(waitUntil([&] {
        context->checkCancelDone();
        return !context->needCancel();
    }));
    for (const auto& server : tp_broadcast_servers_) {
        EXPECT_EQ(server->service()->getBroadcastTpCancelCallCount(), 1);
        server->service()->setLeaseStatus(true, 1, 1, true);
    }
    // Even an all-stopped response cannot release the target while a kickoff
    // operation may still publish a READ call.
    EXPECT_FALSE(context->needLeasePoll());
    context->pollLeaseIfNeeded(tp_broadcast_client_);
    EXPECT_TRUE(context->resourceHoldPending());
    EXPECT_FALSE(weak_resource.expired());

    auto read_result = std::make_shared<P2PBroadcastClient::Result>(context->uniqueKey());
    auto server_result = std::make_shared<DecodeLoadHelper::Result>();
    server_result->done_ = true;
    EXPECT_TRUE(context->setCallResults(read_result, server_result));
    context->cancel(tp_broadcast_client_);
    ASSERT_TRUE(waitUntil([&] {
        context->pollLeaseIfNeeded(tp_broadcast_client_);
        return !context->resourceHoldPending();
    }));
    EXPECT_TRUE(weak_resource.expired());
}

TEST_F(P2PConnectorSchedulerTest, AsyncReadPendingContextCanBeCancelledBeforeKickoff) {
    auto resource  = createValidKVCacheResource(1, 1);
    auto collector = std::make_shared<P2PConnectorMetricsCollector>();
    auto context   = std::make_shared<P2PConnectorAsyncReadContext>(
        resource, "cancel-before-kickoff", collector, /*lease_query_timeout_ms=*/0);

    context->cancel(nullptr);
    context->waitDone();

    EXPECT_TRUE(context->done());
    EXPECT_FALSE(context->success());
    EXPECT_TRUE(context->cancelRequested());
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::CANCELLED);
}

TEST_F(P2PConnectorSchedulerTest, AsyncReadStartingContextRetainsKvOnCancelBeforeCallsReady) {
    auto resource  = createValidKVCacheResource(1, 1);
    auto collector = std::make_shared<P2PConnectorMetricsCollector>();
    auto context   = std::make_shared<P2PConnectorAsyncReadContext>(
        resource, "cancel-during-kickoff", collector, /*lease_query_timeout_ms=*/0);

    ASSERT_TRUE(context->beginKickoff());
    context->cancel(nullptr);

    EXPECT_TRUE(context->cancelRequested());
    EXPECT_TRUE(context->done());
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::CANCELLED);
    EXPECT_TRUE(context->resourceHoldPending());

    context->markStartFailed(ErrorInfo(ErrorCode::CANCELLED, "kickoff cancelled before calls were created"));
    EXPECT_TRUE(context->done());
    EXPECT_FALSE(context->success());
    EXPECT_FALSE(context->resourceHoldPending());
}

// ==================== asyncRead 测试 (Decode 端功能) ====================
TEST_F(P2PConnectorSchedulerTest, AsyncRead_ReturnNotNull_AllSuccess) {
    auto resource = createValidKVCacheResource(2, 2);
    auto meta     = createMockMeta(2001, "test_async_read_1", currentTimeMs() + 5000);

    // block_range: {start_block_idx, block_count}, use -1 for block_count to include all blocks
    auto result = decode_scheduler_->asyncRead(resource, meta, {0, -1});
    ASSERT_TRUE(result.ok());
    auto async_context = result.context;
    ASSERT_NE(async_context, nullptr);

    waitAsyncContextDone(async_context);

    EXPECT_TRUE(async_context->done());
    EXPECT_TRUE(async_context->success());

    // 验证 BroadcastTp 和 StartLoad 都被调用
    for (size_t i = 0; i < tp_broadcast_servers_.size(); ++i) {
        EXPECT_EQ(tp_broadcast_servers_[i]->service()->getP2PRequestCallCount(P2PConnectorBroadcastType::READ), 1);
    }
    EXPECT_EQ(prefill_server_->service()->getStartLoadCallCount(), 1);
}

TEST_F(P2PConnectorSchedulerTest, AsyncReadCpSendsCompletePagesToEachDecodeWorker) {
    rebuildSchedulerWithLayerAttnTypes({CacheGroupType::FULL}, /*cp_size=*/2);

    auto resource = std::make_shared<KVCacheResource>();
    resource->initGroups(makeSingleMlaTopology());
    resource->mutableBlockIds(resource->groupTags().at(0)).assign({10, 11, 12, 13});
    resource->cacheKeys() = {100, 101, 102, 103};
    auto meta = createMockMeta(2012, "test_async_read_cp", currentTimeMs() + 5000);
    meta->setPrefillTpSize(2);
    meta->setPrefillCpSize(2);

    auto result = decode_scheduler_->asyncRead(resource, meta, {0, -1});
    ASSERT_TRUE(result.ok());
    ASSERT_NE(result.context, nullptr);
    waitAsyncContextDone(result.context);
    ASSERT_TRUE(result.context->success());

    std::set<int> all_routes;
    for (const auto& server : tp_broadcast_servers_) {
        const auto request = server->service()->getLastBroadcastTpRequest();
        std::map<int64_t, uint32_t> received;
        for (const auto& route : request.routes()) {
            all_routes.insert(route.route_id());
            for (const auto& layer : route.layer_blocks()) {
                ASSERT_EQ(layer.cache_keys_size(), layer.block_ids_size());
                for (int i = 0; i < layer.cache_keys_size(); ++i) {
                    EXPECT_TRUE(received.emplace(layer.cache_keys(i), layer.block_ids(i)).second);
                }
            }
        }
        EXPECT_EQ(received, (std::map<int64_t, uint32_t>{{100, 10}, {101, 11}, {102, 12}, {103, 13}}));
    }
    const auto request = prefill_server_->service()->getLastStartLoadRequest();
    EXPECT_EQ(std::set<int>(request.active_route_ids().begin(), request.active_route_ids().end()), all_routes);
}

TEST_F(P2PConnectorSchedulerTest, AsymmetricPipelineProjectsSharedBlocksAndExactPrefillLayers) {
    // Both Stage 0 allocators own both tags. Global layer 2 belongs to Prefill
    // Stage 0 and Decode Stage 1, where local layer 0 uses a different tag
    // from the allocator's own local layer 0.
    const std::vector<std::vector<int>> layer_groups{{0}, {1}, {1}, {0}, {1}, {0}};
    const auto                          make_layout = [&](const std::vector<int>& cuts, RoleType role) {
        P2PPipelineLayoutPB layout;
        layout.set_worker_count(cuts.size());
        int begin = 0;
        for (int count : cuts) {
            auto* stage = layout.add_stages();
            stage->set_main_layer_begin(begin);
            if (begin == 0) {
                stage->add_block_table_tags("group0");
                stage->add_block_table_tags("group1");
            }
            stage->set_main_layer_count(count);
            stage->add_worker_indices(layout.stages_size() - 1);
            const std::vector<std::vector<int>> groups(layer_groups.begin() + begin,
                                                       layer_groups.begin() + begin + count);
            const auto                          topology = test::makeTestCacheTopology(2, count, groups);
            encodeP2PShardLayout(ShardLayoutFactory::fromTopology(*topology, ParallelismConfig{}, role, 1),
                                 stage->mutable_main_shards());
            for (const auto& layer : topology->layers()) {
                auto* layer_pb = stage->add_layers();
                for (const auto& tag : layer.group_tags) {
                    layer_pb->add_tags(tag);
                }
            }
            begin += count;
        }
        return layout.SerializeAsString();
    };
    P2PConnectorSchedulerConfig decode_config;
    decode_config.parallelism_config.pp_size = 2;
    decode_config.topology                   = test::makeTestCacheTopology(2, 2, {{0}, {1}});
    decode_config.pipeline_layout            = make_layout({2, 4}, RoleType::DECODE);
    decode_config.worker_grpc_addrs          = tp_broadcast_addrs_;
    for (size_t i = 0; i < tp_broadcast_servers_.size(); ++i) {
        decode_config.worker_addrs.push_back("127.0.0.1:" + std::to_string(12345 + i) + ":"
                                             + std::to_string(tp_broadcast_servers_[i]->listenPort()));
    }
    rebuildSchedulers(decode_config);
    const auto prefill_layout = make_layout({3, 3}, RoleType::PREFILL);
    auto       resource       = createValidKVCacheResource();
    resource->mutableBlockIds("group0").assign({10, 11});
    resource->mutableBlockIds("group1").assign({20, 21});
    auto meta = createMockMeta(2013, "pp-shared-block-table", currentTimeMs() + 5000);
    meta->setPrefillPipelineLayout(prefill_layout);
    auto read = decode_scheduler_->asyncRead(resource, meta, {1, 1});
    ASSERT_TRUE(read.ok()) << read.error_info.ToString();
    ASSERT_NE(read.context, nullptr);
    waitAsyncContextDone(read.context);
    ASSERT_TRUE(read.context->success());

    const auto request = prefill_server_->service()->getLastStartLoadRequest();
    ASSERT_EQ(request.decode_pipeline_layout(), decode_config.pipeline_layout);
    P2PConnectorSchedulerConfig prefill_config;
    prefill_config.parallelism_config.pp_size = 2;
    prefill_config.worker_grpc_addrs          = tp_broadcast_addrs_;
    prefill_config.topology                   = test::makeTestCacheTopology(2, 3, {{0}, {1}, {1}});
    prefill_config.pipeline_layout            = prefill_layout;
    P2PSchedulerPrefillRead prefill(prefill_config, nullptr, nullptr);
    const auto prepared = prefill.preparePlan(1,
                                              request.plan_digest(),
                                              request.unique_key(),
                                              request.workers_size(),
                                              {request.active_route_ids().begin(), request.active_route_ids().end()},
                                              request.decode_pipeline_layout());
    ASSERT_TRUE(prepared.ok()) << prepared.status().ToString();
    const auto sends = prefill.buildPrefillRankRoutes(*prepared.value().plan, 2, prepared.value().active_route_ids);
    std::map<std::pair<int, int>, std::pair<int, int>> endpoints;
    for (int stage = 0; stage < 2; ++stage) {
        for (const auto& route : sends[stage]) {
            for (int local_layer : route.prefill_layer_ids()) {
                EXPECT_TRUE(endpoints
                                .emplace(std::make_pair(route.route_id(), 3 * stage + local_layer),
                                         std::make_pair(stage, route.peer_index()))
                                .second);
            }
        }
    }
    std::set<int> received_layers;
    for (int stage = 0; stage < 2; ++stage) {
        const auto rd = tp_broadcast_servers_[stage]->service()->getLastBroadcastTpRequest();
        EXPECT_EQ(rd.plan_digest(), request.plan_digest());
        for (const auto& route : rd.routes()) {
            for (const auto& layer : route.layer_blocks()) {
                const int  global_layer = 2 * stage + layer.layer_id();
                const auto sender       = endpoints.find({route.route_id(), global_layer});
                ASSERT_NE(sender, endpoints.end());
                EXPECT_EQ(sender->second, std::make_pair(global_layer / 3, stage));
                ASSERT_EQ(layer.cache_keys_size(), 1);
                EXPECT_EQ(layer.cache_keys(0), 1001);
                EXPECT_EQ(layer.block_ids(0), layer_groups.at(global_layer).front() == 0 ? 11 : 21);
                EXPECT_TRUE(received_layers.insert(global_layer).second);
            }
        }
    }
    EXPECT_EQ(received_layers, (std::set<int>{0, 1, 2, 3, 4, 5}));
    EXPECT_EQ(endpoints.size(), received_layers.size());
}

TEST_F(P2PConnectorSchedulerTest, PageRRFanoutCarriesOnlyDecodeMissingKeysToPrefill) {
    auto cfg                                                  = P2PConnectorSchedulerConfig{};
    cfg.topology                                              = test::makeTestCacheTopology(1, 2, {{0}, {0}});
    cfg.parallelism_config.tp_size                            = 8;
    cfg.parallelism_config.role_type                          = RoleType::DECODE;
    cfg.parallelism_config.prefill_cp_config.kv_cache_sharded = true;
    cfg.local_full_page_rr                                    = true;
    P2PSchedulerDecodeRead  decode(cfg, nullptr, nullptr);
    P2PSchedulerPrefillRead prefill(cfg, nullptr, nullptr);

    KVCacheResource resource;
    resource.initGroups(cfg.topology);
    for (int i = 0; i < 18; ++i) {
        resource.cacheKeys().push_back(100 + i);
    }
    resource.mutableBlockIds("group0").assign({10, 11, 12});

    TransferPlan plan;
    auto&        route        = plan.routes.emplace_back();
    route.route_id            = 42;
    route.cache_tag           = "group0";
    route.src_keys.modulus    = 8;
    route.src_keys.residue    = 1;
    route.dst_rank            = 1;
    route.dst_group_policy    = cfg.topology->group(route.cache_tag).policy;
    route.dst_physical_cp_size = 8;
    route.dst_physical_cp_rank = 1;
    route.filter_prefill_keys = true;
    route.layers              = {{0, 0, 0}, {1, 1, 1}};

    std::vector<PrefillRouteKeysPB> filters;
    const auto                      reads = decode.buildDecodeRankRoutes(plan, resource, {2, 16}, 8, filters);
    ASSERT_TRUE(reads.ok()) << reads.status().ToString();
    ASSERT_EQ(filters.size(), 1u);
    EXPECT_EQ(filters.front().route_id(), 42);
    const std::vector<int64_t> keys{filters.front().cache_keys().begin(), filters.front().cache_keys().end()};
    EXPECT_EQ(keys, (std::vector<int64_t>{109, 117}));
    ASSERT_EQ(reads.value().at(1).size(), 1u);
    ASSERT_EQ(reads.value().at(1).front().layer_blocks_size(), 2);
    const auto writes = prefill.buildPrefillRankRoutes(plan, 4, {42}, {{42, keys}});
    ASSERT_EQ(writes.at(0).size(), 1u);
    const auto& wire_keys = writes.at(0).front().prefill_cache_keys();
    EXPECT_EQ(std::vector<int64_t>(wire_keys.begin(), wire_keys.end()), keys);
}

TEST_F(P2PConnectorSchedulerTest, DecodeCompactStateProjectsUsingPrefillCpSize) {
    P2PConnectorSchedulerConfig cfg;
    cfg.topology                   = test::makeTestCacheTopology(1, 1, {{0}}, 4, {CacheGroupType::SWA});
    cfg.parallelism_config.tp_size = 1;
    cfg.parallelism_config.prefill_cp_config.kv_cache_sharded = true;
    cfg.parallelism_config.prefill_cp_config.prefill_cp_size = 4;
    cfg.parallelism_config.prefill_cp_config.method = CPRotateMethod::PREFILL_CP;
    P2PSchedulerDecodeRead scheduler(cfg, nullptr, nullptr);
    KVCacheResource resource;
    resource.initGroups(cfg.topology);
    resource.cacheKeys() = {100, 101, 102, 103, 104, 105, 106, 107, 108};
    resource.mutableBlockIds("group0").assign({10, 11, 12});
    TransferPlan plan;
    TransferRoute route;
    route.cache_tag = "group0";
    route.src_keys.modulus = 4;
    route.src_keys.residue = 3;
    route.src_keys.include_final_key = true;
    route.src_keys.tail_count = 2;
    plan.routes.push_back(route);
    std::vector<PrefillRouteKeysPB> filters;
    auto                            result = scheduler.buildDecodeRankRoutes(plan, resource, {0, -1}, 1, filters);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_TRUE(filters.empty());
    const auto& ranks = result.value();
    ASSERT_EQ(ranks.size(), 1u);
    ASSERT_EQ(ranks[0].size(), 1u);
    ASSERT_EQ(ranks[0][0].layer_blocks_size(), 1);
    const auto& layer = ranks[0][0].layer_blocks(0);
    ASSERT_EQ(layer.cache_keys_size(), 2);
    EXPECT_EQ(layer.cache_keys(0), 107);
    EXPECT_EQ(layer.cache_keys(1), 108);
    EXPECT_EQ(layer.block_ids(0), 11);
    EXPECT_EQ(layer.block_ids(1), 12);
}

TEST_F(P2PConnectorSchedulerTest, DecodeRouteUsesDraftModuleBlocksFromTagOwnership) {
    auto cache_config =
        test::makeSimpleMhaCacheConfig(/*layer_num=*/1, /*block_num=*/16, /*tokens_per_block=*/1, DataType::TYPE_FP16);
    auto draft_config = test::makeSimpleMhaCacheConfig(
        /*layer_num=*/1, /*block_num=*/16, /*tokens_per_block=*/1, DataType::TYPE_FP16);
    auto draft = test::declareModuleScopedTags(draft_config.topology());
    ::rtp_llm::test::CacheConfigTestAccess::addMtpModule(
        cache_config, std::move(draft), draft_config.moduleMetadata(0));
    ASSERT_EQ(cache_config.moduleCount(), 2u);
    const auto main_tag  = cache_config.topology(0).groupTags().front();
    const auto draft_tag = cache_config.topology(1).groupTags().front();
    ASSERT_NE(main_tag, draft_tag);

    RuntimeConfig     runtime_config;
    CacheStoreConfig  cache_store_config;
    ParallelismConfig parallelism_config;
    parallelism_config.tp_size = 1;
    PDSepConfig pd_sep_config;
    pd_sep_config.role_type = RoleType::DECODE;
    auto p2p_config                = P2PConnectorConfig::create(runtime_config,
                                                 cache_store_config,
                                                 parallelism_config,
                                                 pd_sep_config,
                                                 cache_config.layer_num(),
                                                 /*llm_kv_block_size_bytes=*/0,
                                                 cache_config);
    auto cfg                       = p2p_config.scheduler_config;
    cfg.parallelism_config.tp_size = 1;
    P2PSchedulerDecodeRead scheduler(cfg, nullptr, nullptr);

    KVCacheResource resource;
    resource.initGroups(cache_config);
    resource.cacheKeys() = {100};
    resource.mutableBlockIds(main_tag).assign({10});
    resource.mutableBlockIds(draft_tag).assign({20});

    TransferPlan plan;
    for (const auto& tag : {main_tag, draft_tag}) {
        TransferRoute route;
        route.route_id  = static_cast<int>(plan.routes.size());
        route.cache_tag = tag;
        plan.routes.push_back(std::move(route));
    }
    std::vector<PrefillRouteKeysPB> filters;
    auto result = scheduler.buildDecodeRankRoutes(plan, resource, {0, -1}, 1, filters);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    const auto& ranks = result.value();
    ASSERT_EQ(ranks.size(), 1u);
    ASSERT_EQ(ranks[0].size(), 2u);

    std::map<std::string, int> block_by_tag;
    for (const auto& route : ranks[0]) {
        ASSERT_EQ(route.layer_blocks_size(), 1);
        const auto& layer = route.layer_blocks(0);
        ASSERT_EQ(layer.cache_keys_size(), 1);
        ASSERT_EQ(layer.block_ids_size(), 1);
        block_by_tag.emplace(layer.cache_tag(), layer.block_ids(0));
    }
    EXPECT_EQ(block_by_tag, (std::map<std::string, int>{{main_tag, 10}, {draft_tag, 20}}));
}

TEST(P2PDraftRouteTest, LastStageDraftReadsIndependentRootRowAcrossDifferentStageCuts) {
    const auto main         = test::makeSimpleMhaCacheConfig(2, 16, 1, DataType::TYPE_FP16);
    auto       draft_config = test::makeSimpleMhaCacheConfig(2, 16, 1, DataType::TYPE_FP16);
    const auto draft        = test::declareModuleScopedTags(draft_config.topology());
    const auto& main_tag  = main.groupTags().front();
    const auto& draft_tag = draft->groupTags().front();

    const auto layout_for = [&](int first_count, RoleType role, bool allocate_draft) {
        P2PPipelineLayoutPB pb;
        pb.set_worker_count(2);
        for (int i = 0; i < 2; ++i) {
            auto* stage = pb.add_stages();
            const int count = i == 0 ? first_count : 4 - first_count;
            stage->set_main_layer_count(count);
            stage->set_main_layer_begin(i == 0 ? 0 : first_count);
            stage->add_worker_indices(i);
            encodeP2PShardLayout(ShardLayoutFactory::fromTopology(main.topology(), ParallelismConfig{}, role, 1),
                                 stage->mutable_main_shards());
            for (int layer = 0; layer < count; ++layer) {
                stage->add_layers()->add_tags(main_tag);
            }
            if (i == 0) {
                stage->add_block_table_tags(main_tag);
                if (allocate_draft) {
                    stage->add_block_table_tags(draft_tag);
                }
            } else {
                for (uint32_t layer = 0; layer < draft_config.layer_num(); ++layer) {
                    stage->add_layers()->add_tags(draft_tag);
                }
                encodeP2PShardLayout(ShardLayoutFactory::fromTopology(*draft, ParallelismConfig{}, role, 1),
                                     stage->mutable_draft_shards());
            }
        }
        return pb.SerializeAsString();
    };

    const auto prefill_layout = layout_for(1, RoleType::PREFILL, true);
    const auto decode_layout  = layout_for(2, RoleType::DECODE, true);
    const auto missing_row = planP2PPipelineTransfer(prefill_layout, layout_for(2, RoleType::DECODE, false), "draft");
    ASSERT_FALSE(missing_row.ok());
    EXPECT_NE(missing_row.error.ToString().find(draft_tag), std::string::npos);

    auto root = main;
    ::rtp_llm::test::CacheConfigTestAccess::addMtpModule(
        root, CacheTopology::create(draft->groups(), {}), draft_config.moduleMetadata(0));
    const auto config_for = [](const CacheConfig& cfg, RoleType role) {
        PDSepConfig pd;
        pd.role_type = role;
        return P2PConnectorConfig::create(
            RuntimeConfig{}, CacheStoreConfig{}, ParallelismConfig{}, pd, cfg.layer_num(), 0, cfg);
    };
    const auto root_config = config_for(root, RoleType::DECODE);
    EXPECT_EQ(root_config.worker_config.topology->groupTags(), main.groupTags());
    EXPECT_EQ(root_config.worker_config.module_layers.stage_offsets.at(draft_tag), main.layer_num());
    EXPECT_NE(root_config.worker_config.wireLayerId(draft_tag, 0), root_config.worker_config.wireLayerId(main_tag, 0));
    KVCacheResource resource;
    resource.initGroups(root);
    resource.cacheKeys() = {101};
    resource.mutableBlockIds(main_tag).assign({11});
    resource.mutableBlockIds(draft_tag).assign({27});

    const auto plan = planP2PPipelineTransfer(prefill_layout, decode_layout, "draft");
    ASSERT_TRUE(plan.ok()) << plan.error.ToString();
    const auto draft_routes = plan.plan.forTag(draft_tag);
    ASSERT_EQ(draft_routes.size(), 1u);
    ASSERT_EQ(draft_routes.front()->layers.size(), 2u);
    const auto& mapping = draft_routes.front()->layers.front();
    EXPECT_EQ(mapping.global_id, 4);
    EXPECT_EQ(mapping.src_stage_layer_id, 3);
    EXPECT_EQ(mapping.dst_stage_layer_id, 2);
    const auto& next_mapping = draft_routes.front()->layers.back();
    EXPECT_EQ(next_mapping.global_id, 5);
    EXPECT_EQ(next_mapping.src_stage_layer_id, 4);
    EXPECT_EQ(next_mapping.dst_stage_layer_id, 3);

    P2PSchedulerDecodeRead scheduler(root_config.scheduler_config, nullptr, nullptr);
    std::vector<PrefillRouteKeysPB> filters;
    const auto projected = scheduler.buildDecodeRankRoutes(plan.plan, resource, {0, -1}, 2, filters);
    ASSERT_TRUE(projected.ok()) << projected.status().ToString();
    size_t draft_buffers = 0;
    for (size_t rank = 0; rank < projected.value().size(); ++rank) {
        for (const auto& route : projected.value()[rank]) {
            for (const auto& layer : route.layer_blocks()) {
                ASSERT_EQ(layer.block_ids_size(), 1);
                if (route.cache_tag() == draft_tag) {
                    ++draft_buffers;
                    EXPECT_EQ(rank, 1u);
                    EXPECT_EQ(layer.layer_id(), 1u + draft_buffers);
                    EXPECT_EQ(layer.block_ids(0), 27);
                } else {
                    EXPECT_EQ(layer.block_ids(0), 11);
                }
            }
        }
    }
    EXPECT_EQ(draft_buffers, 2u);

    auto prefill = test::makeSimpleMhaCacheConfig(3, 16, 1, DataType::TYPE_FP16);
    ::rtp_llm::test::CacheConfigTestAccess::addMtpModule(prefill, draft, draft_config.moduleMetadata(0));
    KVCacheResource source_resource;
    source_resource.initGroups(prefill);
    source_resource.cacheKeys() = {101};
    source_resource.mutableBlockIds(main_tag).assign({15});
    source_resource.mutableBlockIds(draft_tag).assign({31});
    auto published = LayerCacheBufferUtil::convertLayerTag(
        source_resource, draft->group(draft_tag), 0, 0, -1, 0, 1, 1, /*module_id=*/1);
    ASSERT_TRUE(published.ok()) << published.status().ToString();
    ASSERT_NE(published.value(), nullptr);
    EXPECT_EQ(published.value()->moduleId(), 1u);
    EXPECT_EQ(published.value()->getBlockId(101), 31);

    auto decode = main;
    ::rtp_llm::test::CacheConfigTestAccess::addMtpModule(decode, draft, draft_config.moduleMetadata(0));
    auto sender = config_for(prefill, RoleType::PREFILL).worker_config;
    auto receiver = config_for(decode, RoleType::DECODE).worker_config;
    sender.global_layer_begin = 1;
    receiver.global_layer_begin = 2;
    EXPECT_EQ(sender.wireLayerId(draft_tag, 0), 4u);
    EXPECT_EQ(receiver.wireLayerId(draft_tag, 0), 4u);
    EXPECT_EQ(sender.wireLayerId(draft_tag, 1), 5u);
    EXPECT_EQ(receiver.wireLayerId(draft_tag, 1), 5u);
    EXPECT_EQ(sender.wireLayerId(main_tag, 0), 1u);
    EXPECT_EQ(receiver.wireLayerId(main_tag, 0), 2u);
}

TEST_F(P2PConnectorSchedulerTest, AsyncReadCpAcceptsDifferentAdvertisedSourceCpSize) {
    rebuildSchedulerWithLayerAttnTypes({CacheGroupType::FULL}, /*cp_size=*/2);

    auto resource = std::make_shared<KVCacheResource>();
    resource->initGroups(makeSingleMlaTopology());
    resource->mutableBlockIds(resource->groupTags().at(0)).assign({10, 11});
    resource->cacheKeys() = {100, 101};
    auto meta = createMockMeta(2013, "test_async_read_cp_mismatch", currentTimeMs() + 5000);
    // TP happens to match Decode, but effective KV-cache CP does not.
    meta->setPrefillTpSize(2);
    meta->setPrefillCpSize(1);

    auto result = decode_scheduler_->asyncRead(resource, meta, {0, -1});
    ASSERT_TRUE(result.ok()) << result.error_info.ToString();
    ASSERT_NE(result.context, nullptr);
    waitAsyncContextDone(result.context);
    EXPECT_TRUE(result.context->success());
    EXPECT_EQ(prefill_server_->service()->getStartLoadCallCount(), 1);
}

TEST_F(P2PConnectorSchedulerTest, AsyncRead_NoTransferCompletesWithoutDecodeBuffers) {
    auto resource = createValidKVCacheResource(2, 2);
    auto meta     = createMockMeta(2011, "test_async_read_no_transfer", currentTimeMs() + 5000);

    auto result = decode_scheduler_->asyncRead(resource, meta, {2, 0}, /*no_transfer=*/true);
    ASSERT_TRUE(result.ok());
    ASSERT_NE(result.context, nullptr);

    waitAsyncContextDone(result.context);

    EXPECT_TRUE(result.context->done());
    EXPECT_TRUE(result.context->success());
    ASSERT_EQ(prefill_server_->service()->getStartLoadCallCount(), 1);
    EXPECT_TRUE(prefill_server_->service()->getLastStartLoadRequest().no_transfer());
    for (const auto& server : tp_broadcast_servers_) {
        EXPECT_EQ(server->service()->getBroadcastTpCallCount(), 0);
    }
}

TEST_F(P2PConnectorSchedulerTest, AsyncRead_NoDecodeComputeOverridesRequestedTransfer) {
    auto resource = std::make_shared<KVCacheResource>();
    auto meta     = createMockMeta(2011, "test_async_read_no_decode", currentTimeMs() + 5000);

    meta->setDecodeNeedsCompute(false);
    auto result = decode_scheduler_->asyncRead(resource, meta, {0, 2}, /*no_transfer=*/false);
    ASSERT_TRUE(result.ok());
    ASSERT_NE(result.context, nullptr);

    waitAsyncContextDone(result.context);

    EXPECT_TRUE(result.context->done());
    EXPECT_TRUE(result.context->success());
    ASSERT_EQ(prefill_server_->service()->getStartLoadCallCount(), 1);
    EXPECT_TRUE(prefill_server_->service()->getLastStartLoadRequest().no_transfer());
    for (const auto& server : tp_broadcast_servers_) {
        EXPECT_EQ(server->service()->getBroadcastTpCallCount(), 0);
    }
}

// 验证 P2PConnectorAsyncReadContext::waitDone() 在 checkDone() 置 done 后由 condition_variable 唤醒
TEST_F(P2PConnectorSchedulerTest, AsyncRead_WaitDone_UnblocksWhenCheckDoneCompletes) {
    auto resource = createValidKVCacheResource(2, 2);
    auto meta     = createMockMeta(2010, "test_async_read_wait_done", currentTimeMs() + 5000);

    auto result = decode_scheduler_->asyncRead(resource, meta, {0, -1});
    ASSERT_TRUE(result.ok());
    auto async_context = result.context;
    ASSERT_NE(async_context, nullptr);

    std::atomic<bool> wait_done_thread_finished{false};
    std::thread       wait_thread([&]() {
        async_context->waitDone();
        wait_done_thread_finished.store(true);
    });

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!async_context->done() && std::chrono::steady_clock::now() < deadline) {
        // The background checker must wake from completion, with no test-side progress.
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    ASSERT_TRUE(async_context->done()) << "async read should complete within timeout";
    wait_thread.join();
    EXPECT_TRUE(wait_done_thread_finished.load());
    EXPECT_TRUE(async_context->success());

    for (size_t i = 0; i < tp_broadcast_servers_.size(); ++i) {
        EXPECT_GE(tp_broadcast_servers_[i]->service()->getP2PRequestCallCount(P2PConnectorBroadcastType::READ), 1);
    }
    EXPECT_EQ(prefill_server_->service()->getStartLoadCallCount(), 1);
}

TEST_F(P2PConnectorSchedulerTest, AsyncRead_ReturnNull_NullResource) {
    auto meta = createMockMeta(2002, "test_async_read_null_resource", currentTimeMs() + 5000);

    auto result = decode_scheduler_->asyncRead(nullptr, meta, {0, -1});

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.context, nullptr);

    // 验证 BroadcastTp 和 StartLoad 都没有被调用
    for (size_t i = 0; i < tp_broadcast_servers_.size(); ++i) {
        EXPECT_EQ(tp_broadcast_servers_[i]->service()->getBroadcastTpCallCount(), 0);
    }
    EXPECT_EQ(prefill_server_->service()->getStartLoadCallCount(), 0);
}

TEST_F(P2PConnectorSchedulerTest, AsyncRead_ReturnNull_EmptyResource) {
    auto resource = std::make_shared<KVCacheResource>();
    auto meta     = createMockMeta(2003, "test_async_read_empty", currentTimeMs() + 5000);

    auto result = decode_scheduler_->asyncRead(resource, meta, {0, -1});

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.context, nullptr);

    // 验证 BroadcastTp 和 StartLoad 都没有被调用
    for (size_t i = 0; i < tp_broadcast_servers_.size(); ++i) {
        EXPECT_EQ(tp_broadcast_servers_[i]->service()->getBroadcastTpCallCount(), 0);
    }
    EXPECT_EQ(prefill_server_->service()->getStartLoadCallCount(), 0);
}

TEST_F(P2PConnectorSchedulerTest, AsyncRead_ReturnFalse_BroadcastFailed) {
    tp_broadcast_servers_[0]->service()->setP2PResponseSuccess(false);

    auto resource = createValidKVCacheResource(2, 2);
    auto meta     = createMockMeta(2004, "test_async_read_broadcast_fail", currentTimeMs() + 5000);

    auto result = decode_scheduler_->asyncRead(resource, meta, {0, -1});
    ASSERT_TRUE(result.ok());
    auto async_context = result.context;
    ASSERT_NE(async_context, nullptr);

    waitAsyncContextDone(async_context);

    EXPECT_TRUE(async_context->done());
    EXPECT_FALSE(async_context->success());

    // 验证 BroadcastTp 和 StartLoad 都被调用
    for (size_t i = 0; i < tp_broadcast_servers_.size(); ++i) {
        EXPECT_EQ(tp_broadcast_servers_[i]->service()->getP2PRequestCallCount(P2PConnectorBroadcastType::READ), 1);
    }
    EXPECT_EQ(prefill_server_->service()->getStartLoadCallCount(), 1);
}

TEST_F(P2PConnectorSchedulerTest, AsyncRead_ReturnFalse_LoadFailed) {
    prefill_server_->service()->setStartLoadResponseSuccess(false);

    auto resource = createValidKVCacheResource(2, 2);
    auto meta     = createMockMeta(2005, "test_async_read_load_fail", currentTimeMs() + 5000);

    auto result = decode_scheduler_->asyncRead(resource, meta, {0, -1});
    ASSERT_TRUE(result.ok());
    auto async_context = result.context;
    ASSERT_NE(async_context, nullptr);

    waitAsyncContextDone(async_context);

    EXPECT_TRUE(async_context->done());
    EXPECT_FALSE(async_context->success());

    // 验证 BroadcastTp 和 StartLoad 都被调用
    for (size_t i = 0; i < tp_broadcast_servers_.size(); ++i) {
        EXPECT_EQ(tp_broadcast_servers_[i]->service()->getP2PRequestCallCount(P2PConnectorBroadcastType::READ), 1);
    }
    EXPECT_EQ(prefill_server_->service()->getStartLoadCallCount(), 1);
}

TEST_F(P2PConnectorSchedulerTest, AsyncRead_ReturnFalse_BothFailed) {
    tp_broadcast_servers_[0]->service()->setP2PResponseSuccess(false);
    prefill_server_->service()->setStartLoadResponseSuccess(false);

    auto resource = createValidKVCacheResource(2, 2);
    auto meta     = createMockMeta(2006, "test_async_read_both_fail", currentTimeMs() + 5000);

    auto result = decode_scheduler_->asyncRead(resource, meta, {0, -1});
    ASSERT_TRUE(result.ok());
    auto async_context = result.context;
    ASSERT_NE(async_context, nullptr);

    waitAsyncContextDone(async_context);

    EXPECT_TRUE(async_context->done());
    EXPECT_FALSE(async_context->success());
    EXPECT_TRUE(async_context->resourceHoldPending());
}

// 测试: prefill server 超时, 返回失败
TEST_F(P2PConnectorSchedulerTest, AsyncRead_ReturnFalse_PrefillTimeout) {
    // 设置 prefill server 延迟响应
    prefill_server_->service()->setSleepMillis(500);

    auto resource = createValidKVCacheResource(2, 2);
    auto meta     = createMockMeta(2007, "test_async_read_prefill_timeout", currentTimeMs() + 50);

    auto result = decode_scheduler_->asyncRead(resource, meta, {0, -1});
    ASSERT_TRUE(result.ok());
    auto async_context = result.context;
    ASSERT_NE(async_context, nullptr);

    waitAsyncContextDone(async_context);

    EXPECT_TRUE(async_context->done());
    EXPECT_FALSE(async_context->success());  // prefill server 超时导致失败

    // 验证 StartLoad 被调用
    EXPECT_EQ(prefill_server_->service()->getStartLoadCallCount(), 1);
}

// 测试: broadcast worker 慢于 gRPC deadline，checkDone 标记失败
TEST_F(P2PConnectorSchedulerTest, AsyncRead_ReturnFalse_BroadcastTimeout) {
    tp_broadcast_servers_[0]->service()->setSleepMillis(500);

    decode_scheduler_->stopChecker();

    auto resource = createValidKVCacheResource(2, 2);
    auto meta     = createMockMeta(2008, "test_async_read_broadcast_timeout", currentTimeMs() + 50);

    auto result = decode_scheduler_->asyncRead(resource, meta, {0, -1});
    ASSERT_TRUE(result.ok());
    auto async_context = result.context;
    ASSERT_NE(async_context, nullptr);

    waitAsyncContextDone(async_context, 5000, /*check_done=*/true);

    EXPECT_TRUE(async_context->done());
    EXPECT_FALSE(async_context->success());
    EXPECT_TRUE(async_context->resourceHoldPending());
}

// 测试: asyncread prefill 失败, 取消broadcast
TEST_F(P2PConnectorSchedulerTest, AsyncRead_CancelBroadcast_WhenPrefillFailed) {
    // 设置 prefill server 立即返回失败
    prefill_server_->service()->setStartLoadResponseSuccess(false);

    // 设置 broadcast server 延迟响应，确保 prefill 先完成
    for (auto& server : tp_broadcast_servers_) {
        server->service()->setSleepMillis(200);
    }

    auto resource = createValidKVCacheResource(2, 2);
    auto meta     = createMockMeta(3001, "test_cancel_broadcast_when_prefill_failed", currentTimeMs() + 5000);

    auto result = decode_scheduler_->asyncRead(resource, meta, {0, -1});
    ASSERT_TRUE(result.ok());
    auto async_context = result.context;
    ASSERT_NE(async_context, nullptr);

    waitAsyncContextDone(async_context);

    EXPECT_TRUE(async_context->done());
    EXPECT_FALSE(async_context->success());

    // 验证 StartLoad 被调用
    EXPECT_EQ(prefill_server_->service()->getStartLoadCallCount(), 1);

    // 验证 BroadcastTp 被调用
    for (size_t i = 0; i < tp_broadcast_servers_.size(); ++i) {
        EXPECT_EQ(tp_broadcast_servers_[i]->service()->getP2PRequestCallCount(P2PConnectorBroadcastType::READ), 1);
    }

    // 验证 CANCEL_READ 被发送给所有 worker（因为 prefill 失败，需要取消 broadcast）
    for (size_t i = 0; i < tp_broadcast_servers_.size(); ++i) {
        const auto cancel_check_deadline = currentTimeMs() + 1000;
        while (tp_broadcast_servers_[i]->service()->getBroadcastTpCancelCallCount() == 0
               && currentTimeMs() < cancel_check_deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        EXPECT_EQ(tp_broadcast_servers_[i]->service()->getBroadcastTpCancelCallCount(), 1);
    }
}

// 测试: asyncread broadcast 失败, 取消prefill
TEST_F(P2PConnectorSchedulerTest, AsyncRead_CancelPrefill_WhenBroadcastFailed) {
    // 设置 broadcast server 立即返回失败
    for (auto& server : tp_broadcast_servers_) {
        server->service()->setP2PResponseSuccess(false);
    }

    // 设置 prefill server 延迟响应，确保 broadcast 先完成
    prefill_server_->service()->setSleepMillis(200);

    auto resource = createValidKVCacheResource(2, 2);
    auto meta     = createMockMeta(3002, "test_cancel_prefill_when_broadcast_failed", currentTimeMs() + 5000);

    auto result = decode_scheduler_->asyncRead(resource, meta, {0, -1});
    ASSERT_TRUE(result.ok());
    auto async_context = result.context;
    ASSERT_NE(async_context, nullptr);

    waitAsyncContextDone(async_context);

    EXPECT_TRUE(async_context->done());
    EXPECT_FALSE(async_context->success());

    // 验证 BroadcastTp 被调用
    for (size_t i = 0; i < tp_broadcast_servers_.size(); ++i) {
        EXPECT_EQ(tp_broadcast_servers_[i]->service()->getP2PRequestCallCount(P2PConnectorBroadcastType::READ), 1);
    }

    // 验证 StartLoad 被调用（可能被取消）
    EXPECT_EQ(prefill_server_->service()->getStartLoadCallCount(), 1);

    // 注意：prefill 请求会被取消，但由于 grpc TryCancel 的实现，
    // 服务端可能已经开始处理请求，所以这里不验证取消是否成功
}

// Prefill：worker 极慢导致超过 deadline，返回超时错误
TEST_F(P2PConnectorSchedulerTest, SendKVCache_ReturnError_WhenBroadcastExceedsDeadline) {
    for (auto& server : tp_broadcast_servers_) {
        server->service()->setSleepMillis(200);
        server->service()->setP2PResponseSuccess(true);
    }

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});

    const int64_t deadline_ms = currentTimeMs() + 80;
    ErrorInfo     error_info  = sendPrefillKV(
        "test_prefill_broadcast_past_deadline", 4006, decode_transfer_servers, deadline_ms, nullptr, deadline_ms);

    EXPECT_TRUE(error_info.hasError());
    EXPECT_EQ(error_info.code(), ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_TIMEOUT);
}

// StartLoad 返回 TRANSFER_NOT_DONE：请求立即失败完成，
// 但 checker 继续持有 Decode 目标资源并轮询 lease。
TEST_F(P2PConnectorSchedulerTest, AsyncRead_TransferNotDone_CompletesRequestAndRetainsResource) {
    rebuildSchedulerWithLeaseQueryTimeoutMs(120);

    prefill_server_->service()->setStartLoadApplicationError(ErrorCodePB::P2P_CONNECTOR_WORKER_READ_TRANSFER_NOT_DONE,
                                                             "test transfer not done");
    for (auto& server : tp_broadcast_servers_) {
        server->service()->setP2PResponseSuccess(true);
        server->service()->setSleepMillis(0);
    }

    decode_scheduler_->stopChecker();

    auto resource = createValidKVCacheResource(2, 2);
    auto meta     = createMockMeta(5010, "test_transfer_not_done_hold", currentTimeMs() + 5000);

    auto result = decode_scheduler_->asyncRead(resource, meta, {0, -1});
    ASSERT_TRUE(result.ok());
    auto async_context = result.context;
    ASSERT_NE(async_context, nullptr);

    for (int i = 0; i < 3000 && !async_context->done(); ++i) {
        async_context->checkDone();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    EXPECT_TRUE(async_context->done());
    EXPECT_FALSE(async_context->success());
    EXPECT_EQ(async_context->errorInfo().code(), ErrorCode::P2P_CONNECTOR_WORKER_READ_TRANSFER_NOT_DONE);
    EXPECT_TRUE(async_context->resourceHoldPending());
    EXPECT_TRUE(async_context->needCancel());
    EXPECT_FALSE(async_context->needLeasePoll());

    async_context->cancel(tp_broadcast_client_);
    for (int i = 0; i < 1000 && async_context->needCancel(); ++i) {
        async_context->checkCancelDone();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_FALSE(async_context->needCancel());

    for (size_t i = 0; i < tp_broadcast_servers_.size(); ++i) {
        const auto cancel_check_deadline = currentTimeMs() + 1000;
        while (tp_broadcast_servers_[i]->service()->getBroadcastTpCancelCallCount() == 0
               && currentTimeMs() < cancel_check_deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        EXPECT_EQ(tp_broadcast_servers_[i]->service()->getBroadcastTpCancelCallCount(), 1);
    }
}

// Invalid zero timeout is clamped so it cannot disable target-KV protection.
TEST_F(P2PConnectorSchedulerTest, AsyncRead_TransferNotDone_ZeroTimeoutStillRetainsResource) {
    rebuildSchedulerWithLeaseQueryTimeoutMs(0);

    prefill_server_->service()->setStartLoadApplicationError(ErrorCodePB::P2P_CONNECTOR_WORKER_READ_TRANSFER_NOT_DONE,
                                                             "test transfer not done");
    for (auto& server : tp_broadcast_servers_) {
        server->service()->setP2PResponseSuccess(true);
        server->service()->setSleepMillis(0);
    }

    decode_scheduler_->stopChecker();

    auto resource = createValidKVCacheResource(2, 2);
    auto meta     = createMockMeta(5011, "test_transfer_not_done_zero_hold", currentTimeMs() + 5000);

    auto result = decode_scheduler_->asyncRead(resource, meta, {0, -1});
    ASSERT_TRUE(result.ok());
    auto async_context = result.context;
    ASSERT_NE(async_context, nullptr);

    for (int i = 0; i < 500 && !async_context->done(); ++i) {
        async_context->checkDone();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    ASSERT_TRUE(async_context->done());
    EXPECT_FALSE(async_context->success());
    EXPECT_EQ(async_context->errorInfo().code(), ErrorCode::P2P_CONNECTOR_WORKER_READ_TRANSFER_NOT_DONE);
    EXPECT_TRUE(async_context->resourceHoldPending());
}

TEST_F(P2PConnectorSchedulerTest, AsyncRead_StartLoadTimeout_HoldsTargetUntilAllRanksStop) {
    auto allocated = createAllocatedConnectorResource();
    rebuildSchedulerForDeadlineTest(allocated.config.topologyPtr());
    prefill_server_->service()->setStartLoadSleepMillis(500);
    for (auto& server : tp_broadcast_servers_) {
        // Keep READ physically unconfirmed when StartLoad expires; a successful READ already proves completion.
        server->service()->setP2PRequestSleepMillis(P2PConnectorBroadcastType::READ, 500);
        server->service()->setLeaseStatus(true, 1, 0, false);
    }

    auto result = decode_scheduler_->asyncRead(
        allocated.resource, createMockMeta(5100, "start_load_timeout_hold", currentTimeMs() + 150), {0, -1});
    allocated.resource.reset();
    ASSERT_TRUE(result.ok());
    auto context = result.context;
    ASSERT_NE(context, nullptr);
    ASSERT_TRUE(waitUntil([&]() { return context->done(); }));
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::GENERATE_TIMEOUT);
    ASSERT_TRUE(waitUntil([&]() {
        return tp_broadcast_servers_[0]->service()->getP2PRequestCallCount(
                   P2PConnectorBroadcastType::QUERY_LEASE_STATUS)
               > 0;
    }));
    EXPECT_EQ(allocated.allocator->freeBlocksNum(), allocated.free_blocks_before - allocated.held_blocks);
    EXPECT_TRUE(allBlocksHeldByLease(allocated));

    std::weak_ptr<P2PConnectorAsyncReadContext> weak_context = context;
    context.reset();
    result.context.reset();
    EXPECT_FALSE(weak_context.expired());
    for (auto& server : tp_broadcast_servers_) {
        server->service()->setLeaseStatus(true, 1, 1, true);
    }
    EXPECT_TRUE(waitUntil([&]() { return allocated.allocator->freeBlocksNum() == allocated.free_blocks_before; }));
    EXPECT_TRUE(allBlocksReclaimed(allocated));
    // Releasing blocks precedes the checker's final shared_ptr destruction.
    ASSERT_TRUE(waitUntil([&]() { return weak_context.expired(); }));
}

TEST_F(P2PConnectorSchedulerTest, AsyncRead_ReadPerRankTimeout_WaitsForEveryRankLease) {
    auto allocated = createAllocatedConnectorResource();
    rebuildSchedulerForDeadlineTest(allocated.config.topologyPtr());
    for (auto& server : tp_broadcast_servers_) {
        server->service()->setP2PRequestSleepMillis(P2PConnectorBroadcastType::READ, 500);
        server->service()->setLeaseStatus(true, 1, 0, false);
    }

    auto result = decode_scheduler_->asyncRead(
        allocated.resource, createMockMeta(5101, "read_per_rank_timeout_hold", currentTimeMs() + 150), {0, -1});
    allocated.resource.reset();
    ASSERT_TRUE(result.ok());
    auto context = result.context;
    ASSERT_NE(context, nullptr);
    ASSERT_TRUE(waitUntil([&]() { return context->done(); }));
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::GENERATE_TIMEOUT);
    ASSERT_TRUE(waitUntil([&]() {
        return tp_broadcast_servers_[1]->service()->getP2PRequestCallCount(
                   P2PConnectorBroadcastType::QUERY_LEASE_STATUS)
               > 0;
    }));

    tp_broadcast_servers_[0]->service()->setLeaseStatus(true, 1, 1, true);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_TRUE(context->resourceHoldPending());
    EXPECT_EQ(allocated.allocator->freeBlocksNum(), allocated.free_blocks_before - allocated.held_blocks);
    EXPECT_TRUE(allBlocksHeldByLease(allocated));

    tp_broadcast_servers_[1]->service()->setLeaseStatus(true, 1, 1, true);
    ASSERT_TRUE(waitUntil([&]() { return !context->resourceHoldPending(); }));
    context.reset();
    result.context.reset();
    EXPECT_TRUE(waitUntil([&]() { return allocated.allocator->freeBlocksNum() == allocated.free_blocks_before; }));
    EXPECT_TRUE(allBlocksReclaimed(allocated));
}

TEST_F(P2PConnectorSchedulerTest, AsyncRead_CancelTimeout_DoesNotQueryOrReleaseTargetEarly) {
    auto allocated = createAllocatedConnectorResource();
    rebuildSchedulerForDeadlineTest(allocated.config.topologyPtr());
    for (auto& server : tp_broadcast_servers_) {
        server->service()->setP2PRequestSleepMillis(P2PConnectorBroadcastType::READ, 500);
        server->service()->setP2PRequestSleepMillis(P2PConnectorBroadcastType::CANCEL_READ, 200);
        server->service()->setLeaseStatus(true, 1, 1, true);
    }

    auto result = decode_scheduler_->asyncRead(
        allocated.resource, createMockMeta(5102, "cancel_timeout_hold", currentTimeMs() + 150), {0, -1});
    allocated.resource.reset();
    ASSERT_TRUE(result.ok());
    auto context = result.context;
    ASSERT_NE(context, nullptr);
    ASSERT_TRUE(waitUntil([&]() { return context->done(); }));
    ASSERT_TRUE(waitUntil([&]() {
        return tp_broadcast_servers_[0]->service()->getP2PRequestCallCount(P2PConnectorBroadcastType::CANCEL_READ)
               > 0;
    }));
    EXPECT_EQ(tp_broadcast_servers_[0]->service()->getP2PRequestCallCount(
                  P2PConnectorBroadcastType::QUERY_LEASE_STATUS),
              0);
    EXPECT_EQ(allocated.allocator->freeBlocksNum(), allocated.free_blocks_before - allocated.held_blocks);
    EXPECT_TRUE(allBlocksHeldByLease(allocated));

    for (auto& server : tp_broadcast_servers_) {
        server->service()->setP2PRequestSleepMillis(P2PConnectorBroadcastType::CANCEL_READ, 0);
    }
    ASSERT_TRUE(waitUntil([&]() {
        return tp_broadcast_servers_[0]->service()->getP2PRequestCallCount(
                   P2PConnectorBroadcastType::QUERY_LEASE_STATUS)
               > 0;
    }));
    ASSERT_TRUE(waitUntil([&]() { return !context->resourceHoldPending(); }));
    context.reset();
    result.context.reset();
    EXPECT_TRUE(waitUntil([&]() { return allocated.allocator->freeBlocksNum() == allocated.free_blocks_before; }));
    EXPECT_TRUE(allBlocksReclaimed(allocated));
}

TEST_F(P2PConnectorSchedulerTest, AsyncRead_QueryTimeout_RetainsTargetUntilRetryConfirmsStop) {
    auto allocated = createAllocatedConnectorResource();
    rebuildSchedulerForDeadlineTest(allocated.config.topologyPtr());
    for (auto& server : tp_broadcast_servers_) {
        server->service()->setP2PRequestSleepMillis(P2PConnectorBroadcastType::READ, 500);
        server->service()->setP2PRequestSleepMillis(P2PConnectorBroadcastType::QUERY_LEASE_STATUS, 700);
        server->service()->setLeaseStatus(true, 1, 1, true);
    }

    auto result = decode_scheduler_->asyncRead(
        allocated.resource, createMockMeta(5103, "query_timeout_hold", currentTimeMs() + 150), {0, -1});
    allocated.resource.reset();
    ASSERT_TRUE(result.ok());
    auto context = result.context;
    ASSERT_NE(context, nullptr);
    ASSERT_TRUE(waitUntil([&]() { return context->done(); }));
    ASSERT_TRUE(waitUntil([&]() {
        return tp_broadcast_servers_[0]->service()->getP2PRequestCallCount(
                   P2PConnectorBroadcastType::QUERY_LEASE_STATUS)
               > 0;
    }));
    std::this_thread::sleep_for(std::chrono::milliseconds(550));
    EXPECT_TRUE(context->resourceHoldPending());
    EXPECT_EQ(allocated.allocator->freeBlocksNum(), allocated.free_blocks_before - allocated.held_blocks);
    EXPECT_TRUE(allBlocksHeldByLease(allocated));

    for (auto& server : tp_broadcast_servers_) {
        server->service()->setP2PRequestSleepMillis(P2PConnectorBroadcastType::QUERY_LEASE_STATUS, 0);
    }
    ASSERT_TRUE(waitUntil([&]() { return !context->resourceHoldPending(); }));
    context.reset();
    result.context.reset();
    EXPECT_TRUE(waitUntil([&]() { return allocated.allocator->freeBlocksNum() == allocated.free_blocks_before; }));
    EXPECT_TRUE(allBlocksReclaimed(allocated));
}

TEST_F(P2PConnectorSchedulerTest, LeaseQueriesForDistinctRequestsCanOverlap) {
    for (const auto& server : tp_broadcast_servers_) {
        server->service()->setP2PRequestSleepMillis(P2PConnectorBroadcastType::QUERY_LEASE_STATUS, 700);
        server->service()->setLeaseStatus(true, 1, 1, true);
    }
    const auto deadline          = currentTimeMs() + 5000;
    auto       make_held_context = [&](const std::string& key) {
        auto context =
            std::make_shared<P2PConnectorAsyncReadContext>(nullptr, key, nullptr, 10000, false, deadline, deadline);
        context->markKickoffUnconfirmed(ErrorInfo(ErrorCode::GENERATE_TIMEOUT, "test lease hold"));
        context->cancel_confirmed_.store(true);
        decode_scheduler_->checker_->addContext(context);
        return context;
    };
    const auto first  = make_held_context("lease-overlap-first");
    const auto second = make_held_context("lease-overlap-second");
    ASSERT_TRUE(waitUntil([&] { return first->control_active_.load() && second->control_active_.load(); }));
    EXPECT_TRUE(first->resourceHoldPending());
    EXPECT_TRUE(second->resourceHoldPending());
    for (const auto& server : tp_broadcast_servers_) {
        server->service()->setP2PRequestSleepMillis(P2PConnectorBroadcastType::QUERY_LEASE_STATUS, 0);
    }
    ASSERT_TRUE(waitUntil([&] { return !first->resourceHoldPending() && !second->resourceHoldPending(); }));
}

TEST(P2PRetainedKvTest, OldLeaseStillRequiresPhysicalStopConfirmation) {
    const auto deadline = currentTimeMs() - 1;
    auto make_context = [&](const std::string& key) {
        auto context = std::make_shared<P2PConnectorAsyncReadContext>(
            std::make_shared<KVCacheResource>(), key, nullptr, 1, false, deadline, deadline);
        EXPECT_TRUE(context->beginKickoff());
        EXPECT_TRUE(context->expireTransferDeadlineIfNeeded());
        return context;
    };
    auto expired = make_context("expired-hold");
    expired->lease_hold_start_us_.store(P2PNotification::nowUs() - 10 * 60 * 1000 * 1000LL);
    expired->markKickoffUnconfirmed({ErrorCode::GENERATE_TIMEOUT, "late kickoff error"});
    EXPECT_TRUE(expired->resourceHoldPending());
    EXPECT_TRUE(expired->resource_);
    EXPECT_FALSE(expired->lease_all_ranks_stopped_.load());
    EXPECT_EQ(expired->nextWakeupMs(false), std::numeric_limits<int64_t>::max());
}

TEST(P2PRetainedKvTest, CapacityIncludesAllDecodePipelineStages) {
    P2PConnectorSchedulerConfig config;
    config.topology                               = test::makeTestCacheTopology(2, 2, {{0}, {1}});
    P2PStageStartupPB stage0;
    (*stage0.mutable_decode_block_bytes())["group0"] = 100;
    (*stage0.mutable_decode_block_bytes())["group1"] = 200;
    P2PStageStartupPB stage1;
    (*stage1.mutable_decode_block_bytes())["group1"] = 600;
    ASSERT_TRUE(accumulateDecodeStageBlockBytes(stage0, 2, config.physical_block_bytes_by_group));
    ASSERT_TRUE(accumulateDecodeStageBlockBytes(stage1, 2, config.physical_block_bytes_by_group));

    auto pool = std::make_shared<autil::LockFreeThreadPool>(1, 16, nullptr, "P2PPipelineCapacity");
    ASSERT_TRUE(pool->start());
    P2PConnectorAsyncReadContextChecker checker;
    ASSERT_TRUE(checker.init(nullptr, nullptr, pool, config));
    EXPECT_EQ(checker.group_block_bytes_.at("group0"), 200u);
    EXPECT_EQ(checker.group_block_bytes_.at("group1"), 1600u);
    EXPECT_EQ(checker.total_kv_bytes_, 28800u);

    auto resource = std::make_shared<KVCacheResource>();
    resource->initGroups(config.topology);
    resource->mutableBlockIds("group1").assign({1, 2, 3, 4, 5});
    const auto deadline = currentTimeMs() - 1;
    auto       context =
        std::make_shared<P2PConnectorAsyncReadContext>(resource, "pp-held", nullptr, 1, false, deadline, deadline);
    ASSERT_TRUE(context->beginKickoff());
    ASSERT_TRUE(context->expireTransferDeadlineIfNeeded());
    checker.checkRetainedResources({context, context});
    EXPECT_TRUE(context->resourceHoldPending());
    checker.stop();
    pool->stop();
}

TEST_F(P2PConnectorSchedulerTest, ControlWorkProgressesWhileEveryKickoffThreadIsBlocked) {
    auto gate_release = std::make_shared<std::promise<void>>();
    auto gate = gate_release->get_future().share();
    auto cleanup = std::shared_ptr<void>(nullptr, [gate_release](void*) { gate_release->set_value(); });
    std::vector<std::future<void>> entered;
    for (int i = 0; i < 4; ++i) {
        auto signal = std::make_shared<std::promise<void>>();
        entered.push_back(signal->get_future());
        ASSERT_EQ(decode_scheduler_->async_read_pool_->pushTask([signal, gate] {
            signal->set_value();
            gate.wait();
        }), autil::ThreadPoolBase::ERROR_NONE);
    }
    for (auto& ready : entered) {
        ASSERT_EQ(ready.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    }
    ASSERT_NE(decode_scheduler_->control_pool_, decode_scheduler_->async_read_pool_);
    EXPECT_EQ(decode_scheduler_->checker_->control_pool_, decode_scheduler_->control_pool_);
    auto executed = std::make_shared<std::promise<void>>();
    auto done = executed->get_future();
    ASSERT_EQ(decode_scheduler_->control_pool_->pushTask([executed] { executed->set_value(); }),
              autil::ThreadPoolBase::ERROR_NONE);
    EXPECT_EQ(done.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    cleanup.reset();
}

TEST(P2PReadNotificationTest, GenerationRetainsNotificationBeforeWait) {
    P2PNotification notification;
    auto            observed = notification.generation();
    notification.notify();
    auto waiting = std::async(std::launch::async, [&] {
        notification.waitUntil(observed, std::numeric_limits<int64_t>::max());
    });
    const auto ready = waiting.wait_for(std::chrono::seconds(5));
    // There is no deadline to release a lost notification; rescue only on failure.
    if (ready != std::future_status::ready) notification.notify();
    EXPECT_EQ(ready, std::future_status::ready);
    waiting.get();
}

TEST(P2PReadNotificationTest, CheckerWakesForBothRegistrationOrderings) {
    auto pool = std::make_shared<autil::LockFreeThreadPool>(1, 16, nullptr, "P2PEventTest");
    ASSERT_TRUE(pool->start());
    auto checker = std::make_shared<P2PConnectorAsyncReadContextChecker>();
    P2PConnectorSchedulerConfig config;
    config.topology = test::makeTestCacheTopology(1, 1, {{0}});
    ASSERT_TRUE(checker->init(nullptr, nullptr, pool, config));
    for (const bool complete_before_add : {false, true}) {
        const auto deadline = currentTimeMs() + 5000;
        auto       context  = std::make_shared<P2PConnectorAsyncReadContext>(
            nullptr, "event-ready", std::make_shared<P2PConnectorMetricsCollector>(), 1000, true, deadline, deadline);
        auto server = std::make_shared<DecodeLoadHelper::Result>();
        server->response.mutable_payload()->set_has_first_generate_token(true);
        auto broadcast = std::make_shared<P2PBroadcastClient::Result>("event-ready");
        ASSERT_TRUE(context->beginKickoff());
        if (complete_before_add) {
            server->complete(true);
            context->setCallResults(broadcast, server);
            checker->addContext(context);
        } else {
            checker->addContext(context);
            context->setCallResults(broadcast, server);
            server->complete(true);
        }
        auto done = std::async(std::launch::async, [context] { context->waitDone(); });
        EXPECT_EQ(done.wait_for(std::chrono::seconds(2)), std::future_status::ready);
        if (!context->done()) {
            context->markStartFailed(ErrorInfo(ErrorCode::CANCELLED, "test cleanup"));
        }
        done.get();
        EXPECT_TRUE(context->success());
    }
    checker->stop();
    pool->stop();
}

TEST(P2PReadNotificationTest, DeadlineWakesWithoutAnyCompletion) {
    auto pool = std::make_shared<autil::LockFreeThreadPool>(1, 16, nullptr, "P2PDeadlineTest");
    ASSERT_TRUE(pool->start());
    auto checker = std::make_shared<P2PConnectorAsyncReadContextChecker>();
    P2PConnectorSchedulerConfig config;
    config.topology = test::makeTestCacheTopology(1, 1, {{0}});
    ASSERT_TRUE(checker->init(nullptr, nullptr, pool, config));
    const auto deadline = currentTimeMs() + 50;
    auto       context  = std::make_shared<P2PConnectorAsyncReadContext>(
        nullptr, "deadline", std::make_shared<P2PConnectorMetricsCollector>(), 1000, false, deadline, deadline);
    checker->addContext(context);  // Still queued: timeout must not create a lease hold.
    auto done = std::async(std::launch::async, [context] { context->waitDone(); });
    EXPECT_EQ(done.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    if (!context->done()) {
        context->markStartFailed(ErrorInfo(ErrorCode::CANCELLED, "test cleanup"));
    }
    done.get();
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::GENERATE_TIMEOUT);
    EXPECT_FALSE(context->resourceHoldPending());
    checker->stop();
    pool->stop();
}

TEST(P2PReadNotificationTest, BusyControlJobDoesNotDelayDeadline) {
    auto pool = std::make_shared<autil::LockFreeThreadPool>(1, 16, nullptr, "P2PBusyControlTest");
    ASSERT_TRUE(pool->start());
    auto checker = std::make_shared<P2PConnectorAsyncReadContextChecker>();
    P2PConnectorSchedulerConfig config;
    config.topology = test::makeTestCacheTopology(1, 1, {{0}});
    ASSERT_TRUE(checker->init(nullptr, nullptr, pool, config));
    auto entered        = std::make_shared<std::promise<void>>();
    auto entered_future = entered->get_future();
    auto release        = std::make_shared<std::promise<void>>();
    auto gate           = release->get_future().share();
    checker->control_inflight_->store(1);
    ASSERT_EQ(pool->pushTask(
                  [entered, gate] {
                      entered->set_value();
                      gate.wait_for(std::chrono::seconds(3));
                  },
                  false),
              autil::ThreadPoolBase::ERROR_NONE);
    EXPECT_EQ(entered_future.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    const auto deadline = currentTimeMs() + 50;
    auto       context  = std::make_shared<P2PConnectorAsyncReadContext>(
        nullptr, "busy-control", std::make_shared<P2PConnectorMetricsCollector>(), 1000, false, deadline, deadline);
    checker->addContext(context);
    auto done = std::async(std::launch::async, [context] { context->waitDone(); });
    EXPECT_EQ(done.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    release->set_value();
    if (!context->done()) {
        context->markStartFailed(ErrorInfo(ErrorCode::CANCELLED, "test cleanup"));
    }
    done.get();
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::GENERATE_TIMEOUT);
    checker->stop();
    pool->stop();
}

TEST(P2PReadNotificationTest, MetricsWaitForAllBroadcastRanksAndKeepStartLoadIndependent) {
    using Broadcast = P2PBroadcastClient::TpBroadcastResult;
    // Exercise both kickoff/checker registration orders without thread timing.
    for (bool notification_first : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "notification_first=" << notification_first);
        auto a         = std::make_shared<Broadcast::WorkerRpcContext>();
        auto b         = std::make_shared<Broadcast::WorkerRpcContext>();
        auto rpc       = std::make_shared<Broadcast>(std::vector<std::shared_ptr<Broadcast::WorkerRpcContext>>{a, b});
        auto broadcast = std::make_shared<P2PBroadcastClient::Result>("metrics", rpc);
        auto server    = std::make_shared<DecodeLoadHelper::Result>();
        server->response.mutable_payload()->set_has_first_generate_token(true);
        auto collector    = std::make_shared<P2PConnectorMetricsCollector>();
        auto context      = std::make_shared<P2PConnectorAsyncReadContext>(nullptr, "metrics", collector, 1000);
        auto notification = std::make_shared<P2PNotification>();
        if (notification_first) {
            context->setWakeupCallback([notification] { notification->notify(); });
        }
        context->setCallResults(broadcast, server);
        if (!notification_first) {
            context->setWakeupCallback([notification] { notification->notify(); });
        }

        EXPECT_EQ(collector->decode_schedule_tp_sync_cost_time_us.load(), -1);
        EXPECT_EQ(collector->decode_schedule_server_call_cost_time_us.load(), -1);
        auto generation = notification->generation();
        rpc->finishRank(0, true);
        EXPECT_GT(notification->generation(), generation);
        EXPECT_EQ(collector->decode_schedule_tp_sync_cost_time_us.load(), -1);
        EXPECT_EQ(collector->decode_schedule_server_call_cost_time_us.load(), -1);
        generation = notification->generation();
        server->complete(true);
        EXPECT_GT(notification->generation(), generation);
        EXPECT_GE(collector->decode_schedule_server_call_cost_time_us.load(), 0);
        EXPECT_EQ(collector->decode_schedule_tp_sync_cost_time_us.load(), -1);
        generation = notification->generation();
        rpc->finishRank(1, true);
        EXPECT_GT(notification->generation(), generation);
        EXPECT_GE(collector->decode_schedule_tp_sync_cost_time_us.load(), 0);

        // Already-completed results must notify a newly registered checker
        // without changing their recorded durations.
        const auto server_cost       = collector->decode_schedule_server_call_cost_time_us.load();
        const auto read_cost         = collector->decode_schedule_tp_sync_cost_time_us.load();
        auto late_notification = std::make_shared<P2PNotification>();
        context->setWakeupCallback([late_notification] { late_notification->notify(); });
        EXPECT_GT(late_notification->generation(), 0u);
        EXPECT_EQ(collector->decode_schedule_server_call_cost_time_us.load(), server_cost);
        EXPECT_EQ(collector->decode_schedule_tp_sync_cost_time_us.load(), read_cost);
    }
}

TEST(P2PReadNotificationTest, FatalBroadcastBeforeRegistrationStillPublishesCallsReady) {
    using Broadcast = P2PBroadcastClient::TpBroadcastResult;
    for (bool notification_first : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "notification_first=" << notification_first);
        auto worker = std::make_shared<Broadcast::WorkerRpcContext>();
        auto rpc = std::make_shared<Broadcast>(
            std::vector<std::shared_ptr<Broadcast::WorkerRpcContext>>{worker});
        rpc->finishRank(0, false);
        auto broadcast = std::make_shared<P2PBroadcastClient::Result>("fatal-registration", rpc);
        auto server = std::make_shared<DecodeLoadHelper::Result>();
        auto collector = std::make_shared<P2PConnectorMetricsCollector>();
        auto context = std::make_shared<P2PConnectorAsyncReadContext>(nullptr, "fatal-registration", collector, 0);
        auto notification = std::make_shared<P2PNotification>();
        if (notification_first) {
            EXPECT_NO_THROW(context->setWakeupCallback([notification] { notification->notify(); }));
        }
        ASSERT_TRUE(context->beginKickoff());
        EXPECT_NO_THROW(context->setCallResults(broadcast, server));
        if (!notification_first) {
            EXPECT_NO_THROW(context->setWakeupCallback([notification] { notification->notify(); }));
        }
        EXPECT_TRUE(context->calls_ready_.load());
        EXPECT_GT(notification->generation(), 0u);
        EXPECT_NO_THROW(context->checkDone());
        EXPECT_TRUE(context->done());
        EXPECT_FALSE(context->success());
        EXPECT_EQ(context->errorInfo().code(), ErrorCode::RPC_FINISH_FAILED);
    }
}

TEST(P2PReadNotificationTest, NoTransferDoesNotEmitReadBroadcastDuration) {
    auto collector = std::make_shared<P2PConnectorMetricsCollector>();
    auto context   = std::make_shared<P2PConnectorAsyncReadContext>(nullptr, "no-transfer", collector, 1000, true);
    auto server    = std::make_shared<DecodeLoadHelper::Result>();
    server->response.mutable_payload()->set_has_first_generate_token(true);
    server->complete(true);
    context->setCallResults(std::make_shared<P2PBroadcastClient::Result>("no-transfer"), server);
    EXPECT_EQ(collector->decode_schedule_tp_sync_cost_time_us.load(), -1);
    EXPECT_GE(collector->decode_schedule_server_call_cost_time_us.load(), 0);
}

TEST(P2PReadNotificationTest, LateCompletionAfterCheckerDestructionIsSafe) {
    auto context = std::make_shared<P2PConnectorAsyncReadContext>(
        nullptr, "late", std::make_shared<P2PConnectorMetricsCollector>(), 1000, true);
    auto server = std::make_shared<DecodeLoadHelper::Result>();
    server->response.mutable_payload()->set_has_first_generate_token(true);
    context->setCallResults(std::make_shared<P2PBroadcastClient::Result>("late"), server);
    {
        auto checker = std::make_shared<P2PConnectorAsyncReadContextChecker>();
        checker->addContext(context);
        checker->stop();
    }
    std::weak_ptr<P2PConnectorAsyncReadContext> weak = context;
    context.reset();
    EXPECT_TRUE(weak.expired());  // No callback cycle keeps the read context alive.
    server->complete(true);
    EXPECT_TRUE(server->success());
}

TEST_F(P2PConnectorSchedulerTest, StartLoadFirstFailureSurvivesLaterReadCancelAndDeadline) {
    auto worker = std::make_shared<P2PBroadcastClient::TpBroadcastResult::WorkerRpcContext>();
    auto tp     = std::make_shared<P2PBroadcastClient::TpBroadcastResult>(
        std::vector<std::shared_ptr<P2PBroadcastClient::TpBroadcastResult::WorkerRpcContext>>{worker});
    auto broadcast = std::make_shared<P2PBroadcastClient::Result>("first-cause", tp);
    auto prefill   = std::make_shared<DecodeLoadHelper::Result>();
    prefill->response.set_error_code(ErrorCodePB::INVALID_PARAMS);
    prefill->response.set_error_message("Prefill plan digest mismatch");
    prefill->complete(true);
    auto collector = std::make_shared<P2PConnectorMetricsCollector>();
    auto context   = std::make_shared<P2PConnectorAsyncReadContext>(createValidKVCacheResource(1, 1),
                                                                  broadcast,
                                                                  prefill,
                                                                  collector,
                                                                  1000,
                                                                  false,
                                                                  currentTimeMs() + 5000,
                                                                  currentTimeMs() - 1);
    context->checkDone();
    EXPECT_TRUE(context->done());
    EXPECT_TRUE(context->resourceHoldPending());
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::INVALID_PARAMS);
    const auto first_error_message = context->errorInfo().ToString();
    EXPECT_NE(first_error_message.find("Prefill plan digest mismatch"), std::string::npos);
    worker->status = grpc::Status(grpc::StatusCode::CANCELLED, "later READ cancellation");
    tp->finishRank(0, true);
    context->checkDone();
    EXPECT_FALSE(context->expireTransferDeadlineIfNeeded());
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_EQ(context->errorInfo().ToString(), first_error_message);
    EXPECT_TRUE(context->resourceHoldPending());
}

}  // namespace rtp_llm

namespace rtp_llm {
TEST(P2PReplicaPlanTest, BothSchedulersRotateMirrorPlansAndBoundCacheBySourceRanks) {
    GroupBase group;
    group.spec                           = test::makeResolvedMlaSpec(DataType::TYPE_FP16, 16, 8, 1);
    group.policy                         = defaultCacheGroupPolicy(CacheGroupType::FULL);
    group.block_num                      = 16;
    auto                        topology = test::makeIndexedTestTopology({group}, {{0}});
    P2PConnectorSchedulerConfig prefill_config;
    prefill_config.topology                   = topology;
    prefill_config.parallelism_config.tp_size = 8;
    prefill_config.worker_grpc_addrs.resize(8);
    P2PConnectorSchedulerConfig decode_config;
    decode_config.topology                   = topology;
    decode_config.parallelism_config.tp_size = 1;
    P2PSchedulerPrefillRead prefill(prefill_config, nullptr, nullptr);
    P2PSchedulerDecodeRead  decode(decode_config, nullptr, nullptr);
    std::set<int>                selected;
    for (int i = 0; i < 64; ++i) {
        const auto key    = "decode-dp-" + std::to_string(i);
        const auto d_plan = decode.planFor(8, 1, key);
        const auto p_plan = prefill.planFor(1, key);
        ASSERT_TRUE(d_plan->ok()) << d_plan->error.ToString();
        ASSERT_TRUE(p_plan->ok()) << p_plan->error.ToString();
        ASSERT_EQ(d_plan->plan.routes.size(), 1u);
        selected.insert(d_plan->plan.routes.front().src_rank);
        EXPECT_EQ(d_plan->plan.digest(), p_plan->plan.digest());
        const auto prepared = prefill.preparePlan(1, d_plan->plan.digest(), key, 1, {});
        ASSERT_TRUE(prepared.ok()) << prepared.status().ToString();
        EXPECT_EQ(prepared.value().plan.get(), &p_plan->plan);
        EXPECT_EQ(decode.planFor(8, 1, key), d_plan);
        EXPECT_EQ(prefill.planFor(1, key), p_plan);
    }
    EXPECT_EQ(selected.size(), 8u);
    EXPECT_EQ(decode.plan_cache_.size(), 8u);
    EXPECT_EQ(prefill.plan_cache_.size(), 8u);
    const auto first      = decode.planFor(8, 1, "decode-dp-0");
    const auto mismatched = prefill.preparePlan(1, first->plan.digest(), "decode-dp-1", 1, {});
    ASSERT_FALSE(mismatched.ok());
    EXPECT_EQ(mismatched.status().code(), ErrorCode::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED);
    EXPECT_NE(mismatched.status().ToString().find("plan digest mismatch"), std::string::npos);
    for (int peer_tp = 2; peer_tp <= 16; ++peer_tp) {
        ASSERT_TRUE(prefill.planFor(peer_tp, "decode-dp-0")->ok());
        EXPECT_LE(prefill.plan_cache_.size(), 8u);
    }
}
}  // namespace rtp_llm
