#include <algorithm>
#include <csignal>
#include <sys/resource.h>
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

    bool allBlockRefsEqual(const AllocatedConnectorResource& allocated, uint32_t expected) const {
        const auto block_pool = allocated.allocator->groupBlockPools().front();
        // refCount() rejects reclaimed blocks; refCountNoCheck() reports the raw
        // count under the pool lock so released blocks can still be observed.
        return std::all_of(allocated.blocks.begin(), allocated.blocks.end(), [&](BlockIdxType block) {
            return block_pool->refCountNoCheck(block) == expected;
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
        prefill_scheduler_ =
            std::make_unique<P2PSchedulerPrefillRead>(scheduler_config, nullptr, tp_broadcast_client_);
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
}

TEST_F(P2PConnectorSchedulerTest, AsyncReadKeepsRequestDeadlineSeparateFromLoadBudget) {
    const int64_t request_deadline_ms = currentTimeMs() + 2000;
    auto meta = createMockMeta(9011, "request_bounds_load", request_deadline_ms);
    auto result = decode_scheduler_->asyncRead(createValidKVCacheResource(), meta, {2, 0}, true);
    ASSERT_TRUE(result.ok());
    waitAsyncContextDone(result.context);
    ASSERT_TRUE(result.context->success());
    auto request = prefill_server_->service()->getLastStartLoadRequest();
    EXPECT_EQ(request.timeout_ms(), 5000);
    EXPECT_EQ(result.context->request_deadline_ms_, request_deadline_ms);
    EXPECT_LE(result.context->transfer_deadline_ms_, request_deadline_ms);
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
        auto meta = createMockMeta(9012, "normalized_load_budget", currentTimeMs() + 60000);
        auto result = decode_scheduler_->asyncRead(createValidKVCacheResource(), meta, {2, 0}, true);
        ASSERT_TRUE(result.ok());
        waitAsyncContextDone(result.context);
        ASSERT_TRUE(result.context->success());
        EXPECT_EQ(prefill_server_->service()->getLastStartLoadRequest().timeout_ms(), expected);
    }
}

TEST(P2PConnectorConfigTest, LoadTimeoutComesFromSharedPDSepConfig) {
    RuntimeConfig runtime;
    CacheStoreConfig cache_store;
    ParallelismConfig parallelism;
    PDSepConfig pd_sep;
    EXPECT_EQ(pd_sep.load_cache_timeout_ms, 60000);
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
        prefill_scheduler_->sendKVCache("test_broadcast_success", 1001, decode_transfer_servers, deadline_ms, nullptr, false, deadline_ms);

    EXPECT_TRUE(error_info.ok());

    // 验证 BroadcastTp 被调用
    for (size_t i = 0; i < tp_broadcast_servers_.size(); ++i) {
        EXPECT_EQ(tp_broadcast_servers_[i]->service()->getBroadcastTpCallCount(), 1);
        EXPECT_EQ(tp_broadcast_servers_[i]->service()->getBroadcastTpCancelCallCount(), 0);
    }
}

TEST_F(P2PConnectorSchedulerTest, HandleReadBroadcastsOnlyDecodeActiveRoutes) {
    const std::string unique_key = "test_active_route_filter";
    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers = {
        {"127.0.0.1", 12345}, {"127.0.0.1", 12346}};
    const auto plan = prefill_scheduler_->planFor(static_cast<int>(decode_transfer_servers.size()), unique_key);
    ASSERT_TRUE(plan->ok()) << plan->error.ToString();
    ASSERT_GT(plan->plan.routes.size(), 1u);

    const int selected_route_id = plan->plan.routes.back().route_id;
    const auto deadline_ms      = currentTimeMs() + 1000;
    const auto error_info       = prefill_scheduler_->sendKVCache(unique_key,
                                                            1011,
                                                            decode_transfer_servers,
                                                            deadline_ms,
                                                            nullptr,
                                                            false,
                                                            deadline_ms,
                                                            {selected_route_id});
    ASSERT_TRUE(error_info.ok()) << error_info.ToString();

    size_t broadcast_route_count = 0;
    for (const auto& server : tp_broadcast_servers_) {
        const auto request = server->service()->getLastBroadcastTpRequest();
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

    ErrorInfo error_info =
        prefill_scheduler_->sendKVCache("test_linear_filter", 1009, decode_transfer_servers, currentTimeMs() + 1000, nullptr, false, currentTimeMs() + 1000);

    ASSERT_TRUE(error_info.ok());
    const auto rank0_request = tp_broadcast_servers_[0]->service()->getLastBroadcastTpRequest();
    const auto rank1_request = tp_broadcast_servers_[1]->service()->getLastBroadcastTpRequest();
    // Plan-driven Prefill broadcasts routes only. Each worker has already
    // produced its local LayerCacheBuffer through the per-layer path.
    EXPECT_EQ(rank0_request.layer_blocks_size(), 0);
    EXPECT_EQ(rank1_request.layer_blocks_size(), 0);
    ASSERT_EQ(rank0_request.routes_size(), 2);
    EXPECT_EQ(rank0_request.routes(0).cache_tag(), "group0");
    EXPECT_EQ(rank0_request.routes(1).cache_tag(), "group1");
    EXPECT_EQ(rank1_request.routes_size(), 0);
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
        prefill_scheduler_->sendKVCache("test_broadcast_all_fail", 1003, decode_transfer_servers, deadline_ms, nullptr, false, deadline_ms);

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
        prefill_scheduler_->sendKVCache("test_broadcast_timeout", 1004, decode_transfer_servers, deadline_ms, nullptr, false, deadline_ms);

    EXPECT_TRUE(error_info.hasError());
    EXPECT_EQ(error_info.code(), ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_TIMEOUT);
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
    auto sending = std::async(std::launch::async, [&] {
        return prefill_scheduler_->sendKVCache("test_exit_after_cancel_done", 4007,
            {{"127.0.0.1", 12345}}, deadline, [&] { return cancelled.load(); }, false, deadline);
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

    ErrorInfo error_info = prefill_scheduler_->sendKVCache("test_broadcast_cancelled", 1005, decode_transfer_servers, deadline_ms, is_cancelled, false, request_deadline_ms);

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

TEST_F(P2PConnectorSchedulerTest, AsyncReadStartingContextIsNotCompletedByCancelBeforeCallsReady) {
    auto resource  = createValidKVCacheResource(1, 1);
    auto collector = std::make_shared<P2PConnectorMetricsCollector>();
    auto context   = std::make_shared<P2PConnectorAsyncReadContext>(
        resource, "cancel-during-kickoff", collector, /*lease_query_timeout_ms=*/0);

    ASSERT_TRUE(context->beginKickoff());
    context->cancel(nullptr);

    EXPECT_TRUE(context->cancelRequested());
    EXPECT_FALSE(context->done());

    context->markStartFailed(ErrorInfo(ErrorCode::CANCELLED, "kickoff cancelled before calls were created"));
    EXPECT_TRUE(context->done());
    EXPECT_FALSE(context->success());
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
    auto result = scheduler.buildDecodeRankRoutes(plan, resource, {0, -1}, 1);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
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

TEST_F(P2PConnectorSchedulerTest, AsyncReadCpRejectsDifferentSourceCpSize) {
    rebuildSchedulerWithLayerAttnTypes({CacheGroupType::FULL}, /*cp_size=*/2);

    auto resource = std::make_shared<KVCacheResource>();
    resource->initGroups(makeSingleMlaTopology());
    resource->mutableBlockIds(resource->groupTags().at(0)).assign({10});
    resource->cacheKeys() = {100, 101};
    auto meta = createMockMeta(2013, "test_async_read_cp_mismatch", currentTimeMs() + 5000);
    // TP happens to match Decode, but effective KV-cache CP does not.
    meta->setPrefillTpSize(2);
    meta->setPrefillCpSize(1);

    auto result = decode_scheduler_->asyncRead(resource, meta, {0, -1});
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.context, nullptr);
    EXPECT_EQ(prefill_server_->service()->getStartLoadCallCount(), 0);
    for (const auto& server : tp_broadcast_servers_) {
        EXPECT_EQ(server->service()->getBroadcastTpCallCount(), 0);
    }
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
    ErrorInfo     error_info  = prefill_scheduler_->sendKVCache("test_prefill_broadcast_past_deadline", 4006, decode_transfer_servers, deadline_ms, nullptr, false, deadline_ms);

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
        server->service()->setLeaseStatus(true, 1, 0, false);
    }

    auto result = decode_scheduler_->asyncRead(
        allocated.resource, createMockMeta(5100, "start_load_timeout_hold", currentTimeMs() + 5000), {0, -1});
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
    EXPECT_TRUE(allBlockRefsEqual(allocated, 1));

    std::weak_ptr<P2PConnectorAsyncReadContext> weak_context = context;
    context.reset();
    result.context.reset();
    EXPECT_FALSE(weak_context.expired());
    for (auto& server : tp_broadcast_servers_) {
        server->service()->setLeaseStatus(true, 1, 1, true);
    }
    EXPECT_TRUE(waitUntil([&]() { return allocated.allocator->freeBlocksNum() == allocated.free_blocks_before; }));
    EXPECT_TRUE(allBlockRefsEqual(allocated, 0));
    EXPECT_TRUE(weak_context.expired());
}

TEST_F(P2PConnectorSchedulerTest, AsyncRead_ReadPerRankTimeout_WaitsForEveryRankLease) {
    auto allocated = createAllocatedConnectorResource();
    rebuildSchedulerForDeadlineTest(allocated.config.topologyPtr());
    for (auto& server : tp_broadcast_servers_) {
        server->service()->setP2PRequestSleepMillis(P2PConnectorBroadcastType::READ, 500);
        server->service()->setLeaseStatus(true, 1, 0, false);
    }

    auto result = decode_scheduler_->asyncRead(
        allocated.resource, createMockMeta(5101, "read_per_rank_timeout_hold", currentTimeMs() + 5000), {0, -1});
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
    EXPECT_TRUE(allBlockRefsEqual(allocated, 1));

    tp_broadcast_servers_[1]->service()->setLeaseStatus(true, 1, 1, true);
    ASSERT_TRUE(waitUntil([&]() { return !context->resourceHoldPending(); }));
    context.reset();
    result.context.reset();
    EXPECT_TRUE(waitUntil([&]() { return allocated.allocator->freeBlocksNum() == allocated.free_blocks_before; }));
    EXPECT_TRUE(allBlockRefsEqual(allocated, 0));
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
        allocated.resource, createMockMeta(5102, "cancel_timeout_hold", currentTimeMs() + 5000), {0, -1});
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
    EXPECT_TRUE(allBlockRefsEqual(allocated, 1));

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
    EXPECT_TRUE(allBlockRefsEqual(allocated, 0));
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
        allocated.resource, createMockMeta(5103, "query_timeout_hold", currentTimeMs() + 5000), {0, -1});
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
    EXPECT_TRUE(allBlockRefsEqual(allocated, 1));

    for (auto& server : tp_broadcast_servers_) {
        server->service()->setP2PRequestSleepMillis(P2PConnectorBroadcastType::QUERY_LEASE_STATUS, 0);
    }
    ASSERT_TRUE(waitUntil([&]() { return !context->resourceHoldPending(); }));
    context.reset();
    result.context.reset();
    EXPECT_TRUE(waitUntil([&]() { return allocated.allocator->freeBlocksNum() == allocated.free_blocks_before; }));
    EXPECT_TRUE(allBlockRefsEqual(allocated, 0));
}

namespace {

void disableRetainedKvCoreDump() {
    const rlimit no_core{0, 0};
    setrlimit(RLIMIT_CORE, &no_core);
}

}  // namespace

TEST(P2PRetainedKvTest, ThresholdUsesCapacityBytesAndStrictGreaterThan) {
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    auto resource = std::make_shared<KVCacheResource>();
    resource->initGroups(test::makeTestCacheTopology(1, 1, {{0}}));
    resource->mutableBlockIds("group0").assign({1, 2, 3});
    const auto deadline = currentTimeMs() - 1;
    auto       context =
        std::make_shared<P2PConnectorAsyncReadContext>(resource, "retained", nullptr, 1, false, deadline, deadline);
    ASSERT_TRUE(context->beginKickoff());
    ASSERT_TRUE(context->expireTransferDeadlineIfNeeded());

    P2PConnectorAsyncReadContextChecker checker;
    checker.group_block_bytes_ = {{"group0", 100}};
    checker.total_kv_bytes_    = 1000;
    // Exactly 30%, including duplicate contexts/blocks, must keep running.
    checker.checkRetainedResources({context, context});
    resource->mutableBlockIds("group0").assign({1, 2, 3, NULL_BLOCK_IDX});
    checker.checkRetainedResources({context});
    resource->mutableBlockIds("group0").assign({1, 2, 3, 4});
    EXPECT_EXIT({
        disableRetainedKvCoreDump();
        checker.checkRetainedResources({context});
    }, ::testing::KilledBySignal(SIGABRT), "");
    checker.kv_leak_exit_threshold_percent_ = 40;
    checker.checkRetainedResources({context});
    checker.kv_leak_exit_threshold_percent_ = 39;
    EXPECT_EXIT({
        disableRetainedKvCoreDump();
        checker.checkRetainedResources({context});
    }, ::testing::KilledBySignal(SIGABRT), "");
}

TEST(P2PRetainedKvTest, DistinctRequestsAccumulateAndGroupsUseDifferentBlockSizes) {
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    const auto topology = test::makeTestCacheTopology(2, 2, {{0}, {1}});
    const auto reordered_topology =
        CacheTopology::create({topology->group("group1"), topology->group("group0")}, topology->layers());
    auto make_context = [&](BlockIndicesType first, BlockIndicesType second, bool reorder = false) {
        auto resource = std::make_shared<KVCacheResource>();
        resource->initGroups(reorder ? reordered_topology : topology);
        resource->mutableBlockIds("group0").assign(std::move(first));
        resource->mutableBlockIds("group1").assign(std::move(second));
        const auto deadline = currentTimeMs() - 1;
        auto       context =
            std::make_shared<P2PConnectorAsyncReadContext>(resource, "weighted", nullptr, 1, false, deadline, deadline);
        EXPECT_TRUE(context->beginKickoff());
        EXPECT_TRUE(context->expireTransferDeadlineIfNeeded());
        return context;
    };
    P2PConnectorAsyncReadContextChecker checker;
    checker.group_block_bytes_ = {{"group0", 100}, {"group1", 400}};
    checker.total_kv_bytes_    = 2000;
    const auto a               = make_context({1}, {1});    // 500 bytes, 25%.
    const auto b               = make_context({1, 2}, {}, true);  // Reordered rows; union is 600 bytes, 30%.
    checker.checkRetainedResources({a, b});
    const auto c = make_context({3}, {});  // Union is 700 bytes, 35%.
    EXPECT_EXIT({
        disableRetainedKvCoreDump();
        checker.checkRetainedResources({a, b, c});
    }, ::testing::KilledBySignal(SIGABRT), "");
    c->lease_hold_pending_.store(false);
    checker.checkRetainedResources({a, b, c});
}

TEST(P2PRetainedKvTest, InitDerivesCapacityAndRejectsInvalidThreshold) {
    auto pool = std::make_shared<autil::LockFreeThreadPool>(1, 16, nullptr, "P2PKvCapacityTest");
    ASSERT_TRUE(pool->start());
    P2PConnectorSchedulerConfig config;
    config.topology = test::makeTestCacheTopology(2, 2, {{0}, {1}});
    P2PConnectorAsyncReadContextChecker checker;
    for (const int invalid : {-1, 0, 101}) {
        config.p2p_kv_leak_exit_threshold_percent = invalid;
        EXPECT_FALSE(checker.init(nullptr, nullptr, pool, config));
    }
    config.p2p_kv_leak_exit_threshold_percent = 45;
    ASSERT_TRUE(checker.init(nullptr, nullptr, pool, config));
    checker.stop();
    size_t expected_bytes = 0;
    for (const auto& group : config.topology->groups()) {
        expected_bytes += group.block_num * config.topology->layerIdsForGroup(group.tag).size()
                          * (group.kvBlockStrideBytes() + group.kvScaleStrideBytes());
    }
    EXPECT_EQ(checker.total_kv_bytes_, expected_bytes);
    EXPECT_EQ(checker.kv_leak_exit_threshold_percent_, 45);
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
    P2PConnectorAsyncReadContextChecker checker;
    P2PConnectorSchedulerConfig config;
    config.topology = test::makeTestCacheTopology(1, 1, {{0}});
    ASSERT_TRUE(checker.init(nullptr, nullptr, pool, config));
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
            checker.addContext(context);
        } else {
            checker.addContext(context);
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
    checker.stop();
    pool->stop();
}

TEST(P2PReadNotificationTest, DeadlineWakesWithoutAnyCompletion) {
    auto pool = std::make_shared<autil::LockFreeThreadPool>(1, 16, nullptr, "P2PDeadlineTest");
    ASSERT_TRUE(pool->start());
    P2PConnectorAsyncReadContextChecker checker;
    P2PConnectorSchedulerConfig config;
    config.topology = test::makeTestCacheTopology(1, 1, {{0}});
    ASSERT_TRUE(checker.init(nullptr, nullptr, pool, config));
    const auto deadline = currentTimeMs() + 50;
    auto       context  = std::make_shared<P2PConnectorAsyncReadContext>(
        nullptr, "deadline", std::make_shared<P2PConnectorMetricsCollector>(), 1000, false, deadline, deadline);
    checker.addContext(context);  // Still queued: timeout must not create a lease hold.
    auto done = std::async(std::launch::async, [context] { context->waitDone(); });
    EXPECT_EQ(done.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    if (!context->done()) {
        context->markStartFailed(ErrorInfo(ErrorCode::CANCELLED, "test cleanup"));
    }
    done.get();
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::GENERATE_TIMEOUT);
    EXPECT_FALSE(context->resourceHoldPending());
    checker.stop();
    pool->stop();
}

TEST(P2PReadNotificationTest, BusyControlJobDoesNotDelayDeadline) {
    auto pool = std::make_shared<autil::LockFreeThreadPool>(1, 16, nullptr, "P2PBusyControlTest");
    ASSERT_TRUE(pool->start());
    P2PConnectorAsyncReadContextChecker checker;
    P2PConnectorSchedulerConfig config;
    config.topology = test::makeTestCacheTopology(1, 1, {{0}});
    ASSERT_TRUE(checker.init(nullptr, nullptr, pool, config));
    auto entered        = std::make_shared<std::promise<void>>();
    auto entered_future = entered->get_future();
    auto release        = std::make_shared<std::promise<void>>();
    auto gate           = release->get_future().share();
    checker.control_busy_->store(true);
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
    checker.addContext(context);
    auto done = std::async(std::launch::async, [context] { context->waitDone(); });
    EXPECT_EQ(done.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    release->set_value();
    if (!context->done()) {
        context->markStartFailed(ErrorInfo(ErrorCode::CANCELLED, "test cleanup"));
    }
    done.get();
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::GENERATE_TIMEOUT);
    checker.stop();
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
            context->setNotification(notification);
        }
        context->setCallResults(broadcast, server);
        if (!notification_first) {
            context->setNotification(notification);
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
        context->setNotification(late_notification);
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
            EXPECT_NO_THROW(context->setNotification(notification));
        }
        ASSERT_TRUE(context->beginKickoff());
        EXPECT_NO_THROW(context->setCallResults(broadcast, server));
        if (!notification_first) {
            EXPECT_NO_THROW(context->setNotification(notification));
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
        P2PConnectorAsyncReadContextChecker checker;
        checker.addContext(context);
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
        EXPECT_TRUE(prefill.checkPlanDigest(1, d_plan->plan.digest(), key).ok());
        EXPECT_EQ(decode.planFor(8, 1, key), d_plan);
        EXPECT_EQ(prefill.planFor(1, key), p_plan);
    }
    EXPECT_EQ(selected.size(), 8u);
    EXPECT_EQ(decode.plan_cache_.size(), 8u);
    EXPECT_EQ(prefill.plan_cache_.size(), 8u);
    const auto first = decode.planFor(8, 1, "decode-dp-0");
    const auto error = prefill.checkPlanDigest(1, first->plan.digest(), "decode-dp-1");
    EXPECT_EQ(error.code(), ErrorCode::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED);
    EXPECT_NE(error.ToString().find("plan digest mismatch"), std::string::npos);
}
}  // namespace rtp_llm
