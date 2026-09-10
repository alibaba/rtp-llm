#include <algorithm>
#include <chrono>
#include <thread>
#include <gtest/gtest.h>
#include "grpc++/grpc++.h"

#include "autil/NetUtil.h"
#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/cache/SingleTypeKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PBroadcastClient.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorSchedulerDecode.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorSchedulerPrefill.h"
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
        std::shared_ptr<SingleTypeKVCacheAllocator> allocator;
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

    // 创建有效的 KVCacheResource（使用 initGroups + groupBlocks/blocks/cacheKeys 公开 API）
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
                resource->mutableBlockIds(layer_id).add({i});
            }
        }

        for (int i = 0; i < num_layers * blocks_per_layer; ++i) {
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
        result.allocator = std::make_shared<SingleTypeKVCacheAllocator>(result.config, AllocationType::HOST);
        EXPECT_TRUE(result.allocator->init());

        auto block_pool = result.allocator->getDeviceBlockPool();
        EXPECT_NE(block_pool, nullptr);
        result.free_blocks_before = result.allocator->freeBlocksNum();
        result.blocks             = block_pool->malloc(2).value();
        result.held_blocks        = result.blocks.size();
        block_pool->incRef(result.blocks);

        KVCacheResource source;
        source.initGroups(result.config.topologyPtr());
        source.cacheKeys() = {101, 102};
        source.mutableBlockIds(0).assign(result.blocks);
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
        const auto block_pool = allocated.allocator->getDeviceBlockPool();
        return std::all_of(allocated.blocks.begin(), allocated.blocks.end(), [&](BlockIdxType block) {
            return block_pool->refCount(block) == expected;
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
        group.layer_ids                 = {0};
        group.block_num                 = 16;
        group.seq_size_per_block        = 1;
        group.kernel_seq_size_per_block = 1;
        group.kv_block_stride_bytes     = spec->block_size_bytes();
        group.kv_scale_stride_bytes     = spec->scale_block_size_bytes();
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
            std::make_unique<P2PConnectorSchedulerPrefill>(scheduler_config, nullptr, tp_broadcast_client_);
        decode_scheduler_ =
            std::make_unique<P2PConnectorSchedulerDecode>(std::move(scheduler_config), nullptr, tp_broadcast_client_);
        ASSERT_TRUE(decode_scheduler_->init("p2p_connector_scheduler_test"));
    }

protected:
    std::vector<std::unique_ptr<TestRpcServer>> tp_broadcast_servers_;
    std::vector<std::string>                    tp_broadcast_addrs_;
    std::unique_ptr<TestRpcServer>              prefill_server_;
    std::string                                 prefill_addr_;
    std::shared_ptr<P2PBroadcastClient>             tp_broadcast_client_;
    std::unique_ptr<P2PConnectorSchedulerPrefill>   prefill_scheduler_;
    std::unique_ptr<P2PConnectorSchedulerDecode>    decode_scheduler_;
};

TEST_F(P2PConnectorSchedulerTest, AsyncReadUsesConfiguredLoadBudgetAndRequestDeadline) {
    P2PConnectorSchedulerConfig config;
    config.worker_grpc_addrs = tp_broadcast_addrs_;
    config.worker_addrs.push_back("127.0.0.1:12345:" + std::to_string(prefill_server_->listenPort()));
    config.load_cache_timeout_ms = 1000;
    rebuildSchedulers(config);
    const int64_t before = currentTimeMs();
    const int64_t request_deadline_ms = before + 60000;
    auto meta = createMockMeta(9010, "configured_load_budget", request_deadline_ms);
    auto result = decode_scheduler_->asyncRead(createValidKVCacheResource(), meta, {2, 0}, true);
    const int64_t after = currentTimeMs();
    ASSERT_TRUE(result.ok());
    waitAsyncContextDone(result.context);
    ASSERT_TRUE(result.context->success());
    auto request = prefill_server_->service()->getLastStartLoadRequest();
    EXPECT_EQ(request.request_deadline_ms(), request_deadline_ms);
    EXPECT_GE(request.deadline_ms(), before + 1000);
    EXPECT_LE(request.deadline_ms(), after + 1000);
}

TEST_F(P2PConnectorSchedulerTest, AsyncReadLoadBudgetCannotExceedRemainingRequestTime) {
    const int64_t request_deadline_ms = currentTimeMs() + 2000;
    auto meta = createMockMeta(9011, "request_bounds_load", request_deadline_ms);
    auto result = decode_scheduler_->asyncRead(createValidKVCacheResource(), meta, {2, 0}, true);
    ASSERT_TRUE(result.ok());
    waitAsyncContextDone(result.context);
    ASSERT_TRUE(result.context->success());
    auto request = prefill_server_->service()->getLastStartLoadRequest();
    EXPECT_EQ(request.deadline_ms(), request_deadline_ms);
    EXPECT_EQ(request.request_deadline_ms(), request_deadline_ms);
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
        prefill_scheduler_->sendKVCache("test_broadcast_success", 1001, decode_transfer_servers, deadline_ms, nullptr, false, deadline_ms);

    EXPECT_TRUE(error_info.ok());

    // 验证 BroadcastTp 被调用
    for (size_t i = 0; i < tp_broadcast_servers_.size(); ++i) {
        EXPECT_EQ(tp_broadcast_servers_[i]->service()->getBroadcastTpCallCount(), 1);
        EXPECT_EQ(tp_broadcast_servers_[i]->service()->getBroadcastTpCancelCallCount(), 0);
    }
}

TEST_F(P2PConnectorSchedulerTest, HandleRead_FiltersLinearLayersByAttentionType) {
    rebuildSchedulerWithLayerAttnTypes({CacheGroupType::FULL, CacheGroupType::LINEAR});

    auto topology = test::makeTestCacheTopology(
        2, 2, {{0}, {1}}, /*kernel_blocks_per_kv_block=*/1, {CacheGroupType::FULL, CacheGroupType::LINEAR});
    auto resource = std::make_shared<KVCacheResource>();
    resource->initGroups(topology);
    resource->mutableBlockIds(0).assign({10, 11, 12, 13});
    resource->mutableBlockIds(1).assign({NULL_BLOCK_IDX, 21, NULL_BLOCK_IDX, 25});
    resource->cacheKeys() = {1000, 1001, 1002, 1003};

    const auto layer_buffers = LayerCacheBufferUtil::convert(*resource, *topology);
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

// 修复 5/22 现场 B 类(7 笔 handleRead cost > 60s,最大 ~1h)的回归测试:
// cancel 已发且 cancel_result 已 done(成功送达 or cancel 自身 gRPC timeout)后,
// 即使原 HANDLE_READ broadcast 的 result 仍未 done(worker hang / channel down),
// waitForBroadcastCompletion 也应立即退出,不再盲等 client 业务 deadline_ms。
//
// 复现 corner case 的近似手法:让 broadcast worker 慢响应 3s(模拟 worker hang),
// 把 deadline_ms 设到 10s 远大于 worker sleep(避免靠 gRPC client deadline 退出),
// 100ms 后触发 cancel。test server 的 cancel 路径立即 ack,所以 cancel_result 几 ms done。
// - 修复前:主循环只看 result->done(),要等 worker sleep 结束 ~3s 才返回
// - 修复后:cancel done 立即 break,函数在 ~120ms 内返回
TEST_F(P2PConnectorSchedulerTest, HandleRead_ExitsImmediately_AfterCancelDoneEvenIfBroadcastSlow) {
    for (auto& server : tp_broadcast_servers_) {
        server->service()->setSleepMillis(3000);  // 3s 慢 worker,模拟 hang
    }

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});

    // deadline_ms 设大,确保不靠 gRPC client deadline 退出循环
    auto deadline_ms = currentTimeMs() + 10000;

    std::atomic<bool> cancelled{false};
    auto              is_cancelled = [&cancelled]() { return cancelled.load(); };
    std::thread       cancel_thread([&cancelled]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        cancelled = true;
    });

    auto      start_ms   = currentTimeMs();
    ErrorInfo error_info = prefill_scheduler_->sendKVCache("test_exit_after_cancel_done", 4007, decode_transfer_servers, deadline_ms, is_cancelled, false, deadline_ms);
    auto duration_ms = currentTimeMs() - start_ms;

    cancel_thread.join();

    EXPECT_TRUE(error_info.hasError());
    EXPECT_EQ(error_info.code(), ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_CANCELLED);

    // 修复后应在 cancel done 后立即返回(~100ms cancel 触发延迟 + 几 ms cancel RPC + break)。
    // 给 CI 噪声留充足余量(1s),但远低于 broadcast worker 的 3s sleep,
    // 确保确实因为 cancel done 早退,而不是等 worker 自然完成。
    EXPECT_LT(duration_ms, 1000) << "function should exit shortly after cancel completes, "
                                 << "but took " << duration_ms << "ms (likely waited for broadcast worker)";

    for (size_t i = 0; i < tp_broadcast_servers_.size(); ++i) {
        const auto cancel_check_deadline = currentTimeMs() + 1000;
        while (tp_broadcast_servers_[i]->service()->getBroadcastTpCancelCallCount() == 0
               && currentTimeMs() < cancel_check_deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        EXPECT_EQ(tp_broadcast_servers_[i]->service()->getBroadcastTpCancelCallCount(), 1);
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
        EXPECT_EQ(tp_broadcast_servers_[i]->service()->getBroadcastTpCallCount(), 1);
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

TEST_F(P2PConnectorSchedulerTest, AsyncRead_UnconfirmedLeaseDeadlineAbortsRankZero) {
    auto tp_ctx = std::make_shared<P2PBroadcastClient::TpBroadcastResult::WorkerRpcContext>();
    tp_ctx->response.mutable_p2p_response()->set_error_code(
        transErrorCodeToRPC(ErrorCode::P2P_CONNECTOR_WORKER_READ_CANCELLED));
    tp_ctx->response.mutable_p2p_response()->set_error_message("worker cancelled");

    auto tp_result = std::make_shared<P2PBroadcastClient::TpBroadcastResult>(
        std::vector<std::shared_ptr<P2PBroadcastClient::TpBroadcastResult::WorkerRpcContext>>{tp_ctx});
    tp_result->finished_[0] = true;
    tp_result->finished_count_.store(1);
    tp_result->already_done_.store(true);
    tp_result->all_request_success_.store(false);

    auto broadcast_result = std::make_shared<P2PBroadcastClient::Result>("cancel-hold", tp_result);

    auto server_result          = std::make_shared<DecodeLoadHelper::Result>();
    server_result->done_        = true;
    server_result->success_     = true;
    server_result->error_code   = ErrorCode::NONE_ERROR;
    server_result->error_message.clear();

    auto resource = createValidKVCacheResource(1, 1);
    auto collector = std::make_shared<DecodeSchedulerMetricsCollector>(nullptr);
    auto context = std::make_shared<P2PConnectorAsyncReadContext>(
        resource, broadcast_result, server_result, collector, /*lease_query_timeout_ms=*/80);

    context->checkDone();

    EXPECT_TRUE(context->done());
    EXPECT_FALSE(context->success());
    EXPECT_TRUE(context->needCancel());
    EXPECT_TRUE(context->resourceHoldPending());
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::P2P_CONNECTOR_WORKER_READ_CANCELLED);

    std::this_thread::sleep_for(std::chrono::milliseconds(120));
    EXPECT_DEATH(context->failStopIfLeaseUnconfirmed(), "");
}

TEST_F(P2PConnectorSchedulerTest, AsyncRead_CompletedBeforeDeadlineObservationRemainsSuccessful) {
    auto tp_ctx = std::make_shared<P2PBroadcastClient::TpBroadcastResult::WorkerRpcContext>();
    tp_ctx->response.mutable_p2p_response()->set_error_code(ErrorCodePB::NONE_ERROR);

    auto tp_result = std::make_shared<P2PBroadcastClient::TpBroadcastResult>(
        std::vector<std::shared_ptr<P2PBroadcastClient::TpBroadcastResult::WorkerRpcContext>>{tp_ctx});
    tp_result->finished_[0] = true;
    tp_result->finished_count_.store(1);
    tp_result->already_done_.store(true);
    tp_result->all_request_success_.store(true);

    auto broadcast_result = std::make_shared<P2PBroadcastClient::Result>("completed-before-observation", tp_result);
    auto server_result     = std::make_shared<DecodeLoadHelper::Result>();
    server_result->done_    = true;
    server_result->success_ = true;
    auto collector          = std::make_shared<DecodeSchedulerMetricsCollector>(nullptr);

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
    tp_result->finished_[0] = true;
    tp_result->finished_count_.store(1);
    tp_result->already_done_.store(true);
    tp_result->all_request_success_.store(true);

    auto collector = std::make_shared<DecodeSchedulerMetricsCollector>(nullptr);
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

TEST_F(P2PConnectorSchedulerTest, AsyncReadPendingContextWaitsForKickoff) {
    auto resource  = createValidKVCacheResource(1, 1);
    auto collector = std::make_shared<DecodeSchedulerMetricsCollector>(nullptr);
    auto context   = std::make_shared<P2PConnectorAsyncReadContext>(
        resource, "pending-kickoff", collector, /*lease_query_timeout_ms=*/0);

    context->checkDone();

    EXPECT_FALSE(context->done());
    EXPECT_FALSE(context->success());
    EXPECT_FALSE(context->needCancel());
    EXPECT_EQ(context->uniqueKey(), "pending-kickoff");
}

TEST_F(P2PConnectorSchedulerTest, AsyncReadPendingContextCanBeCancelledBeforeKickoff) {
    auto resource  = createValidKVCacheResource(1, 1);
    auto collector = std::make_shared<DecodeSchedulerMetricsCollector>(nullptr);
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
    auto collector = std::make_shared<DecodeSchedulerMetricsCollector>(nullptr);
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
        EXPECT_EQ(tp_broadcast_servers_[i]->service()->getBroadcastTpCallCount(), 1);
    }
    EXPECT_EQ(prefill_server_->service()->getStartLoadCallCount(), 1);
}

TEST_F(P2PConnectorSchedulerTest, AsyncReadCpSendsEachWorkerItsRoundRobinKeys) {
    rebuildSchedulerWithLayerAttnTypes({CacheGroupType::FULL}, /*cp_size=*/2);

    auto resource = std::make_shared<KVCacheResource>();
    resource->initGroups(makeSingleMlaTopology());
    resource->mutableBlockIds(0).assign({10, 11});
    resource->cacheKeys() = {100, 101, 102, 103};
    auto meta = createMockMeta(2012, "test_async_read_cp", currentTimeMs() + 5000);
    meta->setPrefillTpSize(2);
    meta->setPrefillCpSize(2);

    auto result = decode_scheduler_->asyncRead(resource, meta, {0, -1});
    ASSERT_TRUE(result.ok());
    ASSERT_NE(result.context, nullptr);
    waitAsyncContextDone(result.context);
    ASSERT_TRUE(result.context->success());

    const auto rank0_request = tp_broadcast_servers_[0]->service()->getLastBroadcastTpRequest();
    const auto rank1_request = tp_broadcast_servers_[1]->service()->getLastBroadcastTpRequest();
    ASSERT_EQ(rank0_request.routes_size(), 1);
    ASSERT_EQ(rank1_request.routes_size(), 1);
    ASSERT_EQ(rank0_request.routes(0).layer_blocks_size(), 1);
    ASSERT_EQ(rank1_request.routes(0).layer_blocks_size(), 1);
    const auto& rank0_layer = rank0_request.routes(0).layer_blocks(0);
    const auto& rank1_layer = rank1_request.routes(0).layer_blocks(0);
    ASSERT_EQ(rank0_layer.cache_keys_size(), 2);
    ASSERT_EQ(rank1_layer.cache_keys_size(), 2);
    EXPECT_EQ(rank0_layer.cache_keys(0), 100);
    EXPECT_EQ(rank0_layer.cache_keys(1), 102);
    EXPECT_EQ(rank1_layer.cache_keys(0), 101);
    EXPECT_EQ(rank1_layer.cache_keys(1), 103);
    ASSERT_EQ(rank0_layer.block_ids_size(), 2);
    ASSERT_EQ(rank1_layer.block_ids_size(), 2);
    EXPECT_EQ(rank0_layer.block_ids(0), 10);
    EXPECT_EQ(rank0_layer.block_ids(1), 11);
    EXPECT_EQ(rank1_layer.block_ids(0), 10);
    EXPECT_EQ(rank1_layer.block_ids(1), 11);
}

TEST_F(P2PConnectorSchedulerTest, AsyncReadCpRejectsDifferentSourceCpSize) {
    rebuildSchedulerWithLayerAttnTypes({CacheGroupType::FULL}, /*cp_size=*/2);

    auto resource = std::make_shared<KVCacheResource>();
    resource->initGroups(makeSingleMlaTopology());
    resource->mutableBlockIds(0).assign({10});
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
        async_context->checkDone();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    ASSERT_TRUE(async_context->done()) << "async read should complete within timeout";
    wait_thread.join();
    EXPECT_TRUE(wait_done_thread_finished.load());
    EXPECT_TRUE(async_context->success());

    for (size_t i = 0; i < tp_broadcast_servers_.size(); ++i) {
        EXPECT_GE(tp_broadcast_servers_[i]->service()->getBroadcastTpCallCount(), 1);
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
        EXPECT_EQ(tp_broadcast_servers_[i]->service()->getBroadcastTpCallCount(), 1);
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
        EXPECT_EQ(tp_broadcast_servers_[i]->service()->getBroadcastTpCallCount(), 1);
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
        EXPECT_EQ(tp_broadcast_servers_[i]->service()->getBroadcastTpCallCount(), 1);
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
        EXPECT_EQ(tp_broadcast_servers_[i]->service()->getBroadcastTpCallCount(), 1);
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
    EXPECT_EQ(async_context->errorInfo().code(), ErrorCode::GENERATE_TIMEOUT);
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
    EXPECT_EQ(async_context->errorInfo().code(), ErrorCode::GENERATE_TIMEOUT);
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

}  // namespace rtp_llm
