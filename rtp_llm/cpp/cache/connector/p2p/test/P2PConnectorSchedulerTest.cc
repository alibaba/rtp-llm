#include <chrono>
#include <thread>
#include <gtest/gtest.h>
#include "grpc++/grpc++.h"

#include "autil/NetUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorScheduler.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
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
    void SetUp() override {
        // 创建测试用的 RPC 服务器（用于 P2PBroadcastClient）
        for (int i = 0; i < 2; ++i) {
            auto service = std::make_unique<TestRpcService>();
            auto server  = std::make_unique<TestRpcServer>(std::move(service));
            ASSERT_TRUE(server->start());
            tp_broadcast_servers_.push_back(std::move(server));
            tp_broadcast_addrs_.push_back("127.0.0.1:" + std::to_string(tp_broadcast_servers_.back()->listenPort()));
        }

        // 创建测试用的 RPC 服务器（用于 PrefillLoadCaller）
        auto prefill_service = std::make_unique<TestRpcService>();
        prefill_server_      = std::make_unique<TestRpcServer>(std::move(prefill_service));
        ASSERT_TRUE(prefill_server_->start());
        prefill_addr_ = "127.0.0.1:" + std::to_string(prefill_server_->listenPort());

        P2PConnectorSchedulerConfig scheduler_config;
        scheduler_config.worker_grpc_addrs = tp_broadcast_addrs_;
        scheduler_config.worker_addrs.push_back("127.0.0.1:12345:" + std::to_string(prefill_server_->listenPort()));

        scheduler_ = std::make_unique<P2PConnectorScheduler>(std::move(scheduler_config), nullptr);
        ASSERT_TRUE(scheduler_->init());
    }

    void TearDown() override {
        scheduler_.reset();
        tp_broadcast_servers_.clear();
        prefill_server_.reset();
    }

    // 创建有效的 KVCacheResource（使用 initGroups + groupBlocks/blocks/cacheKeys 公开 API）
    KVCacheResourcePtr createValidKVCacheResource(int num_layers = 2, int blocks_per_layer = 2) {
        auto             resource = std::make_shared<KVCacheResource>();
        std::vector<int> layer_to_group(num_layers);
        for (int i = 0; i < num_layers; ++i) {
            layer_to_group[i] = i;
        }
        resource->initGroups(num_layers, num_layers, layer_to_group);

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

    std::shared_ptr<MockGenerateStream>
    createMockStream(int64_t request_id, const std::string& unique_key, int64_t deadline_ms) {
        std::string prefill_ip   = "127.0.0.1";
        uint32_t    prefill_port = static_cast<uint32_t>(prefill_server_->listenPort());
        auto        stream       = std::make_shared<MockGenerateStream>(prefill_ip, prefill_port);
        stream->setRequestId(request_id);
        stream->setUniqueKey(unique_key);
        stream->setDeadlineMs(deadline_ms);
        return stream;
    }

    KVCacheResourcePtr createInvalidKVCacheResource() {
        auto resource = std::make_shared<KVCacheResource>();
        return resource;
    }

    // 等待 async context 完成，调用 checkDone() 以便异常在测试线程中抛出
    // 注意：如果需要在测试中捕获超时异常，请先调用 scheduler_->stopChecker() 停止后台线程
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

    void rebuildSchedulerWithResourceHoldMs(int64_t hold_ms) {
        scheduler_.reset();
        P2PConnectorSchedulerConfig cfg;
        cfg.worker_grpc_addrs = tp_broadcast_addrs_;
        cfg.worker_addrs.push_back("127.0.0.1:12345:" + std::to_string(prefill_server_->listenPort()));
        cfg.p2p_transfer_not_done_resource_hold_ms = hold_ms;
        scheduler_                                 = std::make_unique<P2PConnectorScheduler>(std::move(cfg), nullptr);
        ASSERT_TRUE(scheduler_->init());
    }

protected:
    std::vector<std::unique_ptr<TestRpcServer>> tp_broadcast_servers_;
    std::vector<std::string>                    tp_broadcast_addrs_;
    std::unique_ptr<TestRpcServer>              prefill_server_;
    std::string                                 prefill_addr_;
    std::unique_ptr<P2PConnectorScheduler>      scheduler_;
};

// ==================== sendKVCache 测试 (Prefill 端功能) ====================

// 测试：resource 转换不到 layer_cache_buffers
TEST_F(P2PConnectorSchedulerTest, HandleRead_ReturnError_LayerCacheBuffersEmpty) {
    auto invalid_resource = createInvalidKVCacheResource();

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});

    auto deadline_ms = currentTimeMs() + 1000;

    ErrorInfo error_info =
        scheduler_->sendKVCache(invalid_resource, "test_unique_key", 1001, decode_transfer_servers, deadline_ms);

    EXPECT_TRUE(error_info.hasError());

    // 验证 BroadcastTp 没有被调用
    for (size_t i = 0; i < tp_broadcast_servers_.size(); ++i) {
        EXPECT_EQ(tp_broadcast_servers_[i]->service()->getBroadcastTpCallCount(), 0);
    }
}

// 测试：broadcast 成功
TEST_F(P2PConnectorSchedulerTest, HandleRead_ReturnOK_BroadcastSuccess) {
    auto valid_resource = createValidKVCacheResource(2, 2);

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});
    decode_transfer_servers.push_back({"127.0.0.1", 12346});

    auto deadline_ms = currentTimeMs() + 1000;

    ErrorInfo error_info =
        scheduler_->sendKVCache(valid_resource, "test_broadcast_success", 1001, decode_transfer_servers, deadline_ms);

    EXPECT_TRUE(error_info.ok());

    // 验证 BroadcastTp 被调用
    for (size_t i = 0; i < tp_broadcast_servers_.size(); ++i) {
        EXPECT_EQ(tp_broadcast_servers_[i]->service()->getBroadcastTpCallCount(), 1);
        EXPECT_EQ(tp_broadcast_servers_[i]->service()->getBroadcastTpCancelCallCount(), 0);
    }
}

// 测试：broadcast 返回失败（所有响应失败）
TEST_F(P2PConnectorSchedulerTest, HandleRead_ReturnError_BroadcastPartialFailed) {
    for (auto& server : tp_broadcast_servers_) {
        server->service()->setP2PResponseSuccess(false);
        break;
    }

    auto valid_resource = createValidKVCacheResource(2, 2);

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});

    auto deadline_ms = currentTimeMs() + 1000;

    ErrorInfo error_info =
        scheduler_->sendKVCache(valid_resource, "test_broadcast_all_fail", 1003, decode_transfer_servers, deadline_ms);

    EXPECT_TRUE(error_info.hasError());

    // 验证 BroadcastTp 被调用
    for (size_t i = 0; i < tp_broadcast_servers_.size(); ++i) {
        EXPECT_EQ(tp_broadcast_servers_[i]->service()->getBroadcastTpCallCount(), 1);
    }
}

// 测试: broadcast worker 慢于 gRPC deadline，checkDone 路径抛 RTPException
TEST_F(P2PConnectorSchedulerTest, HandleRead_ThrowException_BroadcastTimeout) {
    for (auto& server : tp_broadcast_servers_) {
        server->service()->setSleepMillis(500);  // 延迟 500ms
        break;
    }

    auto valid_resource = createValidKVCacheResource(2, 2);

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});

    auto deadline_ms = currentTimeMs() + 50;

    EXPECT_THROW(
        scheduler_->sendKVCache(valid_resource, "test_broadcast_timeout", 1004, decode_transfer_servers, deadline_ms),
        RTPException);
}

// 测试: handleRead 被 client 取消, 返回失败
TEST_F(P2PConnectorSchedulerTest, HandleRead_ReturnFalse_BroadcastCancelled) {
    // 设置 TP worker 延迟响应，以便有足够的时间触发取消
    for (auto& server : tp_broadcast_servers_) {
        server->service()->setSleepMillis(200);
    }

    auto valid_resource = createValidKVCacheResource(2, 2);

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});

    auto deadline_ms = currentTimeMs() + 5000;

    // 使用 atomic 来控制取消状态
    std::atomic<bool> cancelled{false};
    auto              is_cancelled = [&cancelled]() { return cancelled.load(); };

    // 在另一个线程中延迟设置取消标志
    std::thread cancel_thread([&cancelled]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));  // 50ms 后取消
        cancelled = true;
    });

    ErrorInfo error_info = scheduler_->sendKVCache(
        valid_resource, "test_broadcast_cancelled", 1005, decode_transfer_servers, deadline_ms, is_cancelled);

    cancel_thread.join();

    // 由于被取消，handleRead 应该返回错误码
    EXPECT_TRUE(error_info.hasError());
    EXPECT_EQ(error_info.code(), ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_CANCELLED);

    // 验证 BroadcastTp 被调用，且 CANCEL_HANDLE_READ 也被发送
    for (size_t i = 0; i < tp_broadcast_servers_.size(); ++i) {
        EXPECT_EQ(tp_broadcast_servers_[i]->service()->getBroadcastTpCallCount(), 1);
        EXPECT_EQ(tp_broadcast_servers_[i]->service()->getBroadcastTpCancelCallCount(), 1);
    }
}

// ==================== asyncRead 测试 (Decode 端功能) ====================
TEST_F(P2PConnectorSchedulerTest, AsyncRead_ReturnNotNull_AllSuccess) {
    auto resource        = createValidKVCacheResource(2, 2);
    auto generate_stream = createMockStream(2001, "test_async_read_1", currentTimeMs() + 5000);

    // block_range: {start_block_idx, block_count}, use -1 for block_count to include all blocks
    auto result = scheduler_->asyncRead(resource, generate_stream, {0, -1});
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

// 验证 P2PConnectorAsyncReadContext::waitDone() 在 checkDone() 置 done 后由 condition_variable 唤醒
TEST_F(P2PConnectorSchedulerTest, AsyncRead_WaitDone_UnblocksWhenCheckDoneCompletes) {
    auto resource        = createValidKVCacheResource(2, 2);
    auto generate_stream = createMockStream(2010, "test_async_read_wait_done", currentTimeMs() + 5000);

    auto result = scheduler_->asyncRead(resource, generate_stream, {0, -1});
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
    auto generate_stream = createMockStream(2002, "test_async_read_null_resource", currentTimeMs() + 5000);

    auto result = scheduler_->asyncRead(nullptr, generate_stream, {0, -1});

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.context, nullptr);

    // 验证 BroadcastTp 和 StartLoad 都没有被调用
    for (size_t i = 0; i < tp_broadcast_servers_.size(); ++i) {
        EXPECT_EQ(tp_broadcast_servers_[i]->service()->getBroadcastTpCallCount(), 0);
    }
    EXPECT_EQ(prefill_server_->service()->getStartLoadCallCount(), 0);
}

TEST_F(P2PConnectorSchedulerTest, AsyncRead_ReturnNull_EmptyResource) {
    auto resource        = std::make_shared<KVCacheResource>();
    auto generate_stream = createMockStream(2003, "test_async_read_empty", currentTimeMs() + 5000);

    auto result = scheduler_->asyncRead(resource, generate_stream, {0, -1});

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

    auto resource        = createValidKVCacheResource(2, 2);
    auto generate_stream = createMockStream(2004, "test_async_read_broadcast_fail", currentTimeMs() + 5000);

    auto result = scheduler_->asyncRead(resource, generate_stream, {0, -1});
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

    auto resource        = createValidKVCacheResource(2, 2);
    auto generate_stream = createMockStream(2005, "test_async_read_load_fail", currentTimeMs() + 5000);

    auto result = scheduler_->asyncRead(resource, generate_stream, {0, -1});
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

    auto resource        = createValidKVCacheResource(2, 2);
    auto generate_stream = createMockStream(2006, "test_async_read_both_fail", currentTimeMs() + 5000);

    auto result = scheduler_->asyncRead(resource, generate_stream, {0, -1});
    ASSERT_TRUE(result.ok());
    auto async_context = result.context;
    ASSERT_NE(async_context, nullptr);

    waitAsyncContextDone(async_context);

    EXPECT_TRUE(async_context->done());
    EXPECT_FALSE(async_context->success());
}

// 测试: prefill server 超时, 返回失败
TEST_F(P2PConnectorSchedulerTest, AsyncRead_ReturnFalse_PrefillTimeout) {
    // 设置 prefill server 延迟响应
    prefill_server_->service()->setSleepMillis(500);

    auto resource        = createValidKVCacheResource(2, 2);
    auto generate_stream = createMockStream(2007, "test_async_read_prefill_timeout", currentTimeMs() + 50);

    auto result = scheduler_->asyncRead(resource, generate_stream, {0, -1});
    ASSERT_TRUE(result.ok());
    auto async_context = result.context;
    ASSERT_NE(async_context, nullptr);

    waitAsyncContextDone(async_context);

    EXPECT_TRUE(async_context->done());
    EXPECT_FALSE(async_context->success());  // prefill server 超时导致失败

    // 验证 StartLoad 被调用
    EXPECT_EQ(prefill_server_->service()->getStartLoadCallCount(), 1);
}

// 测试: broadcast worker 慢于 gRPC deadline，checkDone 抛 RTPException
TEST_F(P2PConnectorSchedulerTest, AsyncRead_ThrowException_BroadcastTimeout) {
    tp_broadcast_servers_[0]->service()->setSleepMillis(500);

    scheduler_->stopChecker();

    auto resource        = createValidKVCacheResource(2, 2);
    auto generate_stream = createMockStream(2008, "test_async_read_broadcast_timeout", currentTimeMs() + 50);

    auto result = scheduler_->asyncRead(resource, generate_stream, {0, -1});
    ASSERT_TRUE(result.ok());
    auto async_context = result.context;
    ASSERT_NE(async_context, nullptr);

    EXPECT_THROW(waitAsyncContextDone(async_context, 500, /*check_done=*/true), RTPException);
}

// 测试: asyncread prefill 失败, 取消broadcast
TEST_F(P2PConnectorSchedulerTest, AsyncRead_CancelBroadcast_WhenPrefillFailed) {
    // 设置 prefill server 立即返回失败
    prefill_server_->service()->setStartLoadResponseSuccess(false);

    // 设置 broadcast server 延迟响应，确保 prefill 先完成
    for (auto& server : tp_broadcast_servers_) {
        server->service()->setSleepMillis(200);
    }

    auto resource        = createValidKVCacheResource(2, 2);
    auto generate_stream = createMockStream(3001, "test_cancel_broadcast_when_prefill_failed", currentTimeMs() + 5000);

    auto result = scheduler_->asyncRead(resource, generate_stream, {0, -1});
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

    auto resource        = createValidKVCacheResource(2, 2);
    auto generate_stream = createMockStream(3002, "test_cancel_prefill_when_broadcast_failed", currentTimeMs() + 5000);

    auto result = scheduler_->asyncRead(resource, generate_stream, {0, -1});
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

// Prefill：worker 极慢导致 gRPC DEADLINE_EXCEEDED 时抛 RTPException（与 BroadcastManager 行为一致）
TEST_F(P2PConnectorSchedulerTest, SendKVCache_ThrowException_WhenBroadcastExceedsDeadline) {
    for (auto& server : tp_broadcast_servers_) {
        server->service()->setSleepMillis(120000);
        server->service()->setP2PResponseSuccess(true);
    }

    auto                                          valid_resource = createValidKVCacheResource(2, 2);
    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});

    const int64_t deadline_ms = currentTimeMs() + 80;
    EXPECT_THROW(
        scheduler_->sendKVCache(
            valid_resource, "test_prefill_broadcast_past_deadline", 4006, decode_transfer_servers, deadline_ms),
        RTPException);
}

// StartLoad 返回 TRANSFER_NOT_DONE 且 hold_ms>0：checkDone 进入保留窗口，done 仍为 false 且 needCancel 为 false；hold
// 结束后 done
TEST_F(P2PConnectorSchedulerTest, AsyncRead_TransferNotDone_HoldDelaysDoneAndSuppressesNeedCancel) {
    rebuildSchedulerWithResourceHoldMs(120);

    prefill_server_->service()->setStartLoadApplicationError(ErrorCodePB::P2P_CONNECTOR_WORKER_READ_TRANSFER_NOT_DONE,
                                                             "test transfer not done");
    for (auto& server : tp_broadcast_servers_) {
        server->service()->setP2PResponseSuccess(true);
        server->service()->setSleepMillis(0);
    }

    scheduler_->stopChecker();

    auto resource        = createValidKVCacheResource(2, 2);
    auto generate_stream = createMockStream(5010, "test_transfer_not_done_hold", currentTimeMs() + 5000);

    auto result = scheduler_->asyncRead(resource, generate_stream, {0, -1});
    ASSERT_TRUE(result.ok());
    auto async_context = result.context;
    ASSERT_NE(async_context, nullptr);

    bool saw_hold = false;
    for (int i = 0; i < 3000 && !saw_hold; ++i) {
        async_context->checkDone();
        if (!async_context->done()
            && async_context->errorInfo().code() == ErrorCode::P2P_CONNECTOR_WORKER_READ_TRANSFER_NOT_DONE
            && !async_context->needCancel()) {
            saw_hold = true;
            break;
        }
        if (async_context->done()) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_TRUE(saw_hold) << "expected hold phase: not done, TRANSFER_NOT_DONE, needCancel false";

    ASSERT_FALSE(async_context->done());
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    async_context->checkDone();

    EXPECT_TRUE(async_context->done());
    EXPECT_FALSE(async_context->success());
    EXPECT_EQ(async_context->errorInfo().code(), ErrorCode::P2P_CONNECTOR_WORKER_READ_TRANSFER_NOT_DONE);

    for (size_t i = 0; i < tp_broadcast_servers_.size(); ++i) {
        EXPECT_EQ(tp_broadcast_servers_[i]->service()->getBroadcastTpCancelCallCount(), 0);
    }
}

// hold_ms==0 时不进入保留窗口，应立即 done 且失败
TEST_F(P2PConnectorSchedulerTest, AsyncRead_TransferNotDone_ZeroHold_CompletesImmediately) {
    rebuildSchedulerWithResourceHoldMs(0);

    prefill_server_->service()->setStartLoadApplicationError(ErrorCodePB::P2P_CONNECTOR_WORKER_READ_TRANSFER_NOT_DONE,
                                                             "test transfer not done");
    for (auto& server : tp_broadcast_servers_) {
        server->service()->setP2PResponseSuccess(true);
        server->service()->setSleepMillis(0);
    }

    scheduler_->stopChecker();

    auto resource        = createValidKVCacheResource(2, 2);
    auto generate_stream = createMockStream(5011, "test_transfer_not_done_zero_hold", currentTimeMs() + 5000);

    auto result = scheduler_->asyncRead(resource, generate_stream, {0, -1});
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
}

}  // namespace rtp_llm
