#include <thread>
#include <gtest/gtest.h>
#include "grpc++/grpc++.h"

#include "autil/NetUtil.h"
#include "rtp_llm/cpp/disaggregate/p2p_connector/P2PConnectorScheduler.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/disaggregate/p2p_connector/test/TestRpcServer.h"

namespace rtp_llm {

// Mock implementation of IGenerateStream for testing
class MockGenerateStream: public IGenerateStream {
public:
    MockGenerateStream(const std::string& prefill_ip, uint32_t prefill_port):
        prefill_ip_(prefill_ip), prefill_port_(prefill_port) {}

    void             appendTokenId(int batch_id, int token_id) override {}
    std::vector<int> currentExecuteTokens(int batch_id) override {
        return {};
    }
    void                                                            appendSPInfo(const std::vector<int>& propose_tokens,
                                                                                 const TensorPB&         propose_probs,
                                                                                 const TensorPB&         propose_hidden) override {}
    std::optional<std::tuple<std::vector<int>, TensorPB, TensorPB>> getSPInfoPB() override {
        return std::nullopt;
    }
    int reuseBlockNum() override {
        return 0;
    }
    std::tuple<int, int, int> getReuseLength() override {
        return {0, 0, 0};
    }
    void setPrefillReuseLength(int reuse_length, int local_reuse_length, int remote_reuse_length) override {}
    std::pair<std::string, uint32_t> getPrefillAddr() override {
        return {prefill_ip_, prefill_port_};
    }
    std::vector<int32_t> getContextPositionIdsPB() override {
        return {};
    }
    void setContextPositionIds(const std::vector<int32_t>& ids) override {}
    bool waitForRemoteGenerate() override {
        return true;
    }

private:
    std::string prefill_ip_;
    uint32_t    prefill_port_;
};

class P2PConnectorSchedulerTest: public ::testing::Test {
protected:
    void SetUp() override {
        // 创建测试用的 RPC 服务器（用于 TPBroadcastClient）
        for (int i = 0; i < 2; ++i) {
            auto service = std::make_unique<TestRpcService>();
            auto server  = std::make_unique<TestRpcServer>(std::move(service));
            ASSERT_TRUE(server->start());
            tp_broadcast_servers_.push_back(std::move(server));
            tp_broadcast_addrs_.push_back("127.0.0.1:" + std::to_string(tp_broadcast_servers_.back()->listenPort()));
        }

        // 创建测试用的 RPC 服务器（用于 P2PConnectorServerCaller）
        auto prefill_service = std::make_unique<TestRpcService>();
        prefill_server_      = std::make_unique<TestRpcServer>(std::move(prefill_service));
        ASSERT_TRUE(prefill_server_->start());
        prefill_addr_ = "127.0.0.1:" + std::to_string(prefill_server_->listenPort());

        // 创建 RuntimeConfig
        runtime_config_.worker_grpc_addrs = tp_broadcast_addrs_;
        // worker_addrs 格式: "ip:cache_store_port:grpc_port"
        runtime_config_.worker_addrs.push_back("127.0.0.1:12345:" + std::to_string(prefill_server_->listenPort()));

        // 创建 P2PConnectorScheduler
        scheduler_ = std::make_unique<P2PConnectorScheduler>(runtime_config_, nullptr);
        ASSERT_TRUE(scheduler_->init());
    }

    void TearDown() override {
        scheduler_.reset();
        tp_broadcast_servers_.clear();
        prefill_server_.reset();
    }

    // 创建有效的 KVCacheResource
    KVCacheResourcePtr createValidKVCacheResource(int num_layers = 2, int blocks_per_layer = 2) {
        auto resource = std::make_shared<KVCacheResource>();

        // 设置 layer_block_ids
        for (int layer_id = 0; layer_id < num_layers; ++layer_id) {
            auto block_ids = std::make_shared<BlockIds>();
            for (int i = 0; i < blocks_per_layer; ++i) {
                block_ids->block_indices.push_back(i);
            }
            resource->layer_block_ids.push_back(block_ids);
        }

        // 设置 cache_keys
        for (int i = 0; i < num_layers * blocks_per_layer; ++i) {
            resource->cache_keys.push_back(1000 + i);
        }

        return resource;
    }

    // 创建无效的 KVCacheResource（无法转换为 layer_cache_buffers）
    KVCacheResourcePtr createInvalidKVCacheResource() {
        auto resource = std::make_shared<KVCacheResource>();
        // 不设置 layer_block_ids，这样 convert 会返回空
        return resource;
    }

    // 等待 async context 完成
    void waitAsyncContextDone(std::shared_ptr<P2PConnectorAsyncReadContext>& context, int timeout_ms = 5000) {
        int waited_ms = 0;
        while (!context->done() && waited_ms < timeout_ms) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            waited_ms += 10;
        }
    }

protected:
    std::vector<std::unique_ptr<TestRpcServer>> tp_broadcast_servers_;
    std::vector<std::string>                    tp_broadcast_addrs_;
    std::unique_ptr<TestRpcServer>              prefill_server_;
    std::string                                 prefill_addr_;
    RuntimeConfig                               runtime_config_;
    std::unique_ptr<P2PConnectorScheduler>      scheduler_;
};

// ==================== handleRead 测试 (Server 端功能) ====================

// 测试：resource 转换不到 layer_cache_buffers
TEST_F(P2PConnectorSchedulerTest, HandleRead_ReturnError_LayerCacheBuffersEmpty) {
    auto invalid_resource = createInvalidKVCacheResource();

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});

    auto deadline_ms = currentTimeMs() + 1000;

    auto success =
        scheduler_->handleRead(invalid_resource, "test_unique_key", 1001, decode_transfer_servers, deadline_ms);

    EXPECT_FALSE(success);

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

    auto success =
        scheduler_->handleRead(valid_resource, "test_broadcast_success", 1001, decode_transfer_servers, deadline_ms);

    EXPECT_TRUE(success);

    // 验证 BroadcastTp 被调用
    for (size_t i = 0; i < tp_broadcast_servers_.size(); ++i) {
        EXPECT_EQ(tp_broadcast_servers_[i]->service()->getBroadcastTpCallCount(), 1);
        EXPECT_EQ(tp_broadcast_servers_[i]->service()->getBroadcastTpCancelCallCount(), 0);
    }
}

// 测试：broadcast 失败（所有响应失败）
TEST_F(P2PConnectorSchedulerTest, HandleRead_ReturnError_BroadcastAllFailed) {
    for (auto& server : tp_broadcast_servers_) {
        server->service()->setP2PResponseSuccess(false);
    }

    auto valid_resource = createValidKVCacheResource(2, 2);

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});

    auto deadline_ms = currentTimeMs() + 1000;

    auto success =
        scheduler_->handleRead(valid_resource, "test_broadcast_all_fail", 1003, decode_transfer_servers, deadline_ms);

    EXPECT_FALSE(success);

    // 验证 BroadcastTp 被调用
    for (size_t i = 0; i < tp_broadcast_servers_.size(); ++i) {
        EXPECT_EQ(tp_broadcast_servers_[i]->service()->getBroadcastTpCallCount(), 1);
    }
}

// ==================== asyncRead 测试 (Client 端功能) ====================

TEST_F(P2PConnectorSchedulerTest, AsyncRead_ReturnNotNull_AllSuccess) {
    auto resource = createValidKVCacheResource(2, 2);

    std::string unique_key      = "test_async_read_1";
    int64_t     request_id      = 2001;
    std::string prefill_ip      = "127.0.0.1";
    uint32_t    prefill_port    = static_cast<uint32_t>(prefill_server_->listenPort());
    int64_t     deadline_ms     = currentTimeMs() + 5000;
    auto        generate_stream = std::make_shared<MockGenerateStream>(prefill_ip, prefill_port);

    // block_range: {start_block_idx, block_count}, use -1 for block_count to include all blocks
    auto async_context = scheduler_->asyncRead(resource, request_id, unique_key, deadline_ms, generate_stream, {0, -1});
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

TEST_F(P2PConnectorSchedulerTest, AsyncRead_ReturnNull_NullResource) {
    std::string unique_key      = "test_async_read_null_resource";
    int64_t     request_id      = 2002;
    std::string prefill_ip      = "127.0.0.1";
    uint32_t    prefill_port    = static_cast<uint32_t>(prefill_server_->listenPort());
    int64_t     deadline_ms     = currentTimeMs() + 5000;
    auto        generate_stream = std::make_shared<MockGenerateStream>(prefill_ip, prefill_port);

    auto async_context = scheduler_->asyncRead(nullptr, request_id, unique_key, deadline_ms, generate_stream, {0, -1});

    EXPECT_EQ(async_context, nullptr);

    // 验证 BroadcastTp 和 StartLoad 都没有被调用
    for (size_t i = 0; i < tp_broadcast_servers_.size(); ++i) {
        EXPECT_EQ(tp_broadcast_servers_[i]->service()->getBroadcastTpCallCount(), 0);
    }
    EXPECT_EQ(prefill_server_->service()->getStartLoadCallCount(), 0);
}

TEST_F(P2PConnectorSchedulerTest, AsyncRead_ReturnNull_EmptyResource) {
    auto resource = std::make_shared<KVCacheResource>();

    std::string unique_key      = "test_async_read_empty";
    int64_t     request_id      = 2003;
    std::string prefill_ip      = "127.0.0.1";
    uint32_t    prefill_port    = static_cast<uint32_t>(prefill_server_->listenPort());
    int64_t     deadline_ms     = currentTimeMs() + 5000;
    auto        generate_stream = std::make_shared<MockGenerateStream>(prefill_ip, prefill_port);

    auto async_context = scheduler_->asyncRead(resource, request_id, unique_key, deadline_ms, generate_stream, {0, -1});

    EXPECT_EQ(async_context, nullptr);

    // 验证 BroadcastTp 和 StartLoad 都没有被调用
    for (size_t i = 0; i < tp_broadcast_servers_.size(); ++i) {
        EXPECT_EQ(tp_broadcast_servers_[i]->service()->getBroadcastTpCallCount(), 0);
    }
    EXPECT_EQ(prefill_server_->service()->getStartLoadCallCount(), 0);
}

TEST_F(P2PConnectorSchedulerTest, AsyncRead_ReturnFalse_BroadcastFailed) {
    tp_broadcast_servers_[0]->service()->setP2PResponseSuccess(false);

    auto resource = createValidKVCacheResource(2, 2);

    std::string unique_key      = "test_async_read_broadcast_fail";
    int64_t     request_id      = 2004;
    std::string prefill_ip      = "127.0.0.1";
    uint32_t    prefill_port    = static_cast<uint32_t>(prefill_server_->listenPort());
    int64_t     deadline_ms     = currentTimeMs() + 5000;
    auto        generate_stream = std::make_shared<MockGenerateStream>(prefill_ip, prefill_port);

    auto async_context = scheduler_->asyncRead(resource, request_id, unique_key, deadline_ms, generate_stream, {0, -1});
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

    std::string unique_key      = "test_async_read_load_fail";
    int64_t     request_id      = 2005;
    std::string prefill_ip      = "127.0.0.1";
    uint32_t    prefill_port    = static_cast<uint32_t>(prefill_server_->listenPort());
    int64_t     deadline_ms     = currentTimeMs() + 5000;
    auto        generate_stream = std::make_shared<MockGenerateStream>(prefill_ip, prefill_port);

    auto async_context = scheduler_->asyncRead(resource, request_id, unique_key, deadline_ms, generate_stream, {0, -1});
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

    std::string unique_key      = "test_async_read_both_fail";
    int64_t     request_id      = 2006;
    std::string prefill_ip      = "127.0.0.1";
    uint32_t    prefill_port    = static_cast<uint32_t>(prefill_server_->listenPort());
    int64_t     deadline_ms     = currentTimeMs() + 5000;
    auto        generate_stream = std::make_shared<MockGenerateStream>(prefill_ip, prefill_port);

    auto async_context = scheduler_->asyncRead(resource, request_id, unique_key, deadline_ms, generate_stream, {0, -1});
    ASSERT_NE(async_context, nullptr);

    waitAsyncContextDone(async_context);

    EXPECT_TRUE(async_context->done());
    EXPECT_FALSE(async_context->success());
}

}  // namespace rtp_llm
