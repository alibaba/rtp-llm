#include <thread>
#include <gtest/gtest.h>
#include "grpc++/grpc++.h"

#include "autil/NetUtil.h"
#include "rtp_llm/cpp/disaggregate/p2p_connector/P2PConnectorDecodeScheduler.h"
#include "rtp_llm/cpp/cache_new/BatchKVCacheResource.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/disaggregate/p2p_connector/test/TestRpcServer.h"

namespace rtp_llm {

class P2PConnectorDecodeSchedulerTest: public ::testing::Test {
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

        // 创建测试用的 RPC 服务器（用于 PrefillLoadClient）
        auto prefill_service = std::make_unique<TestRpcService>();
        prefill_server_      = std::make_unique<TestRpcServer>(std::move(prefill_service));
        ASSERT_TRUE(prefill_server_->start());
        prefill_addr_ = "127.0.0.1:" + std::to_string(prefill_server_->listenPort());

        // 创建 GptInitParameter
        gpt_init_parameter_.worker_grpc_addrs_ = tp_broadcast_addrs_;
        // worker_addrs_ 格式: "ip:cache_store_port:grpc_port"
        gpt_init_parameter_.worker_addrs_.push_back("127.0.0.1:12345:" + std::to_string(prefill_server_->listenPort()));

        // 创建 P2PConnectorDecodeScheduler
        scheduler_ = std::make_unique<P2PConnectorDecodeScheduler>(gpt_init_parameter_);
        ASSERT_TRUE(scheduler_->init());
    }

    void TearDown() override {
        scheduler_.reset();
        tp_broadcast_servers_.clear();
        prefill_server_.reset();
    }

    // 创建测试用的 KVCacheResourceV1
    std::shared_ptr<KVCacheResourceV1> createKVCacheResource(int num_layers = 2, int blocks_per_layer = 2) {
        auto resource = std::make_shared<KVCacheResourceV1>();

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

protected:
    std::vector<std::unique_ptr<TestRpcServer>>  tp_broadcast_servers_;
    std::vector<std::string>                     tp_broadcast_addrs_;
    std::unique_ptr<TestRpcServer>               prefill_server_;
    std::string                                  prefill_addr_;
    GptInitParameter                             gpt_init_parameter_;
    std::unique_ptr<P2PConnectorDecodeScheduler> scheduler_;
};

// ---------------------------- asyncRead ----------------------------

TEST_F(P2PConnectorDecodeSchedulerTest, AsyncRead_ReturnNotNull_AllSuccess) {
    auto resource = createKVCacheResource(2, 2);

    std::string unique_key   = "test_async_read_1";
    int64_t     request_id   = 1001;
    std::string prefill_ip   = "127.0.0.1";
    uint32_t    prefill_port = static_cast<uint32_t>(prefill_server_->listenPort());
    int64_t     deadline_ms  = currentTimeMs() + 5000;

    // 执行 asyncRead
    auto async_context = scheduler_->asyncRead(resource, request_id, unique_key, prefill_ip, prefill_port, deadline_ms);
    ASSERT_NE(async_context, nullptr);

    // 等待完成
    async_context->waitDone();

    // 验证结果
    EXPECT_TRUE(async_context->success());

    // 验证 BroadcastTp 和 StartLoad 都被调用
    for (size_t i = 0; i < tp_broadcast_servers_.size(); ++i) {
        EXPECT_EQ(tp_broadcast_servers_[i]->service()->getBroadcastTpCallCount(), 1);
        EXPECT_EQ(tp_broadcast_servers_[i]->service()->getBroadcastTpCancelCallCount(), 0);
    }
    EXPECT_EQ(prefill_server_->service()->getStartLoadCallCount(), 1);
}

TEST_F(P2PConnectorDecodeSchedulerTest, AsyncRead_ReturnNull_NullResource) {
    std::string unique_key   = "test_async_read_null_resource";
    int64_t     request_id   = 1002;
    std::string prefill_ip   = "127.0.0.1";
    uint32_t    prefill_port = static_cast<uint32_t>(prefill_server_->listenPort());
    int64_t     deadline_ms  = currentTimeMs() + 5000;

    // 执行 asyncRead，传入 nullptr
    auto async_context = scheduler_->asyncRead(nullptr, request_id, unique_key, prefill_ip, prefill_port, deadline_ms);

    // 应该返回 nullptr
    EXPECT_EQ(async_context, nullptr);

    // 验证 BroadcastTp 和 StartLoad 都没有被调用（因为 resource 为 null）
    for (size_t i = 0; i < tp_broadcast_servers_.size(); ++i) {
        EXPECT_EQ(tp_broadcast_servers_[i]->service()->getBroadcastTpCallCount(), 0);
        EXPECT_EQ(tp_broadcast_servers_[i]->service()->getBroadcastTpCancelCallCount(), 0);
    }
    EXPECT_EQ(prefill_server_->service()->getStartLoadCallCount(), 0);
}

TEST_F(P2PConnectorDecodeSchedulerTest, AsyncRead_ReturnNull_EmptyResource) {
    // 创建空的 resource（没有 layer_block_ids）
    auto resource = std::make_shared<KVCacheResourceV1>();

    std::string unique_key   = "test_async_read_empty";
    int64_t     request_id   = 1003;
    std::string prefill_ip   = "127.0.0.1";
    uint32_t    prefill_port = static_cast<uint32_t>(prefill_server_->listenPort());
    int64_t     deadline_ms  = currentTimeMs() + 5000;

    // 执行 asyncRead
    auto async_context = scheduler_->asyncRead(resource, request_id, unique_key, prefill_ip, prefill_port, deadline_ms);

    // 应该返回 nullptr（因为 layer_cache_buffers 为空）
    EXPECT_EQ(async_context, nullptr);

    // 验证 BroadcastTp 和 StartLoad 都没有被调用（因为 layer_cache_buffers 为空）
    for (size_t i = 0; i < tp_broadcast_servers_.size(); ++i) {
        EXPECT_EQ(tp_broadcast_servers_[i]->service()->getBroadcastTpCallCount(), 0);
        EXPECT_EQ(tp_broadcast_servers_[i]->service()->getBroadcastTpCancelCallCount(), 0);
    }
    EXPECT_EQ(prefill_server_->service()->getStartLoadCallCount(), 0);
}

TEST_F(P2PConnectorDecodeSchedulerTest, AsyncRead_ReturnFalse_BroadcastFailed) {
    // 设置第一个服务器返回失败
    tp_broadcast_servers_[0]->service()->setP2PResponseSuccess(false);

    auto resource = createKVCacheResource(2, 2);

    std::string unique_key   = "test_async_read_broadcast_fail";
    int64_t     request_id   = 1004;
    std::string prefill_ip   = "127.0.0.1";
    uint32_t    prefill_port = static_cast<uint32_t>(prefill_server_->listenPort());
    int64_t     deadline_ms  = currentTimeMs() + 5000;

    // 执行 asyncRead
    auto async_context = scheduler_->asyncRead(resource, request_id, unique_key, prefill_ip, prefill_port, deadline_ms);
    ASSERT_NE(async_context, nullptr);

    // 等待完成
    async_context->waitDone();

    // 验证结果（broadcast 失败，所以整体应该失败）
    EXPECT_FALSE(async_context->success());

    // 验证 BroadcastTp 被调用，StartLoad 也被调用（即使 broadcast 失败，load 也会执行）
    for (size_t i = 0; i < tp_broadcast_servers_.size(); ++i) {
        EXPECT_EQ(tp_broadcast_servers_[i]->service()->getBroadcastTpCallCount(), 1);
        EXPECT_EQ(tp_broadcast_servers_[i]->service()->getBroadcastTpCancelCallCount(), 0);
    }
    EXPECT_EQ(prefill_server_->service()->getStartLoadCallCount(), 1);
}

TEST_F(P2PConnectorDecodeSchedulerTest, AsyncRead_ReturnFalse_LoadFailed) {
    // 设置 prefill 服务器返回失败
    prefill_server_->service()->setStartLoadResponseSuccess(false);

    auto resource = createKVCacheResource(2, 2);

    std::string unique_key   = "test_async_read_load_fail";
    int64_t     request_id   = 1005;
    std::string prefill_ip   = "127.0.0.1";
    uint32_t    prefill_port = static_cast<uint32_t>(prefill_server_->listenPort());
    int64_t     deadline_ms  = currentTimeMs() + 5000;

    // 执行 asyncRead
    auto async_context = scheduler_->asyncRead(resource, request_id, unique_key, prefill_ip, prefill_port, deadline_ms);
    ASSERT_NE(async_context, nullptr);

    // 等待完成
    async_context->waitDone();

    // 验证结果（load 失败，所以整体应该失败）
    EXPECT_FALSE(async_context->success());

    // 验证 BroadcastTp 和 StartLoad 都被调用
    for (size_t i = 0; i < tp_broadcast_servers_.size(); ++i) {
        EXPECT_EQ(tp_broadcast_servers_[i]->service()->getBroadcastTpCallCount(), 1);
        EXPECT_EQ(tp_broadcast_servers_[i]->service()->getBroadcastTpCancelCallCount(), 0);
    }
    EXPECT_EQ(prefill_server_->service()->getStartLoadCallCount(), 1);
}

}  // namespace rtp_llm
