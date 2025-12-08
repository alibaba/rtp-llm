#include <thread>
#include <gtest/gtest.h>
#include "grpc++/grpc++.h"

#include "autil/NetUtil.h"
#include "rtp_llm/cpp/disaggregate/p2p_connector/P2PConnectorPrefillScheduler.h"
#include "rtp_llm/cpp/disaggregate/p2p_connector/test/TestRpcServer.h"
#include "rtp_llm/cpp/cache_new/BatchKVCacheResource.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

class P2PConnectorPrefillSchedulerTest: public ::testing::Test {
protected:
    void SetUp() override {
        // 创建测试用的 RPC 服务器（用于 TPBroadcastClient）
        for (int i = 0; i < 2; ++i) {
            auto service = std::make_unique<TestRpcService>();
            auto server  = std::make_unique<TestRpcServer>(std::move(service));
            ASSERT_TRUE(server->start());
            servers_.push_back(std::move(server));
            server_addrs_.push_back("127.0.0.1:" + std::to_string(servers_.back()->listenPort()));
        }

        // 创建 GptInitParameter
        gpt_init_parameter_.worker_grpc_addrs_ = server_addrs_;

        // 创建 P2PConnectorPrefillScheduler
        scheduler_ = std::make_unique<P2PConnectorPrefillScheduler>(gpt_init_parameter_);
        ASSERT_TRUE(scheduler_->init());
    }

    void TearDown() override {
        scheduler_.reset();
        servers_.clear();
    }

    // 创建测试用的 KVCacheResourceV1（可以转换为 layer_cache_buffers）
    std::shared_ptr<KVCacheResourceV1> createValidKVCacheResource(int num_layers = 2, int blocks_per_layer = 2) {
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

    // 创建测试用的 KVCacheResourceV1（无法转换为 layer_cache_buffers，返回空）
    std::shared_ptr<KVCacheResourceV1> createInvalidKVCacheResource() {
        auto resource = std::make_shared<KVCacheResourceV1>();
        // 不设置 layer_block_ids，这样 convert 会返回空
        return resource;
    }

protected:
    std::vector<std::unique_ptr<TestRpcServer>>   servers_;
    std::vector<std::string>                      server_addrs_;
    GptInitParameter                              gpt_init_parameter_;
    std::unique_ptr<P2PConnectorPrefillScheduler> scheduler_;
};

// ---------------------------- write ----------------------------

// 测试：resource 转换不到 layer_cache_buffers
TEST_F(P2PConnectorPrefillSchedulerTest, Write_ReturnError_LayerCacheBuffersEmpty) {
    // 创建无效的 KVCacheResourceV1（无法转换为 layer_cache_buffers）
    auto invalid_resource = createInvalidKVCacheResource();

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});

    // 执行 write
    auto status = scheduler_->write(invalid_resource, "test_unique_key", 1001, decode_transfer_servers, 10000);

    // 应该返回错误状态
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.error_code(), grpc::StatusCode::INTERNAL);
    EXPECT_EQ(status.error_message(), "layer_cache_buffers is empty");

    // 验证 BroadcastTp 没有被调用（因为转换失败，不会调用 broadcast）
    for (size_t i = 0; i < servers_.size(); ++i) {
        EXPECT_EQ(servers_[i]->service()->getBroadcastTpCallCount(), 0);
    }
}

// 测试：broadcast 成功
TEST_F(P2PConnectorPrefillSchedulerTest, Write_ReturnOK_BroadcastSuccess) {
    // 创建有效的 KVCacheResourceV1
    auto valid_resource = createValidKVCacheResource(2, 2);

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});
    decode_transfer_servers.push_back({"127.0.0.1", 12346});

    // 执行 write
    auto status = scheduler_->write(valid_resource, "test_broadcast_success", 1001, decode_transfer_servers, 10000);

    // 应该返回成功状态
    EXPECT_TRUE(status.ok());

    // 验证 BroadcastTp 被调用（每个服务器应该被调用一次）
    for (size_t i = 0; i < servers_.size(); ++i) {
        EXPECT_EQ(servers_[i]->service()->getBroadcastTpCallCount(), 1);
        EXPECT_EQ(servers_[i]->service()->getBroadcastTpCancelCallCount(), 0);
    }
}

// 测试：broadcast 失败（部分响应失败）
TEST_F(P2PConnectorPrefillSchedulerTest, Write_ReturnError_BroadcastPartialFailed) {
    // 设置第一个服务器返回失败
    servers_[0]->service()->setP2PResponseSuccess(false);

    // 创建有效的 KVCacheResourceV1
    auto valid_resource = createValidKVCacheResource(2, 2);

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});

    // 执行 write
    auto status =
        scheduler_->write(valid_resource, "test_broadcast_partial_fail", 1002, decode_transfer_servers, 10000);

    // 应该返回错误状态（因为 broadcast result success() 返回 false）
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.error_code(), grpc::StatusCode::INTERNAL);
    EXPECT_EQ(status.error_message(), "broadcast result wait done failed");

    // 验证 BroadcastTp 被调用
    for (size_t i = 0; i < servers_.size(); ++i) {
        EXPECT_EQ(servers_[i]->service()->getBroadcastTpCallCount(), 1);
        EXPECT_EQ(servers_[i]->service()->getBroadcastTpCancelCallCount(), 0);
    }
}

// 测试：broadcast 失败（所有响应失败）
TEST_F(P2PConnectorPrefillSchedulerTest, Write_ReturnError_BroadcastAllFailed) {
    // 设置所有服务器返回失败
    for (auto& server : servers_) {
        server->service()->setP2PResponseSuccess(false);
    }

    // 创建有效的 KVCacheResourceV1
    auto valid_resource = createValidKVCacheResource(2, 2);

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});

    // 执行 write
    auto status = scheduler_->write(valid_resource, "test_broadcast_all_fail", 1003, decode_transfer_servers, 10000);

    // 应该返回错误状态
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.error_code(), grpc::StatusCode::INTERNAL);
    EXPECT_EQ(status.error_message(), "broadcast result wait done failed");

    // 验证 BroadcastTp 被调用
    for (size_t i = 0; i < servers_.size(); ++i) {
        EXPECT_EQ(servers_[i]->service()->getBroadcastTpCallCount(), 1);
        EXPECT_EQ(servers_[i]->service()->getBroadcastTpCancelCallCount(), 0);
    }
}

// 测试：broadcast 失败（broadcast 返回 nullptr）
TEST_F(P2PConnectorPrefillSchedulerTest, Write_ReturnError_BroadcastReturnNull) {
    // 设置 RPC 响应状态为错误，导致 broadcast 可能返回 nullptr 或失败
    // 注意：根据 TPBroadcastClient 的实现，如果 broadcast 返回 nullptr，write 应该返回错误
    // 但实际测试中，broadcast 可能不会返回 nullptr，而是返回一个失败的 result
    // 这里我们测试 RPC 状态失败的情况
    servers_[0]->service()->setRpcResponseStatus(::grpc::Status(grpc::StatusCode::INTERNAL, "Internal error"));

    // 创建有效的 KVCacheResourceV1
    auto valid_resource = createValidKVCacheResource(2, 2);

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});

    // 执行 write
    auto status = scheduler_->write(valid_resource, "test_broadcast_rpc_fail", 1004, decode_transfer_servers, 10000);

    // 根据实现，如果 broadcast result->success() 返回 false，应该返回错误
    // 如果 RPC 失败，result->success() 应该返回 false
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.error_code(), grpc::StatusCode::INTERNAL);

    // 验证 BroadcastTp 被调用
    EXPECT_EQ(servers_[0]->service()->getBroadcastTpCallCount(), 1);
    EXPECT_EQ(servers_[0]->service()->getBroadcastTpCancelCallCount(), 0);
}

}  // namespace rtp_llm
