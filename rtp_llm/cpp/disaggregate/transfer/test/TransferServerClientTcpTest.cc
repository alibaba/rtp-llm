#pragma once

#include "gtest/gtest.h"
#include "rtp_llm/cpp/disaggregate/transfer/test/TransferServerClientTest.h"

namespace rtp_llm {

class TransferServerClientTcpTest: public TransferServerClientTest {
protected:
    void SetUp() override {
        TransferServerClientTest::SetUp();

        transfer_server_ = std::make_shared<TransferServer>(mock_server_layer_block_convertor_, device_);
        listen_port_     = autil::NetUtil::randomPort();
        bool server_init = transfer_server_->init(false, listen_port_, 2, 4, 0, 0, 0, 5000);
        ASSERT_TRUE(server_init);

        transfer_client_ = std::make_shared<TransferClient>(mock_client_layer_block_convertor_, device_);
        bool client_init = transfer_client_->init(false, 2, 0, 0);
        ASSERT_TRUE(client_init);
    }

    void TearDown() override {
        if (transfer_server_) {
            transfer_server_.reset();
        }
        if (transfer_client_) {
            transfer_client_.reset();
        }
        TransferServerClientTest::TearDown();
    }
};

TEST_F(TransferServerClientTcpTest, RegisterUserMrTcpTest) {
    auto buffer = createTestBuffer(1024);
    EXPECT_TRUE(transfer_client_->registerUserMr(buffer, 0));
    EXPECT_TRUE(transfer_server_->registerUserMr(buffer, 0));
}

TEST_F(TransferServerClientTcpTest, TransferSuccessTest) {
    std::string unique_key = "test_key_2";

    // 在 Server 端预先添加任务（包含多个层）
    auto task_store = transfer_server_->getLayerCacheBufferTaskStore();
    auto buffers =
        std::map<int, std::shared_ptr<LayerCacheBuffer>>{{0, layer_cache_buffer0_}, {1, layer_cache_buffer1_}};
    int64_t deadline_ms = currentTimeMs() + 5000;
    auto    task        = task_store->addTask(unique_key, buffers, deadline_ms);
    ASSERT_NE(task, nullptr);

    // 执行 transfer（传输第一层）
    std::atomic<bool> first_transfer_complete(false);
    std::atomic<bool> first_transfer_success(false);
    std::atomic<bool> second_transfer_complete(false);
    std::atomic<bool> second_transfer_success(false);

    std::string server_ip = "127.0.0.1";
    transfer_client_->transfer(
        server_ip,
        listen_port_,
        unique_key,
        layer_cache_buffer0_,
        1,
        0,
        1,
        0,
        [&first_transfer_complete, &first_transfer_success](bool success) {
            first_transfer_complete = true;
            first_transfer_success  = success;
        },
        5000);

    transfer_client_->transfer(
        server_ip,
        listen_port_,
        unique_key,
        layer_cache_buffer1_,
        1,
        0,
        1,
        0,
        [&second_transfer_complete, &second_transfer_success](bool success) {
            second_transfer_complete = true;
            second_transfer_success  = success;
        },
        5000);

    // 等待传输完成
    int wait_count = 0;
    while (!first_transfer_complete && !second_transfer_complete && wait_count < 50) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        wait_count++;
    }

    EXPECT_TRUE(wait_count < 50);

    EXPECT_TRUE(first_transfer_complete);
    EXPECT_TRUE(first_transfer_success);
    EXPECT_TRUE(second_transfer_complete);
    EXPECT_TRUE(second_transfer_success);

    task->waitDone();
    EXPECT_TRUE(task->success());

    verifyBufferContent(layer_cache_buffer0_, 'A', mock_server_layer_block_convertor_);
    verifyBufferContent(layer_cache_buffer1_, 'C', mock_server_layer_block_convertor_);

    task_store->stealTask(unique_key);
}

TEST_F(TransferServerClientTcpTest, ClientLayerBlockConvertorFailedTest) {
    std::string unique_key = "test_key_4";

    // 在 Server 端预先添加任务（包含多个层）
    auto    task_store  = transfer_server_->getLayerCacheBufferTaskStore();
    auto    buffers     = std::map<int, std::shared_ptr<LayerCacheBuffer>>{{0, layer_cache_buffer0_}};
    int64_t deadline_ms = currentTimeMs() + 5000;
    auto    task        = task_store->addTask(unique_key, buffers, deadline_ms);
    ASSERT_NE(task, nullptr);

    // 移除 Client 的 LayerBlockConvertor 中的 buffer
    mock_client_layer_block_convertor_->removeBuffer(layer_cache_buffer0_->getLayerId(), 1);

    // 执行 transfer（传输第一层）
    std::atomic<bool> transfer_complete(false);
    std::atomic<bool> transfer_success(false);

    std::string server_ip = "127.0.0.1";
    transfer_client_->transfer(
        server_ip,
        listen_port_,
        unique_key,
        layer_cache_buffer0_,
        1,
        0,
        1,
        0,
        [&transfer_complete, &transfer_success](bool success) {
            transfer_complete = true;
            transfer_success  = success;
        },
        5000);

    // 等待传输完成
    int wait_count = 0;
    while (!transfer_complete && wait_count < 50) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        wait_count++;
    }

    EXPECT_TRUE(wait_count < 50);

    EXPECT_TRUE(transfer_complete);
    EXPECT_FALSE(transfer_success);

    task->waitDone();
    EXPECT_FALSE(task->success());

    task_store->stealTask(unique_key);
}

TEST_F(TransferServerClientTcpTest, ConnectFailedTest) {
    std::string unique_key = "test_key_3";

    // 在 Server 端预先添加任务（包含多个层）
    auto    task_store  = transfer_server_->getLayerCacheBufferTaskStore();
    auto    buffers     = std::map<int, std::shared_ptr<LayerCacheBuffer>>{{0, layer_cache_buffer0_}};
    int64_t deadline_ms = currentTimeMs() + 5000;
    auto    task        = task_store->addTask(unique_key, buffers, deadline_ms);
    ASSERT_NE(task, nullptr);

    // 执行 transfer（传输第一层）
    std::atomic<bool> transfer_complete(false);
    std::atomic<bool> transfer_success(false);

    std::string server_ip = "127.0.0.1";
    transfer_client_->transfer(
        server_ip,
        autil::NetUtil::randomPort(),
        unique_key,
        layer_cache_buffer0_,
        1,
        0,
        1,
        0,
        [&transfer_complete, &transfer_success](bool success) {
            transfer_complete = true;
            transfer_success  = success;
        },
        2000);

    // 等待传输完成
    int wait_count = 0;
    while (!transfer_complete && wait_count < 50) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        wait_count++;
    }

    EXPECT_TRUE(wait_count < 50);

    EXPECT_TRUE(transfer_complete);
    EXPECT_FALSE(transfer_success);

    task->waitDone();
    EXPECT_FALSE(task->success());

    task_store->stealTask(unique_key);
}

TEST_F(TransferServerClientTcpTest, TransferTimeoutTest) {
    std::string unique_key = "test_key_5";

    // 在 Server 端预先添加任务（包含多个层）
    auto    task_store  = transfer_server_->getLayerCacheBufferTaskStore();
    auto    buffers     = std::map<int, std::shared_ptr<LayerCacheBuffer>>{{0, layer_cache_buffer0_}};
    int64_t deadline_ms = currentTimeMs() + 5000;
    auto    task        = task_store->addTask(unique_key, buffers, deadline_ms);
    ASSERT_NE(task, nullptr);

    // 执行 transfer（传输第一层）
    std::atomic<bool> transfer_complete(false);
    std::atomic<bool> transfer_success(false);

    std::string server_ip = "127.0.0.1";
    transfer_client_->transfer(
        server_ip,
        listen_port_,
        unique_key,
        layer_cache_buffer0_,
        1,
        0,
        1,
        0,
        [&transfer_complete, &transfer_success](bool success) {
            transfer_complete = true;
            transfer_success  = success;
        },
        2);  // 20ms timeout

    // 等待传输完成
    int wait_count = 0;
    while (!transfer_complete && wait_count < 50) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        wait_count++;
    }

    EXPECT_TRUE(wait_count < 50);

    EXPECT_TRUE(transfer_complete);
    EXPECT_FALSE(transfer_success);

    task->waitDone();
    //! 注意这里 client 侧认为任务完成了. 因为 server 侧认为任务超时了, 没有通知 client 侧. 这种情况下的正确性需要结合
    //! client 侧的实现来保证
    EXPECT_TRUE(task->success());

    task_store->stealTask(unique_key);
}

TEST_F(TransferServerClientTcpTest, ServerLayerBlockConvertorFailedTest) {
    std::string unique_key = "test_key_6";

    // 在 Server 端预先添加任务（包含多个层）
    auto    task_store  = transfer_server_->getLayerCacheBufferTaskStore();
    auto    buffers     = std::map<int, std::shared_ptr<LayerCacheBuffer>>{{0, layer_cache_buffer0_}};
    int64_t deadline_ms = currentTimeMs() + 5000;
    auto    task        = task_store->addTask(unique_key, buffers, deadline_ms);
    ASSERT_NE(task, nullptr);

    // 移除 Server 的 LayerBlockConvertor 中的 buffer
    mock_server_layer_block_convertor_->removeBuffer(layer_cache_buffer0_->getLayerId(), 1);

    // 执行 transfer（传输第一层）
    std::atomic<bool> transfer_complete(false);
    std::atomic<bool> transfer_success(false);

    std::string server_ip = "127.0.0.1";
    transfer_client_->transfer(
        server_ip,
        listen_port_,
        unique_key,
        layer_cache_buffer0_,
        1,
        0,
        1,
        0,
        [&transfer_complete, &transfer_success](bool success) {
            transfer_complete = true;
            transfer_success  = success;
        },
        5000);

    // 等待传输完成
    int wait_count = 0;
    while (!transfer_complete && wait_count < 50) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        wait_count++;
    }

    EXPECT_TRUE(wait_count < 50);

    EXPECT_TRUE(transfer_complete);
    EXPECT_FALSE(transfer_success);

    task->waitDone();
    EXPECT_FALSE(task->success());

    task_store->stealTask(unique_key);
}

TEST_F(TransferServerClientTcpTest, testServerBufferMismatchTest) {
    std::string unique_key = "test_key_2";

    // 在 Server 端预先添加任务（包含多个层）
    auto server_layer_cache_buffer = createLayerCacheBuffer(0, 3);
    addBufferToConvertor(server_layer_cache_buffer, mock_server_layer_block_convertor_);

    auto    task_store  = transfer_server_->getLayerCacheBufferTaskStore();
    auto    buffers     = std::map<int, std::shared_ptr<LayerCacheBuffer>>{{0, server_layer_cache_buffer}};
    int64_t deadline_ms = currentTimeMs() + 5000;
    auto    task        = task_store->addTask(unique_key, buffers, deadline_ms);
    ASSERT_NE(task, nullptr);

    // 执行 transfer（传输第一层）
    std::atomic<bool> first_transfer_complete(false);
    std::atomic<bool> first_transfer_success(false);

    std::string server_ip = "127.0.0.1";
    transfer_client_->transfer(
        server_ip,
        listen_port_,
        unique_key,
        layer_cache_buffer0_,
        1,
        0,
        1,
        0,
        [&first_transfer_complete, &first_transfer_success](bool success) {
            first_transfer_complete = true;
            first_transfer_success  = success;
        },
        5000);

    // 等待传输完成
    int wait_count = 0;
    while (!first_transfer_complete && wait_count < 50) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        wait_count++;
    }
    EXPECT_TRUE(wait_count < 50);

    EXPECT_TRUE(first_transfer_complete);
    EXPECT_FALSE(first_transfer_success);

    task->waitDone();
    EXPECT_FALSE(task->success());

    task_store->stealTask(unique_key);
}

}  // namespace rtp_llm