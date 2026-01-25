#include "gtest/gtest.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/test/TransferServerClientTest.h"
#include <thread>
#include <chrono>
#include <atomic>
#include <map>

namespace rtp_llm {

class TransferServerClientRdmaTest: public TransferServerClientTest {
protected:
    void SetUp() override {
        TransferServerClientTest::SetUp();

        transfer_server_ = std::make_shared<TransferServer>(mock_server_layer_block_convertor_, device_);
        listen_port_     = autil::NetUtil::randomPort();
        bool server_init = transfer_server_->init(true,  // use_rdma = true
                                                  listen_port_,
                                                  2,    // tcp_io_thread_count
                                                  4,    // tcp_worker_thread_count
                                                  2,    // rdma_io_thread_count
                                                  4,    // rdma_worker_thread_count
                                                  1,    // rdma_connections_per_host
                                                  5000  // connect_timeout_ms
        );
        ASSERT_TRUE(server_init);

        transfer_client_ = std::make_shared<TransferClient>(mock_client_layer_block_convertor_, device_);
        bool client_init = transfer_client_->init(true,  // use_rdma = true
                                                  2,     // tcp_io_thread_count
                                                  2,     // rdma_io_thread_count
                                                  4      // rdma_worker_thread_count
        );
        ASSERT_TRUE(client_init);

        // 等待服务器启动
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // 注册 RDMA MR：为所有 buffer 注册 MR
        registerBuffersForRdma();
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

private:
    // 为所有 buffer 注册 RDMA MR
    void registerBuffersForRdma() {
        // 为 Client 的 buffer 注册 MR
        // 通过 convertIndexToBuffer 获取 buffer 并注册
        for (const auto& [cache_key, block_id] : layer_cache_buffer0_->blockIdMap()) {
            auto buffers = mock_client_layer_block_convertor_->convertIndexToBuffer(
                layer_cache_buffer0_->getLayerId(), block_id, 1, 0);
            for (const auto& buffer : buffers) {
                ASSERT_TRUE(transfer_client_->registerUserMr(buffer, 0));
            }
        }
        for (const auto& [cache_key, block_id] : layer_cache_buffer1_->blockIdMap()) {
            auto buffers = mock_client_layer_block_convertor_->convertIndexToBuffer(
                layer_cache_buffer1_->getLayerId(), block_id, 1, 0);
            for (const auto& buffer : buffers) {
                ASSERT_TRUE(transfer_client_->registerUserMr(buffer, 0));
            }
        }

        // 为 Server 的 buffer 注册 MR
        for (const auto& [cache_key, block_id] : layer_cache_buffer0_->blockIdMap()) {
            auto buffers = mock_server_layer_block_convertor_->convertIndexToBuffer(
                layer_cache_buffer0_->getLayerId(), block_id, 1, 0);
            for (const auto& buffer : buffers) {
                ASSERT_TRUE(transfer_server_->registerUserMr(buffer, 0));
            }
        }
        for (const auto& [cache_key, block_id] : layer_cache_buffer1_->blockIdMap()) {
            auto buffers = mock_server_layer_block_convertor_->convertIndexToBuffer(
                layer_cache_buffer1_->getLayerId(), block_id, 1, 0);
            for (const auto& buffer : buffers) {
                ASSERT_TRUE(transfer_server_->registerUserMr(buffer, 0));
            }
        }
    }
};

TEST_F(TransferServerClientRdmaTest, RegisterUserMrRdmaTest) {
    auto buffer = createTestBuffer(1024);
    EXPECT_TRUE(transfer_client_->registerUserMr(buffer, 0));
    EXPECT_TRUE(transfer_server_->registerUserMr(buffer, 0));
}

TEST_F(TransferServerClientRdmaTest, TransferSuccessTest) {
    std::string unique_key = "test_key_2";

    // 在 Server 端预先添加任务（包含多个层）
    auto task_store = transfer_server_->getTransferTaskStore();
    auto buffers =
        std::map<int, std::shared_ptr<LayerCacheBuffer>>{{0, layer_cache_buffer0_}, {1, layer_cache_buffer1_}};
    int64_t deadline_ms = currentTimeMs() + 5000;
    auto    task        = task_store->addTask(unique_key, buffers, deadline_ms);
    ASSERT_NE(task, nullptr);

    // 执行 transfer（传输第一层）
    std::atomic<bool> first_transfer_complete(false);
    ErrorCode         first_error_code = ErrorCode::NONE_ERROR;
    std::atomic<bool> second_transfer_complete(false);
    ErrorCode         second_error_code = ErrorCode::NONE_ERROR;

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
        [&first_transfer_complete, &first_error_code](ErrorCode error_code, const std::string&) {
            first_transfer_complete = true;
            first_error_code        = error_code;
        },
        currentTimeMs() + 5000);

    transfer_client_->transfer(
        server_ip,
        listen_port_,
        unique_key,
        layer_cache_buffer1_,
        1,
        0,
        1,
        0,
        [&second_transfer_complete, &second_error_code](ErrorCode error_code, const std::string&) {
            second_transfer_complete = true;
            second_error_code        = error_code;
        },
        currentTimeMs() + 5000);

    // 等待传输完成
    int wait_count = 0;
    while (!first_transfer_complete && !second_transfer_complete && wait_count < 50) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        wait_count++;
    }

    EXPECT_TRUE(wait_count < 50);

    EXPECT_TRUE(first_transfer_complete);
    EXPECT_EQ(first_error_code, ErrorCode::NONE_ERROR);
    EXPECT_TRUE(second_transfer_complete);
    EXPECT_EQ(second_error_code, ErrorCode::NONE_ERROR);

    task->waitDone();
    EXPECT_TRUE(task->success());

    task_store->stealTask(unique_key);
}

TEST_F(TransferServerClientRdmaTest, ClientLayerBlockConvertorFailedTest) {
    std::string unique_key = "test_key_4";

    // 在 Server 端预先添加任务（包含多个层）
    auto    task_store  = transfer_server_->getTransferTaskStore();
    auto    buffers     = std::map<int, std::shared_ptr<LayerCacheBuffer>>{{0, layer_cache_buffer0_}};
    int64_t deadline_ms = currentTimeMs() + 5000;
    auto    task        = task_store->addTask(unique_key, buffers, deadline_ms);
    ASSERT_NE(task, nullptr);

    // 移除 Client 的 LayerBlockConvertor 中的 buffer
    mock_client_layer_block_convertor_->removeBuffer(layer_cache_buffer0_->getLayerId(), 1);

    // 执行 transfer（传输第一层）
    std::atomic<bool> transfer_complete(false);
    ErrorCode         actual_error_code = ErrorCode::NONE_ERROR;

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
        [&transfer_complete, &actual_error_code](ErrorCode error_code, const std::string&) {
            transfer_complete = true;
            actual_error_code = error_code;
        },
        currentTimeMs() + 5000);

    // 等待传输完成
    int wait_count = 0;
    while (!transfer_complete && wait_count < 50) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        wait_count++;
    }

    EXPECT_TRUE(wait_count < 50);

    EXPECT_TRUE(transfer_complete);
    EXPECT_FALSE(transfer_success);
    EXPECT_NE(actual_error_code, ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_TRANSFER_FAILED);

    task->waitDone();
    EXPECT_FALSE(task->success());

    task_store->stealTask(unique_key);
}

TEST_F(TransferServerClientRdmaTest, ConnectFailedTest) {
    std::string unique_key = "test_key_3";

    // 在 Server 端预先添加任务（包含多个层）
    auto    task_store  = transfer_server_->getTransferTaskStore();
    auto    buffers     = std::map<int, std::shared_ptr<LayerCacheBuffer>>{{0, layer_cache_buffer0_}};
    int64_t deadline_ms = currentTimeMs() + 5000;
    auto    task        = task_store->addTask(unique_key, buffers, deadline_ms);
    ASSERT_NE(task, nullptr);

    // 执行 transfer（传输第一层）
    std::atomic<bool> transfer_complete(false);
    std::atomic<bool> transfer_success(false);
    ErrorCode         actual_error_code = ErrorCode::NONE_ERROR;

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
        [&transfer_complete, &transfer_success, &actual_error_code](ErrorCode error_code, const std::string&) {
            transfer_complete = true;
            actual_error_code = error_code;
            transfer_success  = (error_code == ErrorCode::NONE_ERROR);
        },
        currentTimeMs() + 2000);

    // 等待传输完成
    int wait_count = 0;
    while (!transfer_complete && wait_count < 50) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        wait_count++;
    }

    EXPECT_TRUE(wait_count < 50);

    EXPECT_TRUE(transfer_complete);
    EXPECT_FALSE(transfer_success);
    // 验证连接失败时返回的错误码
    EXPECT_EQ(actual_error_code, ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_TRANSFER_FAILED);

    task->waitDone();
    EXPECT_FALSE(task->success());

    task_store->stealTask(unique_key);
}

TEST_F(TransferServerClientRdmaTest, TransferTimeoutTest) {
    std::string unique_key = "test_key_5";

    // 在 Server 端预先添加任务（包含多个层）
    auto    task_store  = transfer_server_->getTransferTaskStore();
    auto    buffers     = std::map<int, std::shared_ptr<LayerCacheBuffer>>{{0, layer_cache_buffer0_}};
    int64_t deadline_ms = currentTimeMs() + 5000;
    auto    task        = task_store->addTask(unique_key, buffers, deadline_ms);
    ASSERT_NE(task, nullptr);

    // 执行 transfer（传输第一层）
    std::atomic<bool> transfer_complete(false);
    ErrorCode         actual_error_code = ErrorCode::NONE_ERROR;

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
        [&transfer_complete, &actual_error_code](ErrorCode error_code, const std::string&) {
            transfer_complete = true;
            actual_error_code = error_code;
        },
        currentTimeMs() + 2);  // 2ms timeout

    // 等待传输完成
    int wait_count = 0;
    while (!transfer_complete && wait_count < 50) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        wait_count++;
    }

    EXPECT_TRUE(wait_count < 50);

    EXPECT_TRUE(transfer_complete);
    EXPECT_NE(actual_error_code, ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_TRANSFER_TIMEOUT);

    task->waitDone();
    //! 注意这里 client 侧认为任务完成了. 因为 server 侧认为任务超时了, 没有通知 client 侧. 这种情况下的正确性需要结合
    //! client 侧的实现来保证
    EXPECT_TRUE(task->success());

    task_store->stealTask(unique_key);
}

TEST_F(TransferServerClientRdmaTest, ServerLayerBlockConvertorFailedTest) {
    std::string unique_key = "test_key_6";

    // 在 Server 端预先添加任务（包含多个层）
    auto    task_store  = transfer_server_->getTransferTaskStore();
    auto    buffers     = std::map<int, std::shared_ptr<LayerCacheBuffer>>{{0, layer_cache_buffer0_}};
    int64_t deadline_ms = currentTimeMs() + 5000;
    auto    task        = task_store->addTask(unique_key, buffers, deadline_ms);
    ASSERT_NE(task, nullptr);

    // 移除 Server 的 LayerBlockConvertor 中的 buffer
    mock_server_layer_block_convertor_->removeBuffer(layer_cache_buffer0_->getLayerId(), 1);

    // 执行 transfer（传输第一层）
    std::atomic<bool> transfer_complete(false);
    ErrorCode         actual_error_code = ErrorCode::NONE_ERROR;

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
        [&transfer_complete, &actual_error_code](ErrorCode error_code, const std::string&) {
            transfer_complete = true;
            actual_error_code = error_code;
        },
        currentTimeMs() + 5000);

    // 等待传输完成
    int wait_count = 0;
    while (!transfer_complete && wait_count < 50) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        wait_count++;
    }

    EXPECT_TRUE(wait_count < 50);

    EXPECT_TRUE(transfer_complete);
    EXPECT_EQ(actual_error_code, ErrorCode::P2P_CONNECTOR_WORKER_READ_BUFFER_MISMATCH);

    task->waitDone();
    EXPECT_FALSE(task->success());

    task_store->stealTask(unique_key);
}

}  // namespace rtp_llm
