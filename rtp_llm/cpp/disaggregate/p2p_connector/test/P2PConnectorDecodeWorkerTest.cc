#include <thread>
#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <map>

#include "rtp_llm/cpp/disaggregate/p2p_connector/P2PConnectorDecodeWorker.h"
#include "rtp_llm/cpp/disaggregate/transfer/LayerCacheBuffer.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/cache_new/KVCacheAllocator.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"

namespace rtp_llm {

class P2PConnectorDecodeWorkerTest: public ::testing::Test {
protected:
    void SetUp() override {
        // 创建测试用的 GptInitParameter
        gpt_init_parameter_.cache_store_rdma_mode_   = false;
        gpt_init_parameter_.cache_store_listen_port_ = 0;

        // 创建测试用的 DeviceBase (使用 nullptr，因为测试不依赖实际设备)
        device_base_ = nullptr;

        // 创建测试用的 KVCacheAllocator (使用 nullptr，因为测试不依赖实际分配器)
        kv_cache_allocator_ = nullptr;

        // 创建 P2PConnectorDecodeWorker（不调用 init，避免创建 TransferServer）
        worker_ = std::make_unique<P2PConnectorDecodeWorker>(gpt_init_parameter_, device_base_, kv_cache_allocator_);

        // 直接设置 LayerCacheBufferTaskStore（友元类可以访问私有成员）
        task_store_ = std::make_shared<LayerCacheBufferTaskStore>();
        worker_->setLayerCacheBufferTaskStore(task_store_);
    }

    void TearDown() override {
        worker_.reset();
        task_store_.reset();
    }

    // 创建测试用的 LayerCacheBuffer
    std::shared_ptr<LayerCacheBuffer> createLayerCacheBuffer(int layer_id, int num_blocks = 2) {
        auto buffer = std::make_shared<LayerCacheBuffer>(layer_id);
        for (int i = 0; i < num_blocks; ++i) {
            int64_t cache_key = layer_id * 1000 + i;
            int     block_id  = i;
            buffer->addBlockId(cache_key, block_id);
        }
        return buffer;
    }

    // 模拟任务完成
    void simulateTaskDone(const std::string& unique_key, const std::vector<int>& layer_ids, bool all_success = true) {
        auto task = task_store_->getTask(unique_key);
        if (task) {
            for (int layer_id : layer_ids) {
                task->setLoading(layer_id);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            for (int layer_id : layer_ids) {
                task->notifyDone(layer_id, all_success);
            }
        }
    }

protected:
    GptInitParameter                           gpt_init_parameter_;
    DeviceBase*                                device_base_;
    std::shared_ptr<KVCacheAllocator>          kv_cache_allocator_;
    std::unique_ptr<P2PConnectorDecodeWorker>  worker_;
    std::shared_ptr<LayerCacheBufferTaskStore> task_store_;
};

// ---------------------------- read ----------------------------

TEST_F(P2PConnectorDecodeWorkerTest, Read_ReturnTrue_AllLayersSuccess) {
    std::string unique_key  = "test_read_success";
    int64_t     request_id  = 1001;
    int64_t     deadline_ms = currentTimeMs() + 5000;

    // 创建 LayerCacheBuffer
    std::vector<std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers;
    layer_cache_buffers.push_back(createLayerCacheBuffer(0, 2));
    layer_cache_buffers.push_back(createLayerCacheBuffer(1, 2));

    // 在后台线程中模拟任务完成
    std::thread completion_thread([this, unique_key]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        simulateTaskDone(unique_key, {0, 1}, true);
    });

    // 执行 read
    bool success = worker_->read(request_id, unique_key, deadline_ms, layer_cache_buffers);

    completion_thread.join();

    // 验证结果
    EXPECT_TRUE(success);
    // 验证任务已从 store 中移除
    EXPECT_EQ(task_store_->getTask(unique_key), nullptr);
}

TEST_F(P2PConnectorDecodeWorkerTest, Read_ReturnFalse_PartialLayersFailed) {
    std::string unique_key  = "test_read_partial_fail";
    int64_t     request_id  = 1002;
    int64_t     deadline_ms = currentTimeMs() + 5000;

    // 创建 LayerCacheBuffer
    std::vector<std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers;
    layer_cache_buffers.push_back(createLayerCacheBuffer(0, 2));
    layer_cache_buffers.push_back(createLayerCacheBuffer(1, 2));

    // 在后台线程中模拟部分失败
    std::thread completion_thread([this, unique_key]() {
        simulateTaskDone(unique_key, {0}, true);   // layer 0 成功
        simulateTaskDone(unique_key, {1}, false);  // layer 1 失败
    });

    // 执行 read
    bool success = worker_->read(request_id, unique_key, deadline_ms, layer_cache_buffers);

    completion_thread.join();

    // 验证结果
    EXPECT_FALSE(success);
    // 验证任务已从 store 中移除
    EXPECT_EQ(task_store_->getTask(unique_key), nullptr);
}

TEST_F(P2PConnectorDecodeWorkerTest, Read_ReturnFalse_Timeout) {
    std::string unique_key  = "test_read_timeout";
    int64_t     request_id  = 1003;
    int64_t     deadline_ms = currentTimeMs() + 10;  // 很短的超时时间

    // 创建 LayerCacheBuffer
    std::vector<std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers;
    layer_cache_buffers.push_back(createLayerCacheBuffer(0, 2));

    // 不模拟任务完成，让它超时
    auto start_time_ms = currentTimeMs();

    // 执行 read
    bool success = worker_->read(request_id, unique_key, deadline_ms, layer_cache_buffers);

    // 验证结果（应该因为超时返回 false）
    EXPECT_FALSE(success);

    // worker 在可控的时间内结束自身处理, 保证worker不会超时
    auto end_time_ms = currentTimeMs();
    EXPECT_GE(end_time_ms - start_time_ms, 10);
    EXPECT_LE(end_time_ms - start_time_ms, 100);

    // 验证任务已从 store 中移除
    EXPECT_EQ(task_store_->getTask(unique_key), nullptr);
}

TEST_F(P2PConnectorDecodeWorkerTest, Read_ReturnTrue_EmptyBuffers) {
    std::string unique_key  = "test_read_empty";
    int64_t     request_id  = 1004;
    int64_t     deadline_ms = currentTimeMs() + 5000;

    // 创建空的 LayerCacheBuffer 列表
    std::vector<std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers;

    // 执行 read
    bool success = worker_->read(request_id, unique_key, deadline_ms, layer_cache_buffers);

    // 验证结果（空列表应该立即返回 true）
    EXPECT_TRUE(success);
}

TEST_F(P2PConnectorDecodeWorkerTest, Read_ReturnFalse_PartialBuffers) {
    std::string unique_key  = "test_read_null";
    int64_t     request_id  = 1005;
    int64_t     deadline_ms = currentTimeMs() + 5000;

    // 创建包含 nullptr 的 LayerCacheBuffer 列表
    std::vector<std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers;
    layer_cache_buffers.push_back(createLayerCacheBuffer(0, 2));
    layer_cache_buffers.push_back(createLayerCacheBuffer(1, 2));

    // 在后台线程中模拟任务完成
    std::thread completion_thread([this, unique_key]() {
        simulateTaskDone(unique_key, {1}, true);  // 只有 layer 1 会被处理
    });

    // 执行 read
    bool success = worker_->read(request_id, unique_key, deadline_ms, layer_cache_buffers);

    completion_thread.join();

    // wait till timeout
    EXPECT_FALSE(success);
    EXPECT_EQ(task_store_->getTask(unique_key), nullptr);
}

// ---------------------------- cancelRead ----------------------------

TEST_F(P2PConnectorDecodeWorkerTest, CancelRead_ReturnSuccess_TaskExists) {
    std::string unique_key  = "test_cancel_success";
    int64_t     request_id  = 1006;
    int64_t     deadline_ms = currentTimeMs() + 5000;

    // 先创建一个任务
    std::map<int, std::shared_ptr<LayerCacheBuffer>> layer_cache_buffer_map;
    layer_cache_buffer_map[0] = createLayerCacheBuffer(0, 2);
    layer_cache_buffer_map[1] = createLayerCacheBuffer(1, 2);

    auto task = task_store_->addTask(unique_key, layer_cache_buffer_map, deadline_ms);
    ASSERT_NE(task, nullptr);

    task->setLoading(0);

    // 在后台线程中等待取消完成
    std::thread cancel_thread([this, request_id, unique_key]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        worker_->cancelRead(request_id, unique_key);
    });

    task->waitDone();
    EXPECT_FALSE(task->success());
    EXPECT_EQ(task_store_->getTask(unique_key), nullptr);

    cancel_thread.join();

    std::thread loading_thread([this, task]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        task->notifyDone(0, true);
    });

    task->waitLoadingDone();

    loading_thread.join();
}

TEST_F(P2PConnectorDecodeWorkerTest, CancelRead_ReturnSuccess_TaskNotExists) {
    std::string unique_key = "test_cancel_not_exist";
    int64_t     request_id = 1007;

    // 执行 cancelRead，任务不存在
    worker_->cancelRead(request_id, unique_key);

    // wait till timeout
    EXPECT_EQ(task_store_->getTask(unique_key), nullptr);
}
}  // namespace rtp_llm
