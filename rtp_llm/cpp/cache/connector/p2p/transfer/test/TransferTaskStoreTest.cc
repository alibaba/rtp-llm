#include <gtest/gtest.h>
#include <memory>
#include <map>
#include <string>
#include <thread>
#include <chrono>

#include "rtp_llm/cpp/cache/connector/p2p/transfer/LayerCacheBuffer.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferTask.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

class TransferTaskStoreTest: public ::testing::Test {
protected:
    void SetUp() override {
        store_ = std::make_unique<TransferTaskStore>();
    }

    void TearDown() override {
        store_.reset();
    }

    // 创建测试用的 LayerCacheBuffer
    std::shared_ptr<LayerCacheBuffer> createLayerCacheBuffer(int layer_id) {
        return std::make_shared<LayerCacheBuffer>(layer_id);
    }

    // 创建测试用的 LayerCacheBuffer 映射
    std::map<int, std::shared_ptr<LayerCacheBuffer>> createLayerCacheBuffers(int num_layers) {
        std::map<int, std::shared_ptr<LayerCacheBuffer>> buffers;
        for (int i = 0; i < num_layers; ++i) {
            buffers[i] = createLayerCacheBuffer(i);
        }
        return buffers;
    }

protected:
    std::unique_ptr<TransferTaskStore> store_;
};

// ==================== 基础功能测试 ====================

TEST_F(TransferTaskStoreTest, AddTaskTest) {
    std::string unique_key  = "test_key_1";
    auto        buffers     = createLayerCacheBuffers(3);
    int64_t     deadline_ms = currentTimeMs() + 1000;

    // test addTask
    auto task = store_->addTask(unique_key, buffers, deadline_ms);

    ASSERT_NE(task, nullptr);
    EXPECT_EQ(task->getLayerCacheBuffer(0)->getLayerId(), 0);
    EXPECT_EQ(task->getLayerCacheBuffer(1)->getLayerId(), 1);
    EXPECT_EQ(task->getLayerCacheBuffer(2)->getLayerId(), 2);

    // test getTask
    auto retrieved_task = store_->getTask(unique_key);
    EXPECT_EQ(retrieved_task, task);
    EXPECT_EQ(retrieved_task->getLayerCacheBuffer(0)->getLayerId(), 0);
    EXPECT_EQ(retrieved_task->getLayerCacheBuffer(1)->getLayerId(), 1);
    EXPECT_EQ(retrieved_task->getLayerCacheBuffer(2)->getLayerId(), 2);

    // test getTask not exist
    auto not_exist_task = store_->getTask("not_exist_key");
    EXPECT_EQ(not_exist_task, nullptr);

    // test stealTask
    auto stolen_task = store_->stealTask(unique_key);
    EXPECT_EQ(stolen_task, task);
    EXPECT_EQ(store_->getTask(unique_key), nullptr);

    // test stealTask not exist
    auto not_exist_stolen_task = store_->stealTask(unique_key);
    EXPECT_EQ(not_exist_stolen_task, nullptr);
}

// 测试并发添加、获取和移除任务
TEST_F(TransferTaskStoreTest, ConcurrentAddGetStealTest) {
    const int                num_threads = 5;
    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([this, t]() {
            std::string unique_key  = "concurrent_key_" + std::to_string(t);
            auto        buffers     = createLayerCacheBuffers(2);
            int64_t     deadline_ms = currentTimeMs() + 1000;

            // 添加任务
            auto task = store_->addTask(unique_key, buffers, deadline_ms);
            EXPECT_NE(task, nullptr);

            // 获取任务
            auto retrieved = store_->getTask(unique_key);
            EXPECT_EQ(retrieved, task);

            // 移除任务
            auto stolen = store_->stealTask(unique_key);
            EXPECT_EQ(stolen, task);

            // 验证已移除
            auto after_steal = store_->getTask(unique_key);
            EXPECT_EQ(after_steal, nullptr);
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

// 测试相同 unique_key 的覆盖行为
TEST_F(TransferTaskStoreTest, OverwriteTaskTest) {
    std::string unique_key  = "overwrite_key";
    auto        buffers1    = createLayerCacheBuffers(2);
    auto        buffers2    = createLayerCacheBuffers(3);
    int64_t     deadline_ms = currentTimeMs() + 1000;

    auto task1 = store_->addTask(unique_key, buffers1, deadline_ms);
    auto task2 = store_->addTask(unique_key, buffers2, deadline_ms);

    // 应该返回新的任务
    EXPECT_NE(task2, nullptr);

    // 获取的任务应该是新的任务
    auto retrieved = store_->getTask(unique_key);
    EXPECT_EQ(retrieved, task2);
    EXPECT_NE(retrieved, task1);

    // 新任务应该有3个层
    EXPECT_NE(retrieved->getLayerCacheBuffer(0), nullptr);
    EXPECT_NE(retrieved->getLayerCacheBuffer(1), nullptr);
    EXPECT_NE(retrieved->getLayerCacheBuffer(2), nullptr);
}

}  // namespace rtp_llm
