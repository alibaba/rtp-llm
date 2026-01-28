#include <gtest/gtest.h>
#include <memory>
#include <map>
#include <thread>
#include <chrono>
#include <vector>
#include <atomic>

#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferTask.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/LayerCacheBuffer.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

class TransferTaskTest: public ::testing::Test {
protected:
    void SetUp() override {
        // 初始化测试环境
    }

    void TearDown() override {
        // 清理测试环境
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

    // 等待任务完成（所有层都已 notifyDone）
    void waitForTaskDone(TransferTask& task, int timeout_ms = 5000) {
        int wait_count = 0;
        int max_wait   = timeout_ms / 10;
        while (!task.success() && !task.isTimeout() && !task.cancelled() && wait_count < max_wait) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            wait_count++;
        }
    }

    // 等待加载完成（没有正在加载的层）
    void waitForLoadingDone(TransferTask& task, int timeout_ms = 5000) {
        int wait_count = 0;
        int max_wait   = timeout_ms / 10;
        while (task.hasLoadingLayer() && wait_count < max_wait) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            wait_count++;
        }
    }

protected:
    // 测试辅助变量
};

// ==================== 基础功能测试 ====================

// 测试构造函数
TEST_F(TransferTaskTest, ConstructorTest) {
    auto    buffers     = createLayerCacheBuffers(3);
    int64_t deadline_ms = currentTimeMs() + 1000;

    TransferTask task(buffers, deadline_ms);

    // 验证初始状态：未完成任何层，success 应该为 false
    EXPECT_FALSE(task.success());
    EXPECT_FALSE(task.cancelled());
    EXPECT_FALSE(task.hasLoadingLayer());

    // 验证可以获取所有层
    for (int i = 0; i < 3; ++i) {
        auto buffer = task.getLayerCacheBuffer(i);
        ASSERT_NE(buffer, nullptr);
        EXPECT_EQ(buffer->getLayerId(), i);
    }

    // 测试获取不存在的层
    auto non_exist_buffer = task.getLayerCacheBuffer(10);
    EXPECT_EQ(non_exist_buffer, nullptr);
}

// 测试 notifyDone - 全部成功
TEST_F(TransferTaskTest, NotifyDoneAllSuccessTest) {
    auto    buffers     = createLayerCacheBuffers(3);
    int64_t deadline_ms = currentTimeMs() + 1000;

    TransferTask task(buffers, deadline_ms);

    // 先将所有层标记为 loading 状态
    task.loadingLayerCacheBuffer(0, 1, 0);
    task.loadingLayerCacheBuffer(1, 1, 0);
    task.loadingLayerCacheBuffer(2, 1, 0);

    // 通知第一层成功完成，但整体未完成
    task.notifyDone(0, true, 1, 0);
    EXPECT_FALSE(task.success());

    task.notifyDone(1, true, 1, 0);
    EXPECT_FALSE(task.success());

    // 所有层完成后，success 为 true
    task.notifyDone(2, true, 1, 0);
    EXPECT_TRUE(task.success());
}

// 测试 notifyDone - 有失败
TEST_F(TransferTaskTest, NotifyDoneWithFailureTest) {
    auto    buffers     = createLayerCacheBuffers(3);
    int64_t deadline_ms = currentTimeMs() + 1000;

    TransferTask task(buffers, deadline_ms);

    // 先将所有层标记为 loading 状态
    task.loadingLayerCacheBuffer(0, 1, 0);
    task.loadingLayerCacheBuffer(1, 1, 0);
    task.loadingLayerCacheBuffer(2, 1, 0);

    // 第一层成功
    task.notifyDone(0, true, 1, 0);
    EXPECT_FALSE(task.success());

    // 第二层失败
    task.notifyDone(1, false, 1, 0);
    EXPECT_FALSE(task.success());

    // 第三层成功，但整体仍为失败（因为第二层失败）
    task.notifyDone(2, true, 1, 0);
    EXPECT_FALSE(task.success());
}

// 测试等待所有层完成
TEST_F(TransferTaskTest, WaitAllLayersCompleteTest) {
    auto    buffers     = createLayerCacheBuffers(3);
    int64_t deadline_ms = currentTimeMs() + 5000;  // 5秒超时，足够完成

    TransferTask task(buffers, deadline_ms);

    // 先将所有层标记为 loading 状态
    task.loadingLayerCacheBuffer(0, 1, 0);
    task.loadingLayerCacheBuffer(1, 1, 0);
    task.loadingLayerCacheBuffer(2, 1, 0);

    // 在另一个线程中通知所有层完成
    std::thread notify_thread([&task]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        task.notifyDone(0, true, 1, 0);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        task.notifyDone(1, true, 1, 0);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        task.notifyDone(2, true, 1, 0);
    });

    // 主线程等待完成
    auto start_time = std::chrono::steady_clock::now();
    waitForTaskDone(task);
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    notify_thread.join();

    // 应该在200ms左右完成（100 + 50 + 50）
    EXPECT_GE(duration, 150);
    EXPECT_LE(duration, 500);
    EXPECT_TRUE(task.success());
}

// 测试并发 notifyDone
TEST_F(TransferTaskTest, ConcurrentNotifyDoneTest) {
    auto    buffers     = createLayerCacheBuffers(10);
    int64_t deadline_ms = currentTimeMs() + 5000;

    TransferTask task(buffers, deadline_ms);

    // 先将所有层标记为 loading 状态
    for (int i = 0; i < 10; ++i) {
        task.loadingLayerCacheBuffer(i, 1, 0);
    }

    // 多个线程并发通知完成
    std::vector<std::thread> threads;
    for (int i = 0; i < 10; ++i) {
        threads.emplace_back([&task, i]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(i * 10));
            task.notifyDone(i, true, 1, 0);
        });
    }

    // 等待所有层完成
    waitForTaskDone(task);

    for (auto& thread : threads) {
        thread.join();
    }

    // 所有层都应该完成
    EXPECT_TRUE(task.success());
}

// 测试超时
TEST_F(TransferTaskTest, TimeoutTest) {
    auto    buffers     = createLayerCacheBuffers(3);
    int64_t deadline_ms = currentTimeMs() + 200;  // 200ms 超时

    TransferTask task(buffers, deadline_ms);

    // 先将所有层标记为 loading 状态
    task.loadingLayerCacheBuffer(0, 1, 0);
    task.loadingLayerCacheBuffer(1, 1, 0);
    task.loadingLayerCacheBuffer(2, 1, 0);

    // 只通知一层完成，不通知其他层，等待超时
    task.notifyDone(0, true, 1, 0);

    auto start_time = std::chrono::steady_clock::now();
    waitForTaskDone(task, 500);
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // 应该在大约200ms后超时返回
    EXPECT_GE(duration, 180);
    EXPECT_LE(duration, 400);
    EXPECT_FALSE(task.success());
    EXPECT_TRUE(task.isTimeout());
}

// 测试取消
TEST_F(TransferTaskTest, CancelledTest) {
    auto    buffers     = createLayerCacheBuffers(3);
    int64_t deadline_ms = currentTimeMs() + 5000;  // 5秒超时

    TransferTask task(buffers, deadline_ms);

    // 在另一个线程中取消任务
    std::thread cancel_thread([&task]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        task.setCancelled();
    });

    // 主线程等待
    auto start_time = std::chrono::steady_clock::now();
    waitForTaskDone(task);
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    cancel_thread.join();

    // 应该在100ms左右被取消
    EXPECT_GE(duration, 80);
    EXPECT_LE(duration, 300);
    EXPECT_TRUE(task.cancelled());
    EXPECT_FALSE(task.success());
}

// ==================== 边界条件测试 ====================
// 测试空 LayerCacheBuffer 映射
TEST_F(TransferTaskTest, EmptyLayerCacheBuffersTest) {
    std::map<int, std::shared_ptr<LayerCacheBuffer>> empty_buffers;
    int64_t                                          deadline_ms = currentTimeMs() + 1000;
    TransferTask                                     task(empty_buffers, deadline_ms);

    // 空映射应该立即完成（success 为 true，因为 0 == 0）
    EXPECT_TRUE(task.success());
}

// 测试重复通知同一层
TEST_F(TransferTaskTest, DuplicateNotifyDoneTest) {
    auto    buffers     = createLayerCacheBuffers(3);
    int64_t deadline_ms = currentTimeMs() + 1000;

    TransferTask task(buffers, deadline_ms);

    // 先将所有层标记为 loading 状态
    task.loadingLayerCacheBuffer(0, 1, 0);
    task.loadingLayerCacheBuffer(1, 1, 0);
    task.loadingLayerCacheBuffer(2, 1, 0);

    // 通知所有层完成
    task.notifyDone(0, true, 1, 0);
    task.notifyDone(0, true, 1, 0);  // 重复（已完成的层会被忽略）
    task.notifyDone(1, true, 1, 0);
    task.notifyDone(2, true, 1, 0);

    EXPECT_TRUE(task.success());
}

// 测试通知不存在的层
TEST_F(TransferTaskTest, NotifyNonExistentLayerTest) {
    auto    buffers     = createLayerCacheBuffers(3);
    int64_t deadline_ms = currentTimeMs() + 1000;

    TransferTask task(buffers, deadline_ms);

    // 先将所有层标记为 loading 状态
    task.loadingLayerCacheBuffer(0, 1, 0);
    task.loadingLayerCacheBuffer(1, 1, 0);
    task.loadingLayerCacheBuffer(2, 1, 0);

    // 通知存在的层
    task.notifyDone(0, true, 1, 0);
    task.notifyDone(1, true, 1, 0);
    task.notifyDone(2, true, 1, 0);

    // 通知不存在的层（应该不影响，因为没有在 loading 状态）
    task.notifyDone(10, false, 1, 0);

    // 所有存在的层都成功，应该为成功
    EXPECT_TRUE(task.success());
}

// ==================== loadingLayerCacheBuffer 测试 ====================

// 测试 loadingLayerCacheBuffer - 正常加载
TEST_F(TransferTaskTest, LoadingLayerCacheBufferTest) {
    auto    buffers     = createLayerCacheBuffers(3);
    int64_t deadline_ms = currentTimeMs() + 1000;

    TransferTask task(buffers, deadline_ms);

    // 没有加载中的层
    EXPECT_FALSE(task.hasLoadingLayer());

    // 开始加载第一层
    auto buffer0 = task.loadingLayerCacheBuffer(0, 1, 0);
    ASSERT_NE(buffer0, nullptr);
    EXPECT_EQ(buffer0->getLayerId(), 0);
    EXPECT_TRUE(task.hasLoadingLayer());

    // 开始加载第二层
    auto buffer1 = task.loadingLayerCacheBuffer(1, 1, 0);
    ASSERT_NE(buffer1, nullptr);
    EXPECT_TRUE(task.hasLoadingLayer());

    // 完成第一层
    task.notifyDone(0, true, 1, 0);
    EXPECT_TRUE(task.hasLoadingLayer());  // 第二层还在加载

    // 完成第二层
    task.notifyDone(1, true, 1, 0);
    EXPECT_FALSE(task.hasLoadingLayer());  // 没有加载中的层了
}

// 测试 loadingLayerCacheBuffer - 加载不存在的层
TEST_F(TransferTaskTest, LoadingNonExistentLayerTest) {
    auto    buffers     = createLayerCacheBuffers(3);
    int64_t deadline_ms = currentTimeMs() + 1000;

    TransferTask task(buffers, deadline_ms);

    // 尝试加载不存在的层
    auto buffer = task.loadingLayerCacheBuffer(10, 1, 0);
    EXPECT_EQ(buffer, nullptr);
    EXPECT_FALSE(task.hasLoadingLayer());
}

// 测试 loadingLayerCacheBuffer - 加载已完成的层
TEST_F(TransferTaskTest, LoadingCompletedLayerTest) {
    auto    buffers     = createLayerCacheBuffers(3);
    int64_t deadline_ms = currentTimeMs() + 1000;

    TransferTask task(buffers, deadline_ms);

    // 先标记为 loading 状态，然后完成该层
    task.loadingLayerCacheBuffer(0, 1, 0);
    task.notifyDone(0, true, 1, 0);

    // 尝试加载已完成的层，应该返回 nullptr
    auto buffer = task.loadingLayerCacheBuffer(0, 1, 0);
    EXPECT_EQ(buffer, nullptr);
}

// 测试 hasLoadingLayer - 所有层完成加载
TEST_F(TransferTaskTest, HasLoadingLayerAllCompleteTest) {
    auto    buffers     = createLayerCacheBuffers(3);
    int64_t deadline_ms = currentTimeMs() + 5000;

    TransferTask task(buffers, deadline_ms);

    // 开始加载所有层
    task.loadingLayerCacheBuffer(0, 1, 0);
    task.loadingLayerCacheBuffer(1, 1, 0);
    task.loadingLayerCacheBuffer(2, 1, 0);
    EXPECT_TRUE(task.hasLoadingLayer());

    // 在另一个线程中完成所有层的加载
    std::thread complete_thread([&task]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        task.notifyDone(0, true, 1, 0);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        task.notifyDone(1, true, 1, 0);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        task.notifyDone(2, true, 1, 0);
    });

    // 主线程等待加载完成
    auto start_time = std::chrono::steady_clock::now();
    waitForLoadingDone(task);
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    complete_thread.join();

    // 应该在200ms左右完成（100 + 50 + 50）
    EXPECT_GE(duration, 150);
    EXPECT_LE(duration, 500);
    EXPECT_FALSE(task.hasLoadingLayer());
}

}  // namespace rtp_llm
