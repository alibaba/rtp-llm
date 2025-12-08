#include <gtest/gtest.h>
#include <memory>
#include <map>
#include <thread>
#include <chrono>
#include <vector>
#include <atomic>

#include "rtp_llm/cpp/disaggregate/transfer/LayerCacheBuffer.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

class LayerCacheBufferTaskTest: public ::testing::Test {
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

protected:
    // 测试辅助变量
};

// ==================== 基础功能测试 ====================

// 测试构造函数
TEST_F(LayerCacheBufferTaskTest, ConstructorTest) {
    auto    buffers     = createLayerCacheBuffers(3);
    int64_t deadline_ms = currentTimeMs() + 1000;

    LayerCacheBufferTask task(buffers, deadline_ms);

    // 验证初始状态
    EXPECT_TRUE(task.success());

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
TEST_F(LayerCacheBufferTaskTest, NotifyDoneAllSuccessTest) {
    auto    buffers     = createLayerCacheBuffers(3);
    int64_t deadline_ms = currentTimeMs() + 1000;

    LayerCacheBufferTask task(buffers, deadline_ms);

    // 通知所有层成功完成
    task.notifyDone(0, true);
    EXPECT_TRUE(task.success());

    task.notifyDone(1, true);
    EXPECT_TRUE(task.success());

    task.notifyDone(2, true);
    EXPECT_TRUE(task.success());
}

// 测试 notifyDone - 有失败
TEST_F(LayerCacheBufferTaskTest, NotifyDoneWithFailureTest) {
    auto    buffers     = createLayerCacheBuffers(3);
    int64_t deadline_ms = currentTimeMs() + 1000;

    LayerCacheBufferTask task(buffers, deadline_ms);

    // 第一层成功
    task.notifyDone(0, true);
    EXPECT_TRUE(task.success());

    // 第二层失败
    task.notifyDone(1, false);
    EXPECT_FALSE(task.success());

    // 第三层成功，但整体仍为失败
    task.notifyDone(2, true);
    EXPECT_FALSE(task.success());
}

// 测试 waitDone - 所有层完成
TEST_F(LayerCacheBufferTaskTest, WaitDoneAllLayersCompleteTest) {
    auto    buffers     = createLayerCacheBuffers(3);
    int64_t deadline_ms = currentTimeMs() + 5000;  // 5秒超时，足够完成

    LayerCacheBufferTask task(buffers, deadline_ms);

    // 在另一个线程中通知所有层完成
    std::thread notify_thread([&task]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        task.notifyDone(0, true);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        task.notifyDone(1, true);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        task.notifyDone(2, true);
    });

    // 主线程等待完成
    auto start_time = std::chrono::steady_clock::now();
    task.waitDone();
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    notify_thread.join();

    // 应该在200ms左右完成（100 + 50 + 50）
    EXPECT_GE(duration, 150);
    EXPECT_LE(duration, 500);
    EXPECT_TRUE(task.success());
}

// 测试并发 notifyDone
TEST_F(LayerCacheBufferTaskTest, ConcurrentNotifyDoneTest) {
    auto    buffers     = createLayerCacheBuffers(10);
    int64_t deadline_ms = currentTimeMs() + 5000;

    LayerCacheBufferTask task(buffers, deadline_ms);

    // 多个线程并发通知完成
    std::vector<std::thread> threads;
    for (int i = 0; i < 10; ++i) {
        threads.emplace_back([&task, i]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(i * 10));
            task.notifyDone(i, true);
        });
    }

    // 等待所有层完成
    task.waitDone();

    for (auto& thread : threads) {
        thread.join();
    }

    // 所有层都应该完成
    EXPECT_TRUE(task.success());
}

// 测试 waitDone - 超时
TEST_F(LayerCacheBufferTaskTest, WaitDoneTimeoutTest) {
    auto    buffers     = createLayerCacheBuffers(3);
    int64_t deadline_ms = currentTimeMs() + 200;  // 200ms 超时

    LayerCacheBufferTask task(buffers, deadline_ms);

    // 只通知一层完成，不通知其他层，等待超时
    task.notifyDone(0, true);

    auto start_time = std::chrono::steady_clock::now();
    task.waitDone();
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // 应该在大约200ms后超时返回
    EXPECT_GE(duration, 180);
    EXPECT_LE(duration, 400);
    EXPECT_FALSE(task.success());
}

// 测试 waitDone - 被取消
TEST_F(LayerCacheBufferTaskTest, WaitDoneCancelledTest) {
    auto    buffers     = createLayerCacheBuffers(3);
    int64_t deadline_ms = currentTimeMs() + 5000;  // 5秒超时

    LayerCacheBufferTask task(buffers, deadline_ms);

    // 在另一个线程中取消任务
    std::thread cancel_thread([&task]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        task.setCancelled();
    });

    // 主线程等待
    auto start_time = std::chrono::steady_clock::now();
    task.waitDone();
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    cancel_thread.join();

    // 应该在100ms左右被取消
    EXPECT_GE(duration, 80);
    EXPECT_LE(duration, 300);
    EXPECT_FALSE(task.success());
}

// ==================== 边界条件测试 ====================
// 测试空 LayerCacheBuffer 映射
TEST_F(LayerCacheBufferTaskTest, EmptyLayerCacheBuffersTest) {
    std::map<int, std::shared_ptr<LayerCacheBuffer>> empty_buffers;
    int64_t                                          deadline_ms = currentTimeMs() + 1000;
    LayerCacheBufferTask                             task(empty_buffers, deadline_ms);
    // 空映射应该立即完成
    auto start_time = std::chrono::steady_clock::now();
    task.waitDone();
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    EXPECT_LT(duration, 100);
    EXPECT_TRUE(task.success());
}

// 测试重复通知同一层
TEST_F(LayerCacheBufferTaskTest, DuplicateNotifyDoneTest) {
    auto    buffers     = createLayerCacheBuffers(3);
    int64_t deadline_ms = currentTimeMs() + 1000;

    LayerCacheBufferTask task(buffers, deadline_ms);

    // 重复通知同一层
    task.notifyDone(0, true);
    task.notifyDone(0, true);  // 重复
    task.notifyDone(1, true);
    task.notifyDone(2, true);

    task.waitDone();
    EXPECT_TRUE(task.success());
}

// 测试通知不存在的层
TEST_F(LayerCacheBufferTaskTest, NotifyNonExistentLayerTest) {
    auto    buffers     = createLayerCacheBuffers(3);
    int64_t deadline_ms = currentTimeMs() + 1000;

    LayerCacheBufferTask task(buffers, deadline_ms);

    // 通知存在的层
    task.notifyDone(0, true);
    task.notifyDone(1, true);
    task.notifyDone(2, true);

    // 通知不存在的层（应该不影响）
    task.notifyDone(10, false);

    task.waitDone();
    // 所有存在的层都成功，应该为成功
    EXPECT_TRUE(task.success());
}

// ==================== waitLoadingDone 测试 ====================

// 测试 waitLoadingDone - 没有加载中的层
TEST_F(LayerCacheBufferTaskTest, WaitLoadingDone_NoLoadingLayersTest) {
    auto    buffers     = createLayerCacheBuffers(3);
    int64_t deadline_ms = currentTimeMs() + 1000;

    LayerCacheBufferTask task(buffers, deadline_ms);

    // 没有设置任何层为加载中，应该立即返回
    auto start_time = std::chrono::steady_clock::now();
    task.waitLoadingDone();
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // 应该立即返回
    EXPECT_LT(duration, 100);
}

// 测试 waitLoadingDone - 所有层完成加载
TEST_F(LayerCacheBufferTaskTest, WaitLoadingDone_AllLayersCompleteTest) {
    auto    buffers     = createLayerCacheBuffers(3);
    int64_t deadline_ms = currentTimeMs() + 5000;

    LayerCacheBufferTask task(buffers, deadline_ms);

    // 设置所有层为加载中
    task.setLoading(0);
    task.setLoading(1);
    task.setLoading(2);

    // 在另一个线程中完成所有层的加载
    std::thread complete_thread([&task]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        task.notifyDone(0, true);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        task.notifyDone(1, true);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        task.notifyDone(2, true);
    });

    // 主线程等待加载完成
    auto start_time = std::chrono::steady_clock::now();
    task.waitLoadingDone();
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    complete_thread.join();

    // 应该在200ms左右完成（100 + 50 + 50）
    EXPECT_GE(duration, 150);
    EXPECT_LE(duration, 500);
}

// 测试 waitLoadingDone - 部分层完成加载
TEST_F(LayerCacheBufferTaskTest, WaitLoadingDone_PartialLayersCompleteTest) {
    auto    buffers     = createLayerCacheBuffers(3);
    int64_t deadline_ms = currentTimeMs() + 5000;

    LayerCacheBufferTask task(buffers, deadline_ms);

    // 设置所有层为加载中
    task.setLoading(0);
    task.setLoading(1);
    task.setLoading(2);

    // 在另一个线程中完成部分层的加载
    std::thread complete_thread([&task]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        task.notifyDone(0, true);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        task.notifyDone(1, true);
        // 不通知 layer 2，但 waitLoadingDone 应该等待所有层完成
    });

    // 主线程等待加载完成（应该等待所有层）
    auto start_time = std::chrono::steady_clock::now();

    // 在另一个线程中完成剩余的层
    std::thread complete_remaining_thread([&task]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        task.notifyDone(2, true);
    });

    task.waitLoadingDone();
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    complete_thread.join();
    complete_remaining_thread.join();

    // 应该等待所有层完成（大约300ms）
    EXPECT_GE(duration, 250);
    EXPECT_LE(duration, 600);
}

// 测试 waitLoadingDone - 并发设置和完成
TEST_F(LayerCacheBufferTaskTest, WaitLoadingDone_ConcurrentSetAndCompleteTest) {
    auto    buffers     = createLayerCacheBuffers(10);
    int64_t deadline_ms = currentTimeMs() + 5000;

    LayerCacheBufferTask task(buffers, deadline_ms);

    // 设置所有层为加载中
    for (int i = 0; i < 10; ++i) {
        task.setLoading(i);
    }

    // 多个线程并发完成加载
    std::vector<std::thread> threads;
    for (int i = 0; i < 10; ++i) {
        threads.emplace_back([&task, i]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(i * 10));
            task.notifyDone(i, true);
        });
    }

    // 等待所有加载完成
    task.waitLoadingDone();

    for (auto& thread : threads) {
        thread.join();
    }

    // 所有层都应该完成加载
    // waitLoadingDone 应该已经返回
}

// 测试 setLoading - 设置存在的层
TEST_F(LayerCacheBufferTaskTest, SetLoading_ValidLayerTest) {
    auto    buffers     = createLayerCacheBuffers(3);
    int64_t deadline_ms = currentTimeMs() + 1000;

    LayerCacheBufferTask task(buffers, deadline_ms);

    // 设置层为加载中
    task.setLoading(0);
    task.setLoading(1);

    // 在另一个线程中完成加载
    std::thread complete_thread([&task]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        task.notifyDone(0, true);
        task.notifyDone(1, true);
    });

    // 等待加载完成
    task.waitLoadingDone();

    complete_thread.join();
}

// 测试 setLoading - 设置不存在的层
TEST_F(LayerCacheBufferTaskTest, SetLoading_InvalidLayerTest) {
    auto    buffers     = createLayerCacheBuffers(3);
    int64_t deadline_ms = currentTimeMs() + 1000;

    LayerCacheBufferTask task(buffers, deadline_ms);

    // 设置不存在的层为加载中（应该记录警告但不崩溃）
    task.setLoading(10);

    // waitLoadingDone 应该立即返回（因为没有有效的加载层）
    auto start_time = std::chrono::steady_clock::now();
    task.waitLoadingDone();
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // 应该立即返回
    EXPECT_LT(duration, 100);
}

// 测试 waitLoadingDone - 与 waitDone 的交互
TEST_F(LayerCacheBufferTaskTest, WaitLoadingDone_WithWaitDoneTest) {
    auto    buffers     = createLayerCacheBuffers(3);
    int64_t deadline_ms = currentTimeMs() + 5000;

    LayerCacheBufferTask task(buffers, deadline_ms);

    // 设置所有层为加载中
    task.setLoading(0);
    task.setLoading(1);
    task.setLoading(2);

    // 在另一个线程中完成所有层
    std::thread complete_thread([&task]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        task.notifyDone(0, true);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        task.notifyDone(1, true);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        task.notifyDone(2, true);
    });

    // 先等待所有层完成（waitDone）
    task.waitDone();

    // 然后等待所有加载完成（waitLoadingDone）
    // 由于 notifyDone 会从 loading_layer_ids_ 中移除，所以应该立即返回
    auto start_time = std::chrono::steady_clock::now();
    task.waitLoadingDone();
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    complete_thread.join();

    // waitLoadingDone 应该立即返回（因为所有层已经通过 notifyDone 完成）
    EXPECT_LT(duration, 100);
    EXPECT_TRUE(task.success());
}

}  // namespace rtp_llm
