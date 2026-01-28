#include <gtest/gtest.h>
#include <memory>
#include <thread>
#include <chrono>
#include <set>

#include "rtp_llm/cpp/cache/connector/p2p/ComputedLayerCacheBuffer.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/LayerCacheBuffer.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

class ComputedLayerCacheBufferTest: public ::testing::Test {
protected:
    void SetUp() override {
        store_ = std::make_unique<ComputedLayerCacheBufferStore>();
    }

    void TearDown() override {
        store_.reset();
    }

    // 创建测试用的 LayerCacheBuffer
    std::shared_ptr<LayerCacheBuffer> createLayerCacheBuffer(int layer_id) {
        return std::make_shared<LayerCacheBuffer>(layer_id);
    }

    // 获取当前时间（毫秒）+ 偏移量
    int64_t getDeadlineMs(int64_t offset_ms = 1000) {
        return currentTimeMs() + offset_ms;
    }

protected:
    std::unique_ptr<ComputedLayerCacheBufferStore> store_;
};

// ==================== ComputedLayerCacheBuffer 类测试 ====================

TEST_F(ComputedLayerCacheBufferTest, nullBufferFirst) {
    int64_t request_id   = 1001;
    int64_t deadline_ms0 = getDeadlineMs();

    ComputedLayerCacheBuffer computed_buffer(request_id, nullptr, deadline_ms0);
    ASSERT_EQ(deadline_ms0, computed_buffer.deadlineMs());

    auto    buffer       = createLayerCacheBuffer(0);
    int64_t deadline_ms1 = getDeadlineMs();

    computed_buffer.addBuffer(buffer, deadline_ms1);
    ASSERT_EQ(deadline_ms1, computed_buffer.deadlineMs());

    auto [layer_count, buffers] = computed_buffer.getBuffers({0, 1});
    EXPECT_EQ(layer_count, 1);
    EXPECT_EQ(buffers.size(), 1);
    EXPECT_EQ(buffer, buffers.at(0));
}

TEST_F(ComputedLayerCacheBufferTest, fullBufferFirst) {
    int64_t request_id   = 1001;
    int64_t deadline_ms0 = getDeadlineMs();
    auto    buffer       = createLayerCacheBuffer(0);

    ComputedLayerCacheBuffer computed_buffer(request_id, buffer, deadline_ms0);
    ASSERT_EQ(deadline_ms0, computed_buffer.deadlineMs());

    int64_t deadline_ms1 = getDeadlineMs();

    computed_buffer.addBuffer(nullptr, deadline_ms1);
    ASSERT_EQ(deadline_ms1, computed_buffer.deadlineMs());

    auto [layer_count, buffers] = computed_buffer.getBuffers({0, 1});
    EXPECT_EQ(layer_count, 1);
    EXPECT_EQ(buffers.size(), 1);
    EXPECT_EQ(buffer, buffers.at(0));
}

TEST_F(ComputedLayerCacheBufferTest, ComputedLayerCacheBuffer_WaitChange) {
    int64_t request_id  = 1005;
    auto    buffer0     = createLayerCacheBuffer(0);
    int64_t deadline_ms = getDeadlineMs();

    auto computed_buffer = std::make_shared<ComputedLayerCacheBuffer>(request_id, buffer0, deadline_ms);

    // 在另一个线程中添加缓冲区
    std::thread producer([computed_buffer, deadline_ms, this]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        auto buffer1 = createLayerCacheBuffer(1);
        computed_buffer->addBuffer(buffer1, deadline_ms);
    });

    // 等待变化
    computed_buffer->waitChange(1, 1000);

    producer.join();

    std::set<int> layer_ids     = {0, 1};
    auto [layer_count, buffers] = computed_buffer->getBuffers(layer_ids);
    EXPECT_EQ(layer_count, 2);
    EXPECT_EQ(buffers.size(), 2);

    // before call wait change, add new buffer
    auto start = std::chrono::steady_clock::now();
    computed_buffer->waitChange(1, 1000);
    auto end     = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    EXPECT_LE(elapsed, 20);
}

TEST_F(ComputedLayerCacheBufferTest, ComputedLayerCacheBuffer_WaitChangeTimeout) {
    int64_t request_id  = 1006;
    auto    buffer0     = createLayerCacheBuffer(0);
    int64_t deadline_ms = getDeadlineMs();

    ComputedLayerCacheBuffer computed_buffer(request_id, buffer0, deadline_ms);

    // 等待变化，但没有新的缓冲区添加，应该超时
    auto start = std::chrono::steady_clock::now();
    computed_buffer.waitChange(1, 100);  // 100ms 超时
    auto end     = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    EXPECT_GE(elapsed, 50);
    EXPECT_LE(elapsed, 200);
}

// ==================== ComputedLayerCacheBufferStore 类测试 ====================

TEST_F(ComputedLayerCacheBufferTest, AddAndGetBuffer) {
    int64_t request_id  = 2001;
    auto    buffer      = createLayerCacheBuffer(0);
    int64_t deadline_ms = getDeadlineMs();

    ASSERT_TRUE(store_->getBuffer(request_id) == nullptr);

    auto computed_buffer = store_->addBuffer(request_id, buffer, deadline_ms);
    ASSERT_NE(computed_buffer, nullptr);
    EXPECT_EQ(computed_buffer->deadlineMs(), deadline_ms);

    auto retrieved = store_->getBuffer(request_id);
    ASSERT_NE(retrieved, nullptr);
    EXPECT_EQ(retrieved->deadlineMs(), deadline_ms);

    // add null buffer
    auto computed_buffer1 = store_->addBuffer(request_id, nullptr, deadline_ms - 10);
    ASSERT_EQ(computed_buffer, computed_buffer1);
    EXPECT_EQ(computed_buffer1->deadlineMs(), deadline_ms);

    // add new buffer
    auto buffer1          = createLayerCacheBuffer(1);
    auto computed_buffer2 = store_->addBuffer(request_id, buffer1, deadline_ms);
    ASSERT_EQ(computed_buffer2, computed_buffer);

    auto [layer_count, buffers] = retrieved->getBuffers({0, 2});
    EXPECT_EQ(layer_count, 2);
    ASSERT_EQ(buffers.size(), 1);
    EXPECT_EQ(buffers[0]->getLayerId(), 0);
}

TEST_F(ComputedLayerCacheBufferTest, RemoveBuffer) {
    int64_t request_id  = 2003;
    auto    buffer      = createLayerCacheBuffer(0);
    int64_t deadline_ms = getDeadlineMs();

    store_->addBuffer(request_id, buffer, deadline_ms);

    auto retrieved = store_->getBuffer(request_id);
    ASSERT_NE(retrieved, nullptr);

    store_->removeBuffer(request_id);
    retrieved = store_->getBuffer(request_id);
    EXPECT_EQ(retrieved, nullptr);
}

TEST_F(ComputedLayerCacheBufferTest, CheckTimeoutMixed) {
    int64_t request_id1  = 3003;
    auto    buffer1      = createLayerCacheBuffer(0);
    int64_t deadline_ms1 = getDeadlineMs(1000);  // 1秒后过期

    int64_t request_id2  = 3004;
    auto    buffer2      = createLayerCacheBuffer(0);
    int64_t deadline_ms2 = getDeadlineMs(-100);  // 已经过期

    store_->addBuffer(request_id1, buffer1, deadline_ms1);
    store_->addBuffer(request_id2, buffer2, deadline_ms2);

    EXPECT_EQ(store_->getBuffersCount(), 2);

    // 检查超时
    store_->checkTimeout();

    // 只有过期的被删除
    auto retrieved1 = store_->getBuffer(request_id1);
    ASSERT_NE(retrieved1, nullptr);

    auto retrieved2 = store_->getBuffer(request_id2);
    EXPECT_EQ(retrieved2, nullptr);

    EXPECT_EQ(store_->getBuffersCount(), 1);
}

}  // namespace rtp_llm
