#include "gtest/gtest.h"

#include "rtp_llm/cpp/devices/CacheStoreAsyncWriter.h"

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

namespace rtp_llm {

class CacheStoreAsyncWriterTest: public ::testing::Test {};

TEST_F(CacheStoreAsyncWriterTest, InitAndWaitBasic) {
    CacheStoreAsyncWriter writer;
    ASSERT_TRUE(writer.isIdle());

    writer.init();
    ASSERT_FALSE(writer.isIdle());

    std::atomic<int> counter{0};
    writer.submit([&counter]() { counter.fetch_add(1); });
    writer.submit([&counter]() { counter.fetch_add(1); });
    writer.submit([&counter]() { counter.fetch_add(1); });

    writer.waitAllDone();
    ASSERT_TRUE(writer.isIdle());
    ASSERT_EQ(3, counter.load());
}

TEST_F(CacheStoreAsyncWriterTest, WaitAllDoneWhileIdleThrows) {
    CacheStoreAsyncWriter writer;
    ASSERT_ANY_THROW(writer.waitAllDone());
}

TEST_F(CacheStoreAsyncWriterTest, SubmitWhileIdleThrows) {
    CacheStoreAsyncWriter writer;
    ASSERT_ANY_THROW(writer.submit([]() {}));
}

TEST_F(CacheStoreAsyncWriterTest, InitWhileRunningThrows) {
    CacheStoreAsyncWriter writer;
    writer.init();

    ASSERT_ANY_THROW(writer.init());

    // Writer should still be functional after the failed second init.
    std::atomic<int> counter{0};
    writer.submit([&counter]() { counter.fetch_add(1); });
    writer.waitAllDone();
    ASSERT_EQ(1, counter.load());
}

TEST_F(CacheStoreAsyncWriterTest, InitWaitCycle) {
    CacheStoreAsyncWriter writer;
    std::vector<int>      order;

    writer.init();
    writer.submit([&]() { order.push_back(1); });
    writer.submit([&]() { order.push_back(2); });
    writer.waitAllDone();

    ASSERT_EQ(2u, order.size());

    writer.init();
    writer.submit([&]() { order.push_back(3); });
    writer.waitAllDone();

    ASSERT_EQ(3u, order.size());
    ASSERT_EQ(3, order.back());
}

TEST_F(CacheStoreAsyncWriterTest, AsyncExecution) {
    CacheStoreAsyncWriter writer;
    writer.init();

    auto              main_tid = std::this_thread::get_id();
    std::atomic<bool> different_thread{false};

    writer.submit([&]() {
        if (std::this_thread::get_id() != main_tid) {
            different_thread.store(true);
        }
    });
    writer.waitAllDone();

    ASSERT_TRUE(different_thread.load());
}

TEST_F(CacheStoreAsyncWriterTest, ExceptionPropagation) {
    CacheStoreAsyncWriter writer;
    writer.init();

    writer.submit([]() { throw std::runtime_error("test error"); });

    ASSERT_THROW(writer.waitAllDone(), std::runtime_error);

    // After exception, writer should be back in IDLE and re-initializable.
    ASSERT_TRUE(writer.isIdle());
    writer.init();
    std::atomic<int> counter{0};
    writer.submit([&counter]() { counter.fetch_add(1); });
    writer.waitAllDone();
    ASSERT_EQ(1, counter.load());
}

TEST_F(CacheStoreAsyncWriterTest, FirstExceptionKeptOnMultipleFailures) {
    CacheStoreAsyncWriter writer;
    writer.init();

    writer.submit([]() { throw std::runtime_error("first"); });
    writer.submit([]() { throw std::runtime_error("second"); });

    try {
        writer.waitAllDone();
        FAIL() << "expected exception";
    } catch (const std::runtime_error& e) {
        ASSERT_STREQ("first", e.what());
    }
}

TEST_F(CacheStoreAsyncWriterTest, MultipleTaskOrdering) {
    CacheStoreAsyncWriter writer;
    writer.init();

    std::vector<int> sequence;
    constexpr int    N = 20;

    for (int i = 0; i < N; ++i) {
        writer.submit([&sequence, i]() { sequence.push_back(i); });
    }
    writer.waitAllDone();

    ASSERT_EQ(static_cast<size_t>(N), sequence.size());
    for (int i = 0; i < N; ++i) {
        ASSERT_EQ(i, sequence[i]);
    }
}

TEST_F(CacheStoreAsyncWriterTest, WaitWithoutSubmit) {
    CacheStoreAsyncWriter writer;
    writer.init();
    writer.waitAllDone();
    ASSERT_TRUE(writer.isIdle());
}

TEST_F(CacheStoreAsyncWriterTest, ManyCycles) {
    CacheStoreAsyncWriter writer;
    std::atomic<int>      total{0};

    for (int cycle = 0; cycle < 50; ++cycle) {
        writer.init();
        for (int i = 0; i < 5; ++i) {
            writer.submit([&total]() { total.fetch_add(1); });
        }
        writer.waitAllDone();
    }
    ASSERT_EQ(250, total.load());
}

TEST_F(CacheStoreAsyncWriterTest, DoubleWaitAllDoneThrows) {
    CacheStoreAsyncWriter writer;
    writer.init();
    writer.waitAllDone();
    ASSERT_ANY_THROW(writer.waitAllDone());
}

}  // namespace rtp_llm
