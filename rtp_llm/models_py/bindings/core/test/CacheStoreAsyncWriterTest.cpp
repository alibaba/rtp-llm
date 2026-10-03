#include "gtest/gtest.h"
#define private public
#include "rtp_llm/models_py/bindings/core/CacheStoreAsyncWriter.h"
#include "rtp_llm/cpp/config/StaticConfig.h"

#include <atomic>
#include <chrono>
#include <future>
#include <mutex>
#include <thread>
#include <vector>

namespace rtp_llm {

class CacheStoreAsyncWriterTest: public ::testing::Test {
protected:
    void SetUp() override {
        saved_core_dump_ = StaticConfig::user_ft_core_dump_on_exception;
        // Invalid-state tests assert exceptions; production deliberately aborts.
        StaticConfig::user_ft_core_dump_on_exception = false;
    }

    void TearDown() override {
        StaticConfig::user_ft_core_dump_on_exception = saved_core_dump_;
    }

private:
    bool saved_core_dump_ = true;
};

TEST_F(CacheStoreAsyncWriterTest, InitAndWaitBasic) {
    CacheStoreAsyncWriter writer;
    ASSERT_TRUE(writer.state_ == CacheStoreAsyncWriter::State::IDLE);

    writer.init();
    ASSERT_FALSE(writer.state_ == CacheStoreAsyncWriter::State::IDLE);

    std::atomic<int> counter{0};
    writer.submit([&counter]() { counter.fetch_add(1); });
    writer.submit([&counter]() { counter.fetch_add(1); });
    writer.submit([&counter]() { counter.fetch_add(1); });

    writer.waitAllDone();
    ASSERT_TRUE(writer.state_ == CacheStoreAsyncWriter::State::IDLE);
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
    std::mutex            order_mutex;
    // Three worker threads may execute submitted tasks concurrently. Submission
    // order is not a completion-order guarantee; protect the test's shared vector.
    const auto append = [&](int value) {
        std::lock_guard<std::mutex> lock(order_mutex);
        order.push_back(value);
    };

    writer.init();
    writer.submit([&]() { append(1); });
    writer.submit([&]() { append(2); });
    writer.waitAllDone();

    ASSERT_EQ(2u, order.size());
    ASSERT_EQ(3, order[0] + order[1]);

    writer.init();
    writer.submit([&]() { append(3); });
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
    ASSERT_TRUE(writer.state_ == CacheStoreAsyncWriter::State::IDLE);
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
        std::string msg = e.what();
        ASSERT_TRUE(msg == "first" || msg == "second") << "unexpected: " << msg;
    }
}

TEST_F(CacheStoreAsyncWriterTest, WaitWithoutSubmit) {
    CacheStoreAsyncWriter writer;
    writer.init();
    writer.waitAllDone();
    ASSERT_TRUE(writer.state_ == CacheStoreAsyncWriter::State::IDLE);
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

TEST_F(CacheStoreAsyncWriterTest, ExceptionCleanupDrainsIdleAndRunningCycles) {
    CacheStoreAsyncWriter writer;
    EXPECT_NO_THROW(writer.drainIfRunning());
    writer.init();
    writer.submit([] { throw std::runtime_error("background reader failed"); });
    EXPECT_THROW(writer.drainIfRunning(), std::runtime_error);
    EXPECT_TRUE(writer.state_ == CacheStoreAsyncWriter::State::IDLE);
    EXPECT_NO_THROW(writer.drainIfRunning());
    writer.init();
    writer.submit([] {});
    EXPECT_NO_THROW(writer.drainIfRunning());
}

TEST_F(CacheStoreAsyncWriterTest, ExceptionCleanupWaitsForExternalReaderAfterCpuFailure) {
    CacheStoreAsyncWriter writer;
    writer.init();
    writer.trackExternalTask();
    writer.submit([] { throw std::runtime_error("partial forward reader failure"); });
    std::promise<void> entered;
    auto               waiter = std::async(std::launch::async, [&] {
        entered.set_value();
        writer.drainIfRunning();
    });
    entered.get_future().wait();
    const auto before_completion = waiter.wait_for(std::chrono::milliseconds(25));
    // Always complete the external claim before an assertion can leave the
    // async future destructor blocked. The stored CPU error propagates later.
    writer.finishExternalTask();
    EXPECT_EQ(before_completion, std::future_status::timeout);
    EXPECT_THROW(waiter.get(), std::runtime_error);
    EXPECT_TRUE(writer.state_ == CacheStoreAsyncWriter::State::IDLE);
    EXPECT_EQ(writer.pending_count_.load(), 0);
    EXPECT_EQ(writer.pending_external_count_.load(), 0);
}

TEST_F(CacheStoreAsyncWriterTest, ExceptionCleanupPropagatesExternalCompletionFailure) {
    CacheStoreAsyncWriter writer;
    writer.init();
    writer.trackExternalTask();
    writer.finishExternalTask(std::make_exception_ptr(std::runtime_error("store callback failed")));
    EXPECT_THROW(writer.drainIfRunning(), std::runtime_error);
    EXPECT_TRUE(writer.state_ == CacheStoreAsyncWriter::State::IDLE);
    EXPECT_NO_THROW(writer.drainIfRunning());
}

}  // namespace rtp_llm
