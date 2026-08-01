#include "gtest/gtest.h"
#define private public
#include "rtp_llm/models_py/bindings/core/CacheStoreAsyncWriter.h"

#include <atomic>
#include <chrono>
#include <future>
#include <stdexcept>
#include <thread>
#include <vector>

namespace rtp_llm {

class CacheStoreAsyncWriterTest: public ::testing::Test {};

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

TEST_F(CacheStoreAsyncWriterTest, WaitIncludesDelayedStoreCompletion) {
    using namespace std::chrono_literals;

    CacheStoreAsyncWriter writer;
    writer.init(/*track_store_completions=*/true);

    std::promise<CacheStoreAsyncWriter::StoreCompletionCallback> registered;
    auto registered_future = registered.get_future();
    writer.submit([&writer, &registered]() { registered.set_value(writer.registerStoreCompletion()); });
    auto completion = registered_future.get();

    auto waiter = std::async(std::launch::async, [&writer]() { writer.waitAllDone(); });
    EXPECT_EQ(std::future_status::timeout, waiter.wait_for(20ms));

    completion(nullptr);
    ASSERT_EQ(std::future_status::ready, waiter.wait_for(1s));
    ASSERT_NO_THROW(waiter.get());
}

TEST_F(CacheStoreAsyncWriterTest, DeferredPublicationBlocksNextCycleButNotSubmissionFinish) {
    using namespace std::chrono_literals;

    CacheStoreAsyncWriter writer;
    writer.init(/*track_store_completions=*/true);

    std::promise<CacheStoreAsyncWriter::StoreCompletionCallback> registered;
    auto registered_future = registered.get_future();
    writer.submit([&writer, &registered]() { registered.set_value(writer.registerStoreCompletion()); });
    auto completion = registered_future.get();

    writer.finishSubmissions();
    ASSERT_TRUE(writer.state_ == CacheStoreAsyncWriter::State::IDLE);
    ASSERT_ANY_THROW(writer.init());

    auto waiter = std::async(std::launch::async, [&writer]() { writer.waitStoreCompletions(); });
    EXPECT_EQ(std::future_status::timeout, waiter.wait_for(20ms));
    completion(nullptr);
    ASSERT_EQ(std::future_status::ready, waiter.wait_for(1s));
    ASSERT_NO_THROW(waiter.get());

    ASSERT_NO_THROW(writer.init());
    ASSERT_NO_THROW(writer.waitAllDone());
}

TEST_F(CacheStoreAsyncWriterTest, StoreCompletionFailureIsPropagated) {
    CacheStoreAsyncWriter writer;
    writer.init(/*track_store_completions=*/true);

    std::promise<CacheStoreAsyncWriter::StoreCompletionCallback> registered;
    auto registered_future = registered.get_future();
    writer.submit([&writer, &registered]() { registered.set_value(writer.registerStoreCompletion()); });
    auto completion = registered_future.get();
    completion(std::make_exception_ptr(std::runtime_error("store publish failed")));

    try {
        writer.waitAllDone();
        FAIL() << "expected store completion exception";
    } catch (const std::runtime_error& e) {
        ASSERT_STREQ("store publish failed", e.what());
    }
    ASSERT_TRUE(writer.state_ == CacheStoreAsyncWriter::State::IDLE);

    writer.init();
    ASSERT_NO_THROW(writer.waitAllDone());
    ASSERT_TRUE(writer.state_ == CacheStoreAsyncWriter::State::IDLE);
}

TEST_F(CacheStoreAsyncWriterTest, StoreCompletionsDoNotDriftAcrossManyCycles) {
    CacheStoreAsyncWriter writer;

    for (int cycle = 0; cycle < 1000; ++cycle) {
        writer.init(/*track_store_completions=*/true);
        std::promise<CacheStoreAsyncWriter::StoreCompletionCallback> registered;
        auto registered_future = registered.get_future();
        writer.submit([&writer, &registered]() { registered.set_value(writer.registerStoreCompletion()); });
        auto completion = registered_future.get();
        completion(nullptr);
        writer.waitAllDone();
        ASSERT_TRUE(writer.state_ == CacheStoreAsyncWriter::State::IDLE);
    }
}

}  // namespace rtp_llm
