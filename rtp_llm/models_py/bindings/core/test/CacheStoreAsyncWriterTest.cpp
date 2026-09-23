#include "gtest/gtest.h"
#define private public
#include "rtp_llm/models_py/bindings/core/CacheStoreAsyncWriter.h"

#include <atomic>
#include <chrono>
#include <mutex>
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

// Contract change: init() while RUNNING now SELF-HEALS (drains the abandoned cycle via reset()
// and starts fresh) instead of throwing. Rationale: PyWrappedModel can skip waitAllDone() when a
// forward throws, and an async prepare (a separate thread) can be orphaned by a cancel, so the
// writer can legitimately be left RUNNING between requests. The old RTP_LLM_CHECK(state==IDLE)
// made init() throw every cycle -> the engine-loop catch-all swallowed and retried -> an unbounded
// "already RUNNING" retry storm that wedged serving. The deeper intent of the old test ("writer
// should still be functional after the second init") is preserved and strengthened.
TEST_F(CacheStoreAsyncWriterTest, InitWhileRunningSelfHeals) {
    CacheStoreAsyncWriter writer;
    writer.init();

    ASSERT_NO_THROW(writer.init());

    // Writer is fully functional after the self-heal.
    std::atomic<int> counter{0};
    writer.submit([&counter]() { counter.fetch_add(1); });
    writer.waitAllDone();
    ASSERT_EQ(1, counter.load());
}

// The self-heal must DRAIN the abandoned cycle's in-flight task before starting the new one, so
// no stale background work races the next request.
TEST_F(CacheStoreAsyncWriterTest, InitSelfHealDrainsAbandonedTasks) {
    CacheStoreAsyncWriter writer;
    writer.init();
    std::atomic<int> abandoned{0};
    writer.submit([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        abandoned.fetch_add(1);
    });
    // Abandon the cycle (never call waitAllDone) and re-init: init() must drain the in-flight task.
    writer.init();
    ASSERT_EQ(1, abandoned.load());
    ASSERT_TRUE(writer.state_ == CacheStoreAsyncWriter::State::RUNNING);
    writer.waitAllDone();
    ASSERT_TRUE(writer.state_ == CacheStoreAsyncWriter::State::IDLE);
}

// reset() is the recovery hook the forward() RAII guard calls: it drains in-flight work, forces
// IDLE, discards the stored exception (does NOT re-throw, unlike waitAllDone), and is idempotent.
TEST_F(CacheStoreAsyncWriterTest, ResetDrainsAndIsIdempotent) {
    CacheStoreAsyncWriter writer;
    writer.init();
    std::atomic<int> counter{0};
    writer.submit([&]() { counter.fetch_add(1); });

    writer.reset();
    ASSERT_TRUE(writer.state_ == CacheStoreAsyncWriter::State::IDLE);
    ASSERT_EQ(1, counter.load());

    // Idempotent: reset() on an IDLE writer is a safe no-op.
    ASSERT_NO_THROW(writer.reset());
    ASSERT_TRUE(writer.state_ == CacheStoreAsyncWriter::State::IDLE);

    // Reusable afterward.
    writer.init();
    writer.submit([&]() { counter.fetch_add(1); });
    writer.waitAllDone();
    ASSERT_EQ(2, counter.load());
}

TEST_F(CacheStoreAsyncWriterTest, InitWaitCycle) {
    CacheStoreAsyncWriter writer;
    std::vector<int>      order;
    std::mutex            order_mutex;

    writer.init();
    writer.submit([&]() {
        std::lock_guard<std::mutex> lock(order_mutex);
        order.push_back(1);
    });
    writer.submit([&]() {
        std::lock_guard<std::mutex> lock(order_mutex);
        order.push_back(2);
    });
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

}  // namespace rtp_llm
