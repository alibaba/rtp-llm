#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <limits>

#include "rtp_llm/cpp/utils/AsyncProfilingScope.h"

namespace rtp_llm {

TEST(AsyncProfilingScopeTest, BackgroundCompletionHasIdentityAndOriginalTimestamps) {
    auto             session = AsyncProfileSession::start();
    const auto       begin   = AsyncProfileSession::nowUs();
    AsyncProfileSpan span("disk_read", 42, 8, 16, begin, 3, 77);
    std::thread      worker([span, begin] { span.finish(begin + 25); });
    worker.join();
    auto events = session->finish(begin + 100);
    ASSERT_EQ(events.size(), 1u);
    EXPECT_EQ(events[0].start_us, begin);
    EXPECT_EQ(events[0].end_us, begin + 25);
    EXPECT_NE(events[0].name.find("request_id=42,batch=8:16,rank=3,transfer_id=77"), std::string::npos);
    EXPECT_FALSE(AsyncProfileSession::current());
}

TEST(AsyncProfilingScopeTest, IncompleteRangesAreClippedAndLateCompletionCannotEnterNextSession) {
    auto             first = AsyncProfileSession::start();
    const auto       begin = AsyncProfileSession::nowUs();
    AsyncProfileSpan pending("staging_wait", 1, 0, 8, begin);
    auto             events = first->finish(begin + 20);
    ASSERT_EQ(events.size(), 1u);
    EXPECT_EQ(events[0].end_us, begin + 20);
    EXPECT_NE(events[0].name.find("incomplete=1"), std::string::npos);
    auto second = AsyncProfileSession::start();
    pending.finish();
    EXPECT_TRUE(second->finish().empty());
    EXPECT_TRUE(first->finish().empty());
}

TEST(AsyncProfilingScopeTest, CaptureWorkThatStartedBeforeSessionAndDoNotClearNewSession) {
    AsyncProfileSpan preceding("h2d", 1, 0, 1);
    auto             first = AsyncProfileSession::start();
    preceding.finish();
    auto second = AsyncProfileSession::start();
    EXPECT_EQ(first->finish().size(), 1u);
    EXPECT_EQ(AsyncProfileSession::current(), second);
    EXPECT_TRUE(second->finish().empty());
}

TEST(AsyncProfilingScopeTest, BoundedBufferDoesNotRetainUnboundedEvents) {
    AsyncProfileSession session(100, 2);
    EXPECT_TRUE(session.add("a", 110));
    EXPECT_TRUE(session.add("b", 120));
    EXPECT_FALSE(session.add("overflow", 130));
    auto events = session.finish(200);
    ASSERT_EQ(events.size(), 3u);
    EXPECT_EQ(events[0].end_us, 200);
    EXPECT_NE(events.back().name.find("profiler.dropped(count=1)"), std::string::npos);
    EXPECT_FALSE(session.add("closed", 140));
}

TEST(AsyncProfilingScopeTest, ConcurrentWorkersCanCompleteWithoutProfilerTLS) {
    auto                     session = AsyncProfileSession::start();
    std::vector<std::thread> workers;
    for (int rank = 0; rank < 8; ++rank) {
        workers.emplace_back([rank] {
            for (size_t batch = 0; batch < 100; ++batch) {
                AsyncProfileSpan span("h2d", rank, batch, batch + 1);
                span.finish();
            }
        });
    }
    for (auto& worker : workers)
        worker.join();
    auto events = session->finish();
    size_t completed = 0, dropped = 0;
    for (const auto& event : events) {
        const auto loss = event.name.find("profiler.dropped(count=");
        if (loss != std::string::npos) {
            dropped += std::stoull(event.name.substr(loss + 22));
            continue;
        }
        ++completed;
        EXPECT_GE(event.end_us, event.start_us);
        EXPECT_EQ(event.name.find("incomplete=1"), std::string::npos);
    }
    EXPECT_EQ(completed + dropped, 800u);
}

TEST(AsyncProfilingScopeTest, AllocationFailureLeavesActiveSessionUsable) {
    auto session = AsyncProfileSession::start(8);
    ASSERT_NE(session, nullptr);
    EXPECT_FALSE(AsyncProfileSession::start(std::numeric_limits<size_t>::max()));
    EXPECT_EQ(AsyncProfileSession::current(), session);
    AsyncProfileSpan span("h2d", 1, 0, 1);
    span.finish();
    EXPECT_EQ(session->finish().size(), 1u);
    EXPECT_FALSE(AsyncProfileSession::enabled());
}

TEST(AsyncProfilingScopeTest, FullBufferDropsWithoutAllocatingOrWaitingForExport) {
    auto session = AsyncProfileSession::start(0);
    ASSERT_NE(session, nullptr);
    for (int i = 0; i < 10000; ++i) {
        AsyncProfileSpan span("h2d", i, 0, 1);
        span.finish();
    }
    auto events = session->finish();
    ASSERT_EQ(events.size(), 1u);
    EXPECT_NE(events[0].name.find("dropped(count=10000)"), std::string::npos);
}

TEST(AsyncProfilingScopeTest, ClosingWhileWorkersRecordDoesNotRetainOrPoisonNextSession) {
    auto session = AsyncProfileSession::start(10000);
    ASSERT_NE(session, nullptr);
    std::atomic<size_t>      started{0};
    std::atomic<bool>        release{false};
    std::vector<std::thread> workers;
    for (int rank = 0; rank < 8; ++rank)
        workers.emplace_back([&, rank] {
            AsyncProfileSpan initial("h2d", rank, 0, 1);
            started.fetch_add(1);
            while (!release.load())
                std::this_thread::yield();
            initial.finish();
            for (int i = 0; i < 10000; ++i) {
                AsyncProfileSpan span("disk_read", rank, i, i + 1);
                span.finish();
            }
        });
    while (started.load() != 8)
        std::this_thread::yield();
    release.store(true);
    auto events = session->finish();
    for (auto& worker : workers)
        worker.join();
    EXPECT_LE(events.size(), 10001u);
    EXPECT_FALSE(AsyncProfileSession::enabled());
    auto next = AsyncProfileSession::start(8);
    ASSERT_NE(next, nullptr);
    EXPECT_TRUE(next->finish().empty());
}

}  // namespace rtp_llm
