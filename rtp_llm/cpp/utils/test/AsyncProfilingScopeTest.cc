#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <limits>
#include <future>
#include <type_traits>

#include "rtp_llm/cpp/utils/AsyncProfilingScope.h"

namespace rtp_llm {

TEST(AsyncProfilingScopeTest, ExecutionKeepsActualWorkerAndInlineThreadIdentities) {
    auto             session    = AsyncProfileSession::start();
    auto             owner_tid  = asyncProfileThreadId();
    int64_t          worker_tid = 0;
    AsyncProfileSpan inline_span("prepare", 1, 0, 1);
    inline_span.finish();
    std::thread worker([&] {
        worker_tid = asyncProfileThreadId();
        AsyncProfileSpan span("disk_read", 1, 0, 1);
        span.finish();
    });
    worker.join();
    auto events = session->finish();
    ASSERT_EQ(events.size(), 2u);
    ASSERT_NE(owner_tid, worker_tid);
    EXPECT_EQ(events[0].record.start_tid, owner_tid);
    EXPECT_EQ(events[0].end_tid, owner_tid);
    EXPECT_EQ(events[1].record.start_tid, worker_tid);
    EXPECT_EQ(events[1].end_tid, worker_tid);
    EXPECT_EQ(events[1].pid, getpid());
    EXPECT_FALSE(events[1].thread_mismatch);
    EXPECT_NE(events[0].span_id, events[1].span_id);
}

TEST(AsyncProfilingScopeTest, WorkerNameCacheRefreshesForEachSessionAndSnapshotsOwnNames) {
    static_assert(std::is_trivially_copyable<AsyncProfileRecord>::value, "worker records must remain bounded POD");
    std::promise<void> proceed[2], recorded[2];
    auto               first = AsyncProfileSession::start();
    std::thread        worker([&] {
        const char* names[] = {"btc_first", "btc_second"};
        for (size_t i = 0; i < 2; ++i) {
            proceed[i].get_future().wait();
#if defined(__linux__)
            EXPECT_EQ(pthread_setname_np(pthread_self(), names[i]), 0);
#elif defined(__APPLE__)
            EXPECT_EQ(pthread_setname_np(names[i]), 0);
#endif
            AsyncProfileSpan span("disk_read", 42, 0, 1);
            span.finish();
            recorded[i].set_value();
        }
    });
    proceed[0].set_value();
    recorded[0].get_future().wait();
    auto a = first->finish();
    first.reset();
    auto second = AsyncProfileSession::start();
    proceed[1].set_value();
    recorded[1].get_future().wait();
    worker.join();
    auto b = second->finish();
    ASSERT_EQ(a.size(), 1u);
    ASSERT_EQ(b.size(), 1u);
    EXPECT_EQ(a[0].record.start_tid, b[0].record.start_tid);
    EXPECT_EQ(a[0].record.start_thread_name, "btc_first");
    EXPECT_EQ(b[0].record.start_thread_name, "btc_second");
}

TEST(AsyncProfilingScopeTest, ExplicitKindsSurviveCrossThreadCompletion) {
    auto             session = AsyncProfileSession::start();
    AsyncProfileSpan total("total", 1, 0, 1, 0, -1, 77, "host_to_device", AsyncProfileKind::Lifecycle);
    AsyncProfileSpan queue("queue", 1, 0, 1, 0, -1, 77, "host_to_device", AsyncProfileKind::Wait);
    AsyncProfileSpan crossed("execution", 1, 0, 1);
    std::thread      worker([&] {
        queue.finish();
        crossed.finish();
        total.finish();
    });
    worker.join();
    auto events = session->finish();
    ASSERT_EQ(events.size(), 3u);
    EXPECT_EQ(events[0].record.kind, AsyncProfileKind::Lifecycle);
    EXPECT_EQ(events[1].record.kind, AsyncProfileKind::Wait);
    EXPECT_TRUE(events[2].thread_mismatch);
    EXPECT_NE(events[2].record.start_tid, events[2].end_tid);
}

TEST(AsyncProfilingScopeTest, CompletionSpansRetainScopeOperationAndDirectionAcrossThreads) {
    auto session = AsyncProfileSession::start();
    auto begin   = AsyncProfileSession::steadyNowUs();
    auto span    = AsyncProfileSpan::fromSteady(
        "completion_execute", 42, 0, 8, begin, -1, 77, "host_to_device", "business_completion", "load");
    std::thread worker([span, begin] { span.finishAtSteady(begin + 25); });
    worker.join();
    auto events = session->finish(AsyncProfileSession::nowUs() + 1000);
    ASSERT_EQ(events.size(), 1u);
    EXPECT_NE(events[0].name.find("cache.transfer.host_to_device.completion_execute"), std::string::npos);
    EXPECT_NE(events[0].name.find("scope=business_completion,operation=load"), std::string::npos);
    EXPECT_NE(events[0].name.find("request_id=42"), std::string::npos);
    EXPECT_NE(events[0].name.find("transfer_id=77"), std::string::npos);
    EXPECT_EQ(events[0].end_us - events[0].start_us, 25);
}

TEST(AsyncProfilingScopeTest, CompletionMetadataDoesNotLeakIntoNextSession) {
    auto first  = AsyncProfileSession::start();
    auto span   = AsyncProfileSpan::fromSteady("completion_total",
                                               42,
                                               0,
                                               8,
                                               AsyncProfileSession::steadyNowUs(),
                                               -1,
                                               77,
                                               "device_to_host",
                                               "coordinator_completion",
                                               "evict");
    auto events = first->finish();
    ASSERT_EQ(events.size(), 1u);
    EXPECT_NE(events[0].name.find("operation=evict"), std::string::npos);
    EXPECT_NE(events[0].name.find("incomplete=1"), std::string::npos);
    auto second = AsyncProfileSession::start();
    span.finish();
    EXPECT_TRUE(second->finish().empty());
}

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
    EXPECT_TRUE(events[0].end_clipped);
    EXPECT_EQ(events[0].end_tid, 0);
    EXPECT_NE(events[0].name.find("incomplete=1"), std::string::npos);
    auto second = AsyncProfileSession::start();
    pending.finish();
    EXPECT_TRUE(second->finish().empty());
    EXPECT_TRUE(first->finish().empty());
}

TEST(AsyncProfilingScopeTest, CaptureWorkThatStartedBeforeSessionAndDoNotClearNewSession) {
    auto preceding = AsyncProfileSpan::fromSteady(
        "h2d", 1, 0, 1, AsyncProfileSession::steadyNowUs() - 1000, -1, 0, "disk_to_device");
    auto first = AsyncProfileSession::start();
    preceding.finish();
    auto second = AsyncProfileSession::start();
    auto events = first->finish();
    ASSERT_EQ(events.size(), 1u);
    EXPECT_TRUE(events[0].start_clipped);
    EXPECT_EQ(events[0].record.start_tid, 0);  // Start occurred before observation was enabled.
    EXPECT_TRUE(events[0].record.start_thread_name.empty());
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
    auto   events    = session->finish();
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

TEST(AsyncProfilingScopeTest, RequestLifecycleUsesAcceptanceAndCommittedTerminal) {
    auto                session  = AsyncProfileSession::start(32);
    const auto          accepted = AsyncProfileSession::nowUs() - 1000;
    AsyncProfileRequest request;
    request.begin(42, accepted, 123);
    request.observe("loading_cache");
    request.observe("loading_cache");  // no phase split on repeated polls
    request.observe("running");
    request.observe("waiting", "finished");
    request.observe("running", "error");  // terminal is exactly once
    const auto events = session->finish();
    ASSERT_EQ(events.size(), 5u);  // root, waiting, loading, running, terminal
    const auto& root = events.front();
    EXPECT_EQ(root.record.stage, "request_lifetime");
    EXPECT_EQ(root.record.client_request_id, 123);
    EXPECT_TRUE(root.start_clipped);
    EXPECT_FALSE(root.end_clipped);
    EXPECT_EQ(root.record.start_us, accepted);
    EXPECT_EQ(events.back().record.stage, "finished");
    for (const auto& e : events) {
        EXPECT_EQ(e.record.request_id, 42);
        EXPECT_EQ(e.record.kind, AsyncProfileKind::Lifecycle);
        EXPECT_GE(e.start_us, root.start_us);
        EXPECT_LE(e.end_us, root.end_us);
    }
}

TEST(AsyncProfilingScopeTest, RequestCrossesCaptureWindowsAndKeepsOpenRoot) {
    AsyncProfileRequest request;
    request.begin(9, AsyncProfileSession::nowUs() - 10000);
    auto first = AsyncProfileSession::start(16);
    request.observe("loading_cache");
    auto events = first->finish();
    ASSERT_FALSE(events.empty());
    EXPECT_TRUE(events.front().start_clipped);
    EXPECT_TRUE(events.front().end_clipped);
    auto second = AsyncProfileSession::start(16);
    request.observe("running");
    request.observe("running", "cancelled");
    events = second->finish();
    ASSERT_FALSE(events.empty());
    EXPECT_TRUE(events.front().start_clipped);
    EXPECT_FALSE(events.front().end_clipped);
    EXPECT_EQ(events.back().record.stage, "cancelled");
}

TEST(AsyncProfilingScopeTest, TaskLinksAreBoundedDeduplicatedAndCaptureLocal) {
    auto session = AsyncProfileSession::start(8);
    for (int i = 0; i < 100; ++i)
        AsyncProfileSession::taskLink(3, 1, 2, 42);
    AsyncProfileSpan stage("h2d", 42, 0, 1, 0, -1, 3);
    stage.finish();
    AsyncProfileSession::taskLink(4, 4, 0, 43, "load", 1);
    auto events = session->finish();
    ASSERT_EQ(events.size(), 2u);
    EXPECT_EQ(events[0].record.workflow_id, 1u);
    EXPECT_EQ(events[0].record.parent_transfer_id, 2u);
    EXPECT_EQ(events[1].record.stage, "workflow_dependency");
    EXPECT_EQ(events[1].record.related_workflow_id, 1u);
    session = AsyncProfileSession::start(8);
    AsyncProfileSession::taskLink(3, 1, 2, 42);
    AsyncProfileSpan next("h2d", 42, 0, 1, 0, -1, 3);
    next.finish();
    const auto second = session->finish();
    ASSERT_EQ(second.size(), 1u);
    EXPECT_EQ(second.front().record.workflow_id, 1u);
}

}  // namespace rtp_llm
