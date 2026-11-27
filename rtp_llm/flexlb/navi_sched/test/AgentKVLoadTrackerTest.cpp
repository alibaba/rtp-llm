#include "aios/navi_sched/resource/AgentKVLoadTracker.h"
#include "unittest/unittest.h"
#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
namespace navi_sched {
namespace {
using Tracker = AgentKVLoadTracker;
const Tracker::Key KEY{99,1234,1};
TEST(AgentKVLoadTrackerTest, OneUnifiedEngineCoreResultCompletesTheRequest) {
    Tracker t(99); ASSERT_TRUE(t.begin(KEY,100));
    EXPECT_EQ(Tracker::State::PREPARING,t.snapshot(KEY)->state);
    auto r=t.complete(KEY,true,200); ASSERT_TRUE(r.terminalChanged);
    ASSERT_TRUE(r.snapshot); EXPECT_EQ(Tracker::Update::READY,r.update);
    EXPECT_EQ(Tracker::State::READY,r.snapshot->state);
    EXPECT_EQ(200,r.snapshot->readyObservedNs);
}
TEST(AgentKVLoadTrackerTest, InvalidAndMissingIdentityCannotDefaultToReady) {
    Tracker t(99);
    for(auto key:std::vector<Tracker::Key>{{0,1234,1},{99,0,1},{99,1234,0}}) {
        EXPECT_FALSE(t.begin(key,100)); EXPECT_EQ(Tracker::Update::INVALID,t.complete(key,true,200).update);
    }
    EXPECT_FALSE(t.begin(KEY,0)); EXPECT_FALSE(t.begin(KEY,-1));
    EXPECT_EQ(0u,t.activeCount());
}
TEST(AgentKVLoadTrackerTest, OldSessionAndUnknownRequestAreIgnored) {
    Tracker t(99); ASSERT_TRUE(t.begin(KEY,100));
    EXPECT_EQ(Tracker::Update::STALE,t.complete({98,1234,1},true,200).update);
    EXPECT_EQ(Tracker::Update::STALE,t.complete({99,4321,1},true,200).update);
    EXPECT_EQ(Tracker::State::PREPARING,t.snapshot(KEY)->state);
}
TEST(AgentKVLoadTrackerTest, ObservedClockCannotPrecedeRegistration) {
    Tracker t(99); ASSERT_TRUE(t.begin(KEY,100));
    EXPECT_EQ(Tracker::Update::INVALID,t.complete(KEY,true,99).update);
    EXPECT_EQ(Tracker::Update::INVALID,t.complete(KEY,true,0).update);
    EXPECT_EQ(Tracker::State::PREPARING,t.snapshot(KEY)->state);
}
TEST(AgentKVLoadTrackerTest, DuplicateCompletionDoesNotRepeatMetricsOrMoveTheClock) {
    Tracker t(99); ASSERT_TRUE(t.begin(KEY,100)); t.complete(KEY,true,200);
    auto r=t.complete(KEY,true,900); EXPECT_EQ(Tracker::Update::DUPLICATE,r.update);
    EXPECT_FALSE(r.terminalChanged); EXPECT_EQ(200,r.snapshot->readyObservedNs);
    auto m=t.takeLatencyWindowStats(); EXPECT_EQ(1u,m.readyCount); EXPECT_EQ(100.0L,m.totalPreparationNs);
}
TEST(AgentKVLoadTrackerTest, FailedUnifiedResultNeverBecomesReady) {
    Tracker t(99); ASSERT_TRUE(t.begin(KEY,100));
    EXPECT_TRUE(t.complete(KEY,false,200).terminalChanged);
    EXPECT_EQ(Tracker::Update::CONFLICT,t.complete(KEY,true,300).update);
    EXPECT_EQ(Tracker::State::FAILED,t.snapshot(KEY)->state);
    EXPECT_EQ(0,t.snapshot(KEY)->readyObservedNs);
    auto m=t.takeLatencyWindowStats(); EXPECT_EQ(0u,m.readyCount); EXPECT_EQ(1u,m.failedCount);
}
TEST(AgentKVLoadTrackerTest, ConflictingLateFailureInvalidatesReadyOnce) {
    Tracker t(99); ASSERT_TRUE(t.begin(KEY,100)); t.complete(KEY,true,200);
    auto r=t.complete(KEY,false,300); EXPECT_TRUE(r.terminalChanged);
    EXPECT_EQ(Tracker::State::FAILED,r.snapshot->state);
    EXPECT_FALSE(t.complete(KEY,false,400).terminalChanged);
    auto m=t.takeLatencyWindowStats(); EXPECT_EQ(1u,m.failedCount);
}
TEST(AgentKVLoadTrackerTest, CancellationAndRetirementIgnoreLaterCompletion) {
    Tracker t(99); ASSERT_TRUE(t.begin(KEY,100)); ASSERT_TRUE(t.cancel(KEY));
    EXPECT_EQ(Tracker::Update::STALE,t.complete(KEY,true,200).update);
    ASSERT_TRUE(t.retire(KEY)); EXPECT_EQ(0u,t.activeCount());
    EXPECT_FALSE(t.retire(KEY)); EXPECT_FALSE(t.cancel(KEY));
    EXPECT_EQ(Tracker::Update::STALE,t.complete(KEY,true,300).update);
    EXPECT_EQ(0u,t.takeLatencyWindowStats().readyCount);
}
TEST(AgentKVLoadTrackerTest, PendingLoadCannotBeSilentlyReplaced) {
    Tracker t(99); ASSERT_TRUE(t.begin(KEY,100));
    EXPECT_FALSE(t.begin({99,1234,2},150)); ASSERT_TRUE(t.cancel(KEY));
    ASSERT_TRUE(t.begin({99,1234,2},200));
    EXPECT_EQ(Tracker::Update::STALE,t.complete(KEY,true,250).update);
    EXPECT_EQ(Tracker::State::PREPARING,t.snapshot({99,1234,2})->state);
    EXPECT_TRUE(t.complete({99,1234,2},true,300).terminalChanged);
}
TEST(AgentKVLoadTrackerTest, RetiredGenerationCannotBeResurrected) {
    Tracker t(99); ASSERT_TRUE(t.begin(KEY,100)); ASSERT_TRUE(t.retire(KEY));
    EXPECT_FALSE(t.begin(KEY,200)); ASSERT_TRUE(t.begin({99,5678,2},200));
    EXPECT_FALSE(t.begin({99,1234,1},300));
}
TEST(AgentKVLoadTrackerTest, RepeatedBeginPreservesClockAndTerminalState) {
    Tracker t(99); ASSERT_TRUE(t.begin(KEY,100));
    ASSERT_EQ(100,t.begin(KEY,150)->registeredNs); t.complete(KEY,true,200);
    auto r=t.begin(KEY,500); ASSERT_TRUE(r); EXPECT_EQ(Tracker::State::READY,r->state);
    EXPECT_EQ(100,r->registeredNs); EXPECT_EQ(200,r->readyObservedNs);
}
TEST(AgentKVLoadTrackerTest, MissingAndStaleSnapshotDoNotLeakNewGeneration) {
    Tracker t(99); EXPECT_FALSE(t.snapshot(KEY)); ASSERT_TRUE(t.begin(KEY,100));
    t.complete(KEY,true,200); ASSERT_TRUE(t.begin({99,1234,2},300));
    EXPECT_FALSE(t.snapshot(KEY)); EXPECT_TRUE(t.snapshot({99,1234,2}));
    EXPECT_FALSE(t.snapshot({98,1234,2}));
}
TEST(AgentKVLoadTrackerTest, CompletionLatencyEndsAtReceiptRatherThanLaterObservation) {
    Tracker t(99); ASSERT_TRUE(t.begin(KEY,100)); t.complete(KEY,true,200);
    EXPECT_EQ(200,t.snapshot(KEY)->readyObservedNs);
    EXPECT_EQ(100,t.takeLatencyWindowStats().maxPreparationNs);
    EXPECT_EQ(0,t.takeLatencyWindowStats().maxPreparationNs);
}
TEST(AgentKVLoadTrackerTest, MultipleRequestMetricsAreWindowedAndFailureIsNotAZeroLatency) {
    Tracker t(99);
    ASSERT_TRUE(t.begin(KEY,100)); t.complete(KEY,true,300);
    ASSERT_TRUE(t.begin({99,2,2},150)); t.complete({99,2,2},true,500);
    ASSERT_TRUE(t.begin({99,3,3},200)); t.complete({99,3,3},false,250);
    auto m=t.takeLatencyWindowStats(); EXPECT_EQ(2u,m.readyCount); EXPECT_EQ(1u,m.failedCount);
    EXPECT_EQ(550.0L,m.totalPreparationNs); EXPECT_EQ(350,m.maxPreparationNs);
    m=t.takeLatencyWindowStats(); EXPECT_EQ(0u,m.readyCount); EXPECT_EQ(0u,m.failedCount);
}
TEST(AgentKVLoadTrackerTest, ConcurrentDuplicateNativeCallbacksHaveOneTransition) {
    Tracker t(99); ASSERT_TRUE(t.begin(KEY,100)); std::atomic<int> changed{0};
    std::vector<std::thread> threads;
    for(int i=0;i<8;++i)threads.emplace_back([&]{if(t.complete(KEY,true,200).terminalChanged)++changed;});
    for(auto &thread:threads)thread.join();
    EXPECT_EQ(1,changed.load()); EXPECT_EQ(1u,t.takeLatencyWindowStats().readyCount);
}
TEST(AgentKVLoadTrackerTest, MeasuresNativeObservationLifecycleWithoutWorkerAggregation) {
    Tracker t(99); constexpr int count=10000; volatile uint64_t checksum=0;
    const auto begin=std::chrono::steady_clock::now();
    for(uint64_t i=1;i<=count;++i){
        const Tracker::Key key{99,i,i};
        auto r=t.begin(key,100); ASSERT_TRUE(r);
        checksum+=t.complete(key,true,200).terminalChanged; ASSERT_TRUE(t.retire(key));
    }
    const auto ns=std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-begin).count()/count;
    std::cout<<"ENGINECORE_KV_LIFECYCLE_BENCHMARK iterations="<<count<<" mean_ns="<<ns<<" checksum="<<checksum<<std::endl;
}
}
}

namespace navi_sched {
TEST(AgentKVLoadTrackerElapsedTest, UsesAgentClockAndFreezesAtReadyWithoutCountingLaterExecution) {
    AgentKVLoadTracker::Snapshot load;load.registeredNs=1000000;
    EXPECT_DOUBLE_EQ(2.5,load.preparationElapsedMs(3500000));
    EXPECT_DOUBLE_EQ(-1.0,load.preparationElapsedMs(999999));
    load.state=AgentKVLoadTracker::State::READY;load.readyObservedNs=3500000;
    EXPECT_DOUBLE_EQ(2.5,load.preparationElapsedMs(99000000));
    load.state=AgentKVLoadTracker::State::FAILED;EXPECT_DOUBLE_EQ(-1.0,load.preparationElapsedMs(99000000));
    load.state=AgentKVLoadTracker::State::CANCELED;EXPECT_DOUBLE_EQ(-1.0,load.preparationElapsedMs(99000000));
}
}
