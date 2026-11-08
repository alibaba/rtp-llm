#include "aios/navi_sched/collector/PPCacheShardReportR.h"
#include "unittest/unittest.h"
#include <functional>
#include <thread>

using namespace navi_sched;
class PPCacheShardReportRTest : public TESTBASE {};
namespace {
class CapturePipe : public navi::AsyncPipe {
public:
    CapturePipe() : navi::AsyncPipe(nullptr, navi::AS_REQUIRED) {}
    navi::ErrorCode setData(const navi::DataPtr &data) override
    {
        if (reject.load()) return navi::EC_ABORT;
        {
            std::lock_guard<std::mutex> lock(mutex);
            values.push_back(std::dynamic_pointer_cast<PPCacheShardUpdateData>(data));
        }
        if (onData) onData();
        return navi::EC_NONE;
    }
    navi::ErrorCode setEof() override
    {
        eof.store(true);
        return navi::EC_NONE;
    }
    void terminate() override { terminated.store(true); }
    std::mutex mutex;
    std::vector<PPCacheShardUpdateDataPtr> values;
    std::atomic<bool> reject{false};
    std::atomic<bool> eof{false};
    std::atomic<bool> terminated{false};
    std::function<void()> onData;
};
}

TEST_F(PPCacheShardReportRTest, testInitialSnapshotPrecedesEveryIncrementalChange)
{
    PPCacheShardReportR report;
    ASSERT_TRUE(report.initializeIdentity("pp-a", 1, 4));
    PPCacheShardUpdate seed;
    seed.addedFull = {5, 3};
    seed.addedTail = {5};
    ASSERT_TRUE(report.publish(seed));
    auto pipe = std::make_shared<CapturePipe>();
    ASSERT_TRUE(report.subscribe("pp-a", 1, 4, "agent", 7, pipe));
    PPCacheShardUpdate delta;
    delta.removedFull = {3};
    ASSERT_TRUE(report.publish(delta));
    ASSERT_EQ(2, pipe->values.size());
    ASSERT_TRUE(pipe->values[0]->update.fullSnapshot);
    ASSERT_EQ(std::vector<uint64_t>({3, 5}), pipe->values[0]->update.addedFull);
    ASSERT_EQ(1, pipe->values[0]->update.shardId);
    ASSERT_EQ(7, pipe->values[0]->update.streamGeneration);
    ASSERT_EQ(1, pipe->values[0]->update.sequence);
    ASSERT_EQ(2, pipe->values[1]->update.sequence);
    ASSERT_EQ(std::vector<uint64_t>({3}), pipe->values[1]->update.removedFull);
}

TEST_F(PPCacheShardReportRTest, testFailureAndCancellationArePublishedWithoutCacheChanges)
{
    PPCacheShardReportR report;
    report.initializeIdentity("pp-a", 0, 1);
    auto pipe = std::make_shared<CapturePipe>();
    report.subscribe("pp-a", 0, 1, "agent", 1, pipe);
    for (auto status : {PPShardRequestStatus::SUCCEEDED, PPShardRequestStatus::FAILED,
                        PPShardRequestStatus::CANCELLED}) {
        PPCacheShardUpdate event;
        event.terminals = {{{10, 1}, status}};
        ASSERT_TRUE(report.publish(event));
        ASSERT_EQ(status, pipe->values.back()->update.terminals.at(0).status);
        ASSERT_TRUE(pipe->values.back()->valid());
    }
    ASSERT_EQ(4, pipe->values.size());
}

TEST_F(PPCacheShardReportRTest, testConcurrentProducersKeepOneOrderedStream)
{
    PPCacheShardReportR report;
    report.initializeIdentity("pp-a", 0, 1);
    auto pipe = std::make_shared<CapturePipe>();
    report.subscribe("pp-a", 0, 1, "agent", 1, pipe);
    std::vector<std::thread> threads;
    for (size_t worker = 0; worker < 8; ++worker) {
        threads.emplace_back([&, worker]() {
            for (size_t index = 0; index < 100; ++index) {
                PPCacheShardUpdate delta;
                delta.addedFull = {worker * 100 + index};
                report.publish(delta);
            }
        });
    }
    for (auto &thread : threads) thread.join();
    ASSERT_EQ(801, pipe->values.size());
    for (size_t i = 0; i < pipe->values.size(); ++i)
        ASSERT_EQ(i + 1, pipe->values[i]->update.sequence);
}

TEST_F(PPCacheShardReportRTest,
       testReconnectionStartsWithCurrentStateAndOldUnsubscribeCannotDetachIt)
{
    PPCacheShardReportR report;
    report.initializeIdentity("pp-a", 0, 1);
    auto oldPipe = std::make_shared<CapturePipe>();
    report.subscribe("pp-a", 0, 1, "agent", 1, oldPipe);
    PPCacheShardUpdate delta;
    delta.addedFull = {8};
    report.publish(delta);
    ASSERT_TRUE(report.unsubscribe("agent", oldPipe));
    auto newPipe = std::make_shared<CapturePipe>();
    ASSERT_TRUE(report.subscribe("pp-a", 0, 1, "agent", 2, newPipe));
    ASSERT_FALSE(report.unsubscribe("agent", oldPipe));
    ASSERT_EQ(1, newPipe->values.size());
    ASSERT_EQ(1, newPipe->values[0]->update.sequence);
    ASSERT_EQ(2, newPipe->values[0]->update.streamGeneration);
    ASSERT_EQ(std::vector<uint64_t>({8}), newPipe->values[0]->update.addedFull);
}

TEST_F(PPCacheShardReportRTest, testPipeFailureTerminatesStreamAndAllowsFreshSubscription)
{
    PPCacheShardReportR report;
    report.initializeIdentity("pp-a", 0, 1);
    auto oldPipe = std::make_shared<CapturePipe>();
    report.subscribe("pp-a", 0, 1, "agent", 1, oldPipe);
    oldPipe->reject.store(true);
    PPCacheShardUpdate event;
    event.terminals = {{{10, 1}, PPShardRequestStatus::FAILED}};
    ASSERT_TRUE(report.publish(event));
    ASSERT_TRUE(oldPipe->eof.load());
    auto newPipe = std::make_shared<CapturePipe>();
    ASSERT_TRUE(report.subscribe("pp-a", 0, 1, "agent", 2, newPipe));
    ASSERT_FALSE(report.unsubscribe("agent", oldPipe));
}

TEST_F(PPCacheShardReportRTest,
       testReentrantProducerDoesNotHoldSubscriberMutexDuringDelivery)
{
    PPCacheShardReportR report;
    report.initializeIdentity("pp-a", 0, 1);
    auto pipe = std::make_shared<CapturePipe>();
    std::atomic<bool> published{false};
    pipe->onData = [&]() {
        if (published.exchange(true)) return;
        PPCacheShardUpdate delta;
        delta.addedFull = {9};
        report.publish(delta);
    };
    ASSERT_TRUE(report.subscribe("pp-a", 0, 1, "agent", 1, pipe));
    ASSERT_EQ(2, pipe->values.size());
    ASSERT_EQ(1, pipe->values[0]->update.sequence);
    ASSERT_EQ(2, pipe->values[1]->update.sequence);
}

TEST_F(PPCacheShardReportRTest, testIdentityCannotChangeAndWrongGroupCannotSubscribe)
{
    PPCacheShardReportR report;
    ASSERT_TRUE(report.initializeIdentity("pp-a", 0, 4));
    ASSERT_FALSE(report.initializeIdentity("pp-b", 0, 4));
    ASSERT_FALSE(report.initializeIdentity("pp-a", 1, 4));
    auto pipe = std::make_shared<CapturePipe>();
    ASSERT_FALSE(report.subscribe("pp-b", 0, 4, "agent", 1, pipe));
    ASSERT_FALSE(report.subscribe("pp-a", 1, 4, "agent", 1, pipe));
    ASSERT_FALSE(report.subscribe("pp-a", 0, 3, "agent", 1, pipe));
    ASSERT_TRUE(report.subscribe("pp-a", 0, 4, "agent", 1, pipe));
}

TEST_F(PPCacheShardReportRTest, testStopEndsStreamAndRejectsNewPublication)
{
    PPCacheShardReportR report;
    report.initializeIdentity("pp-a", 0, 1);
    auto pipe = std::make_shared<CapturePipe>();
    report.subscribe("pp-a", 0, 1, "agent", 1, pipe);
    report.stop();
    ASSERT_TRUE(pipe->eof.load());
    ASSERT_FALSE(report.publish(PPCacheShardUpdate{}));
    ASSERT_FALSE(report.subscribe("pp-a", 0, 1, "agent", 2, pipe));
}

TEST_F(PPCacheShardReportRTest, testPeerStateSurvivesFullCacheReplacementAndReconnect)
{
    PPCacheShardReportR report;
    ASSERT_TRUE(report.initializeIdentity("pp-a", 0, 2));
    auto pipe = std::make_shared<CapturePipe>();
    ASSERT_TRUE(report.subscribe("pp-a", 0, 2, "agent", 1, pipe));
    EXPECT_EQ(0, pipe->values.back()->update.peerCacheReadAvailable);
    PPCacheShardUpdate ready;
    ready.peerCacheReadAvailable = 1;
    ASSERT_TRUE(report.publish(ready));
    EXPECT_EQ(1, pipe->values.back()->update.peerCacheReadAvailable);
    PPCacheShardUpdate replacement;
    replacement.fullSnapshot = true;
    replacement.addedFull = {8};
    ASSERT_TRUE(report.publish(replacement));
    EXPECT_EQ(1, pipe->values.back()->update.peerCacheReadAvailable);
    ASSERT_TRUE(report.unsubscribe("agent", pipe));
    auto next = std::make_shared<CapturePipe>();
    ASSERT_TRUE(report.subscribe("pp-a", 0, 2, "agent", 2, next));
    EXPECT_EQ(1, next->values.back()->update.peerCacheReadAvailable);
    EXPECT_EQ(std::vector<uint64_t>({8}), next->values.back()->update.addedFull);
}
