#include "aios/navi_sched/collector/PPRequestFinishedR.h"
#include "unittest/unittest.h"
#include <thread>

using namespace navi_sched;
class PPRequestFinishedRTest : public TESTBASE {};
namespace {
class CapturePipe : public navi::AsyncPipe {
public:
    CapturePipe() : navi::AsyncPipe(nullptr, navi::AS_REQUIRED) {}
    navi::ErrorCode setData(const navi::DataPtr &data) override
    {
        if (reject) return navi::EC_ABORT;
        values.push_back(std::dynamic_pointer_cast<PPCacheShardUpdateData>(data));
        if (onData) onData();
        return navi::EC_NONE;
    }
    navi::ErrorCode setEof() override
    {
        terminated = true;
        return navi::EC_NONE;
    }
    void terminate() override { terminated = true; }
    std::vector<PPCacheShardUpdateDataPtr> values;
    std::function<void()> onData;
    bool reject = false;
    bool terminated = false;
};
PPShardRequestTerminal finished(uint64_t id = 10)
{
    return {{id, 1}, PPShardRequestStatus::FAILED};
}
}

TEST_F(PPRequestFinishedRTest, testBuffersBeforeKernelSubscriptionAndPreservesOrder)
{
    PPRequestFinishedR source;
    ASSERT_TRUE(source.initialize("pp-a", 4, [](auto, auto) {}));
    ASSERT_TRUE(source.reserve(2, 7));
    ASSERT_TRUE(source.publish(2, 7, finished(10)));
    ASSERT_TRUE(source.publish(2, 7, finished(11)));
    auto pipe = std::make_shared<CapturePipe>();
    ASSERT_TRUE(source.subscribe("pp-a", 2, 4, 7, pipe));
    ASSERT_TRUE(source.publish(2, 7, finished(12)));
    ASSERT_EQ(3, pipe->values.size());
    for (size_t i = 0; i < 3; ++i) {
        const auto &data = pipe->values[i];
        ASSERT_TRUE(data->valid());
        EXPECT_EQ(2, data->update.shardId);
        EXPECT_EQ(7, data->update.streamGeneration);
        EXPECT_EQ(i + 1, data->update.sequence);
        EXPECT_EQ(i + 10, data->update.terminals.at(0).request.requestId);
        EXPECT_TRUE(data->update.addedFull.empty());
        EXPECT_TRUE(data->update.addedTail.empty());
        EXPECT_FALSE(data->update.fullSnapshot);
    }
}

TEST_F(PPRequestFinishedRTest, testFailureAndCancelWithoutKVChanges)
{
    PPRequestFinishedR source;
    source.initialize("pp-a", 2, [](auto, auto) {});
    source.reserve(0, 1);
    auto pipe = std::make_shared<CapturePipe>();
    ASSERT_TRUE(source.subscribe("pp-a", 0, 2, 1, pipe));
    for (auto status : {PPShardRequestStatus::SUCCEEDED, PPShardRequestStatus::FAILED,
                        PPShardRequestStatus::CANCELLED})
        ASSERT_TRUE(source.publish(0, 1, {{10, 1}, status}));
    ASSERT_EQ(3, pipe->values.size());
    EXPECT_EQ(PPShardRequestStatus::CANCELLED,
              pipe->values.back()->update.terminals[0].status);
    EXPECT_FALSE(source.publish(0, 1, {{10, 1}, PPShardRequestStatus::PENDING}));
    EXPECT_FALSE(source.publish(0, 1, {{0, 1}, PPShardRequestStatus::FAILED}));
}

TEST_F(PPRequestFinishedRTest, testOldGenerationCannotPublishOrDetachReplacement)
{
    std::vector<uint64_t> failed;
    PPRequestFinishedR source;
    source.initialize("pp-a", 2,
                      [&](auto, auto generation) { failed.push_back(generation); });
    source.reserve(0, 1);
    auto oldPipe = std::make_shared<CapturePipe>();
    source.subscribe("pp-a", 0, 2, 1, oldPipe);
    ASSERT_TRUE(source.reserve(0, 2));
    EXPECT_TRUE(oldPipe->terminated);
    auto pipe = std::make_shared<CapturePipe>();
    ASSERT_TRUE(source.subscribe("pp-a", 0, 2, 2, pipe));
    EXPECT_FALSE(source.unsubscribe(0, 1, oldPipe));
    EXPECT_FALSE(source.publish(0, 1, finished()));
    ASSERT_TRUE(source.publish(0, 2, finished()));
    EXPECT_EQ(1, pipe->values.size());
    EXPECT_EQ(std::vector<uint64_t>({1}), failed);
    source.stop();
    EXPECT_EQ(std::vector<uint64_t>({1, 2}), failed);
    EXPECT_FALSE(source.publish(0, 2, finished()));
}

TEST_F(PPRequestFinishedRTest, testStreamFailureCompletesMergeWaitAndRetractsCache)
{
    PPCacheCollectorState merge(2);
    for (uint32_t shard = 0; shard < 2; ++shard) {
        merge.connect(shard, 1);
        PPCacheShardUpdate snapshot;
        snapshot.shardId = shard;
        snapshot.streamGeneration = 1;
        snapshot.sequence = 1;
        snapshot.fullSnapshot = true;
        snapshot.addedFull = {8};
        ASSERT_TRUE(merge.apply(snapshot).accepted);
    }
    ASSERT_TRUE(merge.beginRequest({10, 1}).accepted);
    PPCacheShardUpdate completed;
    completed.shardId = 1;
    completed.streamGeneration = 1;
    completed.sequence = 2;
    completed.terminals = {{{10, 1}, PPShardRequestStatus::SUCCEEDED}};
    merge.apply(completed);
    PPCacheCollectorChanges lost;
    PPRequestFinishedR source;
    source.initialize("pp-a", 2, [&](auto shard, auto generation) {
        lost = merge.disconnect(shard, generation);
    });
    source.reserve(0, 1);
    auto pipe = std::make_shared<CapturePipe>();
    pipe->reject = true;
    source.subscribe("pp-a", 0, 2, 1, pipe);
    ASSERT_TRUE(source.publish(0, 1, finished()));
    EXPECT_TRUE(pipe->terminated);
    EXPECT_FALSE(merge.fullReady(8));
    EXPECT_EQ(0, merge.pendingRequestCount());
    ASSERT_EQ(1, lost.completedRequests.size());
    EXPECT_FALSE(lost.completedRequests[0].succeeded);
    EXPECT_EQ(PPShardRequestStatus::DISCONNECTED,
              lost.completedRequests[0].shardStatuses[0]);
}

TEST_F(PPRequestFinishedRTest, testCallbackCanPublishWithoutHoldingSourceMutex)
{
    PPRequestFinishedR source;
    source.initialize("pp-a", 2, [](auto, auto) {});
    source.reserve(0, 1);
    auto pipe = std::make_shared<CapturePipe>();
    pipe->onData = [&]() {
        if (pipe->values.size() == 1) source.publish(0, 1, finished(11));
    };
    ASSERT_TRUE(source.subscribe("pp-a", 0, 2, 1, pipe));
    ASSERT_TRUE(source.publish(0, 1, finished(10)));
    ASSERT_EQ(2, pipe->values.size());
    EXPECT_EQ(10, pipe->values[0]->update.terminals[0].request.requestId);
    EXPECT_EQ(11, pipe->values[1]->update.terminals[0].request.requestId);
}
