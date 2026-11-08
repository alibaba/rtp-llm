#include "aios/navi_sched/collector/PPCacheShardReportR.h"
#include "navi/tester/KernelTesterBuilder.h"
#include "unittest/unittest.h"
using namespace navi_sched;
class PPRequestFinishedSinkKernelTest : public TESTBASE {};
namespace {
class CapturePipe : public navi::AsyncPipe {
public:
    CapturePipe() : navi::AsyncPipe(nullptr, navi::AS_REQUIRED) {}
    navi::ErrorCode setData(const navi::DataPtr &data) override
    {
        values.push_back(std::dynamic_pointer_cast<PPCacheShardUpdateData>(data));
        return navi::EC_NONE;
    }
    std::vector<PPCacheShardUpdateDataPtr> values;
};
PPCacheShardUpdateDataPtr outcome(uint64_t sequence = 1)
{
    auto data = std::make_shared<PPCacheShardUpdateData>();
    data->update.shardId = 2;
    data->update.streamGeneration = 7;
    data->update.sequence = sequence;
    data->update.terminals = {{{10, 1}, PPShardRequestStatus::FAILED}};
    return data;
}
}
TEST_F(PPRequestFinishedSinkKernelTest, testNativeSinkPublishesWithoutChangingKV)
{
    auto report = std::make_shared<PPCacheShardReportR>();
    ASSERT_TRUE(report->initializeIdentity("pp-a", 2, 4));
    PPCacheShardUpdate cache;
    cache.addedFull = {8};
    ASSERT_TRUE(report->publish(cache));
    auto pipe = std::make_shared<CapturePipe>();
    ASSERT_TRUE(report->subscribe("pp-a", 2, 4, "agent", 7, pipe));
    navi::KernelTesterBuilder builder;
    auto tester = builder.kernel("navi_sched.pp_request_finished_sink_k")
                      .input("update")
                      .output("finished")
                      .attrs(
                          "{\"pp_group\":\"pp-a\",\"shard_id\":2,\"shard_count\":4,"
                          "\"collector_id\":\"agent\",\"stream_generation\":7}")
                      .resource(report)
                      .build();
    ASSERT_TRUE(tester);
    ASSERT_EQ(navi::EC_NONE, tester->getErrorCode());
    ASSERT_TRUE(tester->setInput("update", outcome()));
    ASSERT_TRUE(tester->compute());
    ASSERT_EQ(navi::EC_NONE, tester->getErrorCode());
    ASSERT_EQ(2, pipe->values.size());
    EXPECT_EQ(2, pipe->values.back()->update.sequence);
    EXPECT_TRUE(pipe->values.back()->update.addedFull.empty());
    EXPECT_FALSE(pipe->values.back()->update.fullSnapshot);
    EXPECT_EQ(PPShardRequestStatus::FAILED,
              pipe->values.back()->update.terminals.at(0).status);
    ASSERT_TRUE(tester->setInputEof("update"));
    ASSERT_TRUE(tester->compute());
    navi::DataPtr data;
    bool eof = false;
    ASSERT_TRUE(tester->getOutput("finished", data, eof));
    ASSERT_TRUE(eof);
    ASSERT_TRUE(report->unsubscribe("agent", pipe));
    auto next = std::make_shared<CapturePipe>();
    ASSERT_TRUE(report->subscribe("pp-a", 2, 4, "agent", 8, next));
    EXPECT_EQ(std::vector<uint64_t>({8}), next->values.at(0)->update.addedFull);
}
TEST_F(PPRequestFinishedSinkKernelTest, testRejectsOtherConnectionsAndCachePayloads)
{
    auto report = std::make_shared<PPCacheShardReportR>();
    ASSERT_TRUE(report->initializeIdentity("pp-a", 2, 4));
    auto pipe = std::make_shared<CapturePipe>();
    ASSERT_TRUE(report->subscribe("pp-a", 2, 4, "agent", 7, pipe));
    auto data = outcome();
    EXPECT_FALSE(report->publishRequestFinished("pp-b", 4, "agent", *data));
    EXPECT_FALSE(report->publishRequestFinished("pp-a", 3, "agent", *data));
    EXPECT_FALSE(report->publishRequestFinished("pp-a", 4, "other", *data));
    data->update.streamGeneration = 6;
    EXPECT_FALSE(report->publishRequestFinished("pp-a", 4, "agent", *data));
    data = outcome();
    data->update.addedFull = {9};
    EXPECT_FALSE(report->publishRequestFinished("pp-a", 4, "agent", *data));
    data = outcome();
    data->update.peerCacheReadAvailable = 1;
    EXPECT_FALSE(report->publishRequestFinished("pp-a", 4, "agent", *data));
    data = outcome();
    data->update.fullSnapshot = true;
    EXPECT_FALSE(report->publishRequestFinished("pp-a", 4, "agent", *data));
    data = outcome(2);
    EXPECT_FALSE(report->publishRequestFinished("pp-a", 4, "agent", *data));
    data = outcome();
    ASSERT_TRUE(report->publishRequestFinished("pp-a", 4, "agent", *data));
    EXPECT_FALSE(report->publishRequestFinished("pp-a", 4, "agent", *data));
    EXPECT_EQ(2, pipe->values.size());
    ASSERT_TRUE(report->unsubscribe("agent", pipe));
    auto next = std::make_shared<CapturePipe>();
    ASSERT_TRUE(report->subscribe("pp-a", 2, 4, "agent", 8, next));
    EXPECT_FALSE(report->publishRequestFinished("pp-a", 4, "agent", *outcome(2)));
    data = outcome();
    data->update.streamGeneration = 8;
    EXPECT_TRUE(report->publishRequestFinished("pp-a", 4, "agent", *data));
}
