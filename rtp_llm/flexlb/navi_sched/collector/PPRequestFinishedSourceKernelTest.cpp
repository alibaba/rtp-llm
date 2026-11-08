#include "aios/navi_sched/collector/PPRequestFinishedR.h"
#include "navi/tester/KernelTesterBuilder.h"
#include "unittest/unittest.h"

using namespace navi_sched;
class PPRequestFinishedSourceKernelTest : public TESTBASE {};

TEST_F(PPRequestFinishedSourceKernelTest, testBufferedOutcomesAndStopThroughNativeKernel)
{
    auto finished = std::make_shared<PPRequestFinishedR>();
    ASSERT_TRUE(finished->initialize("pp-a", 4, [](uint32_t, uint64_t) {}));
    ASSERT_TRUE(finished->reserve(2, 7));
    ASSERT_TRUE(finished->publish(2, 7, {{10, 1}, PPShardRequestStatus::FAILED}));
    navi::KernelTesterBuilder builder;
    auto tester = builder.kernel("navi_sched.pp_request_finished_source_k")
                      .output("update")
                      .attrs(
                          "{\"pp_group\":\"pp-a\",\"shard_id\":2,\"shard_count\":4,"
                          "\"stream_generation\":7}")
                      .resource(finished)
                      .build();
    ASSERT_TRUE(tester);
    ASSERT_EQ(navi::EC_NONE, tester->getErrorCode());
    ASSERT_TRUE(tester->compute());
    navi::DataPtr data;
    bool eof = false;
    ASSERT_TRUE(tester->getOutput("update", data, eof));
    auto update = std::dynamic_pointer_cast<PPCacheShardUpdateData>(data);
    ASSERT_TRUE(update);
    ASSERT_FALSE(eof);
    ASSERT_FALSE(update->update.fullSnapshot);
    ASSERT_TRUE(update->update.addedFull.empty());
    ASSERT_TRUE(update->update.addedTail.empty());
    ASSERT_EQ(2, update->update.shardId);
    ASSERT_EQ(7, update->update.streamGeneration);
    ASSERT_EQ(1, update->update.sequence);
    ASSERT_EQ(PPShardRequestStatus::FAILED, update->update.terminals.at(0).status);
    ASSERT_TRUE(finished->publish(2, 7, {{11, 1}, PPShardRequestStatus::CANCELLED}));
    ASSERT_TRUE(tester->compute());
    ASSERT_TRUE(tester->getOutput("update", data, eof));
    update = std::dynamic_pointer_cast<PPCacheShardUpdateData>(data);
    ASSERT_TRUE(update);
    ASSERT_EQ(2, update->update.sequence);
    ASSERT_EQ(PPShardRequestStatus::CANCELLED, update->update.terminals.at(0).status);
    finished->stop();
    tester->compute();
    ASSERT_EQ(navi::EC_ABORT, tester->getErrorCode());
}

TEST_F(PPRequestFinishedSourceKernelTest, testWrongGroupCannotConsumeNotifications)
{
    auto finished = std::make_shared<PPRequestFinishedR>();
    ASSERT_TRUE(finished->initialize("pp-a", 4, [](uint32_t, uint64_t) {}));
    ASSERT_TRUE(finished->reserve(0, 1));
    navi::KernelTesterBuilder builder;
    auto tester = builder.kernel("navi_sched.pp_request_finished_source_k")
                      .output("update")
                      .attrs(
                          "{\"pp_group\":\"pp-b\",\"shard_id\":0,\"shard_count\":4,"
                          "\"stream_generation\":1}")
                      .resource(finished)
                      .build();
    ASSERT_TRUE(tester);
    ASSERT_NE(navi::EC_NONE, tester->getErrorCode());
}
