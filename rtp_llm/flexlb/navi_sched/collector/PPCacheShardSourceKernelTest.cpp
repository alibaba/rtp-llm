#include "aios/navi_sched/collector/PPCacheShardReportR.h"
#include "navi/tester/KernelTesterBuilder.h"
#include "unittest/unittest.h"

using namespace navi_sched;
class PPCacheShardSourceKernelTest : public TESTBASE {};

TEST_F(PPCacheShardSourceKernelTest, testNativeGraphReadsTheServicesExactExternalResource)
{
    auto report = std::make_shared<PPCacheShardReportR>();
    ASSERT_TRUE(report->initializeIdentity("pp-a", 2, 4));
    PPCacheShardUpdate seed;
    seed.addedFull = {8, 9};
    ASSERT_TRUE(report->publish(seed));
    navi::KernelTesterBuilder builder;
    auto tester = builder.kernel("navi_sched.pp_cache_shard_source_k")
                      .output("update")
                      .attrs(
                          "{\"pp_group\":\"pp-a\",\"shard_id\":2,\"shard_count\":4,"
                          "\"collector_id\":\"agent\",\"stream_generation\":7}")
                      .resource(report)
                      .build();
    ASSERT_TRUE(tester);
    ASSERT_EQ(navi::EC_NONE, tester->getErrorCode());
    ASSERT_TRUE(tester->compute());
    navi::DataPtr data;
    bool eof = false;
    ASSERT_TRUE(tester->getOutput("update", data, eof));
    ASSERT_FALSE(eof);
    auto snapshot = std::dynamic_pointer_cast<PPCacheShardUpdateData>(data);
    ASSERT_TRUE(snapshot);
    ASSERT_TRUE(snapshot->update.fullSnapshot);
    ASSERT_EQ(std::vector<uint64_t>({8, 9}), snapshot->update.addedFull);
    ASSERT_EQ(2, snapshot->update.shardId);
    ASSERT_EQ(7, snapshot->update.streamGeneration);

    PPCacheShardUpdate terminal;
    terminal.terminals = {{{10, 1}, PPShardRequestStatus::FAILED}};
    ASSERT_TRUE(report->publish(terminal));
    ASSERT_TRUE(tester->compute());
    ASSERT_TRUE(tester->getOutput("update", data, eof));
    auto failed = std::dynamic_pointer_cast<PPCacheShardUpdateData>(data);
    ASSERT_TRUE(failed);
    ASSERT_EQ(PPShardRequestStatus::FAILED, failed->update.terminals.at(0).status);
    ASSERT_EQ(2, failed->update.sequence);
    report->stop();
    tester->compute();
    ASSERT_EQ(navi::EC_ABORT, tester->getErrorCode());
}

TEST_F(PPCacheShardSourceKernelTest, testDifferentPPGroupCannotReadTheShard)
{
    auto report = std::make_shared<PPCacheShardReportR>();
    ASSERT_TRUE(report->initializeIdentity("pp-a", 0, 4));
    navi::KernelTesterBuilder builder;
    auto tester = builder.kernel("navi_sched.pp_cache_shard_source_k")
                      .output("update")
                      .attrs(
                          "{\"pp_group\":\"pp-b\",\"shard_id\":0,\"shard_count\":4,"
                          "\"collector_id\":\"agent\",\"stream_generation\":1}")
                      .resource(report)
                      .build();
    ASSERT_TRUE(tester);
    ASSERT_NE(navi::EC_NONE, tester->getErrorCode());
}
