#include "aios/navi_sched/collector/NaviSchedCollector.h"
#include "unittest/unittest.h"

using namespace navi_sched;
class NaviSchedCollectorTest : public TESTBASE {};

TEST_F(NaviSchedCollectorTest, testKVOnlyRoleLifecycle)
{
    NaviSchedCollectorConfig config;
    config.service.name = "pp_collector_lifecycle_test";
    config.service.enableKmon = false;
    config.service.enableSnapshot = false;
    config.service.waitNoQuery = false;
    config.service.gig.grpcIoThreadNum = 1;
    config.runtime.bizName = "pp_collector_biz";
    config.runtime.threadNum = 1;
    config.ppGroup = "pp-a";
    config.shardCount = 4;
    config.shardId = 2;
    NaviSchedCollector collector(config);
    PPCacheShardUpdate delta;
    delta.addedFull = {8, 9};
    EXPECT_FALSE(collector.publish(delta));
    ASSERT_TRUE(collector.init());
    EXPECT_GT(collector.getGrpcPort(), 0);
    EXPECT_TRUE(collector.init());
    ASSERT_TRUE(collector.publish(delta));
    delta = {};
    delta.terminals = {{{10, 1}, PPShardRequestStatus::FAILED}};
    EXPECT_TRUE(collector.publish(delta));
    collector.stop();
    EXPECT_FALSE(collector.publish(delta));
    EXPECT_FALSE(collector.init());
    collector.stop();
}

TEST_F(NaviSchedCollectorTest, testInvalidIdentityDoesNotStartService)
{
    NaviSchedCollectorConfig config;
    config.runtime.bizName = "pp_collector_biz";
    config.ppGroup = "pp-a";
    config.shardCount = 4;
    config.shardId = 4;
    NaviSchedCollector collector(config);
    EXPECT_FALSE(collector.init());
    EXPECT_LE(collector.getGrpcPort(), 0);
}

TEST_F(NaviSchedCollectorTest, testInvalidResidentPolicyDoesNotStartService)
{
    NaviSchedCollectorConfig config;
    config.runtime.bizName = "pp_collector_biz";
    config.ppGroup = "pp-a";
    config.shardCount = 4;
    config.groupRoles = {PPCacheGroupRole::FULL};
    config.groupBlockScales = {0};
    NaviSchedCollector collector(config);
    EXPECT_FALSE(collector.init());
    EXPECT_FALSE(collector.objectResident({"a", "{}"}));
    EXPECT_FALSE(collector.replaceResidentObjects({}));
    EXPECT_FALSE(collector.objectRemoved("a"));
    EXPECT_LE(collector.getGrpcPort(), 0);
}
