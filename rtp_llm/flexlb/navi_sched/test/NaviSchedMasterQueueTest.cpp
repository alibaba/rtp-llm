#include "rtp_llm/flexlb/navi_sched/graph/MasterCollectorManager.h"
#include "rtp_llm/flexlb/navi_sched/service/NaviSchedMaster.h"
#include "navi/engine/Navi.h"
#include "navi/engine/NaviSnapshot.h"
#include "navi/engine/TaskQueue.h"
#include "unittest/unittest.h"
#include <utility>
#include <vector>

namespace navi_sched {

class NaviSchedMasterQueueTest : public TESTBASE {
protected:
    static NaviSchedMasterConfig makeConfig() {
        NaviSchedMasterConfig config;
        config.runtime.bizName = "master_queue_test";
        config.runtime.threadNum = 16;
        config.service.enableKmon = false;
        config.service.enableSnapshot = false;
        config.service.waitNoQuery = false;
        return config;
    }
};

TEST_F(NaviSchedMasterQueueTest, CollectorQueuesShareTheConfiguredThreadBudget) {
    auto config = makeConfig();
    config.collectorThreadNum = 7;
    config.collectorQueueCount = 3;
    NaviSchedMaster master(config);
    ASSERT_TRUE(master.init());
    auto snapshot = master.getNavi()->getRealNavi()->getSnapshot();
    ASSERT_TRUE(snapshot);
    auto *primary = snapshot->getTaskQueue("")->getThreadPool();
    EXPECT_EQ(9, primary->_threadNum);
    ASSERT_EQ(3, snapshot->_extraTaskQueueMap.size());
    size_t total = primary->_threadNum;
    for (size_t index = 0; index < 3; ++index) {
        const auto name = "nsc" + std::to_string(index);
        auto queue = snapshot->getTaskQueue(name);
        ASSERT_NE(snapshot->getTaskQueue(""), queue);
        EXPECT_EQ(index == 0 ? 3 : 2, queue->getThreadPool()->_threadNum);
        total += queue->getThreadPool()->_threadNum;
        ASSERT_EQ(name, master._collectorManager->_taskQueueNames[index]);
    }
    EXPECT_EQ(config.runtime.threadNum, total);
    snapshot.reset();
    master.stop();
}

TEST_F(NaviSchedMasterQueueTest, DisabledCollectorQueuesPreserveTheBuiltinPool) {
    NaviSchedMaster master(makeConfig());
    ASSERT_TRUE(master.init());
    auto snapshot = master.getNavi()->getRealNavi()->getSnapshot();
    ASSERT_TRUE(snapshot);
    EXPECT_EQ(16, snapshot->getTaskQueue("")->getThreadPool()->_threadNum);
    EXPECT_TRUE(snapshot->_extraTaskQueueMap.empty());
    EXPECT_TRUE(master._collectorManager->_taskQueueNames.empty());
    snapshot.reset();
    master.stop();
}

TEST_F(NaviSchedMasterQueueTest, RejectsEmptyPoolsAndOversubscribedThreadBudgets) {
    for (const auto &allocation : std::vector<std::pair<size_t, size_t>>{
             {0, 1}, {1, 0}, {16, 1}, {17, 1}, {2, 3}}) {
        auto config = makeConfig();
        config.collectorThreadNum = allocation.first;
        config.collectorQueueCount = allocation.second;
        NaviSchedMaster master(config);
        EXPECT_FALSE(master.init()) << allocation.first << "," << allocation.second;
    }
}

} // namespace navi_sched
