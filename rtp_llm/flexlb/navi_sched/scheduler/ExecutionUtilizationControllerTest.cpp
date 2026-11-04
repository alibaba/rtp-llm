#include "aios/navi_sched/scheduler/ExecutionUtilizationController.h"
#include "unittest/unittest.h"
#include <limits>

using namespace navi_sched;

class ExecutionUtilizationControllerTest : public TESTBASE {};

TEST_F(ExecutionUtilizationControllerTest, testFirstCompletionOnlyEstablishesSampleBaseline)
{
    ExecutionUtilizationEstimator estimator(100);
    estimator.begin(721655793);
    ASSERT_TRUE(estimator.end(722949093));
    EXPECT_DOUBLE_EQ(0.0, estimator.value());
    EXPECT_EQ(0, estimator.sampleBusyUs());
    EXPECT_EQ(0, estimator.sampleElapsedUs());
    estimator.begin(723037000);
    ASSERT_TRUE(estimator.end(726373911));
    EXPECT_NEAR(3336911.0 / 3424818.0, estimator.value(), 1e-12);
    EXPECT_EQ(3336911, estimator.sampleBusyUs());
    EXPECT_EQ(3424818, estimator.sampleElapsedUs());
}

TEST_F(ExecutionUtilizationControllerTest, testUnknownExecutionCapacityDoesNotAmplifyQueue)
{
    EXPECT_DOUBLE_EQ(0.5, ExecutionUtilizationController::effectiveUtilization(0.5, 0, 9));
    EXPECT_DOUBLE_EQ(0.95, ExecutionUtilizationController::effectiveUtilization(0.5, 10, 9));
    EXPECT_DOUBLE_EQ(1.4, ExecutionUtilizationController::effectiveUtilization(0.5, 1, 9));
    EXPECT_DOUBLE_EQ(1.4, ExecutionUtilizationController::effectiveUtilization(0.5, 4, 9));
    EXPECT_DOUBLE_EQ(1.4, ExecutionUtilizationController::effectiveUtilization(0.5, 5, 9));
}

TEST_F(ExecutionUtilizationControllerTest, testCompletionSamplesIdleAndBusyWithEma)
{
    ExecutionUtilizationEstimator estimator(100);
    estimator.begin(100);
    ASSERT_TRUE(estimator.end(100));
    estimator.begin(200);
    EXPECT_DOUBLE_EQ(0.0, estimator.value());
    ASSERT_TRUE(estimator.end(300));
    EXPECT_DOUBLE_EQ(0.5, estimator.value());
    EXPECT_EQ(100, estimator.sampleBusyUs());
    EXPECT_EQ(200, estimator.sampleElapsedUs());
    estimator.begin(300);
    ASSERT_TRUE(estimator.end(400));
    EXPECT_DOUBLE_EQ(100.0 / 190.0, estimator.value());
    EXPECT_EQ(100, estimator.sampleBusyUs());
    EXPECT_EQ(100, estimator.sampleElapsedUs());
    estimator.begin(500);
    ASSERT_TRUE(estimator.end(600));
    EXPECT_DOUBLE_EQ(100.0 / 191.0, estimator.value());
    EXPECT_EQ(0u, estimator.active());
    EXPECT_FALSE(estimator.end(700));
    EXPECT_DOUBLE_EQ(100.0 / 191.0, estimator.value());
}

TEST_F(ExecutionUtilizationControllerTest, testShortStepsDoNotOutweighLongIdleInterval)
{
    ExecutionUtilizationEstimator estimator(100);
    estimator.begin(100);
    ASSERT_TRUE(estimator.end(100));
    estimator.begin(5000100);
    ASSERT_TRUE(estimator.end(10000100));
    EXPECT_DOUBLE_EQ(0.5, estimator.value());
    int64_t now = 10000100;
    double busyEma = 5000000.0;
    double elapsedEma = 10000000.0;
    for (int i = 0; i < 10; ++i) {
        estimator.begin(now);
        now += 100;
        ASSERT_TRUE(estimator.end(now));
        busyEma += 0.1 * (100.0 - busyEma);
        elapsedEma += 0.1 * (100.0 - elapsedEma);
    }
    EXPECT_NEAR(busyEma / elapsedEma, estimator.value(), 1e-12);
    EXPECT_LT(estimator.value(), 0.501);
}

TEST_F(ExecutionUtilizationControllerTest, testOverlappingExecutionCountsUnionOnce)
{
    ExecutionUtilizationEstimator estimator(100);
    estimator.begin(100);
    estimator.begin(150);
    ASSERT_TRUE(estimator.end(200));
    EXPECT_EQ(1u, estimator.active());
    EXPECT_DOUBLE_EQ(0.0, estimator.value());
    ASSERT_TRUE(estimator.end(300));
    EXPECT_EQ(0u, estimator.active());
    EXPECT_DOUBLE_EQ(1.0, estimator.value());
    estimator.begin(400);
    ASSERT_TRUE(estimator.end(500));
    EXPECT_DOUBLE_EQ(100.0 / 110.0, estimator.value());
}

TEST_F(ExecutionUtilizationControllerTest, testSameTimestampAndImmediateQueueScaling)
{
    ExecutionUtilizationEstimator estimator(100);
    estimator.begin(100);
    ASSERT_TRUE(estimator.end(100));
    EXPECT_DOUBLE_EQ(0.0, estimator.value());
    EXPECT_EQ(0, estimator.sampleBusyUs());
    EXPECT_EQ(0, estimator.sampleElapsedUs());
    estimator.begin(100);
    ASSERT_TRUE(estimator.end(200));
    EXPECT_DOUBLE_EQ(1.0, estimator.value());
    EXPECT_DOUBLE_EQ(1.0, ExecutionUtilizationController::effectiveUtilization(estimator.value(), 10, 0));
    EXPECT_DOUBLE_EQ(2.0, ExecutionUtilizationController::effectiveUtilization(estimator.value(), 10, 10));
    EXPECT_DOUBLE_EQ(3.0, ExecutionUtilizationController::effectiveUtilization(estimator.value(), 10, 20));
    EXPECT_DOUBLE_EQ(1.0, estimator.value());
}

TEST_F(ExecutionUtilizationControllerTest, testOrdinaryPiAtExactTarget)
{
    ExecutionUtilizationController controller;
    ExecutionUtilizationController::Config config;
    config.target = 1.0;
    EXPECT_DOUBLE_EQ(0.0, controller.update(config, 1.0, 4, 1000000));
    EXPECT_DOUBLE_EQ(0.0, controller.update(config, 1.0, 4, 2000000));
    EXPECT_DOUBLE_EQ(0.0, controller.integral());
}

TEST_F(ExecutionUtilizationControllerTest, testProportionalAndIntegralRecovery)
{
    ExecutionUtilizationController controller;
    ExecutionUtilizationController::Config config;
    config.target = 0.99;
    EXPECT_NEAR(0.0005, controller.update(config, 1.0, 4, 1000000), 1e-12);
    EXPECT_NEAR(0.0006, controller.update(config, 1.0, 4, 2000000), 1e-12);
    EXPECT_NEAR(0.0001, controller.integral(), 1e-12);
    EXPECT_DOUBLE_EQ(0.0, controller.update(config, 0.5, 4, 3000000));
    EXPECT_DOUBLE_EQ(0.0, controller.integral());
    EXPECT_NEAR(0.0006, controller.update(config, 1.0, 4, 4000000), 1e-12);
}

TEST_F(ExecutionUtilizationControllerTest, testEveryBatchUpdatesWithElapsedTime)
{
    ExecutionUtilizationController slow, fast;
    ExecutionUtilizationController::Config config;
    config.target = 1.1;
    slow.update(config, 1.2, 2, 1000000);
    fast.update(config, 1.2, 2, 1000000);
    EXPECT_NEAR(0.0051, fast.update(config, 1.2, 2, 1100000), 1e-12);
    for (int64_t t = 1200000; t <= 2000000; t += 100000) {
        fast.update(config, 1.2, 2, t);
    }
    EXPECT_NEAR(slow.update(config, 1.2, 2, 2000000), fast.probability(), 1e-12);
    EXPECT_NEAR(0.006, fast.probability(), 1e-12);
}

TEST_F(ExecutionUtilizationControllerTest, testMissingInvalidAndRestartReset)
{
    ExecutionUtilizationController controller;
    ExecutionUtilizationController::Config config;
    config.target = 0.9;
    controller.update(config, 1.0, 1, 1000000);
    controller.update(config, 1.0, 1, 2000000);
    EXPECT_DOUBLE_EQ(0.0, controller.update(config, 1.0, 0, 3000000));
    EXPECT_DOUBLE_EQ(0.0, controller.integral());
    EXPECT_DOUBLE_EQ(0.0, controller.update(config,
        std::numeric_limits<double>::quiet_NaN(), 1, 4000000));
    EXPECT_DOUBLE_EQ(0.0, controller.update(config, -0.01, 1, 5000000));
    config.integralGainPerSecond = -1.0;
    EXPECT_FALSE(config.valid());
    EXPECT_DOUBLE_EQ(0.0, controller.update(config, 1.0, 1, 6000000));
}

TEST_F(ExecutionUtilizationControllerTest, testBacklogAboveOneHundredPercent)
{
    ExecutionUtilizationController controller;
    ExecutionUtilizationController::Config config;
    config.target = 1.0;
    EXPECT_NEAR(0.025, controller.update(config, 1.5, 4, 1000000), 1e-12);
    EXPECT_NEAR(0.030, controller.update(config, 1.5, 4, 2000000), 1e-12);
    EXPECT_NEAR(0.005, controller.integral(), 1e-12);
    EXPECT_NEAR(0.005, controller.update(config, 1.0, 4, 3000000), 1e-12);
    EXPECT_NEAR(0.005, controller.update(config, 1.0, 4, 4000000), 1e-12);
    EXPECT_DOUBLE_EQ(0.0, controller.update(config, 0.5, 4, 5000000));
}

TEST_F(ExecutionUtilizationControllerTest, testRequestCountConversion)
{
    EXPECT_DOUBLE_EQ(2.0, ExecutionUtilizationController::effectiveUtilization(1.0, 100, 100));
    EXPECT_DOUBLE_EQ(1.0, ExecutionUtilizationController::effectiveUtilization(0.5, 100, 100));
    EXPECT_DOUBLE_EQ(0.5, ExecutionUtilizationController::effectiveUtilization(0.5, 100, 0));
    EXPECT_DOUBLE_EQ(1.0, ExecutionUtilizationController::effectiveUtilization(1.0, 0, 0));
    EXPECT_DOUBLE_EQ(1.0, ExecutionUtilizationController::effectiveUtilization(1.0, 0, 100));
    EXPECT_DOUBLE_EQ(-1.0, ExecutionUtilizationController::effectiveUtilization(1.0, -1, 100));
    EXPECT_DOUBLE_EQ(-1.0, ExecutionUtilizationController::effectiveUtilization(1.01, 100, 100));
}

TEST_F(ExecutionUtilizationControllerTest, testSaturationDoesNotDelayRecovery)
{
    ExecutionUtilizationController controller;
    ExecutionUtilizationController::Config config;
    config.target = 1.0;
    config.proportionalGain = 0.1;
    config.integralGainPerSecond = 0.2;
    for (int64_t t = 1000000; t <= 100000000; t += 1000000) {
        controller.update(config, 2.0, 1, t);
    }
    EXPECT_DOUBLE_EQ(1.0, controller.probability());
    EXPECT_DOUBLE_EQ(0.9, controller.integral());
    EXPECT_NEAR(0.75, controller.update(config, 0.5, 1, 101000000), 1e-12);
    for (int64_t t = 102000000; t <= 200000000; t += 1000000) {
        controller.update(config, 0.0, 1, t);
    }
    EXPECT_DOUBLE_EQ(0.0, controller.probability());
    EXPECT_DOUBLE_EQ(0.0, controller.integral());
    EXPECT_NEAR(0.3, controller.update(config, 2.0, 1, 201000000), 1e-12);
}

TEST_F(ExecutionUtilizationControllerTest, testGlobalScalingUsesRatioOfSums)
{
    ClusterExecutionUtilization cluster;
    cluster.add(0.5, 10, 90);
    cluster.add(1.0, 90, 10);
    EXPECT_EQ(2u, cluster.nodes);
    EXPECT_EQ(100u, cluster.running);
    EXPECT_EQ(100u, cluster.waiting);
    EXPECT_DOUBLE_EQ(0.75, cluster.average());
    EXPECT_DOUBLE_EQ(1.5, cluster.effective());
    ClusterExecutionUtilization reverse;
    reverse.add(1.0, 90, 10);
    reverse.add(0.5, 10, 90);
    EXPECT_DOUBLE_EQ(cluster.effective(), reverse.effective());
}

TEST_F(ExecutionUtilizationControllerTest, testGlobalScalingFloorAndInvalidCohort)
{
    ClusterExecutionUtilization cluster;
    EXPECT_DOUBLE_EQ(0.0, cluster.effective());
    cluster.add(0.5, 0, 10);
    cluster.add(0.5, 0, 0);
    cluster.add(-1.0, 100, 100);
    cluster.add(1.1, 100, 100);
    cluster.add(std::numeric_limits<double>::quiet_NaN(), 1, 1);
    cluster.add(0.5, -1, 1);
    cluster.add(0.5, 1, -1);
    EXPECT_EQ(2u, cluster.nodes);
    EXPECT_EQ(0u, cluster.running);
    EXPECT_EQ(10u, cluster.waiting);
    EXPECT_DOUBLE_EQ(0.5, cluster.average());
    EXPECT_DOUBLE_EQ(1.0, cluster.effective());
}
