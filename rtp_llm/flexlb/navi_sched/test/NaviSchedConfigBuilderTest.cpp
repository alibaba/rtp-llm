#include "aios/navi_sched/config/NaviSchedConfigBuilder.h"
#include "aios/navi_sched/common/NaviSchedScheduleData.h"
#include "unittest/unittest.h"

using namespace navi_sched;

class NaviSchedConfigBuilderTest : public TESTBASE {};

TEST_F(NaviSchedConfigBuilderTest, testExtraTaskQueuesPreserveCapacityAndThreadCounts)
{
    NaviSchedRuntimeConfig config;
    config.bizName = "master_biz";
    config.threadNum = 9;
    config.queueSize = 37;
    config.processingSize = 71;
    const auto result = NaviSchedConfigBuilder::build(config, {}, false, {}, {}, {}, 0,
                                                      {{"nsc0", 3}, {"nsc1", 2}});
    const auto root = autil::legacy::AnyCast<autil::legacy::json::JsonMap>(
        autil::legacy::json::ParseJson(result));
    const auto engine =
        autil::legacy::AnyCast<autil::legacy::json::JsonMap>(root.at("engine_config"));
    const auto queues = autil::legacy::AnyCast<autil::legacy::json::JsonMap>(
        engine.at("extra_task_queue"));
    ASSERT_EQ(2, queues.size());
    for (const auto &[name, threads] :
         std::map<std::string, size_t>{{"nsc0", 3}, {"nsc1", 2}}) {
        const auto queue =
            autil::legacy::AnyCast<autil::legacy::json::JsonMap>(queues.at(name));
        size_t value = 0;
        autil::legacy::FromJson(value, queue.at("thread_num"));
        EXPECT_EQ(threads, value);
        autil::legacy::FromJson(value, queue.at("queue_size"));
        EXPECT_EQ(37, value);
        autil::legacy::FromJson(value, queue.at("processing_size"));
        EXPECT_EQ(71, value);
    }
    EXPECT_EQ(std::string::npos,
              NaviSchedConfigBuilder::build(config, {}, false).find("extra_task_queue"));
    EXPECT_TRUE(NaviSchedConfigBuilder::build(config, {}, false, {}, {}, {}, 0, {{"", 1}})
                    .empty());
    EXPECT_TRUE(
        NaviSchedConfigBuilder::build(config, {}, false, {}, {}, {}, 0, {{"nsc0", 0}})
            .empty());
}

TEST_F(NaviSchedConfigBuilderTest, testPerformanceDefaults)
{
    NaviSchedConfig roleConfig;
    NaviSchedRuntimeConfig runtimeConfig;
    ASSERT_EQ(20, roleConfig.gig.grpcIoThreadNum);
    ASSERT_EQ(64, runtimeConfig.threadNum);
    ASSERT_EQ(100000000, runtimeConfig.processingSize);
}

TEST_F(NaviSchedConfigBuilderTest, testRejectsEmptyBizName)
{
    NaviSchedClientConfig config;
    ASSERT_TRUE(
        NaviSchedConfigBuilder::build(config.runtime, {"test_kernel"}, true).empty());
}

TEST_F(NaviSchedConfigBuilderTest, testBuildsSubscriptionConfig)
{
    NaviSchedMasterConfig config;
    config.runtime.bizName = "master_biz";
    NaviSchedLocalNode node;
    node.bizName = "agent_biz";
    node.ip = "127.0.0.1";
    node.grpcPort = 12345;
    node.supportHeartbeat = true;
    config.runtime.localNodes.emplace_back(node);
    NaviSchedCm2Subscription cm2;
    cm2.zkHost = "cm2_zk";
    cm2.zkPath = "/cm2/path";
    cm2.cm2Group = {"group_0"};
    config.runtime.cm2Subscriptions.emplace_back(cm2);
    config.runtime.vipSubscription.jmenvDomain = "jmenv.test";
    config.runtime.vipSubscription.services = {"vip.test"};
    config.runtime.vipSubscription.grpcPortDiff = 3;

    auto result = NaviSchedConfigBuilder::build(config.runtime, {"test_kernel"}, true);
    ASSERT_NE(std::string::npos, result.find("master_biz"));
    ASSERT_NE(std::string::npos, result.find("agent_biz"));
    ASSERT_NE(std::string::npos, result.find("127.0.0.1"));
    ASSERT_NE(std::string::npos, result.find("cm2_zk"));
    ASSERT_NE(std::string::npos, result.find("vip.test"));
    ASSERT_NE(std::string::npos, result.find("grpc_port_diff"));
    ASSERT_NE(std::string::npos, result.find("\"support_heartbeat\":true"));
    ASSERT_NE(std::string::npos, result.find("\"query_full_api\":true"));
    ASSERT_EQ(std::string::npos, result.find("\"spectrum\""));
    ASSERT_NE(std::string::npos, result.find("test_kernel"));
    ASSERT_NE(std::string::npos, result.find("support_heartbeat"));
    ASSERT_EQ(std::string::npos, result.find("node_meta"));
}

TEST_F(NaviSchedConfigBuilderTest, testCanSkipGigClient)
{
    NaviSchedAgentConfig config;
    config.runtime.bizName = "agent_biz";
    auto result = NaviSchedConfigBuilder::build(config.runtime, {"test_kernel"}, false);
    ASSERT_EQ(std::string::npos, result.find("navi.buildin.gig_client"));
}

TEST_F(NaviSchedConfigBuilderTest, testConfiguresGrpcClientIoThreads)
{
    NaviSchedRuntimeConfig config;
    config.bizName = "master_biz";
    for (size_t threads : {10, 20}) {
        const auto result = NaviSchedConfigBuilder::build(config, {"test_kernel"}, true,
                                                          {}, {}, {}, threads);
        const auto root = autil::legacy::AnyCast<autil::legacy::json::JsonMap>(
            autil::legacy::json::ParseJson(result));
        const auto resources = autil::legacy::AnyCast<autil::legacy::json::JsonArray>(
            root.at("resource_config"));
        ASSERT_EQ(1u, resources.size());
        const auto resource =
            autil::legacy::AnyCast<autil::legacy::json::JsonMap>(resources[0]);
        const auto body =
            autil::legacy::AnyCast<autil::legacy::json::JsonMap>(resource.at("config"));
        const auto init =
            autil::legacy::AnyCast<autil::legacy::json::JsonMap>(body.at("init_config"));
        ASSERT_EQ(1u, init.count("connection_config"));
        const auto connections = autil::legacy::AnyCast<autil::legacy::json::JsonMap>(
            init.at("connection_config"));
        ASSERT_EQ(1u, connections.size());
        const auto stream = autil::legacy::AnyCast<autil::legacy::json::JsonMap>(
            connections.at("grpc_stream"));
        size_t actual = 0;
        autil::legacy::FromJson(actual, stream.at("thread_num"));
        EXPECT_EQ(threads, actual);
        EXPECT_EQ(1u, init.count("subscribe_config"));
    }
    const auto defaultConfig =
        NaviSchedConfigBuilder::build(config, {"test_kernel"}, true);
    EXPECT_EQ(std::string::npos, defaultConfig.find("connection_config"));
    const auto withoutClient =
        NaviSchedConfigBuilder::build(config, {"test_kernel"}, false, {}, {}, {}, 20);
    EXPECT_EQ(std::string::npos, withoutClient.find("navi.buildin.gig_client"));
}

TEST_F(NaviSchedConfigBuilderTest, testBuildsSpectrumSubscription)
{
    NaviSchedClientConfig config;
    config.runtime.bizName = "client_biz";
    config.runtime.spectrumSubscription.serviceId = "v-c123ec48";
    config.runtime.spectrumSubscription.physicalServiceId = "521ff55f";
    auto result = NaviSchedConfigBuilder::build(config.runtime, {"test_kernel"}, true);
    ASSERT_NE(std::string::npos, result.find("\"spectrum\""));
    ASSERT_NE(std::string::npos, result.find("v-c123ec48"));
    ASSERT_NE(std::string::npos, result.find("521ff55f"));
    ASSERT_NE(std::string::npos, result.find("\"grpc_port_diff\":1"));
}

TEST_F(NaviSchedConfigBuilderTest, testBuildsLatencyKernel)
{
    NaviSchedAgentConfig config;
    config.runtime.bizName = "agent_biz";
    auto result = NaviSchedConfigBuilder::build(config.runtime,
                                                {"navi_sched.agent_latency_k"}, false);
    ASSERT_NE(std::string::npos, result.find("navi_sched.agent_latency_k"));
}

TEST_F(NaviSchedConfigBuilderTest, testBuildsKernelAndResourceConfig)
{
    NaviSchedAgentConfig config;
    config.runtime.bizName = "agent_biz";
    auto result = NaviSchedConfigBuilder::build(
        config.runtime, {"navi_sched.agent_batch_k"}, false,
        {{"navi_sched.agent_batch_k", "{\"batch_timeout_ms\":23}"}},
        {{"test_resource_r", "{\"value\":7}"}});
    ASSERT_NE(std::string::npos, result.find("kernel_config"));
    ASSERT_NE(std::string::npos, result.find("batch_timeout_ms"));
    ASSERT_NE(std::string::npos, result.find("23"));
    ASSERT_NE(std::string::npos, result.find("resource_config"));
    ASSERT_NE(std::string::npos, result.find("test_resource_r"));
}
