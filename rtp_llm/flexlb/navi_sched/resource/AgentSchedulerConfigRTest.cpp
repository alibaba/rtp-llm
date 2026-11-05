#include "aios/navi_sched/resource/AgentSchedulerConfigR.h"

#include "aios/navi_sched/resource/SchedulerConfigR.h"
#include "unittest/unittest.h"

#include <limits>
#include <algorithm>

namespace navi_sched {

TEST(AgentSchedulerConfigRTest, testRetiredLearningRateConfigCannotBeEnabled)
{
    const std::string retiredField = "latency_adaptive_learning_rate";
    EXPECT_EQ(std::string::npos,
              AgentSchedulerConfigR::toJson(AgentSchedulerConfig()).find(retiredField));
    SchedulerConfigR masterConfig;
    const auto before = masterConfig.get();
    SchedulerConfigR::Request request;
    request.update = true;
    request.configJson = R"({"latency_adaptive_learning_rate":{"enabled":true}})";
    SchedulerConfigResponseDataPtr response;
    bool waitingForAgents = false;
    ASSERT_TRUE(masterConfig.processRequest(request, response, waitingForAgents));
    ASSERT_TRUE(response);
    EXPECT_FALSE(response->accepted);
    EXPECT_EQ("unsupported scheduler config field: " + retiredField, response->error);
    EXPECT_FALSE(waitingForAgents);
    EXPECT_EQ(before.generation, masterConfig.get().generation);
    EXPECT_EQ(before.agentConfigHash, masterConfig.get().agentConfigHash);
}

TEST(AgentSchedulerConfigRTest, testServiceConfigCarriesRuntimeParameters)
{
    NaviSchedConfig serviceConfig;
    serviceConfig.latencyOnlineLearningRateScale = 2.5;
    serviceConfig.latencyCurrentSampleWeight = 3.0;
    serviceConfig.latencyParameterLearningRateScales.assign(19, 1.0);
    serviceConfig.latencyParameterLearningRateScales[11] = 0.4;
    serviceConfig.latencyParameterLearningRateScales[14] = 16.0;
    serviceConfig.latencyParameterLearningRateScales[17] = 0.01;
    serviceConfig.latencyNonlinearScale = 2048.0;
    serviceConfig.latencyHistoryBucketCapacity = 17;
    serviceConfig.residualLearningRate = 0.07;
    serviceConfig.residualUpdateErrorFloorMs = 25.0;
    serviceConfig.residualUpdateErrorRatio = 0.2;
    serviceConfig.residualIdleDecayStartMs = 4000;
    serviceConfig.residualIdleDecayEndMs = 9000;

    const auto config = AgentSchedulerConfig::fromServiceConfig(serviceConfig, 1.25);
    EXPECT_DOUBLE_EQ(2.5, config.latencyOnlineLearningRateScale);
    EXPECT_DOUBLE_EQ(3.0, config.latencyCurrentSampleWeight);
    EXPECT_EQ(serviceConfig.latencyParameterLearningRateScales,
              config.latencyParameterLearningRateScales);
    EXPECT_DOUBLE_EQ(2048.0, config.latencyNonlinearScale);
    EXPECT_EQ(17, config.latencyHistoryBucketCapacity);
    EXPECT_DOUBLE_EQ(0.07, config.residualLearningRate);
    EXPECT_DOUBLE_EQ(25.0, config.residualUpdateErrorFloorMs);
    EXPECT_DOUBLE_EQ(0.2, config.residualUpdateErrorRatio);
    EXPECT_EQ(4000, config.residualIdleDecayStartMs);
    EXPECT_EQ(9000, config.residualIdleDecayEndMs);
    EXPECT_DOUBLE_EQ(1.25, config.timeCompression);

    const auto residualConfig = config.residualRuntimeConfig();
    EXPECT_DOUBLE_EQ(0.07, residualConfig.learningRate);
    EXPECT_DOUBLE_EQ(25.0, residualConfig.updateErrorFloorMs);
    EXPECT_DOUBLE_EQ(0.2, residualConfig.updateErrorRatio);
    EXPECT_EQ(4000000, residualConfig.idleDecayStartUs);
    EXPECT_EQ(9000000, residualConfig.idleDecayEndUs);

    AgentSchedulerConfig parsed;
    std::string errorMessage;
    ASSERT_TRUE(AgentSchedulerConfigR::parse(AgentSchedulerConfigR::toJson(config),
                                             parsed, errorMessage))
        << errorMessage;
    EXPECT_DOUBLE_EQ(config.latencyOnlineLearningRateScale,
                     parsed.latencyOnlineLearningRateScale);
    EXPECT_DOUBLE_EQ(config.latencyCurrentSampleWeight,
                     parsed.latencyCurrentSampleWeight);
    EXPECT_EQ(config.latencyParameterLearningRateScales,
              parsed.latencyParameterLearningRateScales);
    EXPECT_DOUBLE_EQ(config.latencyNonlinearScale, parsed.latencyNonlinearScale);
    EXPECT_EQ(config.latencyHistoryBucketCapacity, parsed.latencyHistoryBucketCapacity);
    EXPECT_DOUBLE_EQ(config.residualLearningRate, parsed.residualLearningRate);
    EXPECT_DOUBLE_EQ(config.residualUpdateErrorFloorMs,
                     parsed.residualUpdateErrorFloorMs);
    EXPECT_DOUBLE_EQ(config.residualUpdateErrorRatio, parsed.residualUpdateErrorRatio);
    EXPECT_EQ(config.residualIdleDecayStartMs, parsed.residualIdleDecayStartMs);
    EXPECT_EQ(config.residualIdleDecayEndMs, parsed.residualIdleDecayEndMs);
}

TEST(AgentSchedulerConfigRTest, testFixedLearningRateDefaults)
{
    NaviSchedConfig serviceConfig;
    AgentSchedulerConfig agentConfig =
        AgentSchedulerConfig::fromServiceConfig(serviceConfig);
    EXPECT_DOUBLE_EQ(0.05, serviceConfig.latencyLearningRate);
    EXPECT_DOUBLE_EQ(1.0, serviceConfig.latencyOnlineLearningRateScale);
    EXPECT_DOUBLE_EQ(0.0, serviceConfig.residualLearningRate);
    EXPECT_DOUBLE_EQ(4.0, serviceConfig.latencyCurrentSampleWeight);
    EXPECT_DOUBLE_EQ(0.05, agentConfig.latencyLearningRate);
    EXPECT_DOUBLE_EQ(1.0, agentConfig.latencyOnlineLearningRateScale);
    EXPECT_DOUBLE_EQ(0.0, agentConfig.residualLearningRate);

    AgentSchedulerConfig parsed;
    std::string errorMessage;
    ASSERT_TRUE(AgentSchedulerConfigR::parse("{}", parsed, errorMessage)) << errorMessage;
    EXPECT_DOUBLE_EQ(0.05, parsed.latencyLearningRate);
    EXPECT_DOUBLE_EQ(1.0, parsed.latencyOnlineLearningRateScale);
    EXPECT_DOUBLE_EQ(0.0, parsed.residualLearningRate);
}

TEST(AgentSchedulerConfigRTest, testZeroResidualLearningRateIsValid)
{
    AgentSchedulerConfig config;
    std::string errorMessage;
    ASSERT_TRUE(AgentSchedulerConfigR::parse(R"({"residual_learning_rate":0})", config,
                                             errorMessage))
        << errorMessage;
    EXPECT_DOUBLE_EQ(0.0, config.residualLearningRate);
    EXPECT_TRUE(config.residualRuntimeConfig().valid());

    config.residualLearningRate = -0.01;
    EXPECT_FALSE(config.valid());
}

TEST(AgentSchedulerConfigRTest, testInvalidConfigNamesRejectedField)
{
    AgentSchedulerConfig config;
    std::string errorMessage;
    EXPECT_FALSE(AgentSchedulerConfigR::parse(R"({"residual_learning_rate":-0.01})",
                                              config, errorMessage));
    EXPECT_NE(std::string::npos, errorMessage.find("residual_learning_rate"));

    errorMessage.clear();
    EXPECT_FALSE(AgentSchedulerConfigR::parse(R"({"latency_learning_rate":0})", config,
                                              errorMessage));
    EXPECT_NE(std::string::npos, errorMessage.find("latency_learning_rate"));

    errorMessage.clear();
    EXPECT_FALSE(AgentSchedulerConfigR::parse(R"({"closed_batch_keep_ms":1})", config,
                                              errorMessage));
    EXPECT_NE(std::string::npos, errorMessage.find("closed_batch_keep_ms"));

    AgentSchedulerConfigR resource;
    AgentSchedulerConfigR::Snapshot snapshot;
    errorMessage.clear();
    EXPECT_EQ(AgentSchedulerConfigR::ApplyResult::INVALID,
              resource.apply(2, "nonempty", R"({"residual_learning_rate":-0.01})",
                             snapshot, errorMessage));
    EXPECT_NE(std::string::npos, errorMessage.find("residual_learning_rate"));
}

TEST(AgentSchedulerConfigRTest, testLatencyNonlinearScaleValidationAndHash)
{
    AgentSchedulerConfig config;
    EXPECT_DOUBLE_EQ(4.0, config.latencyCurrentSampleWeight);
    EXPECT_DOUBLE_EQ(1500.0, config.latencyNonlinearScale);
    EXPECT_TRUE(config.valid());

    auto invalid = config;
    invalid.latencyNonlinearScale = 0.0;
    EXPECT_FALSE(invalid.valid());
    invalid.latencyNonlinearScale = -1.0;
    EXPECT_FALSE(invalid.valid());
    invalid.latencyNonlinearScale = std::numeric_limits<double>::infinity();
    EXPECT_FALSE(invalid.valid());
    invalid.latencyNonlinearScale = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(invalid.valid());

    auto changed = config;
    changed.latencyNonlinearScale = 2048.0;
    const auto configJson = AgentSchedulerConfigR::toJson(config);
    const auto changedJson = AgentSchedulerConfigR::toJson(changed);
    EXPECT_NE(configJson, changedJson);
    EXPECT_NE(AgentSchedulerConfigR::hashConfig(config),
              AgentSchedulerConfigR::hashConfig(changed));
    EXPECT_NE(std::string::npos, changedJson.find("\"latency_nonlinear_scale\""));
}

TEST(AgentSchedulerConfigRTest, testLatencyNonlinearScaleCurlPatchAndNewAgentInheritance)
{
    SchedulerConfigR masterConfig;
    SchedulerConfigR::Request request;
    request.update = true;
    request.configJson = R"({"latency_nonlinear_scale":2048.0})";
    SchedulerConfigResponseDataPtr response;
    bool waitingForAgents = false;
    ASSERT_TRUE(masterConfig.processRequest(request, response, waitingForAgents));
    ASSERT_TRUE(response);
    EXPECT_TRUE(response->accepted) << response->error;
    EXPECT_FALSE(waitingForAgents);
    EXPECT_DOUBLE_EQ(2048.0, masterConfig.get().config.agent.latencyNonlinearScale);
    EXPECT_NE(std::string::npos,
              masterConfig._agentConfigPatchJson.find("\"latency_nonlinear_scale\""));

    masterConfig._subscribers.emplace("collector", SchedulerConfigR::Subscriber());
    AgentSchedulerConfig localConfig;
    const auto localJson = AgentSchedulerConfigR::toJson(localConfig);
    const auto localHash = AgentSchedulerConfigR::hashJson(localJson);
    ASSERT_TRUE(masterConfig.activateSubscriber("collector", "agent", "10.0.0.1", 1,
                                                localHash, localJson));

    SchedulerConfigR::PendingAgentUpdate pending;
    ASSERT_TRUE(masterConfig.getPendingAgentUpdate("collector", pending));
    AgentSchedulerConfig inherited;
    std::string errorMessage;
    ASSERT_TRUE(AgentSchedulerConfigR::parse(pending.configJson, inherited, errorMessage))
        << errorMessage;
    EXPECT_DOUBLE_EQ(2048.0, inherited.latencyNonlinearScale);
    EXPECT_EQ(AgentSchedulerConfigR::hashJson(pending.configJson), pending.configHash);
}

TEST(AgentSchedulerConfigRTest,
     testParameterLearningRateScalesValidationPatchAndInheritance)
{
    AgentSchedulerConfig config;
    EXPECT_TRUE(config.latencyParameterLearningRateScales.empty());
    auto invalid = config;
    invalid.latencyParameterLearningRateScales.assign(16, 1.0);
    EXPECT_FALSE(invalid.valid());
    invalid.latencyParameterLearningRateScales.assign(19, 1.0);
    EXPECT_TRUE(invalid.valid());
    invalid.latencyParameterLearningRateScales[11] = 0.0;
    EXPECT_FALSE(invalid.valid());
    invalid.latencyParameterLearningRateScales[11] =
        std::numeric_limits<double>::infinity();
    EXPECT_FALSE(invalid.valid());

    SchedulerConfigR masterConfig;
    SchedulerConfigR::Request request;
    request.update = true;
    std::vector<double> scales(19, 1.0);
    scales[11] = 0.4;
    scales[14] = 16.0;
    scales[17] = 0.01;
    config.latencyParameterLearningRateScales = scales;
    request.configJson = std::string("{\"latency_parameter_learning_rate_scales\":") +
                         autil::legacy::ToJsonString(scales) + "}";
    SchedulerConfigResponseDataPtr response;
    bool waitingForAgents = false;
    ASSERT_TRUE(masterConfig.processRequest(request, response, waitingForAgents));
    ASSERT_TRUE(response);
    EXPECT_TRUE(response->accepted) << response->error;
    EXPECT_FALSE(waitingForAgents);
    EXPECT_EQ(scales, masterConfig.get().config.agent.latencyParameterLearningRateScales);
    EXPECT_NE(AgentSchedulerConfigR::hashConfig(AgentSchedulerConfig()),
              AgentSchedulerConfigR::hashConfig(config));

    masterConfig._subscribers.emplace("collector", SchedulerConfigR::Subscriber());
    AgentSchedulerConfig localConfig;
    const auto localJson = AgentSchedulerConfigR::toJson(localConfig);
    ASSERT_TRUE(masterConfig.activateSubscriber(
        "collector", "agent", "10.0.0.1", 1, AgentSchedulerConfigR::hashJson(localJson),
        localJson));
    SchedulerConfigR::PendingAgentUpdate pending;
    ASSERT_TRUE(masterConfig.getPendingAgentUpdate("collector", pending));
    AgentSchedulerConfig inherited;
    std::string errorMessage;
    ASSERT_TRUE(AgentSchedulerConfigR::parse(pending.configJson, inherited, errorMessage))
        << errorMessage;
    EXPECT_EQ(scales, inherited.latencyParameterLearningRateScales);
}

TEST(AgentSchedulerConfigRTest,
     testMasterDoesNotOverrideIndependentStartupConfigsWithoutPatch)
{
    SchedulerConfigR masterConfig;
    masterConfig._subscribers.emplace("collector_a", SchedulerConfigR::Subscriber());
    masterConfig._subscribers.emplace("collector_b", SchedulerConfigR::Subscriber());

    AgentSchedulerConfig first;
    first.latencyOnlineLearningRateScale = 2.0;
    first.latencyCurrentSampleWeight = 3.0;
    first.latencyHistoryBucketCapacity = 17;
    first.residualLearningRate = 0.04;
    first.residualUpdateErrorFloorMs = 40.0;
    first.residualUpdateErrorRatio = 0.15;
    first.residualIdleDecayStartMs = 4000;
    first.residualIdleDecayEndMs = 9000;
    AgentSchedulerConfig second;
    second.latencyOnlineLearningRateScale = 5.0;
    second.latencyCurrentSampleWeight = 7.0;
    second.latencyHistoryBucketCapacity = 31;
    second.residualLearningRate = 0.08;
    second.residualUpdateErrorFloorMs = 80.0;
    second.residualUpdateErrorRatio = 0.25;
    second.residualIdleDecayStartMs = 12000;
    second.residualIdleDecayEndMs = 18000;

    const auto firstJson = AgentSchedulerConfigR::toJson(first);
    const auto secondJson = AgentSchedulerConfigR::toJson(second);
    AgentSchedulerConfigR firstAgentConfig;
    AgentSchedulerConfigR secondAgentConfig;
    firstAgentConfig._initialConfigJson = firstJson;
    secondAgentConfig._initialConfigJson = secondJson;
    navi::ResourceInitContext context;
    ASSERT_EQ(navi::EC_NONE, firstAgentConfig.init(context));
    ASSERT_EQ(navi::EC_NONE, secondAgentConfig.init(context));
    const auto firstSnapshot = firstAgentConfig.get();
    const auto secondSnapshot = secondAgentConfig.get();
    EXPECT_NE(firstSnapshot.configHash, secondSnapshot.configHash);
    EXPECT_DOUBLE_EQ(0.04, firstSnapshot.config.residualLearningRate);
    EXPECT_DOUBLE_EQ(0.08, secondSnapshot.config.residualLearningRate);

    ASSERT_TRUE(masterConfig.activateSubscriber(
        "collector_a", "agent_a", "10.0.0.1", firstSnapshot.generation,
        firstSnapshot.configHash, firstSnapshot.configJson));
    ASSERT_TRUE(masterConfig.activateSubscriber(
        "collector_b", "agent_b", "10.0.0.2", secondSnapshot.generation,
        secondSnapshot.configHash, secondSnapshot.configJson));

    SchedulerConfigR::PendingAgentUpdate pending;
    EXPECT_FALSE(masterConfig.getPendingAgentUpdate("collector_a", pending));
    EXPECT_FALSE(masterConfig.getPendingAgentUpdate("collector_b", pending));
    EXPECT_TRUE(masterConfig._agentConfigPatchJson.empty());
    EXPECT_EQ(firstSnapshot.configJson,
              masterConfig._subscribers.at("collector_a").configJson);
    EXPECT_EQ(secondSnapshot.configJson,
              masterConfig._subscribers.at("collector_b").configJson);
    EXPECT_NE(masterConfig._subscribers.at("collector_a").configHash,
              masterConfig._subscribers.at("collector_b").configHash);
}

} // namespace navi_sched

namespace navi_sched {
TEST(AgentSchedulerConfigRTest, testCacheSourceLearningRateCoordinateIdentity)
{
    AgentSchedulerConfig config;
    config.latencyParameterLearningRateScales.assign(21, 1.0);
    EXPECT_FALSE(config.valid());
    std::fill(config.latencyParameterLearningRateScales.begin() + 10,
              config.latencyParameterLearningRateScales.begin() + 19, 0.0);
    EXPECT_TRUE(config.valid());
    config.latencyParameterLearningRateScales[19] = 0.0;
    EXPECT_FALSE(config.valid());
    config.latencyParameterLearningRateScales[19] = 1.0;
    config.latencyParameterLearningRateScales[20] = -1.0;
    EXPECT_FALSE(config.valid());
}

} // namespace navi_sched
