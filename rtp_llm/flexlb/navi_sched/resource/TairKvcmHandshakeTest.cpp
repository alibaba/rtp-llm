#include "aios/navi_sched/data/AgentUpdateToMasterData.h"
#include "unittest/unittest.h"

namespace navi_sched {
namespace {
NaviSchedTairKvcmConfig config()
{
    NaviSchedTairKvcmConfig result;
    result.managerEndpoint = "spectrum://v-isolated:6382";
    result.instanceId = "native_deployment_4096";
    result.hashBlockSize = 4096;
    result.nodeCount = 4;
    for (int32_t group = 0; group < 2; ++group) {
        NaviSchedTairKvcmGroupConfig spec;
        spec.groupId = group;
        spec.hashBlockScale = group + 1;
        spec.tail = group == 1;
        spec.objectSizeBytes = 512 * (group + 1);
        for (int rank = 0; rank < result.nodeCount; ++rank)
            spec.locationSpecNames.push_back("group_" + std::to_string(group) + "_N" + std::to_string(rank));
        result.groups.push_back(std::move(spec));
    }
    return result;
}

std::shared_ptr<AgentUpdateToMasterData> handshake()
{
    auto update = std::make_shared<AgentUpdateToMasterData>();
    update->event = NaviSchedAgentEvent::AGENT_HANDSHAKE;
    update->eventTimeUs = 1;
    update->control = std::make_shared<AgentControlToMasterData>();
    update->control->agentStartTimeUs = 1;
    update->control->tairKvcmConfig = config();
    update->requestEvent = std::make_shared<AgentRequestEventToMasterData>();
    update->requestEvent->inflightStatsRevision = 1;
    return update;
}
} // namespace

class TairKvcmHandshakeTest : public TESTBASE {};

TEST_F(TairKvcmHandshakeTest, registeredLayoutRoundTripsInHandshake)
{
    auto sent = handshake();
    ASSERT_TRUE(sent->valid());
    autil::DataBuffer buffer;
    sent->serialize(buffer);
    AgentUpdateToMasterData received;
    received.deserialize(buffer);
    ASSERT_TRUE(received.valid());
    const auto &result = received.control->tairKvcmConfig;
    EXPECT_TRUE(result.valid());
    EXPECT_EQ("native_deployment_4096", result.instanceId);
    EXPECT_EQ(4096, result.hashBlockSize);
    EXPECT_EQ(4, result.nodeCount);
    ASSERT_EQ(2u, result.groups.size());
    EXPECT_EQ(2, result.groups[1].hashBlockScale);
    EXPECT_TRUE(result.groups[1].tail);
    EXPECT_EQ("group_1_N3", result.groups[1].locationSpecNames[3]);
}

TEST_F(TairKvcmHandshakeTest, orderedEnvelopeKeepsConfigurationOnlyOnHandshake)
{
    auto first = handshake();
    auto request = std::make_shared<AgentUpdateToMasterData>();
    request->event = NaviSchedAgentEvent::REQUEST_RECEIVED;
    request->eventTimeUs = 2;
    request->requestEvent = std::make_shared<AgentRequestEventToMasterData>();
    request->requestEvent->requestId = 42;
    request->requestEvent->inflightStatsRevision = 2;
    ASSERT_TRUE(request->valid());
    auto batch = AgentUpdateToMasterData::makeBatch({first, request});
    ASSERT_TRUE(batch);
    autil::DataBuffer buffer;
    batch->serialize(buffer);
    AgentUpdateToMasterData received;
    received.deserialize(buffer);
    ASSERT_TRUE(received.valid());
    ASSERT_EQ(2u, received.updates.size());
    EXPECT_TRUE(received.updates[0]->control->tairKvcmConfig.configured());
    EXPECT_FALSE(received.updates[1]->control);
}

TEST_F(TairKvcmHandshakeTest, disabledV6dResetsReusedConfiguration)
{
    NaviSchedTairKvcmConfig empty;
    ASSERT_TRUE(empty.valid());
    autil::DataBuffer buffer;
    empty.serialize(buffer);
    auto reused = config();
    reused.deserialize(buffer);
    EXPECT_TRUE(reused.valid());
    EXPECT_FALSE(reused.configured());
    EXPECT_TRUE(reused.groups.empty());
    EXPECT_TRUE(reused.instanceId.empty());
}

TEST_F(TairKvcmHandshakeTest, rejectsIncompleteOrAmbiguousPhysicalLayout)
{
    auto value = config();
    value.groups[0].locationSpecNames.pop_back();
    EXPECT_FALSE(value.valid());
    value = config(); value.groups[1].groupId = 0;
    EXPECT_FALSE(value.valid());
    value = config(); value.groups[1].locationSpecNames[0] = value.groups[0].locationSpecNames[0];
    EXPECT_FALSE(value.valid());
    value = config(); value.groups[0].hashBlockScale = 0;
    EXPECT_FALSE(value.valid());
    value = config(); value.managerEndpoint.clear();
    EXPECT_FALSE(value.valid());
    auto update = handshake();
    update->control->tairKvcmConfig.groups[0].locationSpecNames.clear();
    EXPECT_FALSE(update->valid());
}

TEST_F(TairKvcmHandshakeTest, oldHandshakeVersionIsRejected)
{
    auto update = handshake();
    update->version = 37;
    EXPECT_FALSE(update->valid());
}

} // namespace navi_sched
