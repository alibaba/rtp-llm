#include "aios/navi_sched/common/NaviSchedCommon.h"
#include "aios/navi_sched/data/AgentControlToMasterData.h"
#include "aios/navi_sched/data/AgentUpdateToMasterData.h"
#include "aios/navi_sched/debug/NaviSchedScheduleReplayJson.h"
#include "unittest/unittest.h"

#include <limits>
#include "aios/navi_sched/data/ClientRequestToMasterData.h"

namespace navi_sched {
namespace {

std::shared_ptr<NaviSchedNativePPState> nativeState()
{
    auto value = std::make_shared<NaviSchedNativePPState>();
    value->observationId = 7;
    value->occupiedRequestSlotCount = 1;
    value->burst = 1;
    value->waveOpen = true;
    value->lastRealOrdinal = 0;
    value->measuredForwardMs = 0.25;
    value->slots.push_back({0,
                            500,
                            0.3,
                            100,
                            {1.0, 4.0, 4.0, 32.0, 8.0, 16.0, 1.0, 8.0},
                            {{123, 8192, 4096}}});
    value->results.push_back({100, 500, 0, true, false});
    value->results.push_back({101, 400, -1, false, false});
    value->requests.push_back({123, NativePPQueue::RUNNING, 0,
                               NativePPRequestPhase::RUNNING, 16385, 16384, 8192, 0, -1,
                               -1});
    value->requests.back().prefillSlotRelease =
        NativePPPrefillSlotRelease::AFTER_PREFILL_RESULT;
    value->pendingDecision = NaviSchedNativePPDecision{
        102, {1.0, 4.0, 4.0, 48.0, 12.0, 16.0, 1.0, 12.0}, {{123, 12288, 4096}}};
    return value;
}

NaviSchedWorkloadSnapshot workload()
{
    NaviSchedWorkloadSnapshot value;
    value.revision = 10;
    value.eventTimeUs = 1000;
    value.batchCount = value.requestCount = value.remainingPrefillRequestCount = 1;
    value.runningRequestCount = value.runningElapsedRequestCount = 1;
    value.prefillCapacitySplit = true;
    value.maxRunningRequestCount = value.runningCapacitySlotCount =
        value.runningCapacityPrefillRequestCount = 1;
    value.remainingPrefillContextTokenCount =
        value.runningCapacityPrefillContextTokenCount = 8192;
    value.remainingPrefillTokenCount = value.runningCapacityPrefillTokenCount = 8193;
    value.remainingPrefillTokenSquareSum = value.runningCapacityPrefillTokenSquareSum =
        8193.0 * 8193;
    value.contextRemainingPrefillTokenProductSum =
        value.runningCapacityContextRemainingPrefillTokenProductSum = 8192.0 * 8193;
    value.remainingTotalContextTokenCount = value.remainingPrefillContextTokenCount;
    value.remainingTotalTokenCount = value.remainingPrefillTokenCount;
    value.remainingTotalTokenSquareSum = value.remainingPrefillTokenSquareSum;
    value.contextRemainingTotalTokenProductSum =
        value.contextRemainingPrefillTokenProductSum;
    value.totalPromptTokenCount = 16385;
    value.inFlightScheduledTokenCount = 8192;
    value.nativePPState = nativeState();
    value.nativePPObservationAgeUs = 5000;
    return value;
}

TPUPrefillExecutionPolicyConfig policy()
{
    TPUPrefillExecutionPolicyConfig value;
    value.pcpSize = 8;
    value.interleaveSize = 256;
    value.compileSizes = {256, 512, 1024};
    value.maxScheduledTokens = 8192;
    value.hashBlockSize = value.schedulerBlockSize = 4096;
    value.asyncKVLookupEnabled = true;
    value.asyncKVLookupHashBlockSize = 4096;
    return value;
}

} // namespace

TEST(NaviSchedNativePPStateTest, KeepsMetadataResultsAndIndependentReplyReadiness)
{
    const auto value = nativeState();
    ASSERT_TRUE(value->valid());
    ASSERT_EQ(-1, value->results[1].slotOrdinal);
    ASSERT_FALSE(value->results[1].executeReplyReady);
    ASSERT_FALSE(value->results[1].resultReady);
    ASSERT_TRUE(value->results[0].executeReplyReady);
    ASSERT_FALSE(value->results[0].resultReady);
    ASSERT_EQ(uint64_t(102), value->pendingDecision->schedulerOutputId);
}

TEST(NaviSchedNativePPStateTest, ClosedWaveCanStillHaveAnUnfinishedMetadataCall)
{
    NaviSchedNativePPState value;
    value.observationId = 1;
    value.occupiedRequestSlotCount = 0;
    value.results.push_back({10, 20, -1, false, false});
    ASSERT_TRUE(value.valid());
    value.results[0].slotOrdinal = 0;
    ASSERT_FALSE(value.valid());
}

TEST(NaviSchedNativePPStateTest, RejectsIncorrectMomentsAndMissingOrFutureSlotHistory)
{
    auto value = nativeState();
    value->slots[0].features[3] += 1;
    ASSERT_FALSE(value->valid());
    value = nativeState();
    value->slots[0].lastStageReplyAgeUs = 501;
    ASSERT_FALSE(value->valid());
    value = nativeState();
    value->slots[0].lastStageReplyAgeUs = -1;
    ASSERT_TRUE(value->valid());
    value->results[0].slotOrdinal = 1;
    ASSERT_FALSE(value->valid());
    value = nativeState();
    value->slots[0].ordinal = std::numeric_limits<int64_t>::max();
    ASSERT_FALSE(value->valid());
}

TEST(NaviSchedNativePPStateTest, RejectsDuplicatedIdentityAndRequestIdSortingOfAQueue)
{
    auto value = nativeState();
    value->pendingDecision->schedulerOutputId = 100;
    ASSERT_FALSE(value->valid());
    value = nativeState();
    value->requests[0].queueIndex = 1;
    auto other = value->requests[0];
    other.requestId = 999;
    other.queueIndex = 0;
    value->requests.push_back(other);
    value->occupiedRequestSlotCount = 2;
    ASSERT_FALSE(value->valid());
    std::swap(value->requests[0], value->requests[1]);
    ASSERT_TRUE(value->valid());
    value->requests[1].requestId = 999;
    ASSERT_FALSE(value->valid());
}

TEST(NaviSchedNativePPStateTest, UnknownCacheAndQueryAgesRemainUnknownAcrossTheWire)
{
    auto value = nativeState();
    value->requests[0].cacheHitTokens = -1;
    value->requests[0].phase = NativePPRequestPhase::QUERY_PENDING;
    value->requests[0].queryAgeUs = 12345;
    autil::DataBuffer buffer;
    value->serialize(buffer);
    NaviSchedNativePPState decoded;
    decoded.deserialize(buffer);
    ASSERT_TRUE(decoded.valid());
    ASSERT_TRUE(decoded == *value);
    ASSERT_EQ(-1, decoded.requests[0].cacheHitTokens);
    ASSERT_EQ(-1, decoded.requests[0].loadAgeUs);
    ASSERT_EQ(12345, decoded.requests[0].queryAgeUs);
}

TEST(NaviSchedNativePPStateTest,
     WorkloadAndCompactAgentEnvelopeKeepTheOriginalObservationAge)
{
    const auto value = workload();
    ASSERT_TRUE(value.valid());
    autil::DataBuffer buffer;
    value.serialize(buffer);
    NaviSchedWorkloadSnapshot decoded;
    decoded.deserialize(buffer);
    ASSERT_TRUE(decoded.valid());
    ASSERT_TRUE(decoded.sameWorkload(value));
    ASSERT_EQ(5000, decoded.nativePPObservationAgeUs);
    auto update = std::make_shared<AgentUpdateToMasterData>();
    update->event = NaviSchedAgentEvent::WORKLOAD_UPDATED;
    update->eventTimeUs = 1000;
    update->workload = std::make_shared<NaviSchedWorkloadSnapshot>(value);
    ASSERT_TRUE(update->valid());
    autil::DataBuffer ordinary;
    update->serialize(ordinary);
    AgentUpdateToMasterData ordinaryReceived;
    ordinaryReceived.deserialize(ordinary);
    ASSERT_EQ(43, ordinaryReceived.version);
    ASSERT_TRUE(ordinaryReceived.workload->sameWorkload(value));
    auto later = std::make_shared<AgentUpdateToMasterData>();
    later->event = update->event;
    later->eventTimeUs = update->eventTimeUs + 1000;
    later->workload = std::make_shared<NaviSchedWorkloadSnapshot>(value);
    later->workload->revision += 1;
    later->workload->nativePPObservationAgeUs = 6000;
    const auto batch = AgentUpdateToMasterData::makeBatch({update, later});
    ASSERT_TRUE(batch);
    autil::DataBuffer envelope;
    batch->serialize(envelope);
    AgentUpdateToMasterData received;
    received.deserialize(envelope);
    ASSERT_TRUE(received.valid());
    ASSERT_EQ(44, received.version);
    ASSERT_EQ(size_t(2), received.updates.size());
    ASSERT_EQ(43, received.updates[0]->version);
    ASSERT_TRUE(received.updates[0]->workload->sameWorkload(value));
    ASSERT_EQ(6000, received.updates[1]->workload->nativePPObservationAgeUs);
    ASSERT_EQ(received.updates[0]->workload->nativePPState->observationId,
              received.updates[1]->workload->nativePPState->observationId);
    ASSERT_FALSE(received.updates[0]->control);
}

TEST(NaviSchedNativePPStateTest,
     ObservationPresenceAndAgeArePartOfWorkloadValidityAndEquality)
{
    auto value = workload();
    auto other = value;
    other.nativePPObservationAgeUs += 1;
    ASSERT_FALSE(value.sameWorkload(other));
    other.nativePPState.reset();
    ASSERT_FALSE(other.valid());
    other.nativePPObservationAgeUs = -1;
    ASSERT_TRUE(other.valid());
    value.prefillCapacitySplit = false;
    ASSERT_FALSE(value.valid());
}

TEST(NaviSchedNativePPStateTest, UnifiedHandshakeAndBinaryAndJsonReplayKeepTheTypedPolicy)
{
    AgentControlToMasterData control;
    control.unifiedLatencyPrediction = true;
    control.nativePrefillPipelineStages = 4;
    control.prefillExecutionPolicy = policy();
    autil::DataBuffer wire;
    control.serialize(wire);
    AgentControlToMasterData received;
    received.deserialize(wire);
    ASSERT_FALSE(received.executionGeometryEnabled);
    ASSERT_EQ(4, received.nativePrefillPipelineStages);
    ASSERT_EQ(8192, received.prefillExecutionPolicy.maxScheduledTokens);
    ASSERT_TRUE(received.prefillExecutionPolicy.asyncKVLookupEnabled);

    NaviSchedScheduleReplay replay;
    NaviSchedNodeInfo node;
    node.nativePrefillPipelineStages = 4;
    node.prefillExecutionPolicy = policy();
    node.pdUnifiedCapacitySplit = node.unifiedLatencyPrediction = true;
    node.workload = workload();
    replay.nodes.push_back(node);
    autil::DataBuffer saved;
    replay.serialize(saved);
    NaviSchedScheduleReplay loaded;
    loaded.deserialize(saved);
    ASSERT_EQ(size_t(1), loaded.nodes.size());
    ASSERT_TRUE(loaded.nodes[0].workload.sameWorkload(node.workload));
    ASSERT_EQ(4, loaded.nodes[0].nativePrefillPipelineStages);
    ASSERT_EQ(8192, loaded.nodes[0].prefillExecutionPolicy.maxScheduledTokens);
    const auto json = NaviSchedScheduleReplayJson::toJsonString(replay);
    NaviSchedScheduleReplay fromJson;
    NaviSchedScheduleReplayJson::fromJsonString(json, fromJson);
    ASSERT_EQ(size_t(1), fromJson.nodes.size());
    ASSERT_TRUE(fromJson.nodes[0].workload.sameWorkload(node.workload));
    ASSERT_EQ(4, fromJson.nodes[0].nativePrefillPipelineStages);
    ASSERT_TRUE(fromJson.nodes[0].pdUnifiedCapacitySplit);
    ASSERT_EQ(policy().compileSizes,
              fromJson.nodes[0].prefillExecutionPolicy.compileSizes);
    ASSERT_TRUE(fromJson.nodes[0].prefillExecutionPolicy.asyncKVLookupEnabled);
}

TEST(NaviSchedNativePPStateTest, OccupiedSlotsIncludeOtherNativeRequestsAndSurviveReplay)
{
    auto value = workload();
    value.maxRunningRequestCount = 4;
    auto native = nativeState();
    native->occupiedRequestSlotCount = 3;
    value.nativePPState = native;
    ASSERT_TRUE(value.valid());
    auto other = value;
    other.nativePPState = nativeState();
    ASSERT_FALSE(value.sameWorkload(other));
    autil::DataBuffer buffer;
    value.serialize(buffer);
    NaviSchedWorkloadSnapshot received;
    received.deserialize(buffer);
    ASSERT_EQ(3, received.nativePPState->occupiedRequestSlotCount);
    ASSERT_TRUE(value.sameWorkload(received));
    NaviSchedScheduleReplay replay;
    NaviSchedNodeInfo node;
    node.workload = value;
    replay.nodes.push_back(node);
    NaviSchedScheduleReplay loaded;
    NaviSchedScheduleReplayJson::fromJsonString(
        NaviSchedScheduleReplayJson::toJsonString(replay), loaded);
    ASSERT_EQ(size_t(1), loaded.nodes.size());
    ASSERT_EQ(3, loaded.nodes[0].workload.nativePPState->occupiedRequestSlotCount);
    ASSERT_TRUE(value.sameWorkload(loaded.nodes[0].workload));
    native->occupiedRequestSlotCount = 0;
    ASSERT_FALSE(value.valid());
    native->occupiedRequestSlotCount = -1;
    ASSERT_FALSE(value.valid());
}

TEST(NaviSchedNativePPStateTest, RegisteredRequestCannotLoseItsPrefillReleaseCondition)
{
    auto state = nativeState();
    auto &request = state->requests.front();
    request.prefillSlotRelease = NativePPPrefillSlotRelease::UNKNOWN;
    ASSERT_FALSE(state->valid());
    request.prefillSlotRelease = NativePPPrefillSlotRelease::RETAINS_SLOT;
    ASSERT_TRUE(state->valid());
    autil::DataBuffer buffer;
    state->serialize(buffer);
    NaviSchedNativePPState decoded;
    decoded.deserialize(buffer);
    ASSERT_TRUE(decoded.valid());
    ASSERT_TRUE(decoded == *state);
    ASSERT_EQ(NativePPPrefillSlotRelease::RETAINS_SLOT,
              decoded.requests.front().prefillSlotRelease);
}

TEST(NaviSchedNativePPStateTest, LoadMetadataRequestIdsSurviveWire)
{
    auto value = nativeState();
    value->results[0].loadRequestIds = {1001, 1003};
    value->results[1].loadRequestIds = {1007};
    value->pendingDecision->loadRequestIds = {1011, 1013};
    ASSERT_TRUE(value->valid());
    autil::DataBuffer buffer;
    value->serialize(buffer);
    NaviSchedNativePPState received;
    received.deserialize(buffer);
    ASSERT_TRUE(received.valid());
    ASSERT_EQ(value->results[0].loadRequestIds, received.results[0].loadRequestIds);
    ASSERT_EQ(value->results[1].loadRequestIds, received.results[1].loadRequestIds);
    ASSERT_EQ(value->pendingDecision->loadRequestIds,
              received.pendingDecision->loadRequestIds);
}

TEST(NaviSchedNativePPStateTest, LoadCarrierIdsParticipateInValidationAndEquality)
{
    auto value = nativeState();
    auto other = *value;
    other.results[0].loadRequestIds = {9001};
    ASSERT_TRUE(other.valid());
    ASSERT_FALSE(other == *value);
    other = *value;
    other.pendingDecision->loadRequestIds = {9001};
    ASSERT_TRUE(other.valid());
    ASSERT_FALSE(other == *value);
    for (const std::vector<uint64_t> ids : {std::vector<uint64_t>{0}, {1, 1}, {2, 1}}) {
        auto invalid = *value;
        invalid.results[0].loadRequestIds = ids;
        ASSERT_FALSE(invalid.valid());
        invalid = *value;
        invalid.pendingDecision->loadRequestIds = ids;
        ASSERT_FALSE(invalid.valid());
    }
}

TEST(NaviSchedNativePPStateTest, RequestLifetimeSurvivesBothWireMessages)
{
    ASSERT_EQ(28, NAVI_SCHED_DATA_VERSION);
    for (const auto release : {NativePPPrefillSlotRelease::UNKNOWN,
                               NativePPPrefillSlotRelease::AFTER_PREFILL_RESULT,
                               NativePPPrefillSlotRelease::RETAINS_SLOT}) {
        NaviSchedScheduleRequest request;
        request.requestId = 123;
        request.bizName = "prefill";
        request.timeoutMs = 1000;
        request.promptTokenCount = 32;
        request.blockSize = 16;
        request.blockHashes =
            std::make_shared<const NaviSchedBlockHashes>(NaviSchedBlockHashes{11, 22});
        request.nativePrefillSlotRelease = release;
        ASSERT_TRUE(request.valid());
        autil::DataBuffer buffer;
        request.serialize(buffer);
        NaviSchedScheduleRequest received;
        received.deserialize(buffer);
        ASSERT_TRUE(received.valid());
        ASSERT_EQ(release, received.nativePrefillSlotRelease);
        ASSERT_EQ(*request.blockHashes, *received.blockHashes);
        ClientRequestToMasterData rpc;
        rpc.requestId = request.requestId;
        rpc.bizName = request.bizName;
        rpc.timeoutMs = request.timeoutMs;
        rpc.promptTokenCount = request.promptTokenCount;
        rpc.blockSize = request.blockSize;
        rpc.blockHashes = request.blockHashes;
        rpc.nativePrefillSlotRelease = release;
        ASSERT_TRUE(rpc.valid());
        autil::DataBuffer rpcBuffer;
        rpc.serialize(rpcBuffer);
        ClientRequestToMasterData receivedRpc;
        receivedRpc.deserialize(rpcBuffer);
        ASSERT_TRUE(receivedRpc.valid());
        ASSERT_EQ(release, receivedRpc.nativePrefillSlotRelease);
        for (const int32_t bad : {-1, 3, 99}) {
            request.nativePrefillSlotRelease =
                static_cast<NativePPPrefillSlotRelease>(bad);
            rpc.nativePrefillSlotRelease = static_cast<NativePPPrefillSlotRelease>(bad);
            EXPECT_FALSE(request.valid());
            EXPECT_FALSE(rpc.valid());
        }
    }
}

} // namespace navi_sched
