#include "aios/navi_sched/data/BlockHashUtil.h"
#include "aios/navi_sched/resource/ReplayClientR.h"
#include "aios/navi_sched/resource/ReplayControlR.h"
#include "autil/CRC32C.h"
#include "autil/DataBuffer.h"
#include "autil/TimeUtility.h"
#include "unittest/unittest.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <signal.h>
#include <sys/file.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace navi_sched {
namespace {

NaviSchedScheduleRequest makeRequest(uint64_t requestId)
{
    NaviSchedScheduleRequest request;
    request.requestId = requestId;
    request.bizName = "target";
    request.promptTokenCount = 0;
    request.timeoutMs = 1000;
    return request;
}

std::string historyKey(uint64_t value)
{
    std::ostringstream stream;
    stream << std::hex << std::setfill('0') << std::setw(16) << value;
    return stream.str();
}

void appendPersistentHistoryRecord(const std::filesystem::path &path,
                                   const NaviSchedScheduleRequest &request,
                                   int64_t eventTimeUs, uint32_t recordVersion = 1)
{
    constexpr uint32_t RECORD_MAGIC = 0x4E535248;
    autil::DataBuffer buffer;
    // This helper creates the archived v1 history envelope. Its payload
    // uses the frozen v9/v10 request layout, not the current wire protocol.
    const int32_t legacyVersion =
        request.version == NAVI_SCHED_DATA_VERSION ? 10 : request.version;
    buffer.write(legacyVersion);
    buffer.write(request.requestId);
    buffer.write(request.bizName);
    buffer.write(request.promptTokenCount);
    buffer.write(request.blockSize);
    BlockHashUtil::write(buffer, request.blockHashes);
    buffer.write(request.timeoutMs);
    buffer.write(request.forwardPayload);
    const auto payloadSize = static_cast<uint64_t>(buffer.getDataLen());
    const auto checksum = autil::CRC32C::Value(buffer.getData(), buffer.getDataLen());
    std::ofstream output(path, std::ios::binary | std::ios::app);
    ASSERT_TRUE(output);
    output.write(reinterpret_cast<const char *>(&RECORD_MAGIC), sizeof(RECORD_MAGIC));
    output.write(reinterpret_cast<const char *>(&recordVersion), sizeof(recordVersion));
    output.write(reinterpret_cast<const char *>(&eventTimeUs), sizeof(eventTimeUs));
    output.write(reinterpret_cast<const char *>(&payloadSize), sizeof(payloadSize));
    output.write(reinterpret_cast<const char *>(&checksum), sizeof(checksum));
    output.write(buffer.getData(), buffer.getDataLen());
    output.flush();
    ASSERT_TRUE(output);
}

class TempDirectory {
public:
    TempDirectory()
    {
        const auto *testTmpDir = std::getenv("TEST_TMPDIR");
        const auto base = testTmpDir ? std::filesystem::path(testTmpDir)
                                     : std::filesystem::temp_directory_path();
        path = base / ("replay_history_" +
                       std::to_string(autil::TimeUtility::currentTimeInMicroSeconds()));
        std::filesystem::create_directories(path);
    }

    ~TempDirectory()
    {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }

    std::filesystem::path path;
};

int waitForHistoryChild(pid_t child)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    int status = 0;
    while (std::chrono::steady_clock::now() < deadline) {
        const auto result = ::waitpid(child, &status, WNOHANG);
        if (result == child) {
            return status;
        }
        if (result < 0 && errno != EINTR) {
            return -1;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ::kill(child, SIGKILL);
    while (::waitpid(child, &status, 0) < 0 && errno == EINTR) {
    }
    return -1;
}

ReplayControlCommandDataPtr makeStart(ReplayMode mode, int64_t startTimeUs,
                                      int64_t endTimeUs, uint64_t loopCount,
                                      int64_t durationMs, double qps)
{
    auto command = std::make_shared<ReplayControlCommandData>();
    command->action = ReplayAction::START;
    command->mode = mode;
    command->startTimeUs = startTimeUs;
    command->endTimeUs = endTimeUs;
    command->loopCount = loopCount;
    command->durationMs = durationMs;
    command->perClientQps = qps;
    return command;
}

}

class CaptureAsyncPipe : public navi::AsyncPipe {
public:
    CaptureAsyncPipe() : navi::AsyncPipe(nullptr, navi::AS_REQUIRED) {}

    navi::ErrorCode setData(const navi::DataPtr &data) override
    {
        if (failSetData) {
            return navi::EC_ABORT;
        }
        dataItems.emplace_back(data);
        return navi::EC_NONE;
    }

    navi::ErrorCode setEof() override
    {
        eof = true;
        return navi::EC_NONE;
    }

    std::vector<navi::DataPtr> dataItems;
    bool eof = false;
    bool failSetData = false;
};

class ReplayControlRTest : public TESTBASE {};

TEST_F(ReplayControlRTest, testRangeValidationRejectsEmptyClientSet)
{
    ReplayControlR resource;
    ReplayControlR::Request request;
    request.action = "start";
    request.mode = "fixed_qps";
    request.startTimeUs = 100;
    request.endTimeUs = 200;
    request.loopCount = 1;
    request.totalQps = 10.0;
    ReplayControlResponseDataPtr response;
    ASSERT_TRUE(resource.processRequest(request, response));
    ASSERT_TRUE(response);
    EXPECT_FALSE(response->accepted);
    EXPECT_EQ("no replayable samples in requested selection", response->error);
    EXPECT_EQ(ReplayAction::STOP, response->action);
    EXPECT_EQ(0u, response->eligibleClientCount);

    request.mode = "unknown";
    ASSERT_TRUE(resource.processRequest(request, response));
    EXPECT_FALSE(response->accepted);
    EXPECT_NE(std::string::npos, response->error.find("mode"));

    request.action = "query";
    ASSERT_TRUE(resource.processRequest(request, response));
    EXPECT_TRUE(response->accepted);
    EXPECT_EQ(0u, response->clients.size());
}

TEST_F(ReplayControlRTest, testNewSelectionAndStopClearPreviousSelectionState)
{
    ReplayControlR resource;
    auto client = std::make_shared<CaptureAsyncPipe>();
    ASSERT_TRUE(resource.subscribe("client-a", client));

    ReplayHistorySummary summary;
    summary.clientId = "client-a";
    summary.earliestEventTimeUs = 100;
    summary.latestEventTimeUs = 300;
    summary.sampleCount = 3;
    summary.replayable = true;
    summary.selectionGeneration = 7;
    summary.selectedSampleCount = 2;
    resource.updateHistory(summary);

    ReplayControlR::Request request;
    request.action = "start";
    request.mode = "fixed_qps";
    request.startTimeUs = 200;
    request.endTimeUs = 250;
    request.loopCount = 1;
    request.totalQps = 10.0;
    ReplayControlResponseDataPtr response;
    ASSERT_TRUE(resource.processRequest(request, response));
    ASSERT_TRUE(response->accepted);
    ASSERT_EQ(1u, response->clients.size());
    EXPECT_EQ(1u, response->eligibleClientCount);
    EXPECT_EQ(0u, response->clients[0].selectionGeneration);
    EXPECT_EQ(0u, response->clients[0].selectedSampleCount);

    summary.selectionGeneration = 8;
    summary.selectedSampleCount = 1;
    resource.updateHistory(summary);
    request.action = "stop";
    ASSERT_TRUE(resource.processRequest(request, response));
    ASSERT_TRUE(response->accepted);
    ASSERT_EQ(1u, response->clients.size());
    EXPECT_EQ(ReplayAction::STOP, response->action);
    EXPECT_EQ(0u, response->clients[0].selectionGeneration);
    EXPECT_EQ(0u, response->clients[0].selectedSampleCount);
}

TEST_F(ReplayControlRTest, testStartRejectsSelectionOutsideAllKnownHistory)
{
    ReplayControlR resource;
    auto clientA = std::make_shared<CaptureAsyncPipe>();
    auto clientB = std::make_shared<CaptureAsyncPipe>();
    ASSERT_TRUE(resource.subscribe("client-a", clientA));
    ASSERT_TRUE(resource.subscribe("client-b", clientB));

    ReplayHistorySummary summary;
    summary.earliestEventTimeUs = 100;
    summary.latestEventTimeUs = 300;
    summary.sampleCount = 3;
    summary.replayable = true;
    summary.clientId = "client-a";
    resource.updateHistory(summary);
    summary.clientId = "client-b";
    resource.updateHistory(summary);

    ReplayControlR::Request request;
    request.action = "start";
    request.mode = "realtime";
    request.startTimeUs = 400;
    request.endTimeUs = 500;
    request.loopCount = 1;
    ReplayControlResponseDataPtr response;
    ASSERT_TRUE(resource.processRequest(request, response));
    ASSERT_TRUE(response);
    EXPECT_FALSE(response->accepted);
    EXPECT_EQ("no replayable samples in requested selection", response->error);
    EXPECT_EQ(ReplayAction::STOP, response->action);
    EXPECT_EQ(0u, response->eligibleClientCount);
    EXPECT_EQ(0u, response->generation);
    EXPECT_EQ(1u, clientA->dataItems.size());
    EXPECT_EQ(1u, clientB->dataItems.size());
}

TEST_F(ReplayControlRTest, testStartRejectsMissingOrInsufficientHistory)
{
    ReplayControlR resource;
    auto missingClient = std::make_shared<CaptureAsyncPipe>();
    auto shortClient = std::make_shared<CaptureAsyncPipe>();
    ASSERT_TRUE(resource.subscribe("missing", missingClient));
    ASSERT_TRUE(resource.subscribe("short", shortClient));

    ReplayHistorySummary summary;
    summary.clientId = "short";
    summary.earliestEventTimeUs = 100;
    summary.latestEventTimeUs = 300;
    summary.sampleCount = 3;
    summary.replayable = true;
    resource.updateHistory(summary);

    ReplayControlR::Request request;
    request.action = "start";
    request.mode = "fixed_qps";
    request.historyMode = "recent_count";
    request.recentQueryCount = 4;
    request.loopCount = 1;
    request.totalQps = 10.0;
    ReplayControlResponseDataPtr response;
    ASSERT_TRUE(resource.processRequest(request, response));
    ASSERT_TRUE(response);
    EXPECT_FALSE(response->accepted);
    EXPECT_EQ("no replayable samples in requested selection", response->error);
    EXPECT_EQ(ReplayAction::STOP, response->action);
    EXPECT_EQ(0u, response->generation);
}

TEST_F(ReplayControlRTest, testRejectedUpdatePreservesActiveCommand)
{
    ReplayControlR resource;
    auto client = std::make_shared<CaptureAsyncPipe>();
    ASSERT_TRUE(resource.subscribe("client-a", client));

    ReplayHistorySummary summary;
    summary.clientId = "client-a";
    summary.earliestEventTimeUs = 100;
    summary.latestEventTimeUs = 300;
    summary.sampleCount = 3;
    summary.replayable = true;
    resource.updateHistory(summary);

    ReplayControlR::Request request;
    request.action = "start";
    request.mode = "fixed_qps";
    request.startTimeUs = 100;
    request.endTimeUs = 200;
    request.loopCount = 2;
    request.totalQps = 10.0;
    ReplayControlResponseDataPtr response;
    ASSERT_TRUE(resource.processRequest(request, response));
    ASSERT_TRUE(response->accepted);
    const auto generation = response->generation;
    const auto commandCount = client->dataItems.size();

    request.action = "update";
    request.startTimeUs = 400;
    request.endTimeUs = 500;
    request.totalQps = 20.0;
    ASSERT_TRUE(resource.processRequest(request, response));
    ASSERT_TRUE(response);
    EXPECT_FALSE(response->accepted);
    EXPECT_EQ("no replayable samples in requested selection", response->error);
    EXPECT_EQ(ReplayAction::START, response->action);
    EXPECT_EQ(generation, response->generation);
    EXPECT_EQ(1u, response->eligibleClientCount);
    EXPECT_DOUBLE_EQ(10.0, response->perClientQps);
    EXPECT_EQ(commandCount, client->dataItems.size());
}

TEST_F(ReplayControlRTest, testReplayDataRoundTrip)
{
    ReplayControlCommandData command;
    command.action = ReplayAction::START;
    command.mode = ReplayMode::FIXED_QPS;
    command.generation = 7;
    command.startTimeUs = 1000;
    command.endTimeUs = 2000;
    command.durationMs = 3000;
    command.totalQps = 120.5;
    command.perClientQps = 40.25;
    command.loopCount = 4;
    command.historyMode = ReplayHistoryMode::RECENT_COUNT;
    command.recentQueryCount = 19;
    command.selectionGeneration = 23;
    command.pacingIndex = 2;
    command.pacingClientCount = 3;
    command.pacingEpochUs = 123456789;
    command.pacingClockOffsetUs = -12345;

    autil::DataBuffer commandBuffer;
    command.serialize(commandBuffer);
    ReplayControlCommandData commandOutput;
    commandOutput.deserialize(commandBuffer);
    EXPECT_EQ(command.action, commandOutput.action);
    EXPECT_EQ(command.mode, commandOutput.mode);
    EXPECT_EQ(command.generation, commandOutput.generation);
    EXPECT_EQ(command.startTimeUs, commandOutput.startTimeUs);
    EXPECT_EQ(command.endTimeUs, commandOutput.endTimeUs);
    EXPECT_EQ(command.durationMs, commandOutput.durationMs);
    EXPECT_DOUBLE_EQ(command.totalQps, commandOutput.totalQps);
    EXPECT_DOUBLE_EQ(command.perClientQps, commandOutput.perClientQps);
    EXPECT_EQ(command.loopCount, commandOutput.loopCount);
    EXPECT_EQ(command.historyMode, commandOutput.historyMode);
    EXPECT_EQ(command.recentQueryCount, commandOutput.recentQueryCount);
    EXPECT_EQ(command.selectionGeneration, commandOutput.selectionGeneration);
    EXPECT_EQ(command.pacingIndex, commandOutput.pacingIndex);
    EXPECT_EQ(command.pacingClientCount, commandOutput.pacingClientCount);
    EXPECT_EQ(command.pacingEpochUs, commandOutput.pacingEpochUs);
    EXPECT_EQ(command.pacingClockOffsetUs, commandOutput.pacingClockOffsetUs);

    autil::DataBuffer legacyCommandBuffer;
    legacyCommandBuffer.write(command.action);
    legacyCommandBuffer.write(command.mode);
    legacyCommandBuffer.write(command.generation);
    legacyCommandBuffer.write(command.startTimeUs);
    legacyCommandBuffer.write(command.endTimeUs);
    legacyCommandBuffer.write(command.durationMs);
    legacyCommandBuffer.write(command.totalQps);
    legacyCommandBuffer.write(command.perClientQps);
    legacyCommandBuffer.write(command.loopCount);
    legacyCommandBuffer.write(command.historyMode);
    legacyCommandBuffer.write(command.recentQueryCount);
    legacyCommandBuffer.write(command.selectionGeneration);
    legacyCommandBuffer.write(command.pacingIndex);
    legacyCommandBuffer.write(command.pacingClientCount);
    ReplayControlCommandData legacyCommandOutput;
    legacyCommandOutput.deserialize(legacyCommandBuffer);
    EXPECT_EQ(command.pacingClientCount, legacyCommandOutput.pacingClientCount);
    EXPECT_EQ(0, legacyCommandOutput.pacingEpochUs);
    EXPECT_EQ(0, legacyCommandOutput.pacingClockOffsetUs);

    autil::DataBuffer previousCommandBuffer;
    previousCommandBuffer.write(command.action);
    previousCommandBuffer.write(command.mode);
    previousCommandBuffer.write(command.generation);
    previousCommandBuffer.write(command.startTimeUs);
    previousCommandBuffer.write(command.endTimeUs);
    previousCommandBuffer.write(command.durationMs);
    previousCommandBuffer.write(command.totalQps);
    previousCommandBuffer.write(command.perClientQps);
    previousCommandBuffer.write(command.loopCount);
    previousCommandBuffer.write(command.historyMode);
    previousCommandBuffer.write(command.recentQueryCount);
    previousCommandBuffer.write(command.selectionGeneration);
    previousCommandBuffer.write(command.pacingIndex);
    previousCommandBuffer.write(command.pacingClientCount);
    previousCommandBuffer.write(command.pacingEpochUs);
    ReplayControlCommandData previousCommandOutput;
    previousCommandOutput.deserialize(previousCommandBuffer);
    EXPECT_EQ(command.pacingEpochUs, previousCommandOutput.pacingEpochUs);
    EXPECT_EQ(0, previousCommandOutput.pacingClockOffsetUs);

    ReplayClientCapabilityData capability;
    capability.clientId = "client-a";
    capability.summary.clientId = capability.clientId;
    capability.summary.earliestEventTimeUs = 1000;
    capability.summary.latestEventTimeUs = 2000;
    capability.summary.sampleCount = 12;
    capability.summary.replayable = true;
    capability.summary.selectionGeneration = 31;
    capability.summary.selectedSampleCount = 7;
    capability.summary.completedSelectionGeneration = 31;
    capability.summary.reportTimeUs = 987654321;
    autil::DataBuffer capabilityBuffer;
    capability.serialize(capabilityBuffer);
    ReplayClientCapabilityData capabilityOutput;
    capabilityOutput.deserialize(capabilityBuffer);
    EXPECT_EQ(capability.clientId, capabilityOutput.clientId);
    EXPECT_EQ(capability.summary.clientId, capabilityOutput.summary.clientId);
    EXPECT_EQ(capability.summary.earliestEventTimeUs,
              capabilityOutput.summary.earliestEventTimeUs);
    EXPECT_EQ(capability.summary.latestEventTimeUs,
              capabilityOutput.summary.latestEventTimeUs);
    EXPECT_EQ(capability.summary.sampleCount, capabilityOutput.summary.sampleCount);
    EXPECT_EQ(capability.summary.replayable, capabilityOutput.summary.replayable);
    EXPECT_EQ(capability.summary.selectionGeneration,
              capabilityOutput.summary.selectionGeneration);
    EXPECT_EQ(capability.summary.selectedSampleCount,
              capabilityOutput.summary.selectedSampleCount);
    EXPECT_EQ(capability.summary.completedSelectionGeneration,
              capabilityOutput.summary.completedSelectionGeneration);
    EXPECT_EQ(capability.summary.reportTimeUs, capabilityOutput.summary.reportTimeUs);

    autil::DataBuffer legacyCapabilityBuffer;
    legacyCapabilityBuffer.write(capability.clientId);
    legacyCapabilityBuffer.write(capability.summary.clientId);
    legacyCapabilityBuffer.write(capability.summary.earliestEventTimeUs);
    legacyCapabilityBuffer.write(capability.summary.latestEventTimeUs);
    legacyCapabilityBuffer.write(capability.summary.sampleCount);
    legacyCapabilityBuffer.write(capability.summary.replayable);
    legacyCapabilityBuffer.write(capability.summary.status);
    legacyCapabilityBuffer.write(capability.summary.error);
    legacyCapabilityBuffer.write(capability.summary.selectionGeneration);
    legacyCapabilityBuffer.write(capability.summary.selectedSampleCount);
    legacyCapabilityBuffer.write(capability.summary.completedSelectionGeneration);
    ReplayClientCapabilityData legacyCapabilityOutput;
    legacyCapabilityOutput.deserialize(legacyCapabilityBuffer);
    EXPECT_EQ(capability.summary.completedSelectionGeneration,
              legacyCapabilityOutput.summary.completedSelectionGeneration);
    EXPECT_EQ(0, legacyCapabilityOutput.summary.reportTimeUs);
}

TEST_F(ReplayControlRTest, testStartValidationRejectsInvalidRequests)
{
    ReplayControlR resource;
    ReplayControlR::Request request;
    request.action = "start";
    request.mode = "fixed_qps";
    request.startTimeUs = 100;
    request.endTimeUs = 200;
    request.loopCount = 1;
    request.totalQps = 10.0;
    ReplayControlResponseDataPtr response;

    request.startTimeUs = 0;
    ASSERT_TRUE(resource.processRequest(request, response));
    EXPECT_FALSE(response->accepted);
    EXPECT_NE(std::string::npos, response->error.find("start_time_us"));

    request.startTimeUs = 100;
    request.endTimeUs = 99;
    ASSERT_TRUE(resource.processRequest(request, response));
    EXPECT_FALSE(response->accepted);
    EXPECT_NE(std::string::npos, response->error.find("end_time_us"));

    request.endTimeUs = 200;
    request.loopCount = 0;
    request.durationMs = 0;
    ASSERT_TRUE(resource.processRequest(request, response));
    EXPECT_FALSE(response->accepted);
    EXPECT_NE(std::string::npos, response->error.find("loop_count"));

    request.loopCount = 1;
    request.totalQps = 0.0;
    ASSERT_TRUE(resource.processRequest(request, response));
    EXPECT_FALSE(response->accepted);
    EXPECT_NE(std::string::npos, response->error.find("total_qps"));

    request.totalQps = 10.0;
    request.endTimeUs = -1;
    ASSERT_TRUE(resource.processRequest(request, response));
    EXPECT_FALSE(response->accepted);
    EXPECT_NE(std::string::npos, response->error.find("end_time_us"));

    request.endTimeUs = 200;
    request.durationMs = 1;
    request.startTimeUs = std::numeric_limits<int64_t>::max();
    request.endTimeUs = 0;
    ASSERT_TRUE(resource.processRequest(request, response));
    EXPECT_FALSE(response->accepted);
    EXPECT_NE(std::string::npos, response->error.find("overflows"));
}

TEST_F(ReplayControlRTest, testFixedQpsReallocationOnlyOnEligibilityChange)
{
    ReplayControlR resource;
    auto clientA = std::make_shared<CaptureAsyncPipe>();
    auto clientB = std::make_shared<CaptureAsyncPipe>();
    ASSERT_TRUE(resource.subscribe("client-a", clientA));
    ASSERT_TRUE(resource.subscribe("client-b", clientB));

    ReplayHistorySummary summaryA;
    summaryA.clientId = "client-a";
    summaryA.earliestEventTimeUs = 100;
    summaryA.latestEventTimeUs = 300;
    summaryA.sampleCount = 3;
    summaryA.replayable = true;
    ReplayHistorySummary summaryB = summaryA;
    summaryB.clientId = "client-b";
    resource.updateHistory(summaryA);
    resource.updateHistory(summaryB);

    ReplayControlR::Request request;
    request.action = "start";
    request.mode = "fixed_qps";
    request.startTimeUs = 100;
    request.endTimeUs = 200;
    request.loopCount = 1;
    request.totalQps = 100.0;
    ReplayControlResponseDataPtr response;
    ASSERT_TRUE(resource.processRequest(request, response));
    ASSERT_TRUE(response->accepted);
    EXPECT_EQ(2u, response->eligibleClientCount);
    EXPECT_DOUBLE_EQ(50.0, response->perClientQps);
    ASSERT_EQ(2u, response->clients.size());

    const auto beforeStableUpdate = clientA->dataItems.size();
    summaryA.sampleCount = 4;
    resource.updateHistory(summaryA);
    EXPECT_EQ(beforeStableUpdate, clientA->dataItems.size());

    resource.unsubscribe("client-b", clientB);
    ASSERT_GT(clientA->dataItems.size(), beforeStableUpdate);
    auto command =
        std::dynamic_pointer_cast<ReplayControlCommandData>(clientA->dataItems.back());
    ASSERT_TRUE(command);
    EXPECT_EQ(ReplayAction::START, command->action);
    EXPECT_DOUBLE_EQ(100.0, command->perClientQps);

    request.action = "stop";
    ASSERT_TRUE(resource.processRequest(request, response));
    EXPECT_TRUE(response->accepted);
    EXPECT_EQ(ReplayAction::STOP, response->action);
    command =
        std::dynamic_pointer_cast<ReplayControlCommandData>(clientA->dataItems.back());
    ASSERT_TRUE(command);
    EXPECT_EQ(ReplayAction::STOP, command->action);
}

TEST_F(ReplayControlRTest, testFixedQpsDropsConfirmedEmptySelection)
{
    ReplayControlR resource;
    auto clientA = std::make_shared<CaptureAsyncPipe>();
    auto clientB = std::make_shared<CaptureAsyncPipe>();
    ASSERT_TRUE(resource.subscribe("client-a", clientA));
    ASSERT_TRUE(resource.subscribe("client-b", clientB));

    ReplayHistorySummary summaryA;
    summaryA.clientId = "client-a";
    summaryA.earliestEventTimeUs = 100;
    summaryA.latestEventTimeUs = 300;
    summaryA.sampleCount = 3;
    summaryA.replayable = true;
    auto summaryB = summaryA;
    summaryB.clientId = "client-b";
    resource.updateHistory(summaryA);
    resource.updateHistory(summaryB);

    ReplayControlR::Request request;
    request.action = "start";
    request.mode = "fixed_qps";
    request.startTimeUs = 150;
    request.endTimeUs = 200;
    request.loopCount = 1;
    request.totalQps = 100.0;
    ReplayControlResponseDataPtr response;
    ASSERT_TRUE(resource.processRequest(request, response));
    ASSERT_TRUE(response->accepted);
    EXPECT_EQ(2u, response->eligibleClientCount);
    EXPECT_DOUBLE_EQ(50.0, response->perClientQps);

    auto commandA =
        std::dynamic_pointer_cast<ReplayControlCommandData>(clientA->dataItems.back());
    ASSERT_TRUE(commandA);
    ASSERT_GT(commandA->selectionGeneration, 0u);
    summaryA.selectionGeneration = commandA->selectionGeneration;
    summaryA.selectedSampleCount = 1;
    const auto commandCount = clientA->dataItems.size();
    resource.updateHistory(summaryA);
    EXPECT_EQ(commandCount, clientA->dataItems.size());

    summaryB.selectionGeneration = commandA->selectionGeneration;
    summaryB.selectedSampleCount = 0;
    resource.updateHistory(summaryB);
    ASSERT_GT(clientA->dataItems.size(), commandCount);
    commandA =
        std::dynamic_pointer_cast<ReplayControlCommandData>(clientA->dataItems.back());
    ASSERT_TRUE(commandA);
    EXPECT_EQ(ReplayAction::START, commandA->action);
    EXPECT_DOUBLE_EQ(100.0, commandA->perClientQps);
    auto commandB =
        std::dynamic_pointer_cast<ReplayControlCommandData>(clientB->dataItems.back());
    ASSERT_TRUE(commandB);
    EXPECT_EQ(ReplayAction::STOP, commandB->action);
    EXPECT_EQ(commandA->selectionGeneration, commandB->selectionGeneration);

    ASSERT_TRUE(resource.processRequest({"query"}, response));
    ASSERT_TRUE(response->accepted);
    EXPECT_EQ(1u, response->eligibleClientCount);
    EXPECT_DOUBLE_EQ(100.0, response->perClientQps);
}

TEST_F(ReplayControlRTest, testMasterStopsAfterAllClientsCompleteSelection)
{
    ReplayControlR resource;
    auto clientA = std::make_shared<CaptureAsyncPipe>();
    auto clientB = std::make_shared<CaptureAsyncPipe>();
    ASSERT_TRUE(resource.subscribe("client-a", clientA));
    ASSERT_TRUE(resource.subscribe("client-b", clientB));

    ReplayHistorySummary summaryA;
    summaryA.clientId = "client-a";
    summaryA.earliestEventTimeUs = 100;
    summaryA.latestEventTimeUs = 300;
    summaryA.sampleCount = 3;
    summaryA.replayable = true;
    auto summaryB = summaryA;
    summaryB.clientId = "client-b";
    resource.updateHistory(summaryA);
    resource.updateHistory(summaryB);

    ReplayControlR::Request request;
    request.action = "start";
    request.mode = "fixed_qps";
    request.startTimeUs = 100;
    request.endTimeUs = 200;
    request.loopCount = 1;
    request.totalQps = 100.0;
    ReplayControlResponseDataPtr response;
    ASSERT_TRUE(resource.processRequest(request, response));
    ASSERT_TRUE(response->accepted);
    auto command =
        std::dynamic_pointer_cast<ReplayControlCommandData>(clientA->dataItems.back());
    ASSERT_TRUE(command);
    const auto selectionGeneration = command->selectionGeneration;
    ASSERT_NE(0u, selectionGeneration);

    summaryA.selectionGeneration = selectionGeneration;
    summaryA.selectedSampleCount = 2;
    summaryB.selectionGeneration = selectionGeneration;
    summaryB.selectedSampleCount = 1;
    resource.updateHistory(summaryA);
    resource.updateHistory(summaryB);

    summaryA.completedSelectionGeneration = selectionGeneration;
    resource.updateHistory(summaryA);
    ASSERT_TRUE(resource.processRequest({"query"}, response));
    EXPECT_EQ(ReplayAction::START, response->action);

    summaryB.completedSelectionGeneration = selectionGeneration;
    resource.updateHistory(summaryB);
    ASSERT_TRUE(resource.processRequest({"query"}, response));
    EXPECT_EQ(ReplayAction::STOP, response->action);
    EXPECT_EQ(0u, response->eligibleClientCount);
    command =
        std::dynamic_pointer_cast<ReplayControlCommandData>(clientA->dataItems.back());
    ASSERT_TRUE(command);
    EXPECT_EQ(ReplayAction::STOP, command->action);
    EXPECT_EQ(0u, command->selectionGeneration);
}

TEST_F(ReplayControlRTest, testMasterStopsAfterAllSelectionsAreEmpty)
{
    ReplayControlR resource;
    auto client = std::make_shared<CaptureAsyncPipe>();
    ASSERT_TRUE(resource.subscribe("client-a", client));

    ReplayHistorySummary summary;
    summary.clientId = "client-a";
    summary.earliestEventTimeUs = 100;
    summary.latestEventTimeUs = 300;
    summary.sampleCount = 3;
    summary.replayable = true;
    resource.updateHistory(summary);

    ReplayControlR::Request request;
    request.action = "start";
    request.mode = "fixed_qps";
    request.startTimeUs = 150;
    request.endTimeUs = 200;
    request.loopCount = 1;
    request.totalQps = 100.0;
    ReplayControlResponseDataPtr response;
    ASSERT_TRUE(resource.processRequest(request, response));
    auto command =
        std::dynamic_pointer_cast<ReplayControlCommandData>(client->dataItems.back());
    ASSERT_TRUE(command);

    summary.selectionGeneration = command->selectionGeneration;
    summary.selectedSampleCount = 0;
    resource.updateHistory(summary);
    ASSERT_TRUE(resource.processRequest({"query"}, response));
    EXPECT_EQ(ReplayAction::STOP, response->action);
    EXPECT_EQ(0u, response->eligibleClientCount);
}

TEST_F(ReplayControlRTest, testTimeRangeIncludesPartiallyOverlappingHistory)
{
    ReplayControlR resource;
    auto overlappingClient = std::make_shared<CaptureAsyncPipe>();
    auto outsideClient = std::make_shared<CaptureAsyncPipe>();
    ASSERT_TRUE(resource.subscribe("overlapping", overlappingClient));
    ASSERT_TRUE(resource.subscribe("outside", outsideClient));

    ReplayHistorySummary summary;
    summary.clientId = "overlapping";
    summary.earliestEventTimeUs = 150;
    summary.latestEventTimeUs = 175;
    summary.sampleCount = 2;
    summary.replayable = true;
    resource.updateHistory(summary);
    summary.clientId = "outside";
    summary.earliestEventTimeUs = 201;
    summary.latestEventTimeUs = 300;
    resource.updateHistory(summary);

    ReplayControlR::Request request;
    request.action = "start";
    request.mode = "fixed_qps";
    request.startTimeUs = 100;
    request.endTimeUs = 200;
    request.loopCount = 1;
    request.totalQps = 80.0;
    ReplayControlResponseDataPtr response;
    ASSERT_TRUE(resource.processRequest(request, response));
    ASSERT_TRUE(response->accepted);
    EXPECT_EQ(1u, response->eligibleClientCount);
    EXPECT_DOUBLE_EQ(80.0, response->perClientQps);

    auto command = std::dynamic_pointer_cast<ReplayControlCommandData>(
        overlappingClient->dataItems.back());
    ASSERT_TRUE(command);
    EXPECT_EQ(ReplayAction::START, command->action);
    command = std::dynamic_pointer_cast<ReplayControlCommandData>(
        outsideClient->dataItems.back());
    ASSERT_TRUE(command);
    EXPECT_EQ(ReplayAction::STOP, command->action);
}

TEST_F(ReplayControlRTest, testRealtimeDoesNotUseQpsAllocation)
{
    ReplayControlR resource;
    auto client = std::make_shared<CaptureAsyncPipe>();
    ASSERT_TRUE(resource.subscribe("client-a", client));
    ReplayHistorySummary summary;
    summary.clientId = "client-a";
    summary.earliestEventTimeUs = 100;
    summary.latestEventTimeUs = 200;
    summary.sampleCount = 2;
    summary.replayable = true;
    resource.updateHistory(summary);

    ReplayControlR::Request request;
    request.action = "start";
    request.mode = "realtime";
    request.startTimeUs = 100;
    request.endTimeUs = 200;
    request.loopCount = 1;
    request.totalQps = 0.0;
    ReplayControlResponseDataPtr response;
    ASSERT_TRUE(resource.processRequest(request, response));
    EXPECT_TRUE(response->accepted);
    EXPECT_DOUBLE_EQ(0.0, response->perClientQps);

    request.totalQps = 100.0;
    ASSERT_TRUE(resource.processRequest(request, response));
    EXPECT_FALSE(response->accepted);
    EXPECT_NE(std::string::npos, response->error.find("total_qps"));
}

TEST_F(ReplayControlRTest, testMasterSharesRealtimeEpochAcrossClients)
{
    ReplayControlR resource;
    auto clientA = std::make_shared<CaptureAsyncPipe>();
    auto clientB = std::make_shared<CaptureAsyncPipe>();
    ASSERT_TRUE(resource.subscribe("client-a", clientA));
    ASSERT_TRUE(resource.subscribe("client-b", clientB));

    ReplayHistorySummary summary;
    summary.earliestEventTimeUs = 100;
    summary.latestEventTimeUs = 200;
    summary.sampleCount = 2;
    summary.replayable = true;
    summary.clientId = "client-a";
    resource.updateHistory(summary);
    summary.clientId = "client-b";
    resource.updateHistory(summary);

    ReplayControlR::Request request;
    request.action = "start";
    request.mode = "realtime";
    request.startTimeUs = 100;
    request.endTimeUs = 201;
    request.loopCount = 1;
    const auto beforeStartUs = autil::TimeUtility::currentTimeInMicroSeconds();
    ReplayControlResponseDataPtr response;
    ASSERT_TRUE(resource.processRequest(request, response));
    ASSERT_TRUE(response->accepted);

    const auto commandA =
        std::dynamic_pointer_cast<ReplayControlCommandData>(clientA->dataItems.back());
    const auto commandB =
        std::dynamic_pointer_cast<ReplayControlCommandData>(clientB->dataItems.back());
    ASSERT_TRUE(commandA);
    ASSERT_TRUE(commandB);
    EXPECT_EQ(ReplayAction::START, commandA->action);
    EXPECT_EQ(ReplayAction::START, commandB->action);
    EXPECT_GT(commandA->pacingEpochUs, beforeStartUs);
    EXPECT_EQ(commandA->pacingEpochUs, commandB->pacingEpochUs);
}

TEST_F(ReplayControlRTest, testMasterSendsPerClientClockOffsetWithSharedEpoch)
{
    ReplayControlR resource;
    auto clientA = std::make_shared<CaptureAsyncPipe>();
    auto clientB = std::make_shared<CaptureAsyncPipe>();
    ASSERT_TRUE(resource.subscribe("client-a", clientA));
    ASSERT_TRUE(resource.subscribe("client-b", clientB));

    ReplayHistorySummary summaryA;
    summaryA.clientId = "client-a";
    summaryA.earliestEventTimeUs = 100;
    summaryA.latestEventTimeUs = 200;
    summaryA.sampleCount = 2;
    summaryA.replayable = true;
    summaryA.reportTimeUs = autil::TimeUtility::currentTimeInMicroSeconds() + 2'000'000;
    resource.updateHistory(summaryA);

    auto summaryB = summaryA;
    summaryB.clientId = "client-b";
    summaryB.reportTimeUs = autil::TimeUtility::currentTimeInMicroSeconds() - 3'000'000;
    resource.updateHistory(summaryB);

    ReplayControlR::Request request;
    request.action = "start";
    request.mode = "fixed_qps";
    request.startTimeUs = 100;
    request.endTimeUs = 201;
    request.loopCount = 10;
    request.totalQps = 100.0;
    ReplayControlResponseDataPtr response;
    ASSERT_TRUE(resource.processRequest(request, response));
    ASSERT_TRUE(response->accepted);

    auto commandA =
        std::dynamic_pointer_cast<ReplayControlCommandData>(clientA->dataItems.back());
    auto commandB =
        std::dynamic_pointer_cast<ReplayControlCommandData>(clientB->dataItems.back());
    ASSERT_TRUE(commandA);
    ASSERT_TRUE(commandB);
    EXPECT_EQ(commandA->pacingEpochUs, commandB->pacingEpochUs);
    EXPECT_GT(commandA->pacingClockOffsetUs, 1'500'000);
    EXPECT_LE(commandA->pacingClockOffsetUs, 2'000'000);
    EXPECT_LE(commandB->pacingClockOffsetUs, -3'000'000);
    EXPECT_GT(commandB->pacingClockOffsetUs, -3'500'000);

    const auto commandCount = clientA->dataItems.size();
    const auto otherCommandCount = clientB->dataItems.size();
    summaryA.reportTimeUs = autil::TimeUtility::currentTimeInMicroSeconds() + 1'000'000;
    resource.updateHistory(summaryA);
    EXPECT_EQ(commandCount, clientA->dataItems.size());

    summaryA.reportTimeUs = autil::TimeUtility::currentTimeInMicroSeconds() + 2'500'000;
    resource.updateHistory(summaryA);
    ASSERT_GT(clientA->dataItems.size(), commandCount);
    commandA =
        std::dynamic_pointer_cast<ReplayControlCommandData>(clientA->dataItems.back());
    ASSERT_TRUE(commandA);
    EXPECT_GT(commandA->pacingClockOffsetUs, 2'000'000);
    EXPECT_LE(commandA->pacingClockOffsetUs, 2'500'000);
    EXPECT_EQ(otherCommandCount, clientB->dataItems.size());

    ReplayHistoryQueryResponseData queryResponse;
    queryResponse.queryId = 1;
    summaryA.reportTimeUs = autil::TimeUtility::currentTimeInMicroSeconds() + 3'000'000;
    queryResponse.summary = summaryA;
    const auto updatedCommandCount = clientA->dataItems.size();
    resource.completeHistoryQuery(queryResponse);
    ASSERT_EQ(updatedCommandCount + 1, clientA->dataItems.size());
    EXPECT_EQ(otherCommandCount, clientB->dataItems.size());
    commandA =
        std::dynamic_pointer_cast<ReplayControlCommandData>(clientA->dataItems.back());
    ASSERT_TRUE(commandA);
    EXPECT_EQ(commandB->pacingEpochUs, commandA->pacingEpochUs);
    EXPECT_EQ(0u, commandA->pacingIndex);
    EXPECT_EQ(2u, commandA->pacingClientCount);
    EXPECT_DOUBLE_EQ(50.0, commandA->perClientQps);
    EXPECT_GT(commandA->pacingClockOffsetUs, 2'500'000);
}

TEST_F(ReplayControlRTest, testFleetBroadcastAndSingleClientClockUpdate)
{
    constexpr size_t CLIENT_COUNT = 4096;
    ReplayControlR resource;
    std::vector<std::shared_ptr<CaptureAsyncPipe>> clients;
    ReplayHistorySummary summary;
    summary.earliestEventTimeUs = 100;
    summary.latestEventTimeUs = 200;
    summary.sampleCount = 2;
    summary.replayable = true;
    for (size_t index = 0; index < CLIENT_COUNT; ++index) {
        auto pipe = std::make_shared<CaptureAsyncPipe>();
        summary.clientId = historyKey(index);
        ASSERT_TRUE(resource.subscribe(summary.clientId, pipe));
        resource.updateHistory(summary);
        clients.emplace_back(std::move(pipe));
    }
    ReplayControlR::Request request;
    request.action = "start";
    request.mode = "fixed_qps";
    request.startTimeUs = 100;
    request.endTimeUs = 201;
    request.loopCount = 10;
    request.totalQps = 10'000;
    ReplayControlResponseDataPtr response;
    const auto begin = std::chrono::steady_clock::now();
    ASSERT_TRUE(resource.processRequest(request, response));
    const auto startDone = std::chrono::steady_clock::now();
    ASSERT_TRUE(response->accepted);
    ASSERT_EQ(CLIENT_COUNT, response->eligibleClientCount);
    for (size_t index = 0; index < CLIENT_COUNT; ++index) {
        ASSERT_EQ(2u, clients[index]->dataItems.size());
        const auto command = std::dynamic_pointer_cast<ReplayControlCommandData>(
            clients[index]->dataItems.back());
        ASSERT_TRUE(command);
        ASSERT_EQ(index, command->pacingIndex);
        ASSERT_EQ(CLIENT_COUNT, command->pacingClientCount);
        ASSERT_DOUBLE_EQ(10'000.0 / CLIENT_COUNT, command->perClientQps);
    }
    summary.clientId = historyKey(CLIENT_COUNT / 2);
    summary.reportTimeUs = autil::TimeUtility::currentTimeInMicroSeconds() + 1'000'000;
    const auto updateBegin = std::chrono::steady_clock::now();
    resource.updateHistory(summary);
    const auto updateDone = std::chrono::steady_clock::now();
    std::fprintf(
        stderr, "fleet_clients=%zu broadcast_us=%ld clock_update_us=%ld\n", CLIENT_COUNT,
        std::chrono::duration_cast<std::chrono::microseconds>(startDone - begin).count(),
        std::chrono::duration_cast<std::chrono::microseconds>(updateDone - updateBegin)
            .count());
    for (size_t index = 0; index < CLIENT_COUNT; ++index) {
        ASSERT_EQ(index == CLIENT_COUNT / 2 ? 3u : 2u, clients[index]->dataItems.size())
            << index;
    }
}

TEST_F(ReplayControlRTest, testClockUpdateDeliveryFailureRebalancesClients)
{
    for (bool expired : {false, true}) {
        ReplayControlR resource;
        auto clientA = std::make_shared<CaptureAsyncPipe>();
        auto clientB = std::make_shared<CaptureAsyncPipe>();
        ASSERT_TRUE(resource.subscribe("client-a", clientA));
        ASSERT_TRUE(resource.subscribe("client-b", clientB));
        ReplayHistorySummary summary;
        summary.earliestEventTimeUs = 100;
        summary.latestEventTimeUs = 200;
        summary.sampleCount = 2;
        summary.replayable = true;
        summary.clientId = "client-a";
        resource.updateHistory(summary);
        summary.clientId = "client-b";
        resource.updateHistory(summary);
        ReplayControlR::Request request;
        request.action = "start";
        request.mode = "fixed_qps";
        request.startTimeUs = 100;
        request.endTimeUs = 201;
        request.loopCount = 10;
        request.totalQps = 100;
        ReplayControlResponseDataPtr response;
        ASSERT_TRUE(resource.processRequest(request, response));
        ASSERT_TRUE(response->accepted);
        if (expired) {
            clientA.reset();
        } else {
            clientA->failSetData = true;
        }
        summary.clientId = "client-a";
        summary.reportTimeUs = autil::TimeUtility::currentTimeInMicroSeconds();
        resource.updateHistory(summary);
        ASSERT_EQ(1u, resource.getHistory().size());
        const auto command = std::dynamic_pointer_cast<ReplayControlCommandData>(
            clientB->dataItems.back());
        ASSERT_TRUE(command);
        EXPECT_EQ(ReplayAction::START, command->action);
        EXPECT_EQ(0u, command->pacingIndex);
        EXPECT_EQ(1u, command->pacingClientCount);
        EXPECT_DOUBLE_EQ(100.0, command->perClientQps);
    }
}

TEST_F(ReplayControlRTest, testPersistentHistoryRestoresIndexAndLoadsOnDemand)
{
    TempDirectory directory;
    {
        ReplayHistoryStore store;
        ASSERT_TRUE(store.configure(directory.path.string(), 10));
        for (uint64_t requestId = 1; requestId <= 3; ++requestId) {
            auto request = makeRequest(requestId);
            request.forwardPayload = "payload-" + std::to_string(requestId);
            ASSERT_TRUE(store.append(request, 1000 * requestId));
        }
    }

    ReplayHistoryStore restored;
    ASSERT_TRUE(restored.configure(directory.path.string(), 10));
    EXPECT_TRUE(restored.persistent());
    EXPECT_EQ(0u, restored.residentRequestCount());
    const auto summary = restored.getSummary("client-a");
    EXPECT_EQ(3u, summary.sampleCount);
    EXPECT_EQ(1000, summary.earliestEventTimeUs);
    EXPECT_EQ(3000, summary.latestEventTimeUs);

    std::vector<ReplayHistorySample> samples;
    std::string error;
    ASSERT_TRUE(restored.loadTimeRange(1500, 3000, samples, error)) << error;
    ASSERT_EQ(2u, samples.size());
    EXPECT_EQ(2u, samples[0].request.requestId);
    EXPECT_EQ("payload-2", samples[0].request.forwardPayload);
    EXPECT_EQ(3u, samples[1].request.requestId);
    EXPECT_EQ("payload-3", samples[1].request.forwardPayload);

    ASSERT_TRUE(restored.loadRecent(2, samples, error)) << error;
    ASSERT_EQ(2u, samples.size());
    EXPECT_EQ(2u, samples[0].request.requestId);
    EXPECT_EQ(3u, samples[1].request.requestId);
}

TEST_F(ReplayControlRTest, testPersistentHistoryCoordinatesIndependentWriters)
{
    TempDirectory directory;
    ReplayHistoryStore first;
    ReplayHistoryStore second;
    ASSERT_TRUE(first.configure(directory.path.string(), 10));
    ASSERT_TRUE(second.configure(directory.path.string(), 10));
    ASSERT_TRUE(first.append(makeRequest(1), 1000));
    ASSERT_TRUE(second.append(makeRequest(2), 2000));
    ASSERT_TRUE(first.append(makeRequest(3), 3000));

    for (auto *store : {&first, &second}) {
        EXPECT_EQ(3u, store->getSummary("client").sampleCount);
        std::vector<ReplayHistorySample> samples;
        std::string error;
        ASSERT_TRUE(store->loadRecent(3, samples, error)) << error;
        ASSERT_EQ(3u, samples.size());
        for (size_t index = 0; index < samples.size(); ++index) {
            EXPECT_EQ(index + 1, samples[index].request.requestId);
        }
    }
}

TEST_F(ReplayControlRTest, testPersistentHistoryRefreshesSameSizeCompaction)
{
    TempDirectory directory;
    ReplayHistoryStore writer;
    ReplayHistoryStore reader;
    ASSERT_TRUE(writer.configure(directory.path.string(), 2));
    for (uint64_t id = 1; id <= 3; ++id) {
        ASSERT_TRUE(writer.append(makeRequest(id), 1000 * id));
    }
    ASSERT_TRUE(reader.configure(directory.path.string(), 2));
    ASSERT_EQ(3000, reader.getSummary("reader").latestEventTimeUs);
    const auto sizeBefore = std::filesystem::file_size(directory.path / "queries.bin");
    ASSERT_TRUE(writer.append(makeRequest(4), 4000));
    ASSERT_TRUE(writer.append(makeRequest(5), 5000));
    ASSERT_EQ(sizeBefore, std::filesystem::file_size(directory.path / "queries.bin"));

    std::vector<ReplayHistorySample> samples;
    std::string error;
    ASSERT_TRUE(reader.loadRecent(2, samples, error)) << error;
    ASSERT_EQ(2u, samples.size());
    EXPECT_EQ(4u, samples[0].request.requestId);
    EXPECT_EQ(5u, samples[1].request.requestId);
    EXPECT_EQ(5000, reader.getSummary("reader").latestEventTimeUs);
}

TEST_F(ReplayControlRTest, testPersistentHistoryCoordinatesConcurrentProcesses)
{
    TempDirectory referenceDirectory;
    ReplayHistoryStore reference;
    ASSERT_TRUE(reference.configure(referenceDirectory.path.string(), 1));
    auto largestRequest = makeRequest(80);
    largestRequest.forwardPayload = "payload-80";
    ASSERT_TRUE(reference.append(largestRequest, 80000));
    const auto recordSize =
        std::filesystem::file_size(referenceDirectory.path / "queries.bin");
    for (size_t limit : {8u, 128u}) {
        SCOPED_TRACE(limit);
        TempDirectory directory;
        ReplayHistoryStore parent;
        ASSERT_TRUE(parent.configure(directory.path.string(), limit));
        int startPipe[2];
        ASSERT_EQ(0, ::pipe(startPipe));
        const auto child = ::fork();
        if (child == 0) {
            ::close(startPipe[1]);
            char start = 0;
            if (::read(startPipe[0], &start, 1) != 1) {
                ::_exit(1);
            }
            ::close(startPipe[0]);
            ReplayHistoryStore writer;
            if (!writer.configure(directory.path.string(), limit)) {
                ::_exit(2);
            }
            for (uint64_t id = 1; id <= 79; id += 2) {
                auto request = makeRequest(id);
                request.forwardPayload = "payload-" + std::to_string(id);
                if (!writer.append(request, 1000 * id)) {
                    ::_exit(3);
                }
            }
            ::_exit(0);
        }
        ::close(startPipe[0]);
        if (child < 0) {
            ::close(startPipe[1]);
            FAIL() << "fork failed";
        }
        const auto started = ::write(startPipe[1], "s", 1);
        ::close(startPipe[1]);
        bool appended = true;
        for (uint64_t id = 2; id <= 80; id += 2) {
            auto request = makeRequest(id);
            request.forwardPayload = "payload-" + std::to_string(id);
            if (!parent.append(request, 1000 * id)) {
                appended = false;
                break;
            }
        }
        const int status = waitForHistoryChild(child);
        ASSERT_EQ(1, started);
        ASSERT_TRUE(appended);
        ASSERT_NE(-1, status);
        ASSERT_TRUE(WIFEXITED(status));
        ASSERT_EQ(0, WEXITSTATUS(status));
        const size_t count = std::min<size_t>(limit, 80);
        std::vector<ReplayHistorySample> samples;
        std::string error;
        ASSERT_TRUE(parent.loadRecent(count, samples, error)) << error;
        ASSERT_EQ(count, samples.size());
        for (size_t index = 0; index < count; ++index) {
            const uint64_t id = 81 - count + index;
            EXPECT_EQ(id, samples[index].request.requestId);
            EXPECT_EQ("payload-" + std::to_string(id),
                      samples[index].request.forwardPayload);
        }
        ReplayHistoryStore restored;
        ASSERT_TRUE(restored.configure(directory.path.string(), limit));
        EXPECT_EQ(count, restored.getSummary("restored").sampleCount);
        EXPECT_LE(std::filesystem::file_size(directory.path / "queries.bin"),
                  recordSize * limit * 2);
    }
}

TEST_F(ReplayControlRTest, testPersistentHistoryRecoversAfterWriterProcessExits)
{
    TempDirectory directory;
    ReplayHistoryStore observer;
    ASSERT_TRUE(observer.configure(directory.path.string(), 2));
    ASSERT_TRUE(observer.append(makeRequest(1), 1000));
    const auto dataFile = directory.path / "queries.bin";
    const auto validSize = std::filesystem::file_size(dataFile);
    const auto child = ::fork();
    ASSERT_GE(child, 0);
    if (child == 0) {
        const int directoryFd = ::open(directory.path.c_str(), O_RDONLY | O_DIRECTORY);
        if (directoryFd < 0 || ::flock(directoryFd, LOCK_EX) != 0) {
            ::_exit(1);
        }
        {
            std::ofstream output(dataFile, std::ios::binary | std::ios::app);
            output << "partial record";
            output.flush();
            if (!output) {
                ::_exit(2);
            }
            std::ofstream temporary(dataFile.string() + ".tmp");
            temporary << "unfinished compaction";
        }
        ::kill(::getpid(), SIGKILL);
        ::_exit(3);
    }
    const int status = waitForHistoryChild(child);
    ASSERT_NE(-1, status);
    ASSERT_TRUE(WIFSIGNALED(status));
    ASSERT_EQ(SIGKILL, WTERMSIG(status));
    ReplayHistoryStore restored;
    ASSERT_TRUE(restored.configure(directory.path.string(), 2));
    EXPECT_EQ(validSize, std::filesystem::file_size(dataFile));
    EXPECT_FALSE(std::filesystem::exists(dataFile.string() + ".tmp"));
    ASSERT_TRUE(observer.append(makeRequest(2), 2000));
    std::vector<ReplayHistorySample> samples;
    std::string error;
    ASSERT_TRUE(restored.loadRecent(2, samples, error)) << error;
    ASSERT_EQ(2u, samples.size());
    EXPECT_EQ(1u, samples[0].request.requestId);
    EXPECT_EQ(2u, samples[1].request.requestId);
}

TEST_F(ReplayControlRTest, testPersistentHistoryUsesIndependentPayloadVersion)
{
    TempDirectory directory;
    ReplayHistoryStore store;
    ASSERT_TRUE(store.configure(directory.path.string(), 10));
    ASSERT_TRUE(store.append(makeRequest(1), 1000));

    std::ifstream input(directory.path / "queries.bin", std::ios::binary);
    ASSERT_TRUE(input);
    uint32_t magic = 0;
    uint32_t recordVersion = 0;
    int64_t eventTimeUs = 0;
    uint64_t payloadSize = 0;
    uint32_t checksum = 0;
    int32_t payloadVersion = 0;
    input.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    input.read(reinterpret_cast<char *>(&recordVersion), sizeof(recordVersion));
    input.read(reinterpret_cast<char *>(&eventTimeUs), sizeof(eventTimeUs));
    input.read(reinterpret_cast<char *>(&payloadSize), sizeof(payloadSize));
    input.read(reinterpret_cast<char *>(&checksum), sizeof(checksum));
    input.read(reinterpret_cast<char *>(&payloadVersion), sizeof(payloadVersion));
    ASSERT_TRUE(input);
    EXPECT_EQ(0x4E535248u, magic);
    EXPECT_EQ(2u, recordVersion);
    EXPECT_EQ(1000, eventTimeUs);
    EXPECT_GT(payloadSize, sizeof(payloadVersion));
    EXPECT_NE(0u, checksum);
    EXPECT_EQ(1, payloadVersion);
    EXPECT_NE(NAVI_SCHED_DATA_VERSION, payloadVersion);
}

TEST_F(ReplayControlRTest, testPersistentHistoryRestoresCompatibleLegacyVersions)
{
    TempDirectory directory;
    const auto dataFile = directory.path / "queries.bin";
    auto oldRequest = makeRequest(1);
    oldRequest.version = 9;
    appendPersistentHistoryRecord(dataFile, oldRequest, 1000);
    appendPersistentHistoryRecord(dataFile, makeRequest(2), 2000);
    oldRequest.requestId = 3;
    appendPersistentHistoryRecord(dataFile, oldRequest, 3000);
    const auto mixedFileSize = std::filesystem::file_size(dataFile);

    ReplayHistoryStore store;
    ASSERT_TRUE(store.configure(directory.path.string(), 10));
    EXPECT_EQ(mixedFileSize, std::filesystem::file_size(dataFile));
    const auto summary = store.getSummary("client-a");
    ASSERT_EQ(3u, summary.sampleCount);
    EXPECT_EQ(1000, summary.earliestEventTimeUs);
    EXPECT_EQ(3000, summary.latestEventTimeUs);

    std::vector<ReplayHistorySample> samples;
    std::string error;
    ASSERT_TRUE(store.loadRecent(3, samples, error)) << error;
    ASSERT_EQ(3u, samples.size());
    EXPECT_EQ(1u, samples[0].request.requestId);
    EXPECT_EQ(NAVI_SCHED_DATA_VERSION, samples[0].request.version);
    EXPECT_EQ(2u, samples[1].request.requestId);
    EXPECT_EQ(3u, samples[2].request.requestId);

    ASSERT_TRUE(store.append(makeRequest(4), 4000));
    ReplayHistoryStore restored;
    ASSERT_TRUE(restored.configure(directory.path.string(), 10));
    ASSERT_EQ(4u, restored.getSummary("client-a").sampleCount);
    ASSERT_TRUE(restored.loadRecent(4, samples, error)) << error;
    ASSERT_EQ(4u, samples.size());
    EXPECT_EQ(1u, samples[0].request.requestId);
    EXPECT_EQ(4u, samples[3].request.requestId);
}

TEST_F(ReplayControlRTest, testPersistentHistoryDropsUnknownLegacyVersionRecords)
{
    TempDirectory directory;
    const auto dataFile = directory.path / "queries.bin";
    auto unknownRequest = makeRequest(1);
    unknownRequest.version = NAVI_SCHED_DATA_VERSION - 2;
    appendPersistentHistoryRecord(dataFile, unknownRequest, 1000);
    unknownRequest.requestId = 3;
    unknownRequest.version = 2;
    appendPersistentHistoryRecord(dataFile, unknownRequest, 1500, 2);
    appendPersistentHistoryRecord(dataFile, makeRequest(2), 2000);
    const auto mixedFileSize = std::filesystem::file_size(dataFile);

    ReplayHistoryStore store;
    ASSERT_TRUE(store.configure(directory.path.string(), 10));
    EXPECT_LT(std::filesystem::file_size(dataFile), mixedFileSize);
    const auto summary = store.getSummary("client-a");
    ASSERT_EQ(1u, summary.sampleCount);
    EXPECT_EQ(2000, summary.earliestEventTimeUs);
    EXPECT_EQ(2000, summary.latestEventTimeUs);

    std::vector<ReplayHistorySample> samples;
    std::string error;
    ASSERT_TRUE(store.loadRecent(1, samples, error)) << error;
    ASSERT_EQ(1u, samples.size());
    EXPECT_EQ(2u, samples[0].request.requestId);
}

TEST_F(ReplayControlRTest, testHistoryLoadFailureDoesNotAbortControlStream)
{
    TempDirectory directory;
    ReplayClientR client;
    client.setClientId("client-a");
    ASSERT_TRUE(client.configureHistory(directory.path.string(), 10));
    ASSERT_TRUE(client.recordNormalQuery(makeRequest(1), 1000));
    auto capabilityPipe = std::make_shared<CaptureAsyncPipe>();
    ASSERT_TRUE(client.subscribeHistory(capabilityPipe));
    ASSERT_EQ(1u, client.getSummary().sampleCount);

    std::filesystem::resize_file(directory.path / "queries.bin", 0);
    auto command = makeStart(ReplayMode::FIXED_QPS, 1000, 2000, 1, 0, 1.0);
    command->historyMode = ReplayHistoryMode::RECENT_COUNT;
    command->recentQueryCount = 1;
    command->generation = 1;
    command->selectionGeneration = 7;
    ASSERT_TRUE(client.applyCommand(*command));
    EXPECT_FALSE(client.replaying());

    ASSERT_GE(capabilityPipe->dataItems.size(), 2u);
    const auto capability = std::dynamic_pointer_cast<ReplayClientCapabilityData>(
        capabilityPipe->dataItems.back());
    ASSERT_TRUE(capability);
    EXPECT_EQ(7u, capability->summary.selectionGeneration);
    EXPECT_EQ(0u, capability->summary.selectedSampleCount);
}

TEST_F(ReplayControlRTest, testPersistentHistoryRetainsNewestAndRepairsTail)
{
    TempDirectory directory;
    uintmax_t validFileSize = 0;
    {
        ReplayHistoryStore store;
        ASSERT_TRUE(store.configure(directory.path.string(), 2));
        ASSERT_TRUE(store.append(makeRequest(1), 1000));
        ASSERT_TRUE(store.append(makeRequest(2), 2000));
        ASSERT_TRUE(store.append(makeRequest(3), 3000));
        validFileSize = std::filesystem::file_size(directory.path / "queries.bin");
    }
    {
        std::ofstream output(directory.path / "queries.bin",
                             std::ios::binary | std::ios::app);
        ASSERT_TRUE(output);
        output.write("truncated", 9);
    }
    {
        ReplayHistoryStore store;
        ASSERT_TRUE(store.configure(directory.path.string(), 2));
        EXPECT_EQ(validFileSize,
                  std::filesystem::file_size(directory.path / "queries.bin"));
        EXPECT_EQ(2u, store.getSummary("client-a").sampleCount);
        ASSERT_TRUE(store.append(makeRequest(4), 4000));
    }
    ReplayHistoryStore restored;
    ASSERT_TRUE(restored.configure(directory.path.string(), 2));
    std::vector<ReplayHistorySample> samples;
    std::string error;
    ASSERT_TRUE(restored.loadRecent(2, samples, error)) << error;
    ASSERT_EQ(2u, samples.size());
    EXPECT_EQ(3u, samples[0].request.requestId);
    EXPECT_EQ(4u, samples[1].request.requestId);
}

TEST_F(ReplayControlRTest, testPersistentHistoryDropsChecksumCorruptTail)
{
    TempDirectory directory;
    uintmax_t sizeAfterTwoRecords = 0;
    {
        ReplayHistoryStore store;
        ASSERT_TRUE(store.configure(directory.path.string(), 10));
        ASSERT_TRUE(store.append(makeRequest(1), 1000));
        ASSERT_TRUE(store.append(makeRequest(2), 2000));
        sizeAfterTwoRecords = std::filesystem::file_size(directory.path / "queries.bin");
        ASSERT_TRUE(store.append(makeRequest(3), 3000));
    }
    {
        std::fstream file(directory.path / "queries.bin",
                          std::ios::binary | std::ios::in | std::ios::out);
        ASSERT_TRUE(file);
        char lastByte = 0;
        file.seekg(-1, std::ios::end);
        file.get(lastByte);
        ASSERT_TRUE(file);
        file.seekp(-1, std::ios::end);
        file.put(static_cast<char>(lastByte ^ 0xff));
        file.flush();
        ASSERT_TRUE(file);
    }

    ReplayHistoryStore restored;
    ASSERT_TRUE(restored.configure(directory.path.string(), 10));
    EXPECT_EQ(sizeAfterTwoRecords,
              std::filesystem::file_size(directory.path / "queries.bin"));
    EXPECT_EQ(2u, restored.getSummary("client-a").sampleCount);
    ASSERT_TRUE(restored.append(makeRequest(4), 4000));
    std::vector<ReplayHistorySample> samples;
    std::string error;
    ASSERT_TRUE(restored.loadRecent(3, samples, error)) << error;
    ASSERT_EQ(3u, samples.size());
    EXPECT_EQ(1u, samples[0].request.requestId);
    EXPECT_EQ(2u, samples[1].request.requestId);
    EXPECT_EQ(4u, samples[2].request.requestId);
}

TEST_F(ReplayControlRTest, testPersistentHistoryCompactsBeforeRecordLimit)
{
    TempDirectory directory;
    ReplayHistoryStore store;
    ASSERT_TRUE(store.configure(directory.path.string(), 2));
    const auto dataFile = directory.path / "queries.bin";

    ASSERT_TRUE(store.append(makeRequest(1), 1000));
    const auto recordSize = std::filesystem::file_size(dataFile);
    ASSERT_GT(recordSize, 0u);
    ASSERT_TRUE(store.append(makeRequest(2), 2000));
    ASSERT_TRUE(store.append(makeRequest(3), 3000));
    ASSERT_TRUE(store.append(makeRequest(4), 4000));
    EXPECT_EQ(recordSize * 4, std::filesystem::file_size(dataFile));

    ASSERT_TRUE(store.append(makeRequest(5), 5000));
    EXPECT_EQ(recordSize * 3, std::filesystem::file_size(dataFile));
    EXPECT_EQ(2u, store.getSummary("client-a").sampleCount);

    ReplayHistoryStore restored;
    ASSERT_TRUE(restored.configure(directory.path.string(), 2));
    std::vector<ReplayHistorySample> samples;
    std::string error;
    ASSERT_TRUE(restored.loadRecent(2, samples, error)) << error;
    ASSERT_EQ(2u, samples.size());
    EXPECT_EQ(4u, samples[0].request.requestId);
    EXPECT_EQ(5u, samples[1].request.requestId);
}

TEST_F(ReplayControlRTest, testPersistentHistoryMigratesToSingleDirectory)
{
    TempDirectory directory;
    const auto root = directory.path / "replay_history";
    for (uint64_t index = 1; index <= 2; ++index) {
        ReplayHistoryStore store;
        const auto path = root / historyKey(index);
        ASSERT_TRUE(store.configure(path.string(), 10));
        ASSERT_TRUE(store.append(makeRequest(index), 1000 * index));
    }

    ReplayHistoryStore migrated;
    ASSERT_TRUE(migrated.configure(root.string(), 10, (root / historyKey(2)).string()));
    EXPECT_FALSE(std::filesystem::exists(root / historyKey(1)));
    EXPECT_FALSE(std::filesystem::exists(root / historyKey(2)));
    EXPECT_TRUE(std::filesystem::exists(root / "queries.bin"));
    EXPECT_FALSE(std::filesystem::exists(root / "queries.bin.tmp"));
    size_t rootEntryCount = 0;
    for (const auto &entry : std::filesystem::directory_iterator(root)) {
        (void)entry;
        ++rootEntryCount;
    }
    EXPECT_EQ(1u, rootEntryCount);
    std::vector<ReplayHistorySample> samples;
    std::string error;
    ASSERT_TRUE(migrated.loadRecent(1, samples, error)) << error;
    ASSERT_EQ(1u, samples.size());
    EXPECT_EQ(2u, samples[0].request.requestId);
}

TEST_F(ReplayControlRTest, testPersistentHistoryCleansCompactionTempFile)
{
    TempDirectory directory;
    const auto path = directory.path / "replay_history";
    ReplayHistoryStore store;
    ASSERT_TRUE(store.configure(path.string(), 10));
    ASSERT_TRUE(store.append(makeRequest(1), 1000));

    const auto temporary = path / "queries.bin.tmp";
    {
        std::ofstream output(temporary);
        output << "stale compaction";
    }
    ASSERT_TRUE(std::filesystem::exists(temporary));
    ASSERT_TRUE(store.configure(path.string(), 10));
    EXPECT_FALSE(std::filesystem::exists(temporary));
    std::vector<ReplayHistorySample> samples;
    std::string error;
    ASSERT_TRUE(store.loadRecent(1, samples, error)) << error;
    ASSERT_EQ(1u, samples.size());
    EXPECT_EQ(1u, samples[0].request.requestId);
}

TEST_F(ReplayControlRTest, testNormalQueriesDoNotPushHistoryUpdates)
{
    ReplayClientR resource;
    resource.setClientId("client-a");
    auto registrationPipe = std::make_shared<CaptureAsyncPipe>();
    ASSERT_TRUE(resource.subscribeHistory(registrationPipe));
    ASSERT_EQ(1u, registrationPipe->dataItems.size());

    ASSERT_TRUE(resource.recordNormalQuery(makeRequest(1), 1000));
    ASSERT_TRUE(resource.recordNormalQuery(makeRequest(2), 2000));
    EXPECT_EQ(1u, registrationPipe->dataItems.size());
}

TEST_F(ReplayControlRTest, testAsyncHistoryQueryAggregatesAllClients)
{
    ReplayControlR resource;
    auto commandA = std::make_shared<CaptureAsyncPipe>();
    auto commandB = std::make_shared<CaptureAsyncPipe>();
    auto queryA = std::make_shared<CaptureAsyncPipe>();
    auto queryB = std::make_shared<CaptureAsyncPipe>();
    ASSERT_TRUE(resource.subscribe("client-a", commandA, queryA));
    ASSERT_TRUE(resource.subscribe("client-b", commandB, queryB));

    auto responsePipe = std::make_shared<CaptureAsyncPipe>();
    uint64_t queryId = 0;
    ReplayControlResponseDataPtr immediate;
    ASSERT_TRUE(
        resource.processRequestAsync({"query"}, responsePipe, queryId, immediate));
    EXPECT_FALSE(immediate);
    ASSERT_NE(0u, queryId);
    ASSERT_EQ(1u, queryA->dataItems.size());
    ASSERT_EQ(1u, queryB->dataItems.size());

    ReplayHistoryQueryResponseData responseA;
    responseA.queryId = queryId;
    responseA.summary.clientId = "client-a";
    responseA.summary.sampleCount = 2;
    responseA.summary.replayable = true;
    resource.completeHistoryQuery(responseA);
    EXPECT_TRUE(responsePipe->dataItems.empty());

    ReplayHistoryQueryResponseData responseB;
    responseB.queryId = queryId;
    responseB.summary.clientId = "client-b";
    responseB.summary.sampleCount = 3;
    responseB.summary.replayable = true;
    resource.completeHistoryQuery(responseB);
    ASSERT_EQ(1u, responsePipe->dataItems.size());
    auto result = std::dynamic_pointer_cast<ReplayControlResponseData>(
        responsePipe->dataItems.front());
    ASSERT_TRUE(result);
    EXPECT_TRUE(result->accepted);
    EXPECT_EQ(2u, result->expectedClientCount);
    EXPECT_EQ(2u, result->respondedClientCount);
    ASSERT_EQ(2u, result->clients.size());
    EXPECT_EQ("ok", result->clients[0].status);
    EXPECT_EQ("ok", result->clients[1].status);
}

TEST_F(ReplayControlRTest, testAsyncHistoryQueryReportsOfflineAndTimeout)
{
    ReplayControlR resource;
    auto commandA = std::make_shared<CaptureAsyncPipe>();
    auto commandB = std::make_shared<CaptureAsyncPipe>();
    auto commandC = std::make_shared<CaptureAsyncPipe>();
    auto queryA = std::make_shared<CaptureAsyncPipe>();
    auto queryB = std::make_shared<CaptureAsyncPipe>();
    auto queryC = std::make_shared<CaptureAsyncPipe>();
    ASSERT_TRUE(resource.subscribe("client-a", commandA, queryA));
    ASSERT_TRUE(resource.subscribe("client-b", commandB, queryB));
    ASSERT_TRUE(resource.subscribe("client-c", commandC, queryC));

    auto responsePipe = std::make_shared<CaptureAsyncPipe>();
    uint64_t queryId = 0;
    ReplayControlResponseDataPtr immediate;
    ASSERT_TRUE(
        resource.processRequestAsync({"query"}, responsePipe, queryId, immediate));
    ReplayHistoryQueryResponseData responseA;
    responseA.queryId = queryId;
    responseA.summary.clientId = "client-a";
    responseA.summary.sampleCount = 1;
    responseA.summary.replayable = true;
    resource.completeHistoryQuery(responseA);
    resource.unsubscribe("client-b", commandB, queryB);
    EXPECT_TRUE(responsePipe->dataItems.empty());
    resource.timeoutHistoryQuery(queryId);

    ASSERT_EQ(1u, responsePipe->dataItems.size());
    auto result = std::dynamic_pointer_cast<ReplayControlResponseData>(
        responsePipe->dataItems.front());
    ASSERT_TRUE(result);
    EXPECT_FALSE(result->accepted);
    EXPECT_EQ(3u, result->expectedClientCount);
    EXPECT_EQ(1u, result->respondedClientCount);
    ASSERT_EQ(3u, result->clients.size());
    EXPECT_EQ("ok", result->clients[0].status);
    EXPECT_EQ("offline", result->clients[1].status);
    EXPECT_EQ("timeout", result->clients[2].status);
}

TEST_F(ReplayControlRTest, testRecentCountSelectsOnlyCapableClients)
{
    ReplayControlR resource;
    auto clientA = std::make_shared<CaptureAsyncPipe>();
    auto clientB = std::make_shared<CaptureAsyncPipe>();
    ASSERT_TRUE(resource.subscribe("client-a", clientA));
    ASSERT_TRUE(resource.subscribe("client-b", clientB));
    ReplayHistorySummary summary;
    summary.clientId = "client-a";
    summary.sampleCount = 4;
    summary.replayable = true;
    resource.updateHistory(summary);
    summary.clientId = "client-b";
    summary.sampleCount = 2;
    resource.updateHistory(summary);

    ReplayControlR::Request request;
    request.action = "start";
    request.mode = "fixed_qps";
    request.historyMode = "recent_count";
    request.recentQueryCount = 3;
    request.loopCount = 1;
    request.totalQps = 40.0;
    ReplayControlResponseDataPtr response;
    ASSERT_TRUE(resource.processRequest(request, response));
    ASSERT_TRUE(response->accepted);
    EXPECT_EQ(1u, response->eligibleClientCount);
    EXPECT_DOUBLE_EQ(40.0, response->perClientQps);
    auto command =
        std::dynamic_pointer_cast<ReplayControlCommandData>(clientA->dataItems.back());
    ASSERT_TRUE(command);
    EXPECT_EQ(ReplayAction::START, command->action);
    EXPECT_EQ(ReplayHistoryMode::RECENT_COUNT, command->historyMode);
    EXPECT_EQ(3u, command->recentQueryCount);
    command =
        std::dynamic_pointer_cast<ReplayControlCommandData>(clientB->dataItems.back());
    ASSERT_TRUE(command);
    EXPECT_EQ(ReplayAction::STOP, command->action);
}

TEST_F(ReplayControlRTest, testClientJoinAndLeaveReallocateFixedQps)
{
    ReplayControlR resource;
    auto commandA = std::make_shared<CaptureAsyncPipe>();
    auto queryA = std::make_shared<CaptureAsyncPipe>();
    ASSERT_TRUE(resource.subscribe("client-a", commandA, queryA));
    ReplayHistorySummary summaryA;
    summaryA.clientId = "client-a";
    summaryA.earliestEventTimeUs = 100;
    summaryA.latestEventTimeUs = 300;
    summaryA.sampleCount = 3;
    summaryA.replayable = true;
    resource.updateHistory(summaryA);

    ReplayControlR::Request request;
    request.action = "start";
    request.mode = "fixed_qps";
    request.startTimeUs = 100;
    request.endTimeUs = 200;
    request.loopCount = 10;
    request.totalQps = 100.0;
    ReplayControlResponseDataPtr response;
    ASSERT_TRUE(resource.processRequest(request, response));
    ASSERT_TRUE(response->accepted);
    EXPECT_DOUBLE_EQ(100.0, response->perClientQps);
    auto command =
        std::dynamic_pointer_cast<ReplayControlCommandData>(commandA->dataItems.back());
    ASSERT_TRUE(command);
    EXPECT_EQ(0u, command->pacingIndex);
    EXPECT_EQ(1u, command->pacingClientCount);
    ASSERT_GT(command->pacingEpochUs, 0);
    const auto initialPacingEpochUs = command->pacingEpochUs;

    auto commandB = std::make_shared<CaptureAsyncPipe>();
    auto queryB = std::make_shared<CaptureAsyncPipe>();
    ASSERT_TRUE(resource.subscribe("client-b", commandB, queryB));
    ASSERT_EQ(1u, queryB->dataItems.size());
    auto joinQuery =
        std::dynamic_pointer_cast<ReplayHistoryQueryData>(queryB->dataItems.front());
    ASSERT_TRUE(joinQuery);

    ReplayHistoryQueryResponseData responseB;
    responseB.queryId = joinQuery->queryId;
    responseB.summary = summaryA;
    responseB.summary.clientId = "client-b";
    responseB.summary.reportTimeUs =
        autil::TimeUtility::currentTimeInMicroSeconds() + 4'000'000;
    resource.completeHistoryQuery(responseB);
    command =
        std::dynamic_pointer_cast<ReplayControlCommandData>(commandA->dataItems.back());
    ASSERT_TRUE(command);
    EXPECT_DOUBLE_EQ(50.0, command->perClientQps);
    EXPECT_EQ(0u, command->pacingIndex);
    EXPECT_EQ(2u, command->pacingClientCount);
    EXPECT_GE(command->pacingEpochUs, initialPacingEpochUs);
    const auto joinedPacingEpochUs = command->pacingEpochUs;
    command =
        std::dynamic_pointer_cast<ReplayControlCommandData>(commandB->dataItems.back());
    ASSERT_TRUE(command);
    EXPECT_EQ(ReplayAction::START, command->action);
    EXPECT_DOUBLE_EQ(50.0, command->perClientQps);
    EXPECT_EQ(1u, command->pacingIndex);
    EXPECT_EQ(2u, command->pacingClientCount);
    EXPECT_EQ(joinedPacingEpochUs, command->pacingEpochUs);
    EXPECT_GT(command->pacingClockOffsetUs, 3'500'000);
    EXPECT_LE(command->pacingClockOffsetUs, 4'000'000);

    resource.unsubscribe("client-b", commandB, queryB);
    command =
        std::dynamic_pointer_cast<ReplayControlCommandData>(commandA->dataItems.back());
    ASSERT_TRUE(command);
    EXPECT_DOUBLE_EQ(100.0, command->perClientQps);
    EXPECT_EQ(0u, command->pacingIndex);
    EXPECT_EQ(1u, command->pacingClientCount);
    EXPECT_GE(command->pacingEpochUs, joinedPacingEpochUs);
}

TEST_F(ReplayControlRTest, testRestartedClientJoinsActiveReplayAfterRestoringHistory)
{
    ReplayControlR master;
    auto commandA = std::make_shared<CaptureAsyncPipe>();
    auto queryA = std::make_shared<CaptureAsyncPipe>();
    ASSERT_TRUE(master.subscribe("client-a", commandA, queryA));

    ReplayHistorySummary summaryA;
    summaryA.clientId = "client-a";
    summaryA.earliestEventTimeUs = 100;
    summaryA.latestEventTimeUs = 500;
    summaryA.sampleCount = 5;
    summaryA.replayable = true;
    master.updateHistory(summaryA);

    ReplayControlR::Request start;
    start.action = "start";
    start.mode = "fixed_qps";
    start.historyMode = "recent_count";
    start.recentQueryCount = 5;
    start.durationMs = 30'000;
    start.totalQps = 100.0;
    ReplayControlResponseDataPtr response;
    ASSERT_TRUE(master.processRequest(start, response));
    ASSERT_TRUE(response->accepted) << response->error;
    ASSERT_EQ(1u, response->eligibleClientCount);
    EXPECT_DOUBLE_EQ(100.0, response->perClientQps);

    TempDirectory directory;
    {
        ReplayClientR previousClient;
        previousClient.setClientId("client-b");
        ASSERT_TRUE(previousClient.configureHistory(directory.path.string(), 10));
        for (uint64_t requestId = 1; requestId <= 5; ++requestId) {
            ASSERT_TRUE(previousClient.recordNormalQuery(makeRequest(requestId),
                                                         100 * requestId));
        }
    }

    ReplayClientR restartedClient;
    restartedClient.setClientId("client-b");
    ASSERT_TRUE(restartedClient.configureHistory(directory.path.string(), 10));
    restartedClient.setScheduleCallback([](const auto &) {});
    ASSERT_EQ(5u, restartedClient.getSummary().sampleCount);

    auto commandB = std::make_shared<CaptureAsyncPipe>();
    auto queryB = std::make_shared<CaptureAsyncPipe>();
    ASSERT_TRUE(master.subscribe("client-b", commandB, queryB));
    ASSERT_EQ(1u, commandB->dataItems.size());
    auto command =
        std::dynamic_pointer_cast<ReplayControlCommandData>(commandB->dataItems.back());
    ASSERT_TRUE(command);
    EXPECT_EQ(ReplayAction::STOP, command->action);
    EXPECT_EQ(0u, command->selectionGeneration);
    ASSERT_TRUE(restartedClient.applyCommand(*command));
    EXPECT_EQ(0u, restartedClient.getSummary().selectionGeneration);

    ASSERT_EQ(1u, queryB->dataItems.size());
    const auto historyQuery =
        std::dynamic_pointer_cast<ReplayHistoryQueryData>(queryB->dataItems.back());
    ASSERT_TRUE(historyQuery);
    ReplayHistoryQueryResponseData historyResponse;
    historyResponse.queryId = historyQuery->queryId;
    historyResponse.summary = restartedClient.getSummary();
    master.completeHistoryQuery(historyResponse);

    ASSERT_GE(commandB->dataItems.size(), 2u);
    command =
        std::dynamic_pointer_cast<ReplayControlCommandData>(commandB->dataItems.back());
    ASSERT_TRUE(command);
    EXPECT_EQ(ReplayAction::START, command->action);
    EXPECT_GT(command->selectionGeneration, 0u);
    EXPECT_DOUBLE_EQ(100.0, command->totalQps);
    EXPECT_DOUBLE_EQ(50.0, command->perClientQps);
    EXPECT_EQ(2u, command->pacingClientCount);
    ASSERT_TRUE(restartedClient.applyCommand(*command));

    const auto restartedSummary = restartedClient.getSummary();
    EXPECT_EQ(command->selectionGeneration, restartedSummary.selectionGeneration);
    EXPECT_EQ(5u, restartedSummary.selectedSampleCount);
    master.updateHistory(restartedSummary);
    ReplayControlR::Request query;
    query.action = "query";
    ASSERT_TRUE(master.processRequest(query, response));
    ASSERT_TRUE(response->accepted);
    EXPECT_EQ(2u, response->eligibleClientCount);
    EXPECT_DOUBLE_EQ(50.0, response->perClientQps);
    restartedClient.stopReplay();
}

TEST_F(ReplayControlRTest, testQpsOnlyUpdatesPreserveReplayDefinition)
{
    ReplayControlR resource;
    auto commandA = std::make_shared<CaptureAsyncPipe>();
    auto queryA = std::make_shared<CaptureAsyncPipe>();
    auto commandB = std::make_shared<CaptureAsyncPipe>();
    auto queryB = std::make_shared<CaptureAsyncPipe>();
    ASSERT_TRUE(resource.subscribe("client-a", commandA, queryA));
    ASSERT_TRUE(resource.subscribe("client-b", commandB, queryB));

    for (const auto &clientId : {"client-a", "client-b"}) {
        ReplayHistorySummary summary;
        summary.clientId = clientId;
        summary.earliestEventTimeUs = 100;
        summary.latestEventTimeUs = 300;
        summary.sampleCount = 3;
        summary.replayable = true;
        resource.updateHistory(summary);
    }

    ReplayControlR::Request start;
    start.action = "start";
    start.mode = "fixed_qps";
    start.startTimeUs = 100;
    start.endTimeUs = 200;
    start.durationMs = 30000;
    start.loopCount = 0;
    start.totalQps = 100.0;
    ReplayControlResponseDataPtr response;
    ASSERT_TRUE(resource.processRequest(start, response));
    ASSERT_TRUE(response->accepted);
    const auto initialGeneration = response->generation;
    auto initialCommand =
        std::dynamic_pointer_cast<ReplayControlCommandData>(commandA->dataItems.back());
    ASSERT_TRUE(initialCommand);
    auto previousPacingEpochUs = initialCommand->pacingEpochUs;
    auto previousTotalQps = start.totalQps;

    ReplayControlR::Request update;
    update.action = "update";
    for (const auto totalQps : {20.0, 200.0, 5.0, 80.0}) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        update.totalQps = totalQps;
        const auto beforeUpdateUs = autil::TimeUtility::currentTimeInMicroSeconds();
        ASSERT_TRUE(resource.processRequest(update, response));
        const auto afterUpdateUs = autil::TimeUtility::currentTimeInMicroSeconds();
        ASSERT_TRUE(response->accepted) << response->error;
        EXPECT_DOUBLE_EQ(totalQps / 2.0, response->perClientQps);

        auto command = std::dynamic_pointer_cast<ReplayControlCommandData>(
            commandA->dataItems.back());
        ASSERT_TRUE(command);
        EXPECT_EQ(ReplayAction::START, command->action);
        EXPECT_EQ(ReplayMode::FIXED_QPS, command->mode);
        EXPECT_EQ(ReplayHistoryMode::TIME_RANGE, command->historyMode);
        EXPECT_EQ(100, command->startTimeUs);
        EXPECT_EQ(200, command->endTimeUs);
        EXPECT_GT(command->durationMs, 0);
        EXPECT_LE(command->durationMs, 30000);
        EXPECT_EQ(0u, command->loopCount);
        EXPECT_DOUBLE_EQ(totalQps, command->totalQps);
        EXPECT_DOUBLE_EQ(totalQps / 2.0, command->perClientQps);
        EXPECT_EQ(0u, command->pacingIndex);
        EXPECT_EQ(2u, command->pacingClientCount);
        const auto qpsRatio = static_cast<long double>(previousTotalQps) /
                              static_cast<long double>(totalQps);
        const auto inferredUpdateUs =
            (static_cast<long double>(command->pacingEpochUs) -
             qpsRatio * static_cast<long double>(previousPacingEpochUs)) /
            (1.0L - qpsRatio);
        EXPECT_GE(inferredUpdateUs, static_cast<long double>(beforeUpdateUs - 2));
        EXPECT_LE(inferredUpdateUs, static_cast<long double>(afterUpdateUs + 2));
        previousPacingEpochUs = command->pacingEpochUs;
        previousTotalQps = totalQps;
    }
    EXPECT_EQ(initialGeneration + 4, response->generation);

    update.historyMode = "recent_count";
    update.recentQueryCount = 1;
    ASSERT_TRUE(resource.processRequest(update, response));
    EXPECT_FALSE(response->accepted);
    EXPECT_NE(std::string::npos, response->error.find("cannot change"));
}

TEST_F(ReplayControlRTest, testExtremeQpsUpdateKeepsRepresentableEpoch)
{
    ReplayControlR resource;
    auto commandPipe = std::make_shared<CaptureAsyncPipe>();
    ASSERT_TRUE(resource.subscribe("client-a", commandPipe));

    ReplayHistorySummary summary;
    summary.clientId = "client-a";
    summary.earliestEventTimeUs = 100;
    summary.latestEventTimeUs = 300;
    summary.sampleCount = 3;
    summary.replayable = true;
    resource.updateHistory(summary);

    ReplayControlR::Request request;
    request.action = "start";
    request.mode = "fixed_qps";
    request.startTimeUs = 100;
    request.endTimeUs = 200;
    request.durationMs = 30000;
    request.totalQps = 100.0;
    ReplayControlResponseDataPtr response;
    ASSERT_TRUE(resource.processRequest(request, response));
    ASSERT_TRUE(response->accepted);

    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    request = {};
    request.action = "update";
    request.totalQps = std::numeric_limits<double>::min();
    const auto beforeUpdateUs = autil::TimeUtility::currentTimeInMicroSeconds();
    ASSERT_TRUE(resource.processRequest(request, response));
    const auto afterUpdateUs = autil::TimeUtility::currentTimeInMicroSeconds();
    ASSERT_TRUE(response->accepted) << response->error;

    auto command = std::dynamic_pointer_cast<ReplayControlCommandData>(
        commandPipe->dataItems.back());
    ASSERT_TRUE(command);
    EXPECT_GE(command->pacingEpochUs, beforeUpdateUs);
    EXPECT_LE(command->pacingEpochUs, afterUpdateUs);
    EXPECT_GT(command->pacingEpochUs, 0);
}

TEST_F(ReplayControlRTest, testDurationExpiryStopsMasterAndLateClient)
{
    ReplayControlR resource;
    auto commandA = std::make_shared<CaptureAsyncPipe>();
    auto queryA = std::make_shared<CaptureAsyncPipe>();
    ASSERT_TRUE(resource.subscribe("client-a", commandA, queryA));

    ReplayHistorySummary summary;
    summary.clientId = "client-a";
    summary.earliestEventTimeUs = 100;
    summary.latestEventTimeUs = 300;
    summary.sampleCount = 3;
    summary.replayable = true;
    resource.updateHistory(summary);

    ReplayControlR::Request start;
    start.action = "start";
    start.mode = "fixed_qps";
    start.startTimeUs = 100;
    start.endTimeUs = 200;
    start.durationMs = 10;
    start.loopCount = 0;
    start.totalQps = 100.0;
    ReplayControlResponseDataPtr response;
    ASSERT_TRUE(resource.processRequest(start, response));
    ASSERT_TRUE(response->accepted);
    const auto startGeneration = response->generation;

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    ReplayControlR::Request query;
    query.action = "query";
    ASSERT_TRUE(resource.processRequest(query, response));
    ASSERT_TRUE(response->accepted);
    EXPECT_EQ(ReplayAction::STOP, response->action);
    EXPECT_EQ(startGeneration + 1, response->generation);
    EXPECT_EQ(0u, response->eligibleClientCount);
    EXPECT_DOUBLE_EQ(0.0, response->perClientQps);
    ASSERT_FALSE(commandA->dataItems.empty());
    auto command =
        std::dynamic_pointer_cast<ReplayControlCommandData>(commandA->dataItems.back());
    ASSERT_TRUE(command);
    EXPECT_EQ(ReplayAction::STOP, command->action);
    EXPECT_EQ(0u, command->selectionGeneration);

    auto commandB = std::make_shared<CaptureAsyncPipe>();
    auto queryB = std::make_shared<CaptureAsyncPipe>();
    ASSERT_TRUE(resource.subscribe("client-b", commandB, queryB));
    ASSERT_EQ(1u, commandB->dataItems.size());
    command =
        std::dynamic_pointer_cast<ReplayControlCommandData>(commandB->dataItems.back());
    ASSERT_TRUE(command);
    EXPECT_EQ(ReplayAction::STOP, command->action);
    EXPECT_TRUE(queryB->dataItems.empty());

    ReplayControlR::Request update;
    update.action = "update";
    update.totalQps = 50.0;
    ASSERT_TRUE(resource.processRequest(update, response));
    EXPECT_FALSE(response->accepted);
    EXPECT_NE(std::string::npos, response->error.find("no active replay"));
}

TEST_F(ReplayControlRTest, testLateClientReceivesOnlyRemainingDuration)
{
    ReplayControlR resource;
    auto commandA = std::make_shared<CaptureAsyncPipe>();
    auto queryA = std::make_shared<CaptureAsyncPipe>();
    ASSERT_TRUE(resource.subscribe("client-a", commandA, queryA));

    ReplayHistorySummary summary;
    summary.clientId = "client-a";
    summary.earliestEventTimeUs = 100;
    summary.latestEventTimeUs = 300;
    summary.sampleCount = 3;
    summary.replayable = true;
    resource.updateHistory(summary);

    ReplayControlR::Request start;
    start.action = "start";
    start.mode = "fixed_qps";
    start.startTimeUs = 100;
    start.endTimeUs = 200;
    start.durationMs = 1000;
    start.loopCount = 0;
    start.totalQps = 100.0;
    ReplayControlResponseDataPtr response;
    ASSERT_TRUE(resource.processRequest(start, response));
    ASSERT_TRUE(response->accepted);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    auto commandB = std::make_shared<CaptureAsyncPipe>();
    auto queryB = std::make_shared<CaptureAsyncPipe>();
    ASSERT_TRUE(resource.subscribe("client-b", commandB, queryB));
    ASSERT_EQ(1u, queryB->dataItems.size());
    auto historyQuery =
        std::dynamic_pointer_cast<ReplayHistoryQueryData>(queryB->dataItems.back());
    ASSERT_TRUE(historyQuery);

    ReplayHistoryQueryResponseData historyResponse;
    historyResponse.queryId = historyQuery->queryId;
    historyResponse.summary = summary;
    historyResponse.summary.clientId = "client-b";
    resource.completeHistoryQuery(historyResponse);

    ASSERT_GE(commandB->dataItems.size(), 2u);
    auto command =
        std::dynamic_pointer_cast<ReplayControlCommandData>(commandB->dataItems.back());
    ASSERT_TRUE(command);
    EXPECT_EQ(ReplayAction::START, command->action);
    EXPECT_GT(command->durationMs, 0);
    EXPECT_LT(command->durationMs, 1000);
    EXPECT_DOUBLE_EQ(50.0, command->perClientQps);
}

TEST_F(ReplayControlRTest, testQpsUpdateReallocatesAfterDeliveryFailure)
{
    ReplayControlR resource;
    auto commandA = std::make_shared<CaptureAsyncPipe>();
    auto commandB = std::make_shared<CaptureAsyncPipe>();
    ASSERT_TRUE(resource.subscribe("client-a", commandA));
    ASSERT_TRUE(resource.subscribe("client-b", commandB));

    for (const auto &clientId : {"client-a", "client-b"}) {
        ReplayHistorySummary summary;
        summary.clientId = clientId;
        summary.earliestEventTimeUs = 100;
        summary.latestEventTimeUs = 300;
        summary.sampleCount = 3;
        summary.replayable = true;
        resource.updateHistory(summary);
    }

    ReplayControlR::Request start;
    start.action = "start";
    start.mode = "fixed_qps";
    start.startTimeUs = 100;
    start.endTimeUs = 200;
    start.loopCount = 10;
    start.totalQps = 100.0;
    ReplayControlResponseDataPtr response;
    ASSERT_TRUE(resource.processRequest(start, response));
    ASSERT_TRUE(response->accepted);
    EXPECT_DOUBLE_EQ(50.0, response->perClientQps);

    commandB->failSetData = true;
    ReplayControlR::Request update;
    update.action = "update";
    update.totalQps = 80.0;
    ASSERT_TRUE(resource.processRequest(update, response));
    ASSERT_TRUE(response->accepted);
    EXPECT_EQ(1u, response->eligibleClientCount);
    EXPECT_DOUBLE_EQ(80.0, response->perClientQps);

    auto command =
        std::dynamic_pointer_cast<ReplayControlCommandData>(commandA->dataItems.back());
    ASSERT_TRUE(command);
    EXPECT_EQ(ReplayAction::START, command->action);
    EXPECT_DOUBLE_EQ(80.0, command->perClientQps);
}

TEST_F(ReplayControlRTest, testFixedQpsUpdateKeepsReplayCursor)
{
    ReplayClientR resource;
    resource.setClientId("client-a");
    const auto now = autil::TimeUtility::currentTimeInMicroSeconds();
    for (uint64_t requestId = 1; requestId <= 3; ++requestId) {
        auto request = makeRequest(requestId);
        request.promptTokenCount = requestId;
        ASSERT_TRUE(resource.recordNormalQuery(request, now + requestId - 1));
    }

    std::mutex callbackMutex;
    std::vector<uint64_t> promptTokenCounts;
    resource.setScheduleCallback([&](const auto &request) {
        std::lock_guard<std::mutex> lock(callbackMutex);
        promptTokenCounts.push_back(request.promptTokenCount);
    });

    auto first = makeStart(ReplayMode::FIXED_QPS, now, now + 2, 10, 0, 20.0);
    first->generation = 1;
    ASSERT_TRUE(resource.applyCommand(*first));
    for (int i = 0; i < 200; ++i) {
        {
            std::lock_guard<std::mutex> lock(callbackMutex);
            if (promptTokenCounts.size() >= 2) {
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    auto updated = makeStart(ReplayMode::FIXED_QPS, now, now + 2, 10, 0, 100.0);
    updated->generation = 2;
    updated->durationMs = 5000;
    ASSERT_TRUE(resource.applyCommand(*updated));
    for (int i = 0; i < 200; ++i) {
        {
            std::lock_guard<std::mutex> lock(callbackMutex);
            if (promptTokenCounts.size() >= 3) {
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    auto updatedAgain = makeStart(ReplayMode::FIXED_QPS, now, now + 2, 10, 0, 40.0);
    updatedAgain->generation = 3;
    ASSERT_TRUE(resource.applyCommand(*updatedAgain));
    for (int i = 0; i < 200; ++i) {
        {
            std::lock_guard<std::mutex> lock(callbackMutex);
            if (promptTokenCounts.size() >= 5) {
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    resource.stopReplay();

    std::lock_guard<std::mutex> lock(callbackMutex);
    ASSERT_GE(promptTokenCounts.size(), 5u);
    EXPECT_EQ(1u, promptTokenCounts[0]);
    EXPECT_EQ(2u, promptTokenCounts[1]);
    EXPECT_EQ(3u, promptTokenCounts[2]);
    EXPECT_EQ(1u, promptTokenCounts[3]);
    EXPECT_EQ(2u, promptTokenCounts[4]);
}

TEST_F(ReplayControlRTest, testCompletedSelectionIsNotRestartedByReallocation)
{
    for (const auto mode : {ReplayMode::FIXED_QPS, ReplayMode::REALTIME}) {
        SCOPED_TRACE(mode == ReplayMode::FIXED_QPS ? "fixed_qps" : "realtime");
        ReplayClientR resource;
        resource.setClientId("client-a");
        const auto now = autil::TimeUtility::currentTimeInMicroSeconds();
        ASSERT_TRUE(resource.recordNormalQuery(makeRequest(1), now));
        std::atomic<int> callbackCount{0};
        resource.setScheduleCallback([&callbackCount](const auto &) {
            callbackCount.fetch_add(1, std::memory_order_relaxed);
        });

        auto command = makeStart(mode, now, now + 1, 1, 0,
                                 mode == ReplayMode::FIXED_QPS ? 1000.0 : 0.0);
        command->generation = 1;
        command->selectionGeneration = 7;
        ASSERT_TRUE(resource.applyCommand(*command));
        for (int i = 0; i < 200 && resource.replaying(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        ASSERT_FALSE(resource.replaying());
        ASSERT_EQ(1, callbackCount.load(std::memory_order_relaxed));

        command->generation = 2;
        command->perClientQps = mode == ReplayMode::FIXED_QPS ? 2000.0 : 0.0;
        ASSERT_TRUE(resource.applyCommand(*command));
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        EXPECT_FALSE(resource.replaying());
        EXPECT_EQ(1, callbackCount.load(std::memory_order_relaxed));
        resource.stopReplay();
    }
}

TEST_F(ReplayControlRTest, testStopCallbackRunsAfterReplayThreadStops)
{
    std::atomic<int> stopCount{0};
    ReplayClientR resource;
    resource.setClientId("client-a");
    const auto now = autil::TimeUtility::currentTimeInMicroSeconds();
    ASSERT_TRUE(resource.recordNormalQuery(makeRequest(1), now));
    resource.setScheduleCallback([](const auto &) {});
    auto start = makeStart(ReplayMode::FIXED_QPS, now, now + 1, 0, 5000, 1.0);
    start->generation = 1;
    ASSERT_TRUE(resource.applyCommand(*start));
    resource.setStopCallback([&stopCount]() { ++stopCount; });
    ReplayControlCommandData stop;
    stop.action = ReplayAction::STOP;
    stop.generation = 2;
    ASSERT_TRUE(resource.applyCommand(stop));
    EXPECT_EQ(1, stopCount.load());
    EXPECT_FALSE(resource.replaying());

    resource.stopReplay();
    EXPECT_EQ(2, stopCount.load());
}

TEST_F(ReplayControlRTest, testClientPublishesNaturalCompletion)
{
    ReplayClientR resource;
    resource.setClientId("client-a");
    auto capabilityPipe = std::make_shared<CaptureAsyncPipe>();
    ASSERT_TRUE(resource.subscribeHistory(capabilityPipe));
    const auto now = autil::TimeUtility::currentTimeInMicroSeconds();
    ASSERT_TRUE(resource.recordNormalQuery(makeRequest(1), now));
    resource.setScheduleCallback([](const auto &) {});

    auto command = makeStart(ReplayMode::FIXED_QPS, now, now + 1, 1, 0, 1000.0);
    command->generation = 1;
    command->selectionGeneration = 17;
    ASSERT_TRUE(resource.applyCommand(*command));
    for (int i = 0; i < 200 && resource.replaying(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    resource.stopReplay();

    ASSERT_GE(capabilityPipe->dataItems.size(), 3u);
    const auto capability = std::dynamic_pointer_cast<ReplayClientCapabilityData>(
        capabilityPipe->dataItems.back());
    ASSERT_TRUE(capability);
    EXPECT_EQ(17u, capability->summary.selectionGeneration);
    EXPECT_EQ(1u, capability->summary.selectedSampleCount);
    EXPECT_EQ(17u, capability->summary.completedSelectionGeneration);
}

TEST_F(ReplayControlRTest, testContinuousQpsUpdatesDoNotPauseSending)
{
    ReplayClientR resource;
    resource.setClientId("client-a");
    const auto now = autil::TimeUtility::currentTimeInMicroSeconds();
    for (uint64_t requestId = 1; requestId <= 3; ++requestId) {
        auto request = makeRequest(requestId);
        request.promptTokenCount = requestId;
        ASSERT_TRUE(resource.recordNormalQuery(request, now + requestId - 1));
    }

    std::atomic<uint64_t> sentCount = 0;
    resource.setScheduleCallback(
        [&](const auto &) { sentCount.fetch_add(1, std::memory_order_relaxed); });
    auto command = makeStart(ReplayMode::FIXED_QPS, now, now + 2, 0, 2000, 100.0);
    command->generation = 1;
    ASSERT_TRUE(resource.applyCommand(*command));

    const auto beforeUpdates = sentCount.load(std::memory_order_relaxed);
    for (uint64_t generation = 2; generation <= 31; ++generation) {
        command->generation = generation;
        command->totalQps = generation % 2 == 0 ? 40.0 : 200.0;
        command->perClientQps = command->totalQps;
        ASSERT_TRUE(resource.applyCommand(*command));
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    const auto afterUpdates = sentCount.load(std::memory_order_relaxed);
    resource.stopReplay();

    EXPECT_GT(afterUpdates, beforeUpdates)
        << "continuous QPS updates postponed every scheduled send";
    EXPECT_EQ(3u, resource.selectSamples(now, now + 2).size());
}

TEST_F(ReplayControlRTest, testQpsOnlyUpdateDoesNotSendOldDeadline)
{
    ReplayClientR resource;
    resource.setClientId("client-a");
    const auto historyTimeUs = autil::TimeUtility::currentTimeInMicroSeconds();
    ASSERT_TRUE(resource.recordNormalQuery(makeRequest(1), historyTimeUs));

    std::chrono::steady_clock::time_point callbackTime;
    resource.setScheduleCallback(
        [&](const auto &) { callbackTime = std::chrono::steady_clock::now(); });
    auto command =
        makeStart(ReplayMode::FIXED_QPS, historyTimeUs, historyTimeUs + 1, 1, 0, 10.0);
    command->generation = 1;
    command->totalQps = 10.0;
    command->perClientQps = 10.0;
    command->pacingIndex = 0;
    command->pacingClientCount = 1;
    command->pacingEpochUs = autil::TimeUtility::currentTimeInMicroSeconds() + 50'000;
    const auto started = std::chrono::steady_clock::now();
    ASSERT_TRUE(resource.applyCommand(*command));

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    command->generation = 2;
    command->totalQps = 1.0;
    command->perClientQps = 1.0;
    ASSERT_TRUE(resource.applyCommand(*command));

    for (int i = 0; i < 200 && resource.replaying(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    resource.stopReplay();

    ASSERT_NE(std::chrono::steady_clock::time_point(), callbackTime);
    const auto elapsedMs =
        std::chrono::duration_cast<std::chrono::milliseconds>(callbackTime - started)
            .count();
    EXPECT_GE(elapsedMs, 450)
        << "QPS update emitted the deadline calculated with the old rate";
    EXPECT_LT(elapsedMs, 800);
}

TEST_F(ReplayControlRTest, testContinuousQpsProfilesTrackTargetIntegral)
{
    struct ProfileResult {
        double expectedCount = 0.0;
        uint64_t actualCount = 0;
        int64_t maxGapMs = 0;
    };
    const auto runProfile = [](const std::vector<double> &qpsValues,
                               ProfileResult &result) {
        ReplayControlR master;
        auto commandPipe = std::make_shared<CaptureAsyncPipe>();
        ASSERT_TRUE(master.subscribe("client-a", commandPipe));

        ReplayHistorySummary summary;
        summary.clientId = "client-a";
        summary.earliestEventTimeUs = 100;
        summary.latestEventTimeUs = 200;
        summary.sampleCount = 1;
        summary.replayable = true;
        master.updateHistory(summary);

        ReplayClientR client;
        client.setClientId("client-a");
        ASSERT_TRUE(client.recordNormalQuery(makeRequest(1), 150));
        std::mutex callbackMutex;
        std::vector<std::chrono::steady_clock::time_point> callbackTimes;
        client.setScheduleCallback([&](const auto &) {
            std::lock_guard<std::mutex> lock(callbackMutex);
            callbackTimes.emplace_back(std::chrono::steady_clock::now());
        });

        ReplayControlR::Request request;
        request.action = "start";
        request.mode = "fixed_qps";
        request.startTimeUs = 100;
        request.endTimeUs = 200;
        request.durationMs = 5000;
        request.totalQps = qpsValues.front();
        ReplayControlResponseDataPtr response;
        ASSERT_TRUE(master.processRequest(request, response));
        ASSERT_TRUE(response->accepted) << response->error;
        auto applyLatestCommand = [&]() {
            auto command = std::dynamic_pointer_cast<ReplayControlCommandData>(
                commandPipe->dataItems.back());
            ASSERT_TRUE(command);
            ASSERT_TRUE(client.applyCommand(*command));
        };
        applyLatestCommand();

        double expectedCount = 0.0;
        auto activeQps = qpsValues.front();
        auto segmentStart = std::chrono::steady_clock::now();
        request = {};
        request.action = "update";
        for (size_t index = 1; index < qpsValues.size(); ++index) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            const auto segmentEnd = std::chrono::steady_clock::now();
            expectedCount +=
                activeQps *
                std::chrono::duration<double>(segmentEnd - segmentStart).count();
            request.totalQps = qpsValues[index];
            ASSERT_TRUE(master.processRequest(request, response));
            ASSERT_TRUE(response->accepted) << response->error;
            applyLatestCommand();
            activeQps = qpsValues[index];
            segmentStart = std::chrono::steady_clock::now();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        const auto stoppedAt = std::chrono::steady_clock::now();
        expectedCount +=
            activeQps * std::chrono::duration<double>(stoppedAt - segmentStart).count();
        client.stopReplay();

        request = {};
        request.action = "stop";
        ASSERT_TRUE(master.processRequest(request, response));
        ASSERT_TRUE(response->accepted);

        result.expectedCount = expectedCount;
        {
            std::lock_guard<std::mutex> lock(callbackMutex);
            result.actualCount = callbackTimes.size();
            for (size_t index = 1; index < callbackTimes.size(); ++index) {
                result.maxGapMs = std::max<int64_t>(
                    result.maxGapMs,
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        callbackTimes[index] - callbackTimes[index - 1])
                        .count());
            }
        }
    };

    std::vector<double> rising;
    std::vector<double> falling;
    std::vector<double> sine;
    constexpr size_t STEP_COUNT = 40;
    constexpr long double PI = 3.14159265358979323846L;
    for (size_t index = 0; index < STEP_COUNT; ++index) {
        const auto fraction = static_cast<double>(index) / (STEP_COUNT - 1);
        rising.emplace_back(10.0 + 90.0 * fraction);
        falling.emplace_back(100.0 - 90.0 * fraction);
        sine.emplace_back(55.0 + 45.0 * std::sin(2.0L * PI * index / STEP_COUNT));
    }

    for (const auto &profile : {rising, falling, sine}) {
        ProfileResult result;
        runProfile(profile, result);
        const auto tolerance = std::max(4.0, result.expectedCount * 0.08);
        EXPECT_NEAR(result.expectedCount, static_cast<double>(result.actualCount),
                    tolerance);
        EXPECT_LT(result.maxGapMs, 250)
            << "continuous QPS updates paused fixed-QPS replay";
    }
}

TEST_F(ReplayControlRTest, testTinyFixedQpsCanBeStoppedWithoutOverflow)
{
    ReplayClientR resource;
    const auto now = autil::TimeUtility::currentTimeInMicroSeconds();
    ASSERT_TRUE(resource.recordNormalQuery(makeRequest(1), now));
    std::atomic<uint64_t> sentCount = 0;
    resource.setScheduleCallback(
        [&](const auto &) { sentCount.fetch_add(1, std::memory_order_relaxed); });

    auto command = makeStart(ReplayMode::FIXED_QPS, now, now + 1, 0, 0,
                             std::numeric_limits<double>::min());
    command->generation = 1;
    command->selectionGeneration = 1;
    command->totalQps = std::numeric_limits<double>::min();
    command->pacingClientCount = 1;
    command->pacingEpochUs = now;
    ASSERT_TRUE(resource.applyCommand(*command));
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    const auto stopStarted = std::chrono::steady_clock::now();
    resource.stopReplay();
    const auto stopElapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                                   std::chrono::steady_clock::now() - stopStarted)
                                   .count();
    EXPECT_LT(stopElapsedMs, 100);
    EXPECT_EQ(0u, sentCount.load(std::memory_order_relaxed));
}

TEST_F(ReplayControlRTest, testQpsOnlyUpdateRephasesFollowingSendsAcrossClients)
{
    ReplayClientR first;
    ReplayClientR second;
    const auto historyTimeUs = autil::TimeUtility::currentTimeInMicroSeconds();
    ASSERT_TRUE(first.recordNormalQuery(makeRequest(1), historyTimeUs));
    ASSERT_TRUE(second.recordNormalQuery(makeRequest(2), historyTimeUs));

    std::mutex callbackMutex;
    std::vector<std::chrono::steady_clock::time_point> callbackTimes;
    const auto callback = [&](const auto &) {
        std::lock_guard<std::mutex> lock(callbackMutex);
        callbackTimes.emplace_back(std::chrono::steady_clock::now());
    };
    first.setScheduleCallback(callback);
    second.setScheduleCallback(callback);

    auto firstCommand =
        makeStart(ReplayMode::FIXED_QPS, historyTimeUs, historyTimeUs + 1, 0, 2000, 50.0);
    firstCommand->generation = 1;
    firstCommand->selectionGeneration = 29;
    firstCommand->totalQps = 100.0;
    firstCommand->pacingIndex = 0;
    firstCommand->pacingClientCount = 2;
    firstCommand->pacingEpochUs =
        autil::TimeUtility::currentTimeInMicroSeconds() + 100'000;
    auto secondCommand =
        makeStart(ReplayMode::FIXED_QPS, historyTimeUs, historyTimeUs + 1, 0, 2000, 50.0);
    secondCommand->generation = firstCommand->generation;
    secondCommand->selectionGeneration = firstCommand->selectionGeneration;
    secondCommand->totalQps = firstCommand->totalQps;
    secondCommand->pacingIndex = 1;
    secondCommand->pacingClientCount = firstCommand->pacingClientCount;
    secondCommand->pacingEpochUs = firstCommand->pacingEpochUs;

    ASSERT_TRUE(first.applyCommand(*firstCommand));
    ASSERT_TRUE(second.applyCommand(*secondCommand));
    std::this_thread::sleep_for(std::chrono::milliseconds(180));

    const auto updatedAt = std::chrono::steady_clock::now();
    firstCommand->generation = 2;
    firstCommand->totalQps = 10.0;
    firstCommand->perClientQps = 5.0;
    secondCommand->generation = 2;
    secondCommand->totalQps = 10.0;
    secondCommand->perClientQps = 5.0;
    ASSERT_TRUE(first.applyCommand(*firstCommand));
    ASSERT_TRUE(second.applyCommand(*secondCommand));
    std::this_thread::sleep_for(std::chrono::milliseconds(650));
    first.stopReplay();
    second.stopReplay();

    std::vector<std::chrono::steady_clock::time_point> settledCallbacks;
    {
        std::lock_guard<std::mutex> lock(callbackMutex);
        const auto settledAt = updatedAt + std::chrono::milliseconds(150);
        std::copy_if(callbackTimes.begin(), callbackTimes.end(),
                     std::back_inserter(settledCallbacks),
                     [settledAt](const auto &value) { return value >= settledAt; });
    }
    std::sort(settledCallbacks.begin(), settledCallbacks.end());
    ASSERT_GE(settledCallbacks.size(), 4u);
    for (size_t index = 1; index < settledCallbacks.size(); ++index) {
        const auto gapMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                               settledCallbacks[index] - settledCallbacks[index - 1])
                               .count();
        EXPECT_GE(gapMs, 40);
        EXPECT_LT(gapMs, 160)
            << "QPS update left clients clustered on their old pacing phase";
    }
}

TEST_F(ReplayControlRTest, testFixedQpsDoesNotAddScheduleCallbackLatency)
{
    ReplayClientR resource;
    resource.setClientId("client-a");
    const auto now = autil::TimeUtility::currentTimeInMicroSeconds();
    ASSERT_TRUE(resource.recordNormalQuery(makeRequest(1), now));

    std::vector<std::chrono::steady_clock::time_point> callbackTimes;
    resource.setScheduleCallback([&](const auto &) {
        callbackTimes.emplace_back(std::chrono::steady_clock::now());
        std::this_thread::sleep_for(std::chrono::milliseconds(60));
    });
    auto command = makeStart(ReplayMode::FIXED_QPS, now, now + 1, 6, 0, 10.0);
    command->generation = 1;
    ASSERT_TRUE(resource.applyCommand(*command));
    for (int i = 0; i < 400 && resource.replaying(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    resource.stopReplay();

    ASSERT_EQ(6u, callbackTimes.size());
    int64_t totalGapMs = 0;
    for (size_t index = 1; index < callbackTimes.size(); ++index) {
        totalGapMs += std::chrono::duration_cast<std::chrono::milliseconds>(
                          callbackTimes[index] - callbackTimes[index - 1])
                          .count();
    }
    const auto averageGapMs = totalGapMs / static_cast<int64_t>(callbackTimes.size() - 1);
    EXPECT_GE(averageGapMs, 80);
    EXPECT_LT(averageGapMs, 135) << "fixed-QPS pacing included schedule callback latency";
}

TEST_F(ReplayControlRTest, testFixedQpsPacingIndexDelaysFirstSend)
{
    const auto measureFirstSend = [](uint64_t pacingIndex) {
        ReplayClientR resource;
        const auto now = autil::TimeUtility::currentTimeInMicroSeconds();
        EXPECT_TRUE(resource.recordNormalQuery(makeRequest(1), now));
        std::chrono::steady_clock::time_point callbackTime;
        resource.setScheduleCallback([&callbackTime](const auto &) {
            callbackTime = std::chrono::steady_clock::now();
        });
        auto command = makeStart(ReplayMode::FIXED_QPS, now, now + 1, 1, 0, 5.0);
        command->generation = 1;
        command->totalQps = 10.0;
        command->pacingIndex = pacingIndex;
        command->pacingClientCount = 2;
        const auto started = std::chrono::steady_clock::now();
        EXPECT_TRUE(resource.applyCommand(*command));
        for (int i = 0; i < 100 && resource.replaying(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        resource.stopReplay();
        EXPECT_NE(std::chrono::steady_clock::time_point(), callbackTime);
        return std::chrono::duration_cast<std::chrono::milliseconds>(callbackTime -
                                                                     started)
            .count();
    };

    const auto firstClientDelayMs = measureFirstSend(0);
    const auto secondClientDelayMs = measureFirstSend(1);
    EXPECT_GE(firstClientDelayMs, 30);
    EXPECT_LT(firstClientDelayMs, 130);
    EXPECT_GE(secondClientDelayMs, 120);
    EXPECT_LT(secondClientDelayMs, 260);
    EXPECT_GE(secondClientDelayMs - firstClientDelayMs, 60);
}

TEST_F(ReplayControlRTest, testFixedQpsClientsUseSharedPacingEpoch)
{
    ReplayClientR first;
    ReplayClientR second;
    const auto historyTimeUs = autil::TimeUtility::currentTimeInMicroSeconds();
    ASSERT_TRUE(first.recordNormalQuery(makeRequest(1), historyTimeUs));
    ASSERT_TRUE(second.recordNormalQuery(makeRequest(2), historyTimeUs));

    std::chrono::steady_clock::time_point firstCallbackTime;
    std::chrono::steady_clock::time_point secondCallbackTime;
    first.setScheduleCallback(
        [&](const auto &) { firstCallbackTime = std::chrono::steady_clock::now(); });
    second.setScheduleCallback(
        [&](const auto &) { secondCallbackTime = std::chrono::steady_clock::now(); });

    auto firstCommand =
        makeStart(ReplayMode::FIXED_QPS, historyTimeUs, historyTimeUs + 1, 1, 0, 5.0);
    firstCommand->generation = 1;
    firstCommand->totalQps = 10.0;
    firstCommand->pacingIndex = 0;
    firstCommand->pacingClientCount = 2;
    firstCommand->pacingEpochUs =
        autil::TimeUtility::currentTimeInMicroSeconds() + 250000;
    auto secondCommand =
        makeStart(ReplayMode::FIXED_QPS, historyTimeUs, historyTimeUs + 1, 1, 0, 5.0);
    secondCommand->generation = 1;
    secondCommand->totalQps = 10.0;
    secondCommand->pacingIndex = 1;
    secondCommand->pacingClientCount = 2;
    secondCommand->pacingEpochUs = firstCommand->pacingEpochUs;

    ASSERT_TRUE(first.applyCommand(*firstCommand));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    ASSERT_TRUE(second.applyCommand(*secondCommand));
    for (int i = 0; i < 150 && (first.replaying() || second.replaying()); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    first.stopReplay();
    second.stopReplay();

    ASSERT_NE(std::chrono::steady_clock::time_point(), firstCallbackTime);
    ASSERT_NE(std::chrono::steady_clock::time_point(), secondCallbackTime);
    const auto callbackGapMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                                   secondCallbackTime - firstCallbackTime)
                                   .count();
    EXPECT_GE(callbackGapMs, 60);
    EXPECT_LT(callbackGapMs, 140);
}

TEST_F(ReplayControlRTest, testFixedQpsAppliesClientClockOffset)
{
    ReplayClientR resource;
    const auto historyTimeUs = autil::TimeUtility::currentTimeInMicroSeconds();
    ASSERT_TRUE(resource.recordNormalQuery(makeRequest(1), historyTimeUs));

    std::chrono::steady_clock::time_point callbackTime;
    resource.setScheduleCallback(
        [&](const auto &) { callbackTime = std::chrono::steady_clock::now(); });
    auto command =
        makeStart(ReplayMode::FIXED_QPS, historyTimeUs, historyTimeUs + 1, 1, 0, 10.0);
    command->generation = 1;
    command->totalQps = 10.0;
    command->pacingClientCount = 1;
    command->pacingEpochUs = autil::TimeUtility::currentTimeInMicroSeconds() + 200'000;
    command->pacingClockOffsetUs = 200'000;
    const auto appliedAt = std::chrono::steady_clock::now();
    ASSERT_TRUE(resource.applyCommand(*command));
    for (int i = 0; i < 200 && resource.replaying(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    resource.stopReplay();

    ASSERT_NE(std::chrono::steady_clock::time_point(), callbackTime);
    const auto elapsedMs =
        std::chrono::duration_cast<std::chrono::milliseconds>(callbackTime - appliedAt)
            .count();
    EXPECT_GE(elapsedMs, 350);
    EXPECT_LT(elapsedMs, 800);
}

TEST_F(ReplayControlRTest, testRealtimeClientsUseSharedPacingEpoch)
{
    ReplayClientR first;
    ReplayClientR second;
    const auto historyTimeUs = autil::TimeUtility::currentTimeInMicroSeconds();
    ASSERT_TRUE(first.recordNormalQuery(makeRequest(1), historyTimeUs + 100'000));
    ASSERT_TRUE(second.recordNormalQuery(makeRequest(2), historyTimeUs + 100'000));

    std::chrono::steady_clock::time_point firstCallbackTime;
    std::chrono::steady_clock::time_point secondCallbackTime;
    first.setScheduleCallback(
        [&](const auto &) { firstCallbackTime = std::chrono::steady_clock::now(); });
    second.setScheduleCallback(
        [&](const auto &) { secondCallbackTime = std::chrono::steady_clock::now(); });

    auto firstCommand = makeStart(ReplayMode::REALTIME, historyTimeUs,
                                  historyTimeUs + 200'000, 1, 0, 0.0);
    firstCommand->generation = 1;
    firstCommand->pacingEpochUs =
        autil::TimeUtility::currentTimeInMicroSeconds() + 300'000;
    auto secondCommand = makeStart(ReplayMode::REALTIME, historyTimeUs,
                                   historyTimeUs + 200'000, 1, 0, 0.0);
    secondCommand->generation = 1;
    secondCommand->pacingEpochUs = firstCommand->pacingEpochUs;

    ASSERT_TRUE(first.applyCommand(*firstCommand));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    ASSERT_TRUE(second.applyCommand(*secondCommand));
    for (int i = 0; i < 150 && (first.replaying() || second.replaying()); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    first.stopReplay();
    second.stopReplay();

    ASSERT_NE(std::chrono::steady_clock::time_point(), firstCallbackTime);
    ASSERT_NE(std::chrono::steady_clock::time_point(), secondCallbackTime);
    const auto callbackGapMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                                   secondCallbackTime - firstCallbackTime)
                                   .count();
    EXPECT_GT(callbackGapMs, -40);
    EXPECT_LT(callbackGapMs, 40);
}

TEST_F(ReplayControlRTest, testRealtimeLateClientSkipsExpiredEvents)
{
    ReplayClientR resource;
    const auto historyTimeUs = autil::TimeUtility::currentTimeInMicroSeconds();
    ASSERT_TRUE(resource.recordNormalQuery(makeRequest(1), historyTimeUs));
    ASSERT_TRUE(resource.recordNormalQuery(makeRequest(2), historyTimeUs + 300'000));

    std::vector<uint64_t> requestIds;
    resource.setScheduleCallback(
        [&](const auto &request) { requestIds.push_back(request.requestId); });
    auto command = makeStart(ReplayMode::REALTIME, historyTimeUs, historyTimeUs + 400'000,
                             1, 0, 0.0);
    command->generation = 1;
    command->pacingEpochUs = autil::TimeUtility::currentTimeInMicroSeconds() - 200'000;
    ASSERT_TRUE(resource.applyCommand(*command));
    for (int i = 0; i < 100 && resource.replaying(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    resource.stopReplay();

    ASSERT_EQ(1u, requestIds.size());
    EXPECT_EQ(2u, requestIds.front());
}

TEST_F(ReplayControlRTest, testRealtimeDoesNotDropEventsDelayedByCallback)
{
    ReplayClientR resource;
    const auto historyTimeUs = autil::TimeUtility::currentTimeInMicroSeconds();
    ASSERT_TRUE(resource.recordNormalQuery(makeRequest(1), historyTimeUs));
    ASSERT_TRUE(resource.recordNormalQuery(makeRequest(2), historyTimeUs + 1'000));
    ASSERT_TRUE(resource.recordNormalQuery(makeRequest(3), historyTimeUs + 2'000));

    std::vector<uint64_t> requestIds;
    resource.setScheduleCallback([&](const auto &request) {
        requestIds.push_back(request.requestId);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    });
    auto command =
        makeStart(ReplayMode::REALTIME, historyTimeUs, historyTimeUs + 3'000, 1, 0, 0.0);
    command->generation = 1;
    command->pacingEpochUs = autil::TimeUtility::currentTimeInMicroSeconds() + 300'000;
    ASSERT_TRUE(resource.applyCommand(*command));
    for (int i = 0; i < 200 && resource.replaying(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    resource.stopReplay();

    EXPECT_EQ((std::vector<uint64_t>{1, 2, 3}), requestIds);
}

TEST_F(ReplayControlRTest, testFixedQpsUpdateAppliesNewPacingPhase)
{
    ReplayClientR resource;
    const auto now = autil::TimeUtility::currentTimeInMicroSeconds();
    ASSERT_TRUE(resource.recordNormalQuery(makeRequest(1), now));
    std::mutex callbackMutex;
    std::vector<std::chrono::steady_clock::time_point> callbackTimes;
    resource.setScheduleCallback([&](const auto &) {
        std::lock_guard<std::mutex> lock(callbackMutex);
        callbackTimes.emplace_back(std::chrono::steady_clock::now());
    });

    auto command = makeStart(ReplayMode::FIXED_QPS, now, now + 1, 0, 2000, 50.0);
    command->generation = 1;
    command->selectionGeneration = 7;
    command->totalQps = 100.0;
    command->pacingIndex = 0;
    command->pacingClientCount = 2;
    ASSERT_TRUE(resource.applyCommand(*command));
    for (int i = 0; i < 100; ++i) {
        {
            std::lock_guard<std::mutex> lock(callbackMutex);
            if (!callbackTimes.empty()) {
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    const auto updatedAt = std::chrono::steady_clock::now();
    size_t countAtUpdate = 0;
    {
        std::lock_guard<std::mutex> lock(callbackMutex);
        ASSERT_FALSE(callbackTimes.empty());
        countAtUpdate = callbackTimes.size();
    }
    command->generation = 2;
    command->totalQps = 10.0;
    command->perClientQps = 5.0;
    command->pacingIndex = 1;
    ASSERT_TRUE(resource.applyCommand(*command));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    {
        std::lock_guard<std::mutex> lock(callbackMutex);
        EXPECT_EQ(countAtUpdate, callbackTimes.size());
    }
    for (int i = 0; i < 100; ++i) {
        {
            std::lock_guard<std::mutex> lock(callbackMutex);
            if (callbackTimes.size() > countAtUpdate) {
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    resource.stopReplay();

    std::lock_guard<std::mutex> lock(callbackMutex);
    ASSERT_GT(callbackTimes.size(), countAtUpdate);
    const auto delayMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                             callbackTimes[countAtUpdate] - updatedAt)
                             .count();
    EXPECT_GE(delayMs, 120);
    EXPECT_LT(delayMs, 260);
}

TEST_F(ReplayControlRTest, testStopClosesSubscribersAndRejectsNewWork)
{
    ReplayControlR resource;
    auto clientA = std::make_shared<CaptureAsyncPipe>();
    auto clientB = std::make_shared<CaptureAsyncPipe>();
    ASSERT_TRUE(resource.subscribe("client-a", clientA));
    ASSERT_TRUE(resource.subscribe("client-b", clientB));

    resource.stop();
    EXPECT_TRUE(clientA->eof);
    EXPECT_TRUE(clientB->eof);
    EXPECT_TRUE(resource.getHistory().empty());

    auto clientC = std::make_shared<CaptureAsyncPipe>();
    EXPECT_FALSE(resource.subscribe("client-c", clientC));
    EXPECT_FALSE(clientC->eof);

    ReplayControlResponseDataPtr response;
    ASSERT_TRUE(resource.processRequest({"query"}, response));
    ASSERT_TRUE(response);
    EXPECT_FALSE(response->accepted);
    EXPECT_EQ("replay control is stopped", response->error);

    resource.stop();
    EXPECT_TRUE(clientA->eof);
    EXPECT_TRUE(clientB->eof);
}

TEST_F(ReplayControlRTest, testNoHistoryClientIsReportedButNotScheduled)
{
    ReplayControlR resource;
    auto clientA = std::make_shared<CaptureAsyncPipe>();
    auto clientB = std::make_shared<CaptureAsyncPipe>();
    ASSERT_TRUE(resource.subscribe("client-a", clientA));
    ASSERT_TRUE(resource.subscribe("client-b", clientB));

    ReplayHistorySummary summaryA;
    summaryA.clientId = "client-a";
    summaryA.earliestEventTimeUs = 100;
    summaryA.latestEventTimeUs = 300;
    summaryA.sampleCount = 3;
    summaryA.replayable = true;
    resource.updateHistory(summaryA);

    ReplayControlR::Request request;
    request.action = "start";
    request.mode = "fixed_qps";
    request.startTimeUs = 100;
    request.endTimeUs = 200;
    request.loopCount = 1;
    request.totalQps = 100.0;
    ReplayControlResponseDataPtr response;
    ASSERT_TRUE(resource.processRequest(request, response));
    ASSERT_TRUE(response->accepted);
    EXPECT_EQ(1u, response->eligibleClientCount);
    EXPECT_EQ(2u, response->clients.size());

    auto command =
        std::dynamic_pointer_cast<ReplayControlCommandData>(clientB->dataItems.back());
    ASSERT_TRUE(command);
    EXPECT_EQ(ReplayAction::STOP, command->action);

    ASSERT_TRUE(resource.processRequest({"query"}, response));
    ASSERT_TRUE(response->accepted);
    EXPECT_EQ(1u, response->eligibleClientCount);
    EXPECT_EQ(2u, response->clients.size());

    ReplayHistorySummary summaryB = summaryA;
    summaryB.clientId = "client-b";
    resource.updateHistory(summaryB);
    command =
        std::dynamic_pointer_cast<ReplayControlCommandData>(clientB->dataItems.back());
    ASSERT_TRUE(command);
    EXPECT_EQ(ReplayAction::START, command->action);
    EXPECT_DOUBLE_EQ(50.0, command->perClientQps);
}

TEST_F(ReplayControlRTest, testNormalHistoryCanBeSelectedByTimeRange)
{
    ReplayClientR resource;
    resource.setClientId("client-a");
    ASSERT_TRUE(resource.recordNormalQuery(makeRequest(1), 1000));
    ASSERT_TRUE(resource.recordNormalQuery(makeRequest(2), 2000));
    ASSERT_TRUE(resource.recordNormalQuery(makeRequest(3), 3000));
    const auto summary = resource.getSummary();
    EXPECT_EQ("client-a", summary.clientId);
    EXPECT_EQ(3u, summary.sampleCount);
    EXPECT_EQ(1000, summary.earliestEventTimeUs);
    EXPECT_EQ(3000, summary.latestEventTimeUs);
    EXPECT_TRUE(summary.replayable);
    EXPECT_EQ(2u, resource.selectSamples(1500, 3000).size());
    EXPECT_TRUE(resource.selectSamples(3000, 2000).empty());
}

TEST_F(ReplayControlRTest, testValidEmptyRangeDoesNotStartReplay)
{
    ReplayClientR resource;
    resource.setClientId("client-a");
    auto capabilityPipe = std::make_shared<CaptureAsyncPipe>();
    auto queryResponsePipe = std::make_shared<CaptureAsyncPipe>();
    ASSERT_TRUE(resource.subscribeHistory(capabilityPipe));
    ASSERT_TRUE(resource.subscribeHistoryQueryResponse(queryResponsePipe));
    const auto now = autil::TimeUtility::currentTimeInMicroSeconds();
    ASSERT_TRUE(resource.recordNormalQuery(makeRequest(1), now));
    ASSERT_TRUE(resource.recordNormalQuery(makeRequest(2), now + 1'000'000));
    ASSERT_TRUE(resource.selectSamples(now + 100'000, now + 200'000).empty());

    std::atomic<int> callbackCount{0};
    resource.setScheduleCallback([&callbackCount](const auto &) {
        callbackCount.fetch_add(1, std::memory_order_relaxed);
    });
    auto command =
        makeStart(ReplayMode::FIXED_QPS, now + 100'000, now + 200'000, 1, 0, 100.0);
    command->generation = 1;
    command->selectionGeneration = 9;
    ASSERT_TRUE(resource.applyCommand(*command));
    EXPECT_FALSE(resource.replaying());
    EXPECT_EQ(0, callbackCount.load());
    EXPECT_EQ(2u, resource.getSummary().sampleCount);
    ASSERT_EQ(2u, capabilityPipe->dataItems.size());
    const auto capability = std::dynamic_pointer_cast<ReplayClientCapabilityData>(
        capabilityPipe->dataItems.back());
    ASSERT_TRUE(capability);
    EXPECT_EQ(9u, capability->summary.selectionGeneration);
    EXPECT_EQ(0u, capability->summary.selectedSampleCount);

    ASSERT_TRUE(resource.answerHistoryQuery(17));
    ASSERT_EQ(1u, queryResponsePipe->dataItems.size());
    const auto queryResponse = std::dynamic_pointer_cast<ReplayHistoryQueryResponseData>(
        queryResponsePipe->dataItems.back());
    ASSERT_TRUE(queryResponse);
    EXPECT_EQ(17u, queryResponse->queryId);
    EXPECT_EQ(9u, queryResponse->summary.selectionGeneration);
    EXPECT_EQ(0u, queryResponse->summary.selectedSampleCount);
}

TEST_F(ReplayControlRTest, testRealtimeUsesAbsoluteOffsetsWithoutCallbackDrift)
{
    ReplayClientR resource;
    resource.setClientId("client-a");
    const auto now = autil::TimeUtility::currentTimeInMicroSeconds();
    constexpr int64_t OFFSET_US = 300'000;
    ASSERT_TRUE(resource.recordNormalQuery(makeRequest(1), now));
    ASSERT_TRUE(resource.recordNormalQuery(makeRequest(2), now + OFFSET_US));
    ASSERT_TRUE(resource.recordNormalQuery(makeRequest(3), now + 2 * OFFSET_US));

    std::atomic<int> callbackCount{0};
    std::vector<std::chrono::steady_clock::time_point> callbackTimes;
    resource.setScheduleCallback([&](const auto &) {
        callbackTimes.emplace_back(std::chrono::steady_clock::now());
        callbackCount.fetch_add(1, std::memory_order_release);
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
    });
    auto command = makeStart(ReplayMode::REALTIME, now, now + 3 * OFFSET_US, 2, 0, 0.0);
    command->generation = 1;
    ASSERT_TRUE(resource.applyCommand(*command));
    for (int i = 0; i < 400 && callbackCount.load(std::memory_order_acquire) < 6; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    resource.stopReplay();
    ASSERT_EQ(6, callbackCount.load());
    ASSERT_EQ(6u, callbackTimes.size());

    for (size_t loop = 0; loop < 2; ++loop) {
        const auto begin = loop * 3;
        for (size_t index = 1; index < 3; ++index) {
            const auto gapMs =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    callbackTimes[begin + index] - callbackTimes[begin + index - 1])
                    .count();
            EXPECT_GE(gapMs, 250);
            EXPECT_LT(gapMs, 430);
        }
    }
}

TEST_F(ReplayControlRTest, testRealtimePreservesGapFromRequestedRangeStart)
{
    ReplayClientR resource;
    resource.setClientId("client-a");
    const auto now = autil::TimeUtility::currentTimeInMicroSeconds();
    ASSERT_TRUE(resource.recordNormalQuery(makeRequest(1), now + 150'000));

    std::atomic<uint64_t> callbackCount{0};
    resource.setScheduleCallback(
        [&](const auto &) { callbackCount.fetch_add(1, std::memory_order_relaxed); });
    auto command = makeStart(ReplayMode::REALTIME, now, now + 200'000, 1, 0, 0.0);
    command->generation = 1;
    ASSERT_TRUE(resource.applyCommand(*command));

    std::this_thread::sleep_for(std::chrono::milliseconds(60));
    EXPECT_EQ(0u, callbackCount.load(std::memory_order_relaxed));
    for (int i = 0; i < 100 && callbackCount.load(std::memory_order_relaxed) < 1; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    resource.stopReplay();
    EXPECT_EQ(1u, callbackCount.load(std::memory_order_relaxed));
}

TEST_F(ReplayControlRTest, testRealtimeLoopsUseRequestedRangeBoundary)
{
    ReplayClientR resource;
    resource.setClientId("client-a");
    const auto now = autil::TimeUtility::currentTimeInMicroSeconds();
    ASSERT_TRUE(resource.recordNormalQuery(makeRequest(1), now + 20'000));
    ASSERT_TRUE(resource.recordNormalQuery(makeRequest(2), now + 40'000));

    std::mutex callbackMutex;
    std::vector<std::chrono::steady_clock::time_point> callbackTimes;
    resource.setScheduleCallback([&](const auto &) {
        std::lock_guard<std::mutex> lock(callbackMutex);
        callbackTimes.emplace_back(std::chrono::steady_clock::now());
    });
    auto command = makeStart(ReplayMode::REALTIME, now, now + 120'000, 2, 0, 0.0);
    command->generation = 1;
    ASSERT_TRUE(resource.applyCommand(*command));
    for (int i = 0; i < 100; ++i) {
        {
            std::lock_guard<std::mutex> lock(callbackMutex);
            if (callbackTimes.size() == 4) {
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    resource.stopReplay();

    std::lock_guard<std::mutex> lock(callbackMutex);
    ASSERT_EQ(4u, callbackTimes.size());
    const auto loopBoundaryGapMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                                       callbackTimes[2] - callbackTimes[1])
                                       .count();
    EXPECT_GE(loopBoundaryGapMs, 70);
    EXPECT_LT(loopBoundaryGapMs, 170);
}

TEST_F(ReplayControlRTest, testReplayDoesNotAppendToNormalHistory)
{
    ReplayClientR resource;
    resource.setClientId("client-a");
    const auto now = autil::TimeUtility::currentTimeInMicroSeconds();
    ASSERT_TRUE(resource.recordNormalQuery(makeRequest(1), now));
    ASSERT_TRUE(resource.recordNormalQuery(makeRequest(2), now + 1000));
    std::atomic<int> callbackCount{0};
    resource.setScheduleCallback([&callbackCount](const auto &) {
        callbackCount.fetch_add(1, std::memory_order_relaxed);
    });

    auto command = makeStart(ReplayMode::REALTIME, now, now + 1000, 2, 0, 0);
    command->generation = 1;
    ASSERT_TRUE(resource.applyCommand(*command));
    for (int i = 0; i < 100 && callbackCount.load() < 4; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    resource.stopReplay();
    EXPECT_EQ(4, callbackCount.load());
    EXPECT_EQ(2u, resource.getSummary().sampleCount);
}

TEST_F(ReplayControlRTest, testReplayRunsAlongsideNormalTraffic)
{
    ReplayClientR resource;
    resource.setClientId("client-a");
    const auto now = autil::TimeUtility::currentTimeInMicroSeconds();
    ASSERT_TRUE(resource.recordNormalQuery(makeRequest(1), now));
    ASSERT_TRUE(resource.recordNormalQuery(makeRequest(2), now + 1000));
    std::atomic<int> replayCallbackCount{0};
    resource.setScheduleCallback([&replayCallbackCount](const auto &) {
        replayCallbackCount.fetch_add(1, std::memory_order_relaxed);
    });

    auto command = makeStart(ReplayMode::FIXED_QPS, now, now + 1000, 100, 0, 10);
    command->generation = 1;
    ASSERT_TRUE(resource.applyCommand(*command));
    for (int i = 0; i < 100 && replayCallbackCount.load() == 0; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    ASSERT_TRUE(resource.replaying());
    ASSERT_GT(replayCallbackCount.load(), 0);

    ASSERT_TRUE(resource.recordNormalQuery(makeRequest(3), now + 2000));
    EXPECT_EQ(3u, resource.getSummary().sampleCount);
    resource.stopReplay();

    EXPECT_FALSE(resource.replaying());
    EXPECT_EQ(3u, resource.getSummary().sampleCount);
}

TEST_F(ReplayControlRTest, testDurationStopsReplayAfterCurrentInterval)
{
    ReplayClientR resource;
    const auto now = autil::TimeUtility::currentTimeInMicroSeconds();
    ASSERT_TRUE(resource.recordNormalQuery(makeRequest(1), now));
    ASSERT_TRUE(resource.recordNormalQuery(makeRequest(2), now + 2'000'000));
    std::atomic<int> callbackCount{0};
    resource.setScheduleCallback([&callbackCount](const auto &) {
        callbackCount.fetch_add(1, std::memory_order_relaxed);
    });

    auto command = makeStart(ReplayMode::REALTIME, now, now + 2'000'000, 0, 20, 0);
    command->generation = 1;
    ASSERT_TRUE(resource.applyCommand(*command));
    for (int i = 0; i < 100 && resource.replaying(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    EXPECT_EQ(1, callbackCount.load());
    EXPECT_FALSE(resource.replaying());
}

TEST_F(ReplayControlRTest, testRealtimeLoopAndFixedQpsAreIndependent)
{
    ReplayClientR resource;
    resource.setClientId("client-a");
    const auto now = autil::TimeUtility::currentTimeInMicroSeconds();
    ASSERT_TRUE(resource.recordNormalQuery(makeRequest(1), now));
    ASSERT_TRUE(resource.recordNormalQuery(makeRequest(2), now + 10000));
    std::atomic<int> callbackCount{0};
    resource.setScheduleCallback([&callbackCount](const auto &) {
        callbackCount.fetch_add(1, std::memory_order_relaxed);
    });

    auto realtime = makeStart(ReplayMode::REALTIME, now, now + 10000, 2, 0, 0);
    ASSERT_TRUE(resource.applyCommand(*realtime));
    for (int i = 0; i < 100 && callbackCount.load() < 4; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    EXPECT_EQ(4, callbackCount.load());

    callbackCount.store(0);
    auto fixed = makeStart(ReplayMode::FIXED_QPS, now, now + 10000, 2, 0, 1000);
    ASSERT_TRUE(resource.applyCommand(*fixed));
    for (int i = 0; i < 100 && callbackCount.load() < 4; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    EXPECT_EQ(4, callbackCount.load());
    resource.stopReplay();
}

TEST_F(ReplayControlRTest, testStopInterruptsRealtimeWait)
{
    ReplayClientR resource;
    resource.setClientId("client-a");
    const auto now = autil::TimeUtility::currentTimeInMicroSeconds();
    ASSERT_TRUE(resource.recordNormalQuery(makeRequest(1), now));
    ASSERT_TRUE(resource.recordNormalQuery(makeRequest(2), now + 5'000'000));
    std::atomic<int> callbackCount{0};
    resource.setScheduleCallback([&callbackCount](const auto &) {
        callbackCount.fetch_add(1, std::memory_order_relaxed);
    });
    auto command = makeStart(ReplayMode::REALTIME, now, now + 5'000'000, 1, 0, 0);
    ASSERT_TRUE(resource.applyCommand(*command));
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    resource.stopReplay();
    EXPECT_EQ(1, callbackCount.load());
    EXPECT_FALSE(resource.replaying());
}

}
