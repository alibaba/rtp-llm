#pragma once

#include "rtp_llm/flexlb/navi_sched/common/NaviSchedScheduleData.h"
#include "navi/engine/Data.h"

#include <autil/DataBuffer.h>
#include <cstdint>
#include <string>
#include <vector>

namespace navi_sched {

enum class ReplayMode : uint8_t {
    FIXED_QPS = 0,
    REALTIME = 1,
};

enum class ReplayAction : uint8_t {
    STOP = 0,
    START = 1,
};

enum class ReplayHistoryMode : uint8_t {
    TIME_RANGE = 0,
    RECENT_COUNT = 1,
};

struct ReplayHistorySample {
    NaviSchedScheduleRequest request;
    int64_t eventTimeUs = 0;
};

struct ReplayHistorySummary {
    std::string clientId;
    int64_t earliestEventTimeUs = 0;
    int64_t latestEventTimeUs = 0;
    uint64_t sampleCount = 0;
    bool replayable = false;
    std::string status;
    std::string error;
    uint64_t selectionGeneration = 0;
    uint64_t selectedSampleCount = 0;
    uint64_t completedSelectionGeneration = 0;
    int64_t reportTimeUs = 0;
};

class ReplayHistoryQueryData : public navi::Data {
public:
    ReplayHistoryQueryData();
    static const std::string TYPE_ID;
    void serialize(autil::DataBuffer &dataBuffer) const;
    void deserialize(autil::DataBuffer &dataBuffer);

    uint64_t queryId = 0;
};

class ReplayHistoryQueryResponseData : public navi::Data {
public:
    ReplayHistoryQueryResponseData();
    static const std::string TYPE_ID;
    void serialize(autil::DataBuffer &dataBuffer) const;
    void deserialize(autil::DataBuffer &dataBuffer);

    uint64_t queryId = 0;
    ReplayHistorySummary summary;
};

class ReplayHistoryUpdateData : public navi::Data {
public:
    ReplayHistoryUpdateData();
    static const std::string TYPE_ID;
    void serialize(autil::DataBuffer &dataBuffer) const;
    void deserialize(autil::DataBuffer &dataBuffer);

    ReplayHistorySummary summary;
};

class ReplayControlCommandData : public navi::Data {
public:
    ReplayControlCommandData();
    static const std::string TYPE_ID;
    void serialize(autil::DataBuffer &dataBuffer) const;
    void deserialize(autil::DataBuffer &dataBuffer);

    ReplayAction action = ReplayAction::STOP;
    ReplayMode mode = ReplayMode::FIXED_QPS;
    ReplayHistoryMode historyMode = ReplayHistoryMode::TIME_RANGE;
    uint64_t generation = 0;
    int64_t startTimeUs = 0;
    int64_t endTimeUs = 0;
    int64_t durationMs = 0;
    double totalQps = 0.0;
    double perClientQps = 0.0;
    uint64_t loopCount = 0;
    uint64_t recentQueryCount = 0;
    uint64_t selectionGeneration = 0;
    uint64_t pacingIndex = 0;
    uint64_t pacingClientCount = 0;
    int64_t pacingEpochUs = 0;
    int64_t pacingClockOffsetUs = 0;
};

class ReplayClientCapabilityData : public navi::Data {
public:
    ReplayClientCapabilityData();
    static const std::string TYPE_ID;
    void serialize(autil::DataBuffer &dataBuffer) const;
    void deserialize(autil::DataBuffer &dataBuffer);

    std::string clientId;
    ReplayHistorySummary summary;
};

NAVI_TYPEDEF_PTR(ReplayHistoryUpdateData);
NAVI_TYPEDEF_PTR(ReplayControlCommandData);
NAVI_TYPEDEF_PTR(ReplayClientCapabilityData);
NAVI_TYPEDEF_PTR(ReplayHistoryQueryData);
NAVI_TYPEDEF_PTR(ReplayHistoryQueryResponseData);

} // namespace navi_sched
