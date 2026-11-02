#pragma once

#include "navi/engine/Data.h"
#include <autil/DataBuffer.h>
#include <cstdint>
#include <string>

namespace navi_sched::test {

extern const std::string TRAFFIC_REPLAY_START_REQUEST_TYPE_ID;
extern const std::string TRAFFIC_REPLAY_START_RESPONSE_TYPE_ID;
extern const std::string TRAFFIC_REPLAY_START_REQUEST_DATA_NAME;

class TrafficReplayStartRequestData : public navi::Data {
public:
    TrafficReplayStartRequestData();
    void serialize(autil::DataBuffer &dataBuffer) const;
    void deserialize(autil::DataBuffer &dataBuffer);

public:
    std::string replayGroup;
    size_t clientIndex = 0;
    size_t clientCount = 1;
    int64_t firstRequestTimeMs = 0;
    int64_t startDelayMs = 2000;
};

class TrafficReplayStartResponseData : public navi::Data {
public:
    TrafficReplayStartResponseData();
    void serialize(autil::DataBuffer &dataBuffer) const;
    void deserialize(autil::DataBuffer &dataBuffer);

public:
    int64_t firstRequestTimeMs = 0;
    int64_t startTimeMs = 0;
};

}
