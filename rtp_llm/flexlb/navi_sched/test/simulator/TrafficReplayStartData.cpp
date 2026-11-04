#include "rtp_llm/flexlb/navi_sched/test/simulator/TrafficReplayStartData.h"

#include "navi/engine/Type.h"

namespace navi_sched::test {

const std::string TRAFFIC_REPLAY_START_REQUEST_TYPE_ID =
    "NaviSchedTrafficReplayStartRequestType";
const std::string TRAFFIC_REPLAY_START_RESPONSE_TYPE_ID =
    "NaviSchedTrafficReplayStartResponseType";
const std::string TRAFFIC_REPLAY_START_REQUEST_DATA_NAME =
    "navi_sched.traffic_replay_start_request";

TrafficReplayStartRequestData::TrafficReplayStartRequestData()
    : navi::Data(TRAFFIC_REPLAY_START_REQUEST_TYPE_ID)
{
}

void TrafficReplayStartRequestData::serialize(autil::DataBuffer &dataBuffer) const
{
    dataBuffer.write(replayGroup);
    dataBuffer.write(clientIndex);
    dataBuffer.write(clientCount);
    dataBuffer.write(firstRequestTimeMs);
    dataBuffer.write(startDelayMs);
}

void TrafficReplayStartRequestData::deserialize(autil::DataBuffer &dataBuffer)
{
    dataBuffer.read(replayGroup);
    dataBuffer.read(clientIndex);
    dataBuffer.read(clientCount);
    dataBuffer.read(firstRequestTimeMs);
    dataBuffer.read(startDelayMs);
}

TrafficReplayStartResponseData::TrafficReplayStartResponseData()
    : navi::Data(TRAFFIC_REPLAY_START_RESPONSE_TYPE_ID)
{
}

void TrafficReplayStartResponseData::serialize(autil::DataBuffer &dataBuffer) const
{
    dataBuffer.write(firstRequestTimeMs);
    dataBuffer.write(startTimeMs);
}

void TrafficReplayStartResponseData::deserialize(autil::DataBuffer &dataBuffer)
{
    dataBuffer.read(firstRequestTimeMs);
    dataBuffer.read(startTimeMs);
}

template <typename DataType>
class TrafficReplayType : public navi::Type {
public:
    explicit TrafficReplayType(const std::string &id) : navi::Type(NAVI_TYPE_PARAM, id) {}

public:
    navi::TypeErrorCode serialize(navi::TypeContext &ctx,
                                  const navi::DataPtr &data) const override
    {
        auto typedData = dynamic_cast<DataType *>(data.get());
        if (!typedData) {
            return navi::TEC_TYPE_MISMATCH;
        }
        ctx.getDataBuffer().write(typedData);
        return navi::TEC_NONE;
    }
    navi::TypeErrorCode deserialize(navi::TypeContext &ctx,
                                    navi::DataPtr &data) const override
    {
        DataType *typedData = nullptr;
        ctx.getDataBuffer().read(typedData);
        if (!typedData) {
            return navi::TEC_NULL_DATA;
        }
        data.reset(typedData);
        return navi::TEC_NONE;
    }
};

class TrafficReplayStartRequestType
    : public TrafficReplayType<TrafficReplayStartRequestData> {
public:
    TrafficReplayStartRequestType()
        : TrafficReplayType(TRAFFIC_REPLAY_START_REQUEST_TYPE_ID)
    {
    }
};

class TrafficReplayStartResponseType
    : public TrafficReplayType<TrafficReplayStartResponseData> {
public:
    TrafficReplayStartResponseType()
        : TrafficReplayType(TRAFFIC_REPLAY_START_RESPONSE_TYPE_ID)
    {
    }
};

REGISTER_TYPE(TrafficReplayStartRequestType);
REGISTER_TYPE(TrafficReplayStartResponseType);

} // namespace navi_sched::test
