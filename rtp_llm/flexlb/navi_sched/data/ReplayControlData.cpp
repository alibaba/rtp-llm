#include "rtp_llm/flexlb/navi_sched/data/ReplayControlData.h"
#include "navi/engine/Type.h"

namespace navi_sched {
namespace {


void serializeSummary(autil::DataBuffer &buffer, const ReplayHistorySummary &summary)
{
    buffer.write(summary.clientId);
    buffer.write(summary.earliestEventTimeUs);
    buffer.write(summary.latestEventTimeUs);
    buffer.write(summary.sampleCount);
    buffer.write(summary.replayable);
    buffer.write(summary.status);
    buffer.write(summary.error);
    buffer.write(summary.selectionGeneration);
    buffer.write(summary.selectedSampleCount);
    buffer.write(summary.completedSelectionGeneration);
    buffer.write(summary.reportTimeUs);
}

void deserializeSummary(autil::DataBuffer &buffer, ReplayHistorySummary &summary)
{
    buffer.read(summary.clientId);
    buffer.read(summary.earliestEventTimeUs);
    buffer.read(summary.latestEventTimeUs);
    buffer.read(summary.sampleCount);
    buffer.read(summary.replayable);
    if (buffer.getDataLen() > 0) {
        buffer.read(summary.status);
    }
    if (buffer.getDataLen() > 0) {
        buffer.read(summary.error);
    }
    if (buffer.getDataLen() > 0) {
        buffer.read(summary.selectionGeneration);
    }
    if (buffer.getDataLen() > 0) {
        buffer.read(summary.selectedSampleCount);
    }
    if (buffer.getDataLen() > 0) {
        buffer.read(summary.completedSelectionGeneration);
    }
    if (buffer.getDataLen() > 0) {
        buffer.read(summary.reportTimeUs);
    }
}

} // namespace

ReplayHistoryUpdateData::ReplayHistoryUpdateData()
    : navi::Data(TYPE_ID)
{
}

const std::string ReplayHistoryUpdateData::TYPE_ID =
    "NaviSchedReplayHistoryUpdateType";

void ReplayHistoryUpdateData::serialize(autil::DataBuffer &buffer) const
{
    serializeSummary(buffer, summary);
}

void ReplayHistoryUpdateData::deserialize(autil::DataBuffer &buffer)
{
    deserializeSummary(buffer, summary);
}

ReplayControlCommandData::ReplayControlCommandData()
    : navi::Data(TYPE_ID)
{
}

const std::string ReplayControlCommandData::TYPE_ID =
    "NaviSchedReplayControlCommandType";

void ReplayControlCommandData::serialize(autil::DataBuffer &buffer) const
{
    buffer.write(action);
    buffer.write(mode);
    buffer.write(generation);
    buffer.write(startTimeUs);
    buffer.write(endTimeUs);
    buffer.write(durationMs);
    buffer.write(totalQps);
    buffer.write(perClientQps);
    buffer.write(loopCount);
    buffer.write(historyMode);
    buffer.write(recentQueryCount);
    buffer.write(selectionGeneration);
    buffer.write(pacingIndex);
    buffer.write(pacingClientCount);
    buffer.write(pacingEpochUs);
    buffer.write(pacingClockOffsetUs);
}

void ReplayControlCommandData::deserialize(autil::DataBuffer &buffer)
{
    buffer.read(action);
    buffer.read(mode);
    buffer.read(generation);
    buffer.read(startTimeUs);
    buffer.read(endTimeUs);
    buffer.read(durationMs);
    buffer.read(totalQps);
    buffer.read(perClientQps);
    buffer.read(loopCount);
    // Appending fields keeps commands from the previous rolling version readable.
    if (buffer.getDataLen() > 0) {
        buffer.read(historyMode);
    }
    if (buffer.getDataLen() > 0) {
        buffer.read(recentQueryCount);
    }
    if (buffer.getDataLen() > 0) {
        buffer.read(selectionGeneration);
    }
    if (buffer.getDataLen() > 0) {
        buffer.read(pacingIndex);
    }
    if (buffer.getDataLen() > 0) {
        buffer.read(pacingClientCount);
    }
    if (buffer.getDataLen() > 0) {
        buffer.read(pacingEpochUs);
    }
    if (buffer.getDataLen() > 0) {
        buffer.read(pacingClockOffsetUs);
    }
}

ReplayClientCapabilityData::ReplayClientCapabilityData()
    : navi::Data(TYPE_ID)
{
}

ReplayHistoryQueryData::ReplayHistoryQueryData()
    : navi::Data(TYPE_ID)
{
}

const std::string ReplayHistoryQueryData::TYPE_ID =
    "NaviSchedReplayHistoryQueryType";

void ReplayHistoryQueryData::serialize(autil::DataBuffer &buffer) const
{
    buffer.write(queryId);
}

void ReplayHistoryQueryData::deserialize(autil::DataBuffer &buffer)
{
    buffer.read(queryId);
}

ReplayHistoryQueryResponseData::ReplayHistoryQueryResponseData()
    : navi::Data(TYPE_ID)
{
}

const std::string ReplayHistoryQueryResponseData::TYPE_ID =
    "NaviSchedReplayHistoryQueryResponseType";

void ReplayHistoryQueryResponseData::serialize(autil::DataBuffer &buffer) const
{
    buffer.write(queryId);
    serializeSummary(buffer, summary);
}

void ReplayHistoryQueryResponseData::deserialize(autil::DataBuffer &buffer)
{
    buffer.read(queryId);
    deserializeSummary(buffer, summary);
}

const std::string ReplayClientCapabilityData::TYPE_ID =
    "NaviSchedReplayClientCapabilityType";

void ReplayClientCapabilityData::serialize(autil::DataBuffer &buffer) const
{
    buffer.write(clientId);
    serializeSummary(buffer, summary);
}

void ReplayClientCapabilityData::deserialize(autil::DataBuffer &buffer)
{
    buffer.read(clientId);
    deserializeSummary(buffer, summary);
}

template <typename DataType>
class ReplayDataType : public navi::Type {
public:
    explicit ReplayDataType(const std::string &typeId)
        : navi::Type(NAVI_TYPE_PARAM, typeId)
    {
    }

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

class ReplayHistoryUpdateType : public ReplayDataType<ReplayHistoryUpdateData> {
public:
    ReplayHistoryUpdateType()
        : ReplayDataType(ReplayHistoryUpdateData::TYPE_ID)
    {
    }
};

class ReplayControlCommandType : public ReplayDataType<ReplayControlCommandData> {
public:
    ReplayControlCommandType()
        : ReplayDataType(ReplayControlCommandData::TYPE_ID)
    {
    }
};

class ReplayClientCapabilityType
    : public ReplayDataType<ReplayClientCapabilityData> {
public:
    ReplayClientCapabilityType()
        : ReplayDataType(ReplayClientCapabilityData::TYPE_ID)
    {
    }
};

class ReplayHistoryQueryType : public ReplayDataType<ReplayHistoryQueryData> {
public:
    ReplayHistoryQueryType()
        : ReplayDataType(ReplayHistoryQueryData::TYPE_ID)
    {
    }
};

class ReplayHistoryQueryResponseType
    : public ReplayDataType<ReplayHistoryQueryResponseData> {
public:
    ReplayHistoryQueryResponseType()
        : ReplayDataType(ReplayHistoryQueryResponseData::TYPE_ID)
    {
    }
};

REGISTER_TYPE(ReplayHistoryUpdateType);
REGISTER_TYPE(ReplayControlCommandType);
REGISTER_TYPE(ReplayClientCapabilityType);
REGISTER_TYPE(ReplayHistoryQueryType);
REGISTER_TYPE(ReplayHistoryQueryResponseType);

} // namespace navi_sched
