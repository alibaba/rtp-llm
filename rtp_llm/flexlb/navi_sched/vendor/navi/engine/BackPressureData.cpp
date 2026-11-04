#include "navi/engine/BackPressureData.h"
#include "navi/engine/NaviRegisterMacro.h"

namespace navi {

TypeErrorCode BackPressureDataType::serialize(TypeContext &ctx, const DataPtr &data) const {
    if (!data) {
        return TEC_NULL_DATA;
    }
    auto bpData = dynamic_cast<BackPressureData *>(data.get());
    if (!bpData) {
        return TEC_TYPE_MISMATCH;
    }
    auto &dataBuffer = ctx.getDataBuffer();
    dataBuffer.write(bpData);
    return TEC_NONE;
}

TypeErrorCode BackPressureDataType::deserialize(TypeContext &ctx, DataPtr &data) const {
    BackPressureData *bpData = nullptr;
    auto &dataBuffer = ctx.getDataBuffer();
    dataBuffer.read(bpData);
    if (!bpData) {
        return TEC_NULL_DATA;
    }
    data.reset(bpData);
    return TEC_NONE;
}

BackPressureData::BackPressureData()
    : Data(BACK_PRESSURE_DATA_TYPE)
{
}

BackPressureData::~BackPressureData() {
}

void BackPressureData::serialize(autil::DataBuffer &dataBuffer) const {
    dataBuffer.write(maxDataCount);
}

void BackPressureData::deserialize(autil::DataBuffer &dataBuffer) {
    dataBuffer.read(maxDataCount);
}

REGISTER_BUILDIN_TYPE(BackPressureDataType);

}

