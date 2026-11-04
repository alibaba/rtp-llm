#pragma once

#include "navi/engine/Data.h"
#include "navi/engine/Type.h"

namespace navi {

class BackPressureDataType : public Type {
public:
    BackPressureDataType()
        : Type(NAVI_TYPE_PARAM, BACK_PRESSURE_DATA_TYPE)
    {
    }
    ~BackPressureDataType() {
    }
public:
    TypeErrorCode serialize(TypeContext &ctx, const DataPtr &data) const override;
    TypeErrorCode deserialize(TypeContext &ctx, DataPtr &data) const override;
};

class BackPressureData : public Data {
public:
    BackPressureData();
    ~BackPressureData();
private:
    BackPressureData(const BackPressureData &) = delete;
    BackPressureData &operator=(const BackPressureData &) = delete;
public:
    void serialize(autil::DataBuffer &dataBuffer) const;
    void deserialize(autil::DataBuffer &dataBuffer) ;
public:
    int64_t maxDataCount = 0;
};

NAVI_TYPEDEF_PTR(BackPressureData);

}

