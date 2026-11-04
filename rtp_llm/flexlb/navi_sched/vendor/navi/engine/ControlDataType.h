#pragma once

#include "navi/engine/Type.h"

namespace navi {

class ControlDataType : public Type {
public:
    ControlDataType()
        : Type(NAVI_TYPE_PARAM, CONTROL_DATA_TYPE)
    {
    }
    ~ControlDataType() {
    }
public:
    TypeErrorCode serialize(TypeContext &ctx,
                            const DataPtr &data) const override;
    TypeErrorCode deserialize(TypeContext &ctx, DataPtr &data) const override;
};

}
