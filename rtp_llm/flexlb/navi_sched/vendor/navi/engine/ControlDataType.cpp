#include "navi/engine/ControlDataType.h"
#include "navi/engine/ControlData.h"
#include "navi/engine/NaviRegisterMacro.h"

namespace navi {

TypeErrorCode ControlDataType::serialize(TypeContext &ctx,
                                         const DataPtr &data) const
{
    return TEC_NONE;
}

TypeErrorCode ControlDataType::deserialize(TypeContext &ctx,
                                           DataPtr &data) const
{
    return TEC_NONE;
}

REGISTER_BUILDIN_TYPE(ControlDataType);

}

