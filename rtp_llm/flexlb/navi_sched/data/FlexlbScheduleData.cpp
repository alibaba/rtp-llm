#include "aios/navi_sched/data/FlexlbScheduleData.h"
#include "navi/engine/Type.h"

namespace navi_sched {

FlexlbScheduleRequestData::FlexlbScheduleRequestData()
    : navi::Data(FLEXLB_REQUEST_TYPE)
{
}

FlexlbScheduleResponseData::FlexlbScheduleResponseData()
    : navi::Data(FLEXLB_RESPONSE_TYPE)
{
}

class FlexlbScheduleResponseType : public navi::Type {
public:
    FlexlbScheduleResponseType() : navi::Type(NAVI_TYPE_PARAM, FLEXLB_RESPONSE_TYPE) {}

    navi::TypeErrorCode serialize(navi::TypeContext &, const navi::DataPtr &) const override
    {
        return navi::TEC_FAILED;
    }

    navi::TypeErrorCode deserialize(navi::TypeContext &, navi::DataPtr &) const override
    {
        return navi::TEC_FAILED;
    }
};

REGISTER_TYPE(FlexlbScheduleResponseType);

}
