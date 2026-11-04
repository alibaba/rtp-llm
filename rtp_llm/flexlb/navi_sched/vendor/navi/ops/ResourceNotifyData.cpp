#include "navi/ops/ResourceNotifyData.h"

namespace navi {

ResourceNotifyDataType::ResourceNotifyDataType()
    : Type(NAVI_TYPE_PARAM, RESOURCE_NOTIFY_DATA_TYPE_ID)
{
}

ResourceNotifyData::ResourceNotifyData()
    : Data(RESOURCE_NOTIFY_DATA_TYPE_ID)
{
}

ResourceNotifyData::~ResourceNotifyData() {
}

}

