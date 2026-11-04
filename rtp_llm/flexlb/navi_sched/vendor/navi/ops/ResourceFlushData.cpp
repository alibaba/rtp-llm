#include "navi/ops/ResourceFlushData.h"
#include "navi/engine/NaviRegisterMacro.h"

namespace navi {

const std::string ResourceFlushDataType::TYPE_ID = "navi.resource_flush_data_t";

ResourceFlushDataType::ResourceFlushDataType()
    : Type(NAVI_TYPE_PARAM, TYPE_ID)
{
}

REGISTER_BUILDIN_TYPE(ResourceFlushDataType);

ResourceFlushData::ResourceFlushData()
    : Data(ResourceFlushDataType::TYPE_ID)
{
}

ResourceFlushData::~ResourceFlushData() {
}

}

