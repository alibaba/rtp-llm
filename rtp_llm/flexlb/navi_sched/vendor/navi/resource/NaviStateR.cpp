#include "navi/resource/NaviStateR.h"
#include "navi/engine/NaviRegisterMacro.h"

namespace navi {

const std::string NaviStateR::RESOURCE_ID = "navi.state_r";

NaviStateR::NaviStateR()
    : ExternalResource(RESOURCE_ID, RS_EXTERNAL)
{
}

NaviStateR::~NaviStateR() {
}

REGISTER_BUILDIN_RESOURCE(NaviStateR);

}

