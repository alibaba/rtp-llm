#include "navi/resource/TimeoutCheckerR.h"
#include "navi/engine/NaviRegisterMacro.h"

namespace navi {

const std::string TimeoutCheckerR::RESOURCE_ID = "navi.timeout_checker_r";

TimeoutCheckerR::TimeoutCheckerR(TimeoutChecker *timeoutChecker)
    : ExternalResource(RESOURCE_ID, RS_RUN_GRAPH_EXTERNAL)
    , _timeoutChecker(timeoutChecker)
{
}

TimeoutCheckerR::TimeoutCheckerR()
    : ExternalResource(RESOURCE_ID, RS_RUN_GRAPH_EXTERNAL)
{
}

TimeoutCheckerR::~TimeoutCheckerR() {
}

REGISTER_BUILDIN_RESOURCE(TimeoutCheckerR);

}

