#include "navi/engine/RemoteSubGraphBase.h"
#include "navi/engine/RemoteInputPort.h"
#include "navi/engine/RemoteOutputPort.h"
#include "navi/util/ReadyBitMap.h"
#include "navi/util/CommonUtil.h"

namespace navi {

RemoteSubGraphBase::RemoteSubGraphBase(SubGraphDef *subGraphDef,
                                       GraphParam *param,
                                       const GraphDomainPtr &domain,
                                       const SubGraphBorderMapPtr &borderMap,
                                       Graph *graph)
    : SubGraphBase(subGraphDef, param, domain, borderMap, graph)
{
    _logger.addPrefix("remote");
}

RemoteSubGraphBase::~RemoteSubGraphBase() {
}

ErrorCode RemoteSubGraphBase::init() {
    return EC_NONE;
}

ErrorCode RemoteSubGraphBase::run() {
    return EC_NONE;
}

void RemoteSubGraphBase::terminate(ErrorCode ec) {
    NAVI_LOG(SCHEDULE2, "terminate, ec: %s", CommonUtil::getErrorString(ec));
}

}
