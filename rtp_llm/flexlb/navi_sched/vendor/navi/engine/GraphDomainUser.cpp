#include "navi/engine/GraphDomainUser.h"
#include "navi/engine/NaviUserResult.h"

namespace navi {

GraphDomainUser::GraphDomainUser(Graph *graph)
    : GraphDomain(graph, GDT_USER)
{
}

GraphDomainUser::~GraphDomainUser() {
}

void GraphDomainUser::setUserResult(const NaviUserResultPtr &result) {
    autil::ScopedWriteLock lock(_resultLock);
    _result = result;
}

NaviUserResultPtr GraphDomainUser::getUserResult() {
    autil::ScopedReadLock lock(_resultLock);
    return _result;
}

bool GraphDomainUser::run() {
    return true;
}

void GraphDomainUser::notifySend(NaviPartId partId) {
    NAVI_LOG(SCHEDULE2, "notify send, partId: %d", partId);
    auto result = getUserResult();
    if (result) {
        result->notifyData();
    }
}

void GraphDomainUser::doFinish(ErrorCode ec) {
    setUserResult(nullptr);
}

}
