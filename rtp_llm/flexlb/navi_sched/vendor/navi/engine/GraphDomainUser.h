#pragma once

#include "navi/engine/GraphDomain.h"
#include <multi_call/interface/SyncClosure.h>

namespace navi {

class NaviUserResult;

class GraphDomainUser : public GraphDomain
{
public:
    GraphDomainUser(Graph *graph);
    ~GraphDomainUser();
private:
    GraphDomainUser(const GraphDomainUser &);
    GraphDomainUser &operator=(const GraphDomainUser &);
public:
    bool run() override;
    void notifySend(NaviPartId partId) override;
    void doFinish(ErrorCode ec) override;
    void setUserResult(const std::shared_ptr<NaviUserResult> &result);
    std::shared_ptr<NaviUserResult> getUserResult();
private:
    mutable autil::ReadWriteLock _resultLock;
    std::shared_ptr<NaviUserResult> _result;
};

NAVI_TYPEDEF_PTR(GraphDomainUser);

}
