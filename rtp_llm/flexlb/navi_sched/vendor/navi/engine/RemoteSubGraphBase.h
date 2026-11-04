#pragma once

#include "navi/engine/SubGraphBase.h"

namespace navi {

class RemoteSubGraphBase : public SubGraphBase
{
public:
    RemoteSubGraphBase(SubGraphDef *subGraphDef,
                       GraphParam *param,
                       const GraphDomainPtr &domain,
                       const SubGraphBorderMapPtr &borderMap,
                       Graph *graph);
    ~RemoteSubGraphBase();
private:
    RemoteSubGraphBase(const RemoteSubGraphBase &);
    RemoteSubGraphBase &operator=(const RemoteSubGraphBase &);
public:
    ErrorCode init() override;
    ErrorCode run() override;
    void terminate(ErrorCode ec) override;
};

}
