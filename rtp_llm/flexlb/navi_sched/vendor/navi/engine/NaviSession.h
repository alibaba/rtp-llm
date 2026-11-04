#pragma once

#include "navi/common.h"
#include "navi/engine/Biz.h"
#include "navi/engine/NaviWorkerBase.h"
#include "navi/engine/RunGraphParams.h"

namespace navi {

class NaviLogger;
class GraphDef;
class TraceAppender;
class Resource;
class Graph;

class NaviSession : public NaviWorkerBase
{
public:
    NaviSession(TaskQueue *taskQueue,
                BizManager *bizManager,
                GraphDef *graphDef,
                const ResourceMap *resourceMap,
                NaviUserResultClosure *closure = nullptr);
    ~NaviSession();
private:
    NaviSession(const NaviSession &);
    NaviSession &operator=(const NaviSession &);
public:
    bool doInit(GraphParam *param) override;
    void run(const std::shared_ptr<ComputeScope> &scope) override;
    void drop() override;
    void checkGraph();
private:
    bool isTimeout() const override;

protected:
    const ResourceMap *_resourceMap;
    GraphDef *_graphDef;
    Graph *_graph;
};

NAVI_TYPEDEF_PTR(NaviSession);

}
