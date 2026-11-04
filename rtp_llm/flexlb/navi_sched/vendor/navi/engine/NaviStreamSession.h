#pragma once

#include "navi/engine/NaviWorkerBase.h"
#include "navi/engine/Graph.h"
#include <multi_call/stream/GigStreamBase.h>

namespace navi {

class NaviMessage;

typedef std::map<GraphId, std::string> MirrorGraphInfoMap;

class NaviStreamSession : public NaviWorkerBase
{
public:
    NaviStreamSession(TaskQueue *taskQueue,
                      BizManager *bizManager,
                      NaviMessage *request,
                      const ArenaPtr &arena,
                      const multi_call::GigStreamBasePtr &stream);
    ~NaviStreamSession();
private:
    NaviStreamSession(const NaviStreamSession &);
    NaviStreamSession &operator=(const NaviStreamSession &);
public:
    bool doInit(GraphParam *param) override;
    void run(const std::shared_ptr<ComputeScope> &scope) override;
    void drop() override;
private:
    bool decompressGraph(bool collectMalloc);
    void addMirrorGraph(MirrorGraphInfoMap &mirrorInfoMap);
    SubGraphDef *getNewSubGraphDef(GraphId graphId,
                                   NaviUnorderedMap<GraphId, SubGraphDef *> &graphMap);
    void mirrorBorder(GraphId graphId, const BorderDef &borderDef,
                      SubGraphDef *subGraphDef);
    void initDomainMap(const MirrorGraphInfoMap &mirrorInfoMap,
                       const multi_call::GigStreamBasePtr &stream,
                       Graph *graph);
    bool isTimeout() const override;

private:
    ArenaPtr _arena;
    NaviMessage *_request;
    multi_call::GigStreamBasePtr _stream;
    GraphDef *_graphDef;
    Graph *_graph;
};

}
