#pragma once

#include "navi/common.h"
#include "navi/proto/KernelDef.pb.h"
#include "navi/proto/GraphState.pb.h"
#include "navi/util/ReadyBitMap.h"

namespace navi {

class LocalSubGraph;
class Node;
class Edge;
struct EdgeInternal;
class PartInfo;
class LocalPortBase;

struct OutputPortGroupInternal {
    NaviVector<std::pair<IndexType, EdgeInternal *>, 4> _outputs;
    std::shared_ptr<NaviUnorderedMap<IndexType, EdgeInternal *>> _outputsMapPtr;
    IndexType _outputCount = 0;
public:
    size_t getOutputCount() const;
    EdgeInternal *getEdge(size_t index) const;
    size_t getOutputDegree() const;
    void addOutput(IndexType index, EdgeInternal *edge);
    void postInit();
    InlineReadyBitMap createReadyBitMap(const PartInfo &partInfo);
    bool isOk(LocalPortBase *port, LocalSubGraph *graph) const;
    bool isEofOrFinish(LocalPortBase *port, LocalSubGraph *graph) const;
    bool isEofOrFinish(LocalPortBase *port,
                       LocalSubGraph *graph,
                       size_t index) const;
    void forceEof(LocalSubGraph *graph, const Node *callNode) const;
    bool finishInputWork(LocalSubGraph *graph, const Node *callNode);
    void fillState(LocalPortBase *port,
                   LocalSubGraph *graph,
                   KernelOutputGroupStateDef *state) const;

private:
    static bool compare(const std::pair<IndexType, EdgeInternal *> &left,
                        const std::pair<IndexType, EdgeInternal *> &right);
};

}
