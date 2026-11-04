#pragma once

#include "navi/common.h"
#include "navi/builder/GraphDesc.h"

namespace navi {

class GraphBuilder;
class PImpl;

class EImpl
{
public:
    EImpl();
    virtual ~EImpl();
    EImpl(const EImpl &) = delete;
    EImpl &operator=(const EImpl &) = delete;
public:
    bool init(GraphBuilder *builder, GraphId peerGraphId, P from);
    virtual SingleE *addDownStream(P to, EImpl *realEdge) = 0;
    virtual N split(const std::string &splitKernel) = 0;
    virtual N merge(const std::string &mergeKernel) = 0;
    virtual void backPressure(int64_t queueLow, int64_t queueHigh) {
    }
    GraphBuilder *getBuilder() const {
        return _builder;
    }
public:
    static EImpl *createEdge(GraphBuilder *builder, GraphId peerGraphId, P from);
    static std::string getPortName(P p);
protected:
    GraphBuilder *_builder;
    GraphId _fromGraphId;
    P _from;
    std::string _fromPort;
    GraphId _toGraphId;
    NImplBase *_split;
    NImplBase *_merge;
};

}

