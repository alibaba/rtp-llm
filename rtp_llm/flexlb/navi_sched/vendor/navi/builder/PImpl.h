#pragma once

#include "navi/builder/GraphDesc.h"
#include "navi/common.h"

namespace navi {

class NImplBase;

class PImpl
{
public:
    PImpl(NImplBase *node, const std::string &port, bool input);
    ~PImpl();
    PImpl(const PImpl &) = delete;
    PImpl &operator=(const PImpl &) = delete;
public:
    NImplBase *getNode() const;
    const std::string &getNodeName() const;
    int32_t getNodeId() const;
    const std::string &getPort() const;
    GraphId graphId() const;
    E from(IndexType thisIndex, P peer);
    E to(IndexType thisIndex, P peer, EImpl *realEdge);
    E fromForkNodeInput(IndexType thisIndex, const std::string &input, IndexType index);
    E toForkNodeOutput(IndexType thisIndex, const std::string &output, IndexType index);
    E asGraphOutput(IndexType thisIndex, const std::string &name);
    void setConnected(IndexType inputIndex, P from, const E &e);
    void setRealPort(P real);
    P getRealPort() const;
public:
    P asGroup();
    P next(IndexType index);
    P index(IndexType index);
    P autoNext();
    std::vector<std::pair<P, E>> getUpStreams();
    bool hasConnected() const;
    bool isGroup() const;

private:
    static uint64_t getEdgeKey(IndexType index, GraphId graphId) {
        return (((uint64_t)index) << 32) + ((uint32_t)graphId);
    }
public:
    friend std::ostream &operator<<(std::ostream &os, const P &p);
private:
    NImplBase *_node;
    const std::string &_nodeName;
    std::string _port;
    bool _input;
    bool _group;
    IndexType _currentIndex;
    std::vector<std::pair<P, E>> _defaultInputs;
    NaviUnorderedMap<IndexType, std::pair<P, E>> _connectedInputSet;
    NaviUnorderedMap<uint64_t, EImpl *> _edgeMap;
    P _realPort;
};

}
