#pragma once

#include "navi/engine/Edge.h"
#include "navi/engine/GraphDomain.h"

namespace navi {

class Node;

class GraphDomainFork : public GraphDomain
{
public:
    GraphDomainFork(Graph *graph, const BizPtr &biz, Node *node,
                    NaviPartId partCount, NaviPartId partId, bool errorAsEof);
    ~GraphDomainFork();
private:
    GraphDomainFork(const GraphDomainFork &);
    GraphDomainFork &operator=(const GraphDomainFork &);
public:
    bool postInit() override;
    bool run() override;
    void notifySend(NaviPartId partId) override;
    void doFinish(ErrorCode ec) override;
public:
    BizPtr getBiz() const;
    bool bindGraphEdgeList(Edge *edgeVec, int32_t edgeCount);
    Edge *getInputPeer(const std::string &portName) const;
    std::vector<Edge *> getEdgesByPrefix(const std::string &prefix) const;
    std::pair<Edge *, IndexType>
    getOutputPeer(const std::string &portName) const;
    bool checkBind() const;
    void flushData() const;
    NaviPartId getPartCount() const {
        return _partCount;
    }
    NaviPartId getPartId() const {
        return _partId;
    }
private:
    bool initPortDataType();
    void initPartCount();
private:
    BizPtr _biz;
    Node *_node;
    NaviPartId _partCount;
    NaviPartId _partId;
    NaviUnorderedMap<std::string, Edge *> _inputEdgeMap;
    NaviUnorderedMap<std::string, std::pair<Edge *, IndexType>> _outputEdgeMap;
    bool _errorAsEof;
};

}
