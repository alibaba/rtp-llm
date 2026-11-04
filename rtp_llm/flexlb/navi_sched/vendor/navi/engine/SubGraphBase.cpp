#include "navi/engine/SubGraphBase.h"
#include "navi/engine/Graph.h"
#include "navi/log/NaviLogger.h"

namespace navi {

SubGraphBase::SubGraphBase(SubGraphDef *subGraphDef,
                           GraphParam *param,
                           const GraphDomainPtr &domain,
                           const SubGraphBorderMapPtr &borderMap,
                           Graph *graph)
    : _logger(this, "SubGraph", domain->getLogger().logger)
    , _subGraphDef(subGraphDef)
    , _param(param)
    , _domain(domain)
    , _borderMap(borderMap)
    , _graph(graph)
    , _terminated(false)
{
    _logger.addPrefix("%d", getGraphId());
}

SubGraphBase::~SubGraphBase() {
    NAVI_LOG(SCHEDULE1, "deleted");
}


GraphId SubGraphBase::getGraphId() const {
    return _subGraphDef->graph_id();
}

NaviPartId SubGraphBase::getPartCount() const {
    return _subGraphDef->location().part_info().part_count();
}

NaviPartId SubGraphBase::getPartId() const {
    return _subGraphDef->location().this_part_id();
}

const std::string &SubGraphBase::getNodeName(SubGraphBase *graph,
                                             const NodePortDef &nodePortDef) {
    if (!graph) {
        return nodePortDef.node_name();
    }
    auto id = nodePortDef.node_id();
    if (id <= 0) {
        return nodePortDef.node_name();
    }
    auto index = id - 1;
    if (unlikely(index >= graph->_subGraphDef->nodes_size())) {
        static std::string empty_string;
        return empty_string;
    } else {
        return graph->_subGraphDef->nodes(index).name();
    }
}

void SubGraphBase::terminateThis(ErrorCode ec) {
    bool expect = false;
    if (!_terminated.compare_exchange_weak(expect, true)) {
        return;
    }
    terminate(ec);
}

bool SubGraphBase::terminated() const {
    return _terminated.load(std::memory_order_relaxed);
}

void SubGraphBase::setErrorCode(ErrorCode ec) {
    _param->graphResult->setError(_logger.logger, ec);
}

const std::string &SubGraphBase::getBizName() const {
    return _subGraphDef->location().biz_name();
}

Graph *SubGraphBase::getGraph() const {
    return _graph;
}

GraphParam *SubGraphBase::getParam() const {
    return _param;
}

NaviWorkerBase *SubGraphBase::getWorker() const {
    return _param->worker;
}

}
