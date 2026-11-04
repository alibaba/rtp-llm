#include "navi/engine/NaviSession.h"
#include "autil/StringUtil.h"
#include "navi/engine/Graph.h"
#include "navi/engine/ComputeScope.h"
#include "navi/rpc_server/ArpcGraphClosure.h"
#include "navi/log/NaviLogger.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"

namespace navi {

NaviSession::NaviSession(TaskQueue *taskQueue,
                         BizManager *bizManager,
                         GraphDef *graphDef,
                         const ResourceMap *resourceMap,
                         NaviUserResultClosure *closure)
    : NaviWorkerBase(taskQueue, bizManager, closure)
    , _resourceMap(resourceMap)
    , _graphDef(graphDef)
    , _graph(nullptr)
{
    auto graphClosure = dynamic_cast<ArpcGraphClosure *>(closure);
    if (graphClosure) {
        _peerIp = graphClosure->getClientIp();
    }
}

NaviSession::~NaviSession() {
    NAVI_LOG(SCHEDULE1, "destructed");
    clear();
    // todo: fill results
    {
        NaviLoggerScope scope(_logger);
        if (unlikely(_collectFlags.collectMalloc())) {
            _graph->collectMetric();
        }
        delete _graph;
    }
    multi_call::freeProtoMessage(_graphDef);
}

bool NaviSession::doInit(GraphParam *param) {
    _logger.addPrefix("local");
    _graph = new Graph(param, nullptr, true);
    _graph->setResourceMap(_resourceMap);
    NAVI_LOG(SCHEDULE1, "construct");
    return true;
}

void NaviSession::run(const std::shared_ptr<ComputeScope> &scope) {
    NAVI_LOG(SCHEDULE1, "start");
    auto ec = _graph->init(_graphDef);
    if (EC_NONE != ec) {
        if (scope) {
            scope->setErrorCode(ec);
            scope->report();
        }
        _graph->notifyFinishEc(ec);
        return;
    }
    if (!getUserResult()->hasOutput()) {
        if (scope) {
            scope->setErrorCode(EC_NO_GRAPH_OUTPUT);
            scope->report();
        }
        NAVI_LOG(ERROR, "graph has no output");
        _graph->notifyFinishEc(EC_NO_GRAPH_OUTPUT);
        return;
    }
    ec = _graph->run();
    if (EC_NONE != ec) {
        if (scope) {
            scope->setErrorCode(ec);
            scope->report();
        }
        _graph->notifyFinishEc(ec);
        return;
    }
}

void NaviSession::checkGraph() {
    auto ec = _graph->init(_graphDef);
    if (EC_NONE != ec) {
        _graph->notifyFinishEc(ec);
    }
    ec = _graph->run();
    if (EC_NONE != ec) {
        _graph->notifyFinishEc(ec);
    }
}

void NaviSession::drop() {
    NAVI_LOG(SCHEDULE1, "dropped");
    auto ec = EC_QUEUE_FULL;
    if (_graph->isTimeout()) {
        ec = EC_TIMEOUT;
    }
    _graph->notifyFinishEc(ec);
}

bool NaviSession::isTimeout() const {
    auto timeoutChecker = _graph->getTimeoutChecker();
    if (timeoutChecker->timeout()) {
        auto curTime = autil::TimeUtility::currentTime();
        auto beginTime = timeoutChecker->beginTime();
        NAVI_LOG(ERROR,
                 "session timeout, expect time [%ld ms] use time [%ld ms], start "
                 "[%ld], end[%ld]",
                 timeoutChecker->timeoutMs(),
                 (curTime - beginTime) / FACTOR_MS_TO_US,
                 beginTime / FACTOR_MS_TO_US,
                 curTime / FACTOR_MS_TO_US);
        NAVI_LOG(SCHEDULE1, "timeout bt: %s",
                 _logger.logger->firstErrorEvent()
                     ? _logger.logger->firstErrorEvent()->bt.c_str()
                     : "");
        return true;
    }
    return false;
}

}
