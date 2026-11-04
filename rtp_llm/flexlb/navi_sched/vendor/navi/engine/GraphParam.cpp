#include "navi/engine/GraphParam.h"
#include "navi/engine/NaviWorkerBase.h"
#include "navi/engine/KernelComputeContext.h"
#include "navi/engine/ForkGraphResult.h"
#include "navi/engine/ParamTopoTreeNode.h"
#include "navi/log/TraceAppender.h"

namespace navi {

GraphParam::GraphParam() {
}

GraphParam::~GraphParam() {
}

bool GraphParam::init(SessionId id,
                      NaviWorkerBase *worker_,
                      const NaviLoggerPtr &logger_,
                      const RunGraphParams &runParams_,
                      const NaviSymbolTablePtr &naviSymbolTable,
                      const NaviHostInfo *hostInfo,
                      const MemLeakCollectorPtr &memLeakCollector_)
{
    worker = worker_;
    bizManager = worker->getBizManager();
    creatorManager = bizManager->getBuildinCreatorManager().get();
    logger = logger_;
    runParams = runParams_;
    memLeakCollector = memLeakCollector_;
    runParams.setSessionId(id);
    runParams.initGlobalStartTimeUs();
    runParams.initTopoParam();
    graphResult = std::make_shared<GraphResult>();
    graphResult->init(logger,
                      runParams.getCollectFlags(),
                      runParams.getTopoParam(),
                      runParams.getTraceFormatPattern(),
                      naviSymbolTable,
                      hostInfo);
    return true;
}

ErrorCode GraphParam::init(GraphId forkGraphId,
                           NaviPartId partId,
                           const std::string &forkBizName,
                           const std::string &forkNode,
                           const GraphParam &parent,
                           const std::shared_ptr<ParamTopoTreeNode> &paramTopo,
                           const MemLeakCollectorPtr &memLeakCollector_,
                           const NaviLoggerPtr &logger_,
                           const ForkGraphParam &forkParam,
                           int64_t graphEndTime)
{
    NAVI_KERNEL_LOG(TRACE3,
                    "fork param, collectMetric: %d, collectPerf: %d, errorAsEof: %d, timeoutMs: %ld",
                    forkParam.collectMetric,
                    forkParam.collectPerf,
                    forkParam.errorAsEof,
                    forkParam.timeoutMs);
    worker = parent.worker;
    bizManager = parent.bizManager;
    creatorManager = parent.creatorManager;
    logger = std::make_shared<NaviLogger>(*logger_, false);
    querySessionR = parent.querySessionR;
    runParams = parent.runParams;
    memLeakCollector = memLeakCollector_;
    runParams.setParamNode(paramTopo);
    if (forkParam.collectMetric) {
        runParams.setCollectMetric(true);
    }
    if (forkParam.collectPerf) {
        runParams.setCollectPerf(true);
    }
    beginTime = autil::TimeUtility::currentTime();
    runParams.clearNamedData();
    runParams.clearOverrideDatas();
    runParams.setTimeoutMs(std::min(forkParam.timeoutMs, std::max(0l, (graphEndTime - beginTime)) / FACTOR_MS_TO_US));
    runParams.initGlobalStartTimeUs();
    runParams.initTopoParam();
    initTraceAppender(runParams, parent.runParams, logger);
    graphResult =
        std::make_shared<ForkGraphResult>(forkGraphId, forkBizName, forkNode);
    graphResult->init(logger,
                      runParams.getCollectFlags(),
                      runParams.getTopoParam(),
                      runParams.getTraceFormatPattern(),
                      parent.graphResult->getGraphMetric()->getNaviSymbolTable(),
                      parent.graphResult->getHostInfo());
    for (const auto &overrideData : forkParam.overrideDataVec) {
        if (!runParams.overrideEdgeData(overrideData)) {
            NAVI_KERNEL_LOG(
                ERROR,
                "override edge data failed, graphId [%d], partId [%d], outputNode [%s], outputPort [%s], data "
                "count [%lu]",
                overrideData.graphId,
                overrideData.partId,
                overrideData.outputNode.c_str(),
                overrideData.outputPort.c_str(),
                overrideData.datas.size());
            return EC_OVERRIDE_EDGE_DATA;
        }
    }
    for (const auto &namedData : forkParam.namedDataVec) {
        if (!runParams.addNamedData(namedData)) {
            NAVI_KERNEL_LOG(
                ERROR, "add named data failed, graphId [%d], name [%s]", namedData.graphId, namedData.name.c_str());
            return EC_ADD_NAMED_DATA;
        }
    }
    return EC_NONE;
}

void GraphParam::initTraceAppender(const RunGraphParams &runParams,
                                   const RunGraphParams &parentRunParams,
                                   const NaviLoggerPtr &logger)
{
    const auto &topoParam = runParams.getTopoParam();
    auto traceLevel = LOG_LEVEL_DISABLE;
    if (unlikely(topoParam != nullptr)) {
        traceLevel = getLevelByString(topoParam->_traceLevel);
        if (LOG_LEVEL_DISABLE == traceLevel) {
            traceLevel = parentRunParams.getTraceLevelWithoutTopo();
        }
    } else {
        traceLevel = parentRunParams.getTraceLevelWithoutTopo();
    }
    auto traceAppender = std::make_shared<TraceAppender>(traceLevel);
    runParams.fillFilterParam(traceAppender.get());
    logger->setTraceAppender(traceAppender);
}
}
