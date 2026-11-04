#pragma once

#include "navi/common.h"
#include "navi/engine/BizManager.h"
#include "navi/engine/GraphMetric.h"
#include "navi/engine/RunGraphParams.h"
#include "navi/engine/GraphResult.h"
#include "navi/log/NaviLogger.h"

namespace multi_call {
class QuerySession;
} // namespace multi_call

namespace navi {

class NaviWorkerBase;
struct ForkGraphParam;
class QuerySessionR;

struct GraphParam
{
public:
    GraphParam();
    ~GraphParam();
public:
    bool init(SessionId id,
              NaviWorkerBase *worker_,
              const NaviLoggerPtr &logger_,
              const RunGraphParams &runParams_,
              const NaviSymbolTablePtr &naviSymbolTable,
              const NaviHostInfo *hostInfo,
              const MemLeakCollectorPtr &memLeakCollector_);
    ErrorCode init(GraphId forkGraphId,
                   NaviPartId partId,
                   const std::string &forkBizName,
                   const std::string &forkNode,
                   const GraphParam &parent,
                   const std::shared_ptr<ParamTopoTreeNode> &paramTopo,
                   const MemLeakCollectorPtr &memLeakCollector_,
                   const NaviLoggerPtr &logger_,
                   const ForkGraphParam &forkParam,
                   int64_t graphEndTime);
private:
    void initTraceAppender(const RunGraphParams &runParams,
                           const RunGraphParams &parentRunParams,
                           const NaviLoggerPtr &logger);
public:
    NaviWorkerBase *worker = nullptr;
    BizManager *bizManager = nullptr;
    CreatorManager *creatorManager = nullptr;
    NaviLoggerPtr logger;
    std::shared_ptr<QuerySessionR> querySessionR;
    RunGraphParams runParams;
    MemLeakCollectorPtr memLeakCollector;
    GraphResultPtr graphResult;
    int64_t beginTime = -1;
};

}
