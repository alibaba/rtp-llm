#pragma once

#include "navi/common.h"
#include "navi/engine/RunGraphParams.h"
#include "navi/log/NaviLogger.h"
#include "navi/perf/NaviPerfResult.h"
#include <arpc/common/LockFreeQueue.h>

namespace navi {

class NaviPerfWorkerBase;

typedef arpc::common::LockFreeQueue<NaviPerfResultPtr> NaviPerfResultQueue;

class NaviPerfThread {
public:
    NaviPerfThread(pid_t pid);
    ~NaviPerfThread();
public:
    bool init(const NaviLoggerPtr &logger);
    std::vector<NaviPerfWorkerBase *> getWorkers() const;
    CollectFlags setCollectState(const CollectFlags &collectFlags);
    bool beginSample();
    NaviPerfResultPtr endSample();
    int32_t collectResultEntry(NaviPerfEventEntry *entry);
private:
    NaviPerfResultPtr getResult() const;
    void pushQueue(NaviPerfResultQueue &queue,
                   const NaviPerfResultPtr &newResult);
    NaviPerfResultPtr popQueue(NaviPerfResultQueue &queue) const;
    void collectSubResult();
private:
    DECLARE_LOGGER();
    pid_t _pid;
    std::vector<std::shared_ptr<NaviPerfWorkerBase>> _workers;
    arpc::common::LockFreeQueue<NaviPerfResultPtr> _resultQueue;
    arpc::common::LockFreeQueue<NaviPerfResultPtr> _mainQueue;
    NaviPerfResultPtr _resultQueueTail;
    NaviPerfResultPtr _mainQueueHead;
    NaviPerfResultPtr _resultEndList;
};

}
