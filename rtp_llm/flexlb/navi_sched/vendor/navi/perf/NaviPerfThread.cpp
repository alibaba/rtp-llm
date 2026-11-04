#include "navi/perf/NaviPerfThread.h"
#include "navi/perf/NaviPerfWorker.h"
#include "navi/log/NaviLogger.h"
#include "navi/util/CommonUtil.h"
#include <cxxabi.h>
#include <execinfo.h>
#include <sys/epoll.h>
#include <unistd.h>

namespace navi {

NaviPerfThread::NaviPerfThread(pid_t pid)
    : _pid(pid)
{
}

NaviPerfThread::~NaviPerfThread() {
    auto result = _resultEndList;
    while (result) {
        result = result->stealPrev();
    }
}

bool NaviPerfThread::init(const NaviLoggerPtr &logger) {
    _logger.object = this;
    _logger.logger = logger;
    _logger.addPrefix("navi perf thread [%d]", _pid);
    _workers.clear();
    bool hasSuccess = false;
    {
        auto worker = std::make_shared<NaviPerfWorker>(_pid, this);
        if (!worker->init(logger)) {
            NAVI_LOG(ERROR, "navi perf worker init failed, ignored");
            _workers.push_back(nullptr);
        } else {
            hasSuccess = true;
            _workers.push_back(worker);
        }
    }
    {
        auto worker = std::make_shared<NaviContextSwitchWorker>(_pid, this);
        if (!worker->init(logger)) {
            NAVI_LOG(ERROR, "navi perf context switch worker init failed, ignored");
            _workers.push_back(nullptr);
        } else {
            hasSuccess = true;
            _workers.push_back(worker);
        }
    }
    {
        auto worker = std::make_shared<NaviMajorPageFaultWorker>(_pid, this);
        if (!worker->init(logger)) {
            NAVI_LOG(ERROR, "navi perf major page fault worker init failed, ignored");
            _workers.push_back(nullptr);
        } else {
            hasSuccess = true;
            _workers.push_back(worker);
        }
    }
    {
        auto worker = std::make_shared<NaviMinorPageFaultWorker>(_pid, this);
        if (!worker->init(logger)) {
            NAVI_LOG(ERROR, "navi perf minor page fault worker init failed, ignored");
            _workers.push_back(nullptr);
        } else {
            hasSuccess = true;
            _workers.push_back(worker);
        }
    }
    return hasSuccess;
}

std::vector<NaviPerfWorkerBase *> NaviPerfThread::getWorkers() const {
    std::vector<NaviPerfWorkerBase *> workers;
    for (const auto &worker : _workers) {
        if (worker) {
            workers.push_back(worker.get());
        }
    }
    return workers;
}

CollectFlags NaviPerfThread::setCollectState(const CollectFlags &collectFlags) {
    CollectFlags stateFlags;
    {
        const auto &worker = _workers[0];
        if (worker) {
            if (collectFlags.collectPerf()) {
                auto ok = worker->enable();
                stateFlags.setCollectPerf(ok);
            } else {
                if (!worker->disable()) {
                    stateFlags.setCollectPerf(true);
                }
            }
        }
    }
    {
        const auto &worker = _workers[1];
        if (worker) {
            if (collectFlags.collectContextSwitch()) {
                auto ok = worker->enable();
                stateFlags.setCollectContextSwitch(ok);
            } else {
                if (!worker->disable()) {
                    stateFlags.setCollectContextSwitch(true);
                }
            }
        }
    }
    {
        const auto &worker = _workers[2];
        if (worker) {
            if (collectFlags.collectPageFault()) {
                auto ok = worker->enable();
                stateFlags.setCollectPageFault(ok);
            } else {
                if (!worker->disable()) {
                    stateFlags.setCollectPageFault(true);
                }
            }
        }
    }
    {
        const auto &worker = _workers[3];
        if (worker) {
            if (collectFlags.collectPageFault()) {
                auto ok = worker->enable();
                stateFlags.setCollectPageFault(ok);
            } else {
                if (!worker->disable()) {
                    stateFlags.setCollectPageFault(true);
                }
            }
        }
    }
    return stateFlags;
}

bool NaviPerfThread::beginSample() {
    NaviPerfResultPtr newResult(new NaviPerfResult());
    pushQueue(_resultQueue, newResult);
    newResult->setPrev(_resultEndList);
    _resultEndList = newResult;
    NAVI_LOG(SCHEDULE2, "begin thread sample");
    return true;
}

NaviPerfResultPtr NaviPerfThread::endSample() {
    if (!_resultEndList) {
        return nullptr;
    }
    auto retResult = _resultEndList;
    _resultEndList = _resultEndList->stealPrev();
    if (retResult) {
        retResult->stop();
    }
    NAVI_LOG(SCHEDULE2, "end thread sample, %lu", retResult->getEventCount());
    return retResult;
}

void NaviPerfThread::pushQueue(
    arpc::common::LockFreeQueue<NaviPerfResultPtr> &queue,
    const NaviPerfResultPtr &newResult)
{
    queue.Push(newResult);
}

NaviPerfResultPtr NaviPerfThread::popQueue(
    arpc::common::LockFreeQueue<NaviPerfResultPtr> &queue) const
{
    NaviPerfResultPtr ret;
    if (queue.Pop(&ret)) {
        return ret;
    } else {
        return nullptr;
    }
}

void NaviPerfThread::collectSubResult() {
    NaviPerfResultPtr current = _resultQueueTail;
    auto next = popQueue(_resultQueue);
    while (next) {
        if (!current || !current->addSubResult(next)) {
            pushQueue(_mainQueue, next);
            current = next;
        }
        next = popQueue(_resultQueue);
    }
    _resultQueueTail = current;
}

int32_t NaviPerfThread::collectResultEntry(NaviPerfEventEntry *entry) {
    int32_t ret = 0;
    collectSubResult();
    NaviPerfResultPtr current;
    if (!_mainQueueHead) {
        current = popQueue(_mainQueue);
    } else {
        current = _mainQueueHead;
    }
    if (!current) {
        free(entry);
        return ret;
    }
    while (current) {
        ret = current->addEntry(entry);
        if (-1 == ret) {
            // entry too old
            NAVI_LOG(SCHEDULE2, "entry dropped, time [%lu]",
                     NaviPerfResult::getSampleTime(entry));
            free(entry);
            entry = nullptr;
            break;
        } else if (0 == ret) {
            // fail
            current = popQueue(_mainQueue);
        } else if (1 == ret) {
            // success
            entry = nullptr;
            break;
        } else {
            assert(false);
            break;
        }
    }
    if (entry) {
        free(entry);
    }
    _mainQueueHead = current;
    return ret;
}

}
