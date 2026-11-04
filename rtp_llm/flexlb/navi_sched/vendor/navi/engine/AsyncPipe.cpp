#include "navi/engine/AsyncPipe.h"
#include "navi/engine/Node.h"

namespace navi {

AsyncPipe::PeriodicInfo::PeriodicInfo(int64_t period,
                                      int64_t nextTrigger,
                                      int64_t repeat)
    : periodMs(period)
    , nextTriggerMs(nextTrigger)
    , remainingRepeat(repeat)
    , active(true)
{
}

void AsyncPipe::PeriodicInfo::reset(int64_t period,
                                    int64_t nextTrigger,
                                    int64_t repeat)
{
    autil::ScopedLock lock(mutex);
    active.store(false, std::memory_order_release);
    periodMs.store(period, std::memory_order_relaxed);
    nextTriggerMs.store(nextTrigger, std::memory_order_relaxed);
    activatedCount.store(0, std::memory_order_relaxed);
    remainingRepeat.store(repeat, std::memory_order_relaxed);
    stopped = false;
    active.store(true, std::memory_order_release);
}

void AsyncPipe::PeriodicInfo::snapshot(int64_t &activated,
                                       int64_t &remaining,
                                       int64_t &period,
                                       int64_t &nextTrigger) const
{
    activated = activatedCount.load(std::memory_order_relaxed);
    remaining = remainingRepeat.load(std::memory_order_relaxed);
    period = periodMs.load(std::memory_order_relaxed);
    nextTrigger = active.load(std::memory_order_acquire)
        ? nextTriggerMs.load(std::memory_order_relaxed) : -1;
}

bool AsyncPipe::PeriodicInfo::stop()
{
    autil::ScopedLock lock(mutex);
    if (stopped) {
        return false;
    }
    active.store(false, std::memory_order_release);
    stopped = true;
    return true;
}

int64_t AsyncPipe::PeriodicInfo::checkTimeout(int64_t endTimeMs,
                                              bool &activated)
{
    activated = false;
    if (0 != mutex.trylock()) {
        return -1;
    }
    if (!active.load(std::memory_order_relaxed)) {
        mutex.unlock();
        return -1;
    }
    auto nextTrigger = nextTriggerMs.load(std::memory_order_relaxed);
    if (nextTrigger <= endTimeMs) {
        activated = true;
        activatedCount.fetch_add(1, std::memory_order_relaxed);
        auto remaining = remainingRepeat.load(std::memory_order_relaxed);
        if (remaining > 0 && 0 == --remaining) {
            remainingRepeat.store(0, std::memory_order_relaxed);
            active.store(false, std::memory_order_release);
            mutex.unlock();
            return -1;
        }
        if (remaining > 0) {
            remainingRepeat.store(remaining, std::memory_order_relaxed);
        }
        auto period = periodMs.load(std::memory_order_relaxed);
        auto elapsedPeriods = (endTimeMs - nextTrigger) / period + 1;
        nextTrigger += elapsedPeriods * period;
        nextTriggerMs.store(nextTrigger, std::memory_order_relaxed);
    }
    mutex.unlock();
    return nextTrigger;
}

bool AsyncPipe::PeriodicInfo::isActive() const
{
    return active.load(std::memory_order_acquire);
}

AsyncPipe::AsyncPipe(Node *node, ActivateStrategy activateStrategy)
    : _node(node)
    , _activateStrategy(activateStrategy)
    , _terminated(false)
    , _eof(false)
{
    if (!_node) { // only for test
        return;
    }
    incItemCount();
}

AsyncPipe::~AsyncPipe() {
    delete _periodicInfo.load(std::memory_order_relaxed);
}

void AsyncPipe::terminate() {
    {
        std::list<DataPtr> dataList;
        {
            autil::ScopedLock lock(_mutex);
            std::swap(dataList, _dataList);
            if (_terminated) {
                return;
            }
            if (_nodeReadyMap) {
                _nodeReadyMap->setReadyFinish(_pipeIndex, true, true);
            }
            if (_nodeConsumedMap) {
                _nodeConsumedMap->setFinish(_pipeIndex, true);
            }
            _terminated = true;
        }
        NAVI_KERNEL_LOG(SCHEDULE1, "async pipe terminate, index[%lu], dataList count [%lu]",
                        _pipeIndex, dataList.size());
    }
    if (_node) {
        stopPeriodic();
        _node->notifyPipeTerminate();
    }
    decItemCount();
}

ErrorCode AsyncPipe::setData(const DataPtr &data) {
    return setDataOverride(data, false);
}

ErrorCode AsyncPipe::setDataOverride(
        const DataPtr &data,
        bool overrideData,
        bool ignoreData)
{
    NAVI_KERNEL_LOG(SCHEDULE1, "async pipe set data, index[%lu] data[%p].", _pipeIndex, data.get());
    std::list<DataPtr> dataList;
    {
        autil::ScopedLock lock(_mutex);
        if (_terminated) {
            NAVI_KERNEL_LOG(DEBUG, "async pipe set data when terminated, index[%lu].", _pipeIndex);
            return EC_ASYNC_TERMINATED;
        }
        if (_eof) {
            NAVI_KERNEL_LOG(ERROR,
                            "async pipe index[%lu] setData error, nodeReadyMap is finish.",
                            _pipeIndex);
            return EC_DATA_AFTER_EOF;
        }
        if (!ignoreData) {
            if (overrideData) {
                dataList.swap(_dataList);
            }
            _dataList.emplace_back(data);
        }
        _setCount++;
        _nodeReadyMap->setReady(_pipeIndex, true);
        incItemCount();
    }
    scheduleNode();
    decItemCount();
    return EC_NONE;
}

bool AsyncPipe::getData(DataPtr &data, bool &eof) {
    bool ret = false;
    // can only be called by kernel compute
    {
        autil::ScopedLock lock(_mutex);
        if (_terminated) {
            data.reset();
            eof = _eof;
            return false;
        }
        auto empty = _dataList.empty();
        data = nullptr;
        if (!empty) {
            data = _dataList.front();
            _dataList.pop_front();
            empty = _dataList.empty();
            ret = true;
            _getCount++;
        }
        eof = _eof && empty;
        _nodeReadyMap->setReady(_pipeIndex, !empty);
        _nodeConsumedMap->setFinish(_pipeIndex, eof);
        incItemCount();
        NAVI_KERNEL_LOG(SCHEDULE1, "async pipe index[%lu] get data, finish[%d] eof[%d] data[%p] ret[%d]",
                        _pipeIndex, _eof, eof, data.get(), ret);
    }
    scheduleNode();
    decItemCount();
    if (eof) {
        terminate();
    }
    return ret;
}

ErrorCode AsyncPipe::setEof() {
    NAVI_KERNEL_LOG(SCHEDULE1, "async pipe set eof, index[%lu].", _pipeIndex);
    {
        autil::ScopedLock lock(_mutex);
        if (_terminated) {
            NAVI_KERNEL_LOG(DEBUG, "async pipe set eof when terminated, index[%lu].", _pipeIndex);
            return EC_ASYNC_TERMINATED;
        }
        if (_dataList.empty()) {
            _dataList.push_back(nullptr);
        }
        _eof = true;
        _nodeReadyMap->setReadyFinish(_pipeIndex, true, true);
        incItemCount();
    }
    stopPeriodic();
    scheduleNode();
    decItemCount();
    return EC_NONE;
}

size_t AsyncPipe::getDataItemCnt() const {
    autil::ScopedLock lock(_mutex);
    return _dataList.size();
}

void AsyncPipe::switchBitMap(ReadyBitMap *newReadyMap, ReadyBitMap *newConsumedMap, size_t newPipeIndex) {
    autil::ScopedLock lock(_mutex);
    auto oldIndex = _pipeIndex;
    bool optional = _activateStrategy != AS_REQUIRED;
    {
        auto oldReadyMap = _nodeReadyMap;
        auto oldReadySize = oldReadyMap ? oldReadyMap->getBitSize() : 0;
        bool ready = false;
        if (oldIndex < oldReadySize) {
            ready = oldReadyMap->isReady(oldIndex);
        }
        // do not use setReadyFinish when copy finish bit
        newReadyMap->setFinish(newPipeIndex, _eof);
        newReadyMap->setOptional(newPipeIndex, optional);
        newReadyMap->setReady(newPipeIndex, ready);
    }
    {
        auto oldConsumedMap = _nodeConsumedMap;
        auto oldConsumedSize = oldConsumedMap ? oldConsumedMap->getBitSize() : 0;
        bool consumedFinish = false;
        if (oldIndex < oldConsumedSize) {
            consumedFinish = oldConsumedMap->isFinish(oldIndex);
        }
        newConsumedMap->setFinish(newPipeIndex, consumedFinish);
        newConsumedMap->setOptional(newPipeIndex, optional);
    }
    _nodeReadyMap = newReadyMap;
    _nodeConsumedMap = newConsumedMap;
    _pipeIndex = newPipeIndex;
}

bool AsyncPipe::canRecycle() const {
    return _terminated || (_nodeConsumedMap && _nodeConsumedMap->isFinish(_pipeIndex));
}

bool AsyncPipe::registerPeriodic(int64_t periodMs,
                                 int64_t nextTriggerMs,
                                 int64_t repeat)
{
    PeriodicInfo *info = nullptr;
    {
        autil::ScopedLock lock(_mutex);
        if (_terminated || _eof) {
            return false;
        }
        info = _periodicInfo.load(std::memory_order_relaxed);
        if (!info) {
            info = new PeriodicInfo(periodMs, nextTriggerMs, repeat);
            _periodicInfo.store(info, std::memory_order_release);
            return true;
        }
        info->reset(periodMs, nextTriggerMs, repeat);
    }
    return true;
}

bool AsyncPipe::getPeriodicActivationCount(int64_t &activatedCount,
                                           int64_t &remainingCount,
                                           int64_t &periodMs,
                                           int64_t &nextTriggerMs) const
{
    auto info = _periodicInfo.load(std::memory_order_acquire);
    if (!info) {
        return false;
    }
    info->snapshot(
        activatedCount, remainingCount, periodMs, nextTriggerMs);
    return true;
}

bool AsyncPipe::stopPeriodic()
{
    auto info = _periodicInfo.load(std::memory_order_acquire);
    if (!info) {
        return false;
    }
    return info->stop();
}

int64_t AsyncPipe::checkPeriodicTimeout(int64_t endTimeMs, bool &activated)
{
    auto info = _periodicInfo.load(std::memory_order_acquire);
    if (!info) {
        activated = false;
        return -1;
    }
    return info->checkTimeout(endTimeMs, activated);
}

bool AsyncPipe::periodicActive() const
{
    auto info = _periodicInfo.load(std::memory_order_acquire);
    if (!info) {
        return false;
    }
    return info->isActive();
}

void AsyncPipe::incItemCount() {
    auto *work = _node->getGraph()->getParam()->worker;
    work->incItemCount();
}

void AsyncPipe::decItemCount() {
    auto *work = _node->getGraph()->getParam()->worker;
    work->decItemCount();
}

void AsyncPipe::scheduleNode() const {
    _node->schedule(_node);
}

void AsyncPipe::fillState(KernelSingleAsyncPipeStateDef *state) const {
    autil::ScopedLock lock(_mutex);
    state->set_index(_pipeIndex);
    state->set_ready(_nodeReadyMap->isReady(_pipeIndex));
    state->set_finish(_nodeReadyMap->isFinish(_pipeIndex));
    state->set_optional(_nodeReadyMap->isOptional(_pipeIndex));
    state->set_strategy((ActivateStrategyDef)_activateStrategy);
    state->set_terminated(_terminated);
    state->set_set_count(_setCount);
    state->set_get_count(_getCount);
}

} // namespace navi
