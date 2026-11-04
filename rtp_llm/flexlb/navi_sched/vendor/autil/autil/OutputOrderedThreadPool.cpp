#include "autil/OutputOrderedThreadPool.h"

#include <cstddef>
#include <exception>

#include "autil/Log.h"

using namespace std;

namespace autil {
AUTIL_DECLARE_AND_SETUP_LOGGER(autil, OutputOrderedThreadPool);

OutputOrderedThreadPool::OutputOrderedThreadPool(uint32_t threadNum, uint32_t queueSize)
    : _queue(queueSize), _threadPool(threadNum, queueSize) {
    _waitItem = NULL;
    _running = false;
}

OutputOrderedThreadPool::~OutputOrderedThreadPool() { waitStop(ThreadPool::STOP_AND_CLEAR_QUEUE_IGNORE_EXCEPTION); }

bool OutputOrderedThreadPool::start(const string &name) {
    AUTIL_LOG(INFO, "OutputOrderedThreadPool begin start");
    if (_queue.capacity() < 1) {
        return false;
    }

    _running = _threadPool.start(name);
    return _running;
}

bool OutputOrderedThreadPool::pushWorkItem(WorkItem *item) {
    if (!item || !_running)
        return false;
    OrderWorkItem *oItem = NULL;
    {
        ScopedLock lock(_queueCond);
        while (_queue.size() == _queue.capacity()) {
            _queueCond.producerWait();
            if (!_running)
                return false;
        }
        if (!_running) {
            return false;
        }
        _queue.push_back(OrderWorkItem(item, this));
        oItem = &_queue.back();
        _waitItemCount.inc();
    }
    // assert _threadPool is running
    _threadPool.pushWorkItem(oItem, true);
    return true;
}

WorkItem *OutputOrderedThreadPool::popWorkItem() {
    ScopedLock lock(_queueCond);
    while (_queue.empty() || !_queue.front()._processed) {
        if (_queue.empty()) {
            if (!_running) {
                return NULL;
            }
            _waitItem = NULL;
        } else {
            _waitItem = &_queue.front();
        }
        _queueCond.consumerWait();
    }

    WorkItem *item = _queue.front()._item;
    _queue.pop_front();
    _outputItemCount.dec();
    _queueCond.signalProducer();
    return item;
}

bool OutputOrderedThreadPool::waitStop(ThreadPool::STOP_TYPE stopType) {
    _running = false;
    _threadPool.stop(stopType);
    ScopedLock lock(_queueCond);
    std::exception_ptr firstException;
    if (stopType == ThreadPool::STOP_AND_CLEAR_QUEUE_IGNORE_EXCEPTION || stopType == ThreadPool::STOP_AND_CLEAR_QUEUE) {
        while (!_queue.empty()) {
            auto item = _queue.front()._item;
            _queue.pop_front();
            try {
                item->drop();
            } catch (...) {
                if (stopType != ThreadPool::STOP_AND_CLEAR_QUEUE_IGNORE_EXCEPTION && !firstException) {
                    firstException = std::current_exception();
                } else {
                    AUTIL_LOG(ERROR, "Work item drop exception ignore");
                }
            }
        }
    }
    _queueCond.broadcastProducer();
    _queueCond.broadcastConsumer();
    if (firstException) {
        std::rethrow_exception(firstException);
    }
    return true;
}

} // namespace autil
