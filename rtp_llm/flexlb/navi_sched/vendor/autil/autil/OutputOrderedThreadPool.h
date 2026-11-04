#pragma once

#include <memory>
#include <stddef.h>
#include <stdint.h>
#include <string>

#include "autil/AtomicCounter.h"
#include "autil/CircularQueue.h"
#include "autil/Lock.h"
#include "autil/ThreadPool.h"
#include "autil/WorkItem.h"

namespace autil {

class OutputOrderedThreadPool {
public:
    OutputOrderedThreadPool(uint32_t threadNum, uint32_t queueSize);
    virtual ~OutputOrderedThreadPool();

public:
    bool waitStop(ThreadPool::STOP_TYPE stopType = ThreadPool::STOP_AFTER_QUEUE_EMPTY);
    bool start(const std::string &name = "OutputOrderedThreadPool");
    virtual bool pushWorkItem(WorkItem *item);
    virtual WorkItem *popWorkItem();

public:
    uint32_t getWaitItemCount() { return _waitItemCount.getValue(); }
    uint32_t getOutputItemCount() { return _outputItemCount.getValue(); }

private:
    OutputOrderedThreadPool(const OutputOrderedThreadPool &);
    OutputOrderedThreadPool &operator=(const OutputOrderedThreadPool &);

    void afterProcess();

private:
    class OrderWorkItem;

    void signalProcessed(OrderWorkItem *item);
    class OrderWorkItem : public WorkItem {
    public:
        OrderWorkItem(WorkItem *item = NULL, OutputOrderedThreadPool *pool = NULL) {
            _item = item;
            _pool = pool;
            _processed = false;
        }

        void process() {
            _item->process();
            _pool->afterProcess();
            _processed = true;
            _pool->signalProcessed(this);
        }

        void drop() { process(); }
        void destroy() {}

        WorkItem *_item;
        OutputOrderedThreadPool *_pool;
        volatile bool _processed;
    };

private:
    mutable ProducerConsumerCond _queueCond;
    CircularQueue<OrderWorkItem> _queue;
    ThreadPool _threadPool;
    OrderWorkItem *_waitItem;
    AtomicCounter _waitItemCount;
    AtomicCounter _outputItemCount;
    volatile bool _running;
};

inline void OutputOrderedThreadPool::afterProcess() {
    _outputItemCount.inc();
    _waitItemCount.dec();
}

inline void OutputOrderedThreadPool::signalProcessed(OrderWorkItem *item) {
    ScopedLock lock(_queueCond);
    if (_waitItem == NULL || _waitItem == item) {
        _queueCond.signalConsumer();
    }
}

typedef std::shared_ptr<OutputOrderedThreadPool> OutputOrderedThreadPoolPtr;

} // namespace autil
