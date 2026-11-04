#pragma once

#include <deque>

#include "autil/CircularQueue.h"
#include "autil/WorkItem.h"

namespace autil {

class WorkItemQueue {
public:
    WorkItemQueue(){};
    virtual ~WorkItemQueue(){};
    virtual bool push(WorkItem *item) = 0;
    virtual WorkItem *pop() = 0;
    virtual size_t size() const = 0;
};

class WorkItemQueueFactory {
public:
    virtual ~WorkItemQueueFactory(){};
    virtual WorkItemQueue *create(const size_t threadNum, const size_t queueSize) = 0;
};
typedef std::shared_ptr<WorkItemQueueFactory> WorkItemQueueFactoryPtr;

class CircularPoolQueue : public WorkItemQueue {
public:
    CircularPoolQueue(const size_t queueSize) : _queue(queueSize){};
    virtual bool push(WorkItem *item) override {
        if (_queue.size() >= _queue.capacity()) {
            return false;
        }
        _queue.push_back(item);
        return true;
    }
    virtual WorkItem *pop() override {
        if (_queue.empty()) {
            return nullptr;
        }
        WorkItem *item = _queue.front();
        _queue.pop_front();
        return item;
    }
    virtual size_t size() const override { return _queue.size(); }

private:
    CircularQueue<WorkItem *> _queue;
};

class ThreadPoolQueue : public WorkItemQueue {
public:
    ThreadPoolQueue(){};
    virtual bool push(WorkItem *item) override {
        _queue.push_back(item);
        return true;
    }
    virtual WorkItem *pop() override {
        if (_queue.empty()) {
            return nullptr;
        }
        WorkItem *item = _queue.front();
        _queue.pop_front();
        return item;
    }
    virtual size_t size() const override { return _queue.size(); }

private:
    std::deque<WorkItem *> _queue;
};

class ThreadPoolQueueFactory : public WorkItemQueueFactory {
public:
    WorkItemQueue *create(const size_t threadNum, const size_t queueSize) override { return new ThreadPoolQueue(); }
};

} // namespace autil
