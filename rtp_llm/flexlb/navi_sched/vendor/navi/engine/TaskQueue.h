#pragma once
#include "navi/engine/MDPriorityQueue3.h"

#include <atomic>

#include "arpc/common/LockFreeQueue.h"
#include "autil/NoCopyable.h"
#include "navi/common.h"
#include "navi/engine/NaviThreadPool.h"

namespace navi {

class NaviWorkerBase;
class NaviThreadPool;
class ConcurrencyConfig;
class NaviThreadPoolItemBase;
class TaskQueue;

struct TaskQueueStat {
    NaviThreadPoolStat threadPoolStat;
    size_t processingCount = 0;
    size_t queueCount = 0;
    size_t processingCountRatio = 0;
    size_t queueCountRatio = 0;
};

class TaskQueueHolder : autil::NoCopyable {
public:
    TaskQueueHolder(TaskQueue *taskQueue = nullptr) : _taskQueue(taskQueue) {}
    ~TaskQueueHolder() { release(); }
    TaskQueueHolder(TaskQueueHolder &&rhs) : TaskQueueHolder(nullptr) { swap(rhs); }
    TaskQueueHolder &operator=(TaskQueueHolder &&rhs) {
        if (&rhs != this) {
            swap(rhs);
        }
        return *this;
    }

public:
    void release();

private:
    void swap(TaskQueueHolder &rhs) { std::swap(_taskQueue, rhs._taskQueue); }

private:
    TaskQueue *_taskQueue = nullptr;
};

class TaskQueueScheduleItemBase : public NaviThreadPoolItemBase {
public:
    TaskQueueScheduleItemBase();
    virtual ~TaskQueueScheduleItemBase();
    TaskQueueScheduleItemBase(const TaskQueueScheduleItemBase &) = delete;
    const TaskQueueScheduleItemBase &operator=(const TaskQueueScheduleItemBase &) = delete;

public:
    bool syncMode() const override { return _sync; }
    void setSyncMode(bool sync) { _sync = sync; }
    void setTaskQueueHolder(TaskQueueHolder holder) {
        _holder = std::move(holder);
    }

protected:
    TaskQueueHolder stealTaskQueueHolder() {
        return std::move(_holder);
    }

private:
    bool _sync = false;
    TaskQueueHolder _holder;
};

class TaskQueue {
public:
    TaskQueue();
    virtual ~TaskQueue();
    TaskQueue(const TaskQueue &) = delete;
    TaskQueue &operator=(const TaskQueue &) = delete;

public:
    bool init(const std::string &name, const ConcurrencyConfig &config, TestMode testMode);
    void stop();
    bool push(int64_t globalStartTimeUs, TaskQueueScheduleItemBase *item);
    NaviThreadPool *getThreadPool() const { return _threadPool.get(); }
    const NaviThreadPoolPtr &getThreadPoolPtr() const { return _threadPool; }
    void onFinish();
    TaskQueueStat getStat();
    size_t getProcessingCount() const { return atomic_read(&_processingCount); }

private:
    virtual void schedule(); // virtual for test
    void incProcessingCount();
    void decProcessingCount();

private:
    TestMode _testMode;
    std::string _desc;
    std::shared_ptr<NaviThreadPool> _threadPool;
    atomic64_t _processingCount;
    size_t _processingMax = 0;
    size_t _scheduleQueueMax = 0;
    md3::PriorityQueue _scheduleQueue;
};

NAVI_TYPEDEF_PTR(TaskQueue);

} // end namespace navi
