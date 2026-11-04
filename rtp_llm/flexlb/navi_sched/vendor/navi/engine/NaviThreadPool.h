#pragma once

#include "navi/common.h"
#include "navi/config/NaviConfig.h"
#include "navi/engine/ScheduleInfo.h"
#include "navi/engine/PrioritySched.h"
#include "navi/util/CommonUtil.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/atomic.h"
#include <arpc/common/LockFreeQueue.h>
#include <autil/Lock.h>
#include <autil/Thread.h>
#include <vector>
#include <limits>
#include "navi/engine/ThreadPoolScheduleQueue.h"

namespace navi {

class NaviThreadPoolItemBase;
class NaviThreadPool;
thread_local extern int32_t INLINE_DEPTH_TLS;
thread_local extern NaviThreadPool *current_thread_pool;
thread_local extern int64_t thread_pool_signal_cache_count;
thread_local extern size_t current_thread_id;
thread_local extern arpc::common::LockFreeQueue<NaviThreadPoolItemBase *> *inline_item_queue;
thread_local extern arpc::common::LockFreeQueue<NaviThreadPoolItemBase *> *tail_item_queue;
constexpr int32_t INVALID_INLINE_DEPTH = -1;
constexpr int32_t INVALID_SIGNAL_CACHE_COUNT = -1;

enum ThreadStat {
    TS_RUNNING,
    TS_WAIT,
    TS_WAKEUP,
};

struct NaviThreadPoolStat {
    size_t activeThreadCount = 0;
    size_t activeThreadQueueSize = 0;
    size_t idleThreadQueueSize = 0;
};

struct NaviThread {
    NaviThread()
        : tid(-1)
        , stat(TS_RUNNING)
    {
    }
    pid_t tid;
    autil::ThreadPtr thread;
    autil::ThreadCond cond;
    volatile ThreadStat stat;
} __attribute__((aligned(64)));

class NaviThreadPoolItemBase
{
public:
    NaviThreadPoolItemBase() {
        initSchedInfo();
    }
    virtual ~NaviThreadPoolItemBase() {
    }
public:
    virtual void process() = 0;
    virtual void destroy() = 0;
    virtual bool syncMode() const {
        return false;
    }
public:
    void initSchedInfo() {
        _schedInfo.enqueueTime = CommonUtil::getTimelineTimeNs();
        _schedInfo.dequeueTime = _schedInfo.enqueueTime;
        _schedInfo.schedTid = current_thread_id;
    }
    bool inlineCompute() const {
        return _inlineCompute;
    }
    void setInlineCompute(bool value) {
        _inlineCompute = value;
    }
    bool tailCompute() const {
        return _tailCompute;
    }
    void setTailCompute(bool value) {
        _tailCompute = value;
    }
    void setSignalTid(int64_t signalTid) {
        _schedInfo.signalTid = signalTid;
    }
    void setQueueSize(size_t queueSize) {
        _schedInfo.queueSize = queueSize;
    }
    void setRunningThread(int64_t count) {
        _schedInfo.runningThread = count;
    }
    void setScheduleInfo(int64_t dequeueTime, int64_t runningThread,
                         int64_t processTid, int64_t threadCounter,
                         int64_t threadWaitCounter,
                         InlineComputeType inlineType)
    {
        _schedInfo.dequeueTime = dequeueTime;
        if (_schedInfo.runningThread == 0) {
            _schedInfo.runningThread = runningThread;
        }
        _schedInfo.processTid = processTid;
        _schedInfo.threadCounter = threadCounter;
        _schedInfo.threadWaitCounter = threadWaitCounter;
        _schedInfo.inlineType = inlineType;
    }

protected:
    ScheduleInfo _schedInfo;
    bool _inlineCompute = false;
    bool _tailCompute = false;
};

struct PriorityItem {
public:
    bool operator<(const PriorityItem &other) const {
        if (globalStartTimeUs != other.globalStartTimeUs) {
            return globalStartTimeUs < other.globalStartTimeUs;
        } else {
            return kernelId < other.kernelId;
        }
    }
    bool operator<=(const PriorityItem &other) const {
        if (globalStartTimeUs != other.globalStartTimeUs) {
            return globalStartTimeUs <= other.globalStartTimeUs;
        } else {
            return kernelId <= other.kernelId;
        }
    }
    bool operator>=(const PriorityItem &other) const {
        if (globalStartTimeUs != other.globalStartTimeUs) {
            return globalStartTimeUs >= other.globalStartTimeUs;
        } else {
            return kernelId >= other.kernelId;
        }
    }
    bool operator>(const PriorityItem &other) const {
        if (globalStartTimeUs != other.globalStartTimeUs) {
            return globalStartTimeUs > other.globalStartTimeUs;
        } else {
            return kernelId > other.kernelId;
        }
    }
public:
    int64_t globalStartTimeUs = 0;
    int64_t kernelId = 0;
    NaviThreadPoolItemBase *item = nullptr;
};

class NaviThreadPool
{
public:
    NaviThreadPool();
    ~NaviThreadPool();
private:
    NaviThreadPool(const NaviThreadPool &);
    NaviThreadPool &operator=(const NaviThreadPool &);
public:
    bool start(const ConcurrencyConfig &config, const NaviLoggerPtr &logger,
               const std::string &name);
    void stop();
    bool incWorkerCount();
    void decWorkerCount();
    void push(int64_t globalStartTimeUs, int64_t kernelId, NaviThreadPoolItemBase *item, bool collectMalloc);
    void flushSignal(size_t flushCount);
    size_t getActiveThreadCount(size_t queueSize) const;
    int64_t getIdleThreadCount() const;
    int64_t getRunningThreadCount() const;
    size_t getQueueSize();
    size_t getIdleQueueSize() const;
    std::vector<pid_t> getPidVec() const;
    NaviThreadPoolStat getStat();
private:
    bool createThreads(const ConcurrencyConfig &config, const std::string &name);
    void initThreadNumRange(const ConcurrencyConfig &config);
    size_t getCoreNum();
    void backgroundThread();
    void updateThreadNumStep();
    void updateActiveThreadCount();
    void checkTimeout();
    void doPush(int64_t globalStartTimeUs,
                int64_t kernelId,
                NaviThreadPoolItemBase *item,
                bool collectMalloc,
                size_t queueSize);
    void workLoop(int32_t tid);
    void inlineCompute(bool &pool_opened);
    bool tailCompute(bool &pool_opened);
    void runItem(NaviThreadPoolItemBase *item,
                 bool incCount,
                 InlineComputeType inlineType,
                 bool &pool_opened);
    NaviThreadPoolItemBase *pop(size_t &keyH, size_t &keyL);
    int32_t getIdleTid();
    void signal(int32_t tid);
    bool wait(int32_t tid);
    void waitQueueEmpty();
    void waitThreadStop();
    void clear();
private:
    DECLARE_LOGGER();
    volatile bool _run;
    volatile bool _accept;
    atomic64_t _workerCount;
    atomic64_t _runningThread;
    atomic64_t _processingCount;
    int _configThreadNum;
    size_t _threadNum;
    bool _autoScale;
    size_t _minThreadNum;
    size_t _maxThreadNum;
    size_t _maxThreadPoolQueueSize;
    size_t _activeThreadNum;
    NaviThread *_threads;
    autil::ThreadPtr _backgroundThread;
    autil::ThreadCond _backgroundCond;
    atomic64_t _wakeIndex;
    atomic64_t _priorityBase;
    size_t _threadNumStep;
    ThreadPoolScheduleQueue _scheduleQueue;
    arpc::common::LockFreeQueue<int32_t> _idleQueue;
    PrioritySched _prioritySched;
#ifdef THREAD_POOL_DEBUG_PUSH_POP
    arpc::common::LockFreeQueue<std::pair<size_t, void *>> _pushQueue;
    arpc::common::LockFreeQueue<std::pair<size_t, void *>> _popQueue;
#endif
};

class NaviDisableSignalScope {
public:
    NaviDisableSignalScope(bool disable, NaviThreadPool *threadPool = nullptr) {
        if (disable) {
            _threadPool = (threadPool ? threadPool : current_thread_pool);
        }
        if (_threadPool) {
            _prevCount = thread_pool_signal_cache_count;
            thread_pool_signal_cache_count = 0;
        }
    }
    ~NaviDisableSignalScope() {
        if (!_threadPool) {
            return;
        }
        auto flushCount = thread_pool_signal_cache_count;
        auto needSignal = (flushCount > 0);
        if (needSignal && _prevCount > 0) {
            thread_pool_signal_cache_count = 0;
        } else {
            thread_pool_signal_cache_count = _prevCount;
        }
        if (needSignal) {
            _threadPool->flushSignal(flushCount);
        }
    }
private:
    int64_t _prevCount = INVALID_SIGNAL_CACHE_COUNT;
    NaviThreadPool *_threadPool = nullptr;
};

NAVI_TYPEDEF_PTR(NaviThreadPool);

}
