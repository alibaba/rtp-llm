#include "navi/log/NaviLogger.h"
#include "navi/engine/NaviThreadPool.h"
#include "navi/engine/CollectFlags.h"
#include "navi/engine/ThreadLocalInit.h"
#include <malloc.h>
#include <unistd.h>

namespace navi {

// #define THREAD_POOL_DEBUG
// #define THREAD_POOL_DEBUG_PUSH_POP

thread_local int32_t INLINE_DEPTH_TLS = INVALID_INLINE_DEPTH;
thread_local NaviThreadPool *current_thread_pool = nullptr;
thread_local int64_t thread_pool_signal_cache_count = INVALID_SIGNAL_CACHE_COUNT;
thread_local size_t current_thread_id = 0;
thread_local size_t current_thread_counter = 0;
thread_local size_t current_thread_wait_counter = 0;

thread_local CollectFlags current_collect_flags;
thread_local arpc::common::LockFreeQueue<NaviThreadPoolItemBase *> *inline_item_queue = nullptr;
struct LastExpensive {
    NaviThreadPoolItemBase *item = nullptr;
    int64_t globalStartTimeUs = -1;
    int64_t kernelId = -1;
};
thread_local LastExpensive last_expensive;
thread_local arpc::common::LockFreeQueue<NaviThreadPoolItemBase *> *tail_item_queue = nullptr;

constexpr size_t MAX_THREAD_LEVEL = 16;

NaviThreadPool::NaviThreadPool()
    : _run(false)
    , _accept(false)
    , _threadNum(DEFAULT_THREAD_NUMBER)
    , _autoScale(false)
    , _minThreadNum(DEFAULT_THREAD_NUMBER)
    , _maxThreadNum(DEFAULT_THREAD_NUMBER)
    , _maxThreadPoolQueueSize(0u)
    , _activeThreadNum(DEFAULT_THREAD_NUMBER)
    , _threads(nullptr)
    , _threadNumStep(1)
{
    atomic_set(&_workerCount, 0);
    atomic_set(&_runningThread, 0);
    atomic_set(&_processingCount, 0);
    atomic_set(&_wakeIndex, 0);
    atomic_set(&_priorityBase, 0);
}

NaviThreadPool::~NaviThreadPool() {
    clear();
    stop();
    delete[] _threads;
    _prioritySched.stop();
}

bool NaviThreadPool::start(const ConcurrencyConfig &config,
                           const NaviLoggerPtr &logger,
                           const std::string &name)
{
    if (_run) {
        return true;
    }
    _logger.object = this;
    _logger.logger = logger;
    _logger.addPrefix("thread pool %s", name.c_str());
    _run = true;
    _accept = true;
    NAVI_LOG(INFO, "thread pool [%p]", this);
    return createThreads(config, name);
}

void NaviThreadPool::stop() {
    waitQueueEmpty();
    waitThreadStop();
    for (size_t i = 0; i < _threadNum; i++) {
        _threads[i].thread.reset();
    }
    int32_t tid = 0;
    while (_idleQueue.Pop(&tid)) {
    }
}

void NaviThreadPool::clear() {
    while (true) {
        size_t keyH = 0;
        size_t keyL = 0;
        auto item = pop(keyH, keyL);
        if (item) {
            NAVI_KERNEL_LOG(ERROR, "drop item [%p]", item);
            item->destroy();
        } else {
            break;
        }
    }
}

int32_t NaviThreadPool::getIdleTid() {
    int32_t tid = 0;
    if (_idleQueue.Pop(&tid)) {
        return tid;
    }
    return -1;
}

void NaviThreadPool::signal(int32_t tid) {
    if (tid < 0) {
        tid = getIdleTid();
    }
    if (tid >= 0) {
        NAVI_KERNEL_LOG(SCHEDULE3, "pop idle [%d]", tid);
        auto &thread = _threads[tid];
        auto &cond = thread.cond;
        auto &stat = thread.stat;
        if (0 == cond.trylock()) {
            NAVI_KERNEL_LOG(SCHEDULE3, "wake up [%d]", tid);
            if (getQueueSize() != 0) {
                stat = TS_WAKEUP;
                cond.signal();
            }
            cond.unlock();
            return;
        }
        NAVI_KERNEL_LOG(SCHEDULE3, "fail to wakeup idle queue thread: %d", tid);
    }
    while (true) {
        auto queueSize = getQueueSize();
        if (queueSize == 0) {
            return;
        }
        assert(_activeThreadNum > 0);
        auto maxThreadNum = getActiveThreadCount(queueSize);
        int32_t index = atomic_inc_return(&_wakeIndex) % maxThreadNum;
        auto &thread = _threads[index];
        auto &cond = thread.cond;
        if (0 == cond.trylock()) {
            auto &stat = thread.stat;
            if (TS_WAIT == stat) {
                NAVI_KERNEL_LOG(SCHEDULE3, "wake up [%d], maxThreadNum: [%lu]", index, maxThreadNum);
                if (getQueueSize() != 0) {
                    stat = TS_WAKEUP;
                    cond.signal();
                }
            } else {
                NAVI_KERNEL_LOG(SCHEDULE3, "wake up [%d], maxThreadNum: [%lu], stat: %d", index, maxThreadNum, stat);
                stat = TS_WAKEUP;
            }
            cond.unlock();
            break;
        }
    }
}

bool NaviThreadPool::wait(int32_t tid) {
    if (unlikely(!_run)) {
        return false;
    }
    bool suspend = false;
    auto &cond = _threads[tid].cond;
    if (0 == cond.trylock()) {
        auto &stat = _threads[tid].stat;
        if (stat == TS_WAKEUP) {
            NAVI_KERNEL_LOG(SCHEDULE3, "wait success immediately [%d]", tid);
            stat = TS_RUNNING;
            cond.unlock();
            return true;
        }
        stat = TS_WAIT;
        if (tid < getActiveThreadCount(getQueueSize())) {
            NAVI_KERNEL_LOG(SCHEDULE3, "push to idle [%d]", tid);
            _idleQueue.Push(tid);
        } else {
            suspend = true;
            NAVI_KERNEL_LOG(SCHEDULE3, "thread [%d] suspended, queueSize [%lu]", tid, getQueueSize());
        }
        current_thread_wait_counter++;
        cond.wait();
        stat = TS_RUNNING;
        cond.unlock();
        if (suspend) {
            NAVI_KERNEL_LOG(SCHEDULE3, "thread [%d] resumed", tid);
        }
    }
    return true;
}

void NaviThreadPool::waitQueueEmpty() {
    _accept = false;
    size_t count = 0;
    size_t sleepTime = 200;
    while (true) {
        for (size_t i = 0; i < _threadNum; i++) {
            auto &cond = _threads[i].cond;
            autil::ScopedLock lock(cond);
            cond.signal();
        }
        if (0ul == _scheduleQueue.size() &&
            atomic_read(&_workerCount) == 0)
        {
            break;
        }
        if (count++ % 200 == 0) {
             NAVI_KERNEL_LOG(
                 INFO,
                 "thread pool not empty, scheduleQueue size [%lu], workerCount "
                 "[%lld]",
                 _scheduleQueue.size(), atomic_read(&_workerCount));
        }
        usleep(sleepTime);
        sleepTime += 1000;
        if (sleepTime > 10 * 1000) {
            sleepTime = 10 * 1000;
        }
    }
}

void NaviThreadPool::waitThreadStop() {
    _run = false;
    size_t sleepTime = 200;
    while (true) {
        for (size_t i = 0; i < _threadNum; i++) {
            auto &cond = _threads[i].cond;
            autil::ScopedLock lock(cond);
            cond.signal();
        }
        if (0 == atomic_read(&_runningThread)) {
            break;
        }
        usleep(sleepTime);
        sleepTime += 1000;
        if (sleepTime > 10 * 1000) {
            sleepTime = 10 * 1000;
        }
    }
    if (_backgroundThread) {
        {
            autil::ScopedLock lock(_backgroundCond);
            _backgroundCond.signal();
        }
        _backgroundThread.reset();
    }
}

bool NaviThreadPool::createThreads(const ConcurrencyConfig &config,
                                   const std::string &name)
{
    _configThreadNum = config.threadNum;
    int threadNiceValue = config.threadNiceValue;
    _maxThreadPoolQueueSize = config.maxThreadPoolQueueSize;
    if (_configThreadNum <= 0) { // enable auto scale
        initThreadNumRange(config);
        _threadNum = _maxThreadNum;
        _autoScale = true;
    } else {
        _threadNum = (size_t)_configThreadNum;
    }
    _activeThreadNum = _threadNum;
    updateThreadNumStep();
    _threads = new NaviThread[_threadNum];
    _prioritySched.init(_threadNum);
    for (size_t i = 0; i < _threadNum; i++) {
        auto thread = autil::Thread::createThread(
            std::bind(&NaviThreadPool::workLoop, this, (int32_t)i), name, threadNiceValue);
        if (!thread) {
            NAVI_KERNEL_LOG(ERROR, "Create thread fail!");
            _run = false;
            return false;
        }
        _threads[i].thread = thread;
    }
    auto bgThread = autil::Thread::createThread(
            std::bind(&NaviThreadPool::backgroundThread, this), "navi_bg");
    if (!bgThread) {
        NAVI_KERNEL_LOG(ERROR, "Create background thread fail!");
        _run = false;
        return false;
    }
    _backgroundThread = bgThread;
    NAVI_KERNEL_LOG(INFO,
                    "create threads success, autoScale[%d], config[%d],"
                    "threadNum[%lu], minThreadNum[%lu], maxThreadNum[%lu], threadNiceValue[%d]",
                    _autoScale, _configThreadNum,
                    _threadNum, _minThreadNum, _maxThreadNum, threadNiceValue);
    return true;
}

void NaviThreadPool::initThreadNumRange(const ConcurrencyConfig &config) {
    _minThreadNum = config.minThreadNum;
    _maxThreadNum = config.maxThreadNum;
    if (DEFAULT_THREAD_NUMBER == _maxThreadNum) {
        _maxThreadNum = DEFAULT_MAX_THREAD_NUMBER;
    } else {
        _maxThreadNum = std::min(_maxThreadNum, DEFAULT_MAX_THREAD_NUMBER);
    }
    _minThreadNum = std::min(_minThreadNum, _maxThreadNum);
}

size_t NaviThreadPool::getCoreNum() {
    errno = 0;
    auto coreNum = sysconf(_SC_NPROCESSORS_ONLN);
    if (errno == 0 && coreNum > 0) {
        return coreNum;
    } else {
        NAVI_KERNEL_LOG(WARN,
                        "auto scale get core num failed, use default[%lu]",
                        DEFAULT_MAX_THREAD_NUMBER);
        return DEFAULT_MAX_THREAD_NUMBER;
    }
}

void NaviThreadPool::backgroundThread() {
    NaviLoggerScope scope(_logger);
    size_t counter = 0;
    while (_run) {
        if (0 == (counter % 128)) {
            updateActiveThreadCount();
        }
        checkTimeout();
        {
            autil::ScopedLock lock(_backgroundCond);
            _backgroundCond.wait(10 * 1000);
        }
        counter++;
    }
}

void NaviThreadPool::updateThreadNumStep() {
    _threadNumStep = std::max(1lu, (_activeThreadNum + MAX_THREAD_LEVEL - 1) / MAX_THREAD_LEVEL);
}

void NaviThreadPool::updateActiveThreadCount() {
    if (!_autoScale) {
        return;
    }
    assert(_configThreadNum <= 0);
    size_t activeCount = std::max(0, (int)getCoreNum() + _configThreadNum);
    activeCount = std::min(activeCount, _maxThreadNum);
    activeCount = std::max(activeCount, _minThreadNum);
    activeCount = std::max(activeCount, (size_t)1);
    if (activeCount != _activeThreadNum) {
        NAVI_KERNEL_LOG(
            INFO,
            "active thread count changed from [%lu] to [%lu] for auto scale, config [%d]",
            _activeThreadNum, activeCount, _configThreadNum);
        _activeThreadNum = activeCount;
        updateThreadNumStep();
        for (size_t i = 0; i < activeCount; i++) {
            auto &thread = _threads[i];
            auto &cond = thread.cond;
            if (0 == cond.trylock()) {
                auto &stat = thread.stat;
                if (TS_WAIT == stat) {
                    stat = TS_WAKEUP;
                    cond.signal();
                } else {
                    stat = TS_WAKEUP;
                }
                cond.unlock();
                break;
            }
        }
    }
}

void NaviThreadPool::checkTimeout() {
}

bool NaviThreadPool::incWorkerCount() {
    if (_accept) {
        atomic_inc(&_workerCount);
        return true;
    } else {
        return false;
    }
}

void NaviThreadPool::decWorkerCount() {
    atomic_dec(&_workerCount);
}

void NaviThreadPool::push(int64_t globalStartTimeUs, int64_t kernelId, NaviThreadPoolItemBase *item, bool collectMalloc) {
    auto queueSize = getQueueSize();
    item->setQueueSize(queueSize);
    if (unlikely(item->syncMode() || (_maxThreadPoolQueueSize > 0 && queueSize >= _maxThreadPoolQueueSize))) {
        NAVI_KERNEL_LOG(SCHEDULE3, "run sync WorkItem [%p]", item);
        bool pool_opened = false;
        runItem(item, false, ICT_SYNC, pool_opened);
        if (pool_opened) {

        }
        return;
    }
    if (tail_item_queue && item->tailCompute()) {

        tail_item_queue->Push(item);
        return;
    }
    if (inline_item_queue) {
        auto inlineCompute = item->inlineCompute();
        if (inlineCompute) {

            inline_item_queue->Push(item);
            return;
        } else if (!inline_item_queue->Empty()) {
            // Let another worker run ready work while this one drains inline consumers.
            doPush(globalStartTimeUs, kernelId, item, collectMalloc, queueSize);
            return;
        } else {
            if (last_expensive.item) {
                auto current = last_expensive;
                last_expensive.globalStartTimeUs = globalStartTimeUs;
                last_expensive.kernelId = kernelId;
                last_expensive.item = item;
                doPush(current.globalStartTimeUs,
                       current.kernelId,
                       current.item,
                       collectMalloc,
                       queueSize);
            } else {
                last_expensive.globalStartTimeUs = globalStartTimeUs;
                last_expensive.kernelId = kernelId;
                last_expensive.item = item;
            }
            return;
        }
    }
    doPush(globalStartTimeUs, kernelId, item, collectMalloc, queueSize);
}

void NaviThreadPool::doPush(int64_t globalStartTimeUs,
                            int64_t kernelId,
                            NaviThreadPoolItemBase *item,
                            bool collectMalloc,
                            size_t queueSize) {
    int32_t tid = -1;
    if (INVALID_SIGNAL_CACHE_COUNT == thread_pool_signal_cache_count) {
        tid = getIdleTid();
        if (tid >= 0) {
            item->setSignalTid(_threads[tid].tid);
        }
    }
    auto base = atomic_inc_return(&_priorityBase);
    size_t priorityH = globalStartTimeUs;
    size_t priorityL = (kernelId << 32) | (base & ((1l << 32) - 1));
    auto retNode = item;
    {

        if (!_scheduleQueue.insert(priorityH, priorityL, (size_t)item)) {
            NAVI_LOG(WARN,
                     "push WorkItem failed [%p], tid [%d], queueSize [%lu], key [%lu %lu], run sync",
                     item,
                     tid,
                     queueSize,
                     priorityH,
                     priorityL);
            bool pool_opened = false;
            runItem(item, false, ICT_PUSH_FAILED, pool_opened);
            if (pool_opened) {

            }
            return;
        }
    }
#ifdef THREAD_POOL_DEBUG
    _pushQueue.Push(std::make_pair(priority, (void *)retNode));
#endif
#ifdef THREAD_POOL_DEBUG_PUSH_POP
    NAVI_LOG(WARN,
             "push WorkItem [%p], tid [%d], queueSize [%lu], [%p], key [%lu]",
             item,
             tid,
             queueSize,
             retNode,
             priority);
#else
    NAVI_LOG(SCHEDULE3,
             "push WorkItem [%p], tid [%d], queueSize [%lu] cacheCount [%ld], [%p], key [%lu%lu]",
             item,
             tid,
             queueSize,
             thread_pool_signal_cache_count,
             retNode,
             priorityH,
             priorityL);
#endif
    if (INVALID_SIGNAL_CACHE_COUNT == thread_pool_signal_cache_count) {
        signal(tid);
    } else {
        thread_pool_signal_cache_count++;
    }
}

void NaviThreadPool::flushSignal(size_t flushCount) {
    auto total = getQueueSize();
    while (total > 0) {
        auto tid = getIdleTid();
        if (tid >= 0) {
            signal(tid);
            total--;
            total = std::min(total, getQueueSize());
        } else {
            break;
        }
    }
}

size_t NaviThreadPool::getActiveThreadCount(size_t queueSize) const {
    //TODO(zhang7): fix thread count
    return _activeThreadNum;
}

int64_t NaviThreadPool::getIdleThreadCount() const {
    int64_t active = _activeThreadNum;
    int64_t idle = active - atomic_read(&_processingCount);
    return std::max((int64_t)0, idle);
}

int64_t NaviThreadPool::getRunningThreadCount() const {
    return atomic_read(&_processingCount);
}

size_t NaviThreadPool::getQueueSize() {
    return _scheduleQueue.size();
}

size_t NaviThreadPool::getIdleQueueSize() const {
    return _idleQueue.Size();
}

NaviThreadPoolItemBase *NaviThreadPool::pop(size_t &keyH, size_t &keyL) {
    auto ret = (NaviThreadPoolItemBase *)_scheduleQueue.deleteMin(keyH, keyL);
#ifdef THREAD_POOL_DEBUG
    if (ret) {
        _popQueue.Push(std::make_pair(key, (void *)ret));
    }
#endif
    return ret;
}

std::vector<pid_t> NaviThreadPool::getPidVec() const {
    std::vector<pid_t> vec;
    while ((int64_t)_threadNum != atomic_read(&_runningThread)) {
        usleep(100);
    }
    for (size_t i = 0; i < _threadNum; i++) {
        vec.push_back(_threads[i].tid);
    }
    return vec;
}

void NaviThreadPool::workLoop(int32_t tid) {
    ThreadLocalInit::init();
    auto priorityInfo = (UserPriorityInfo *)_prioritySched.alloc();
    current_thread_id = (long)syscall(SYS_gettid);
    _threads[tid].tid = current_thread_id;
    assert(current_collect_flags.getFlags() == 0);
    NAVI_MEMORY_BARRIER();
    atomic_inc(&_runningThread);
    NaviLoggerScope scope(_logger);
    current_thread_pool = this;
    inline_item_queue = new arpc::common::LockFreeQueue<NaviThreadPoolItemBase *>();
    tail_item_queue = new arpc::common::LockFreeQueue<NaviThreadPoolItemBase *>();
    while (_run) {
        size_t keyH = 0;
        size_t keyL = 0;
        auto item = pop(keyH, keyL);
#ifdef THREAD_POOL_DEBUG_PUSH_POP
        // if (item) {
        //     NAVI_KERNEL_LOG(ERROR, "thread pop [%p] queueSize [%lu], key [%lu]", item, getQueueSize(), key);
        // }
#else
        NAVI_KERNEL_LOG(SCHEDULE3, "thread pop [%p] queueSize [%lu], key [%lu%lu]", item, getQueueSize(), keyH, keyL);
#endif
        bool pool_opened = false;
        if (item) {
            if (priorityInfo) {
                priorityInfo->high = keyH;
                priorityInfo->low = keyL;
            }
            NAVI_KERNEL_LOG(SCHEDULE3, "begin process [%d] [%p] queueSize [%lu], wait: %lu", tid, item, getQueueSize(), current_thread_wait_counter);
            runItem(item, true, ICT_NO_INLINE, pool_opened);
            NAVI_KERNEL_LOG(SCHEDULE3, "end process [%d] [%p] queueSize [%lu]", tid, item, getQueueSize());
            if (priorityInfo) {
                priorityInfo->high = 0;
                priorityInfo->low = 0;
            }
        } else {
            NAVI_KERNEL_LOG(SCHEDULE3, "pop failed [%d] [%p] queueSize [%lu]", tid, item, getQueueSize());
        }
        inlineCompute(pool_opened);
        auto cont = tailCompute(pool_opened);
        if (pool_opened) {

        }
        if (cont) {
            continue;
        }
        if (unlikely(tid >= _activeThreadNum)) {
            // transfer to active thread
            signal(-1);
            if (!wait(tid)) {
                break;
            }
            NAVI_KERNEL_LOG(SCHEDULE3,
                            "thread continue running after suspend [%d] queueSize [%lu]", tid, getQueueSize());
        } else {
            if (!item) {
                if (!wait(tid)) {
                    break;
                }
                NAVI_KERNEL_LOG(SCHEDULE3, "thread continue running [%d] queueSize [%lu]", tid, getQueueSize());
            }
        }
    }
    {
        while (!inline_item_queue->Empty() || last_expensive.item) {
            NaviThreadPoolItemBase *item = nullptr;
            while (inline_item_queue->Pop(&item)) {
                if (item) {
                    item->destroy();
                }
            }
            if (last_expensive.item) {
                auto item = last_expensive.item;
                last_expensive.item = nullptr;
                item->destroy();
            }
        }
        assert(inline_item_queue->Empty());
        assert(!last_expensive.item);
        DELETE_AND_SET_NULL(inline_item_queue);
    }
    {
        while (!tail_item_queue->Empty()) {
            NaviThreadPoolItemBase *item = nullptr;
            while (tail_item_queue->Pop(&item)) {
                if (item) {
                    item->destroy();
                }
            }
        }
        assert(tail_item_queue->Empty());
        DELETE_AND_SET_NULL(tail_item_queue);
    }
    NAVI_KERNEL_LOG(INFO, "thread exited");
    atomic_dec(&_runningThread);
    INLINE_DEPTH_TLS = INVALID_INLINE_DEPTH;
    current_thread_pool = nullptr;
}

void NaviThreadPool::inlineCompute(bool &pool_opened) {
    while (!inline_item_queue->Empty() || last_expensive.item) {
        atomic_inc(&_processingCount);
        NaviThreadPoolItemBase *item = nullptr;
        while (inline_item_queue->Pop(&item)) {
            if (item) {
                runItem(item, false, ICT_NORMAL, pool_opened);
            }
        }
        if (last_expensive.item) {
            auto item = last_expensive.item;
            last_expensive.item = nullptr;
            runItem(item, false, ICT_TAIL_RECUR, pool_opened);
        }
        atomic_dec(&_processingCount);
    }
    assert(inline_item_queue->Empty());
    assert(!last_expensive.item);
}

bool NaviThreadPool::tailCompute(bool &pool_opened) {
    bool ret = false;
    while (!tail_item_queue->Empty()) {
        atomic_inc(&_processingCount);
        NaviThreadPoolItemBase *item = nullptr;
        while (tail_item_queue->Pop(&item)) {
            if (item) {
                runItem(item, false, ICT_TAIL_QUEUE, pool_opened);
                ret = true;
            }
        }
        atomic_dec(&_processingCount);
    }
    assert(tail_item_queue->Empty());
    return ret;
}

void NaviThreadPool::runItem(NaviThreadPoolItemBase *item,
                             bool incCount,
                             InlineComputeType inlineType,
                             bool &pool_opened) {
    if (incCount) {
        atomic_inc(&_processingCount);
    }
    auto dequeueTime = CommonUtil::getTimelineTimeNs();
    item->setScheduleInfo(dequeueTime,
                          atomic_read(&_processingCount),
                          current_thread_id,
                          current_thread_counter,
                          current_thread_wait_counter,
                          inlineType);
    INLINE_DEPTH_TLS = 1;
    assert(!incCount || current_collect_flags.getFlags() == 0);
    if (!pool_opened) {

        pool_opened = true;
    }
    item->process();
    item->destroy();
    assert(!incCount || current_collect_flags.getFlags() == 0);
    if (incCount) {
        atomic_dec(&_processingCount);
    }
    current_thread_counter++;
}

NaviThreadPoolStat NaviThreadPool::getStat() {
    NaviThreadPoolStat stat;
    stat.activeThreadCount = getRunningThreadCount();
    stat.activeThreadQueueSize = getQueueSize();
    stat.idleThreadQueueSize = getIdleQueueSize();
    return stat;
}

}
