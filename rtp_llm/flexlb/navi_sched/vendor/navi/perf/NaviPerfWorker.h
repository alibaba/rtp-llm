#pragma once

#include <memory>
#include <linux/perf_event.h>
#include <sys/mman.h>
#include <string.h>
#include <unistd.h>
#include <atomic>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include "navi/log/NaviLogger.h"

namespace navi {

class NaviPerfThread;

class NaviPerfWorkerBase {
public:
    NaviPerfWorkerBase(pid_t pid, NaviPerfThread *thread)
        : _pid(pid)
        , _thread(thread)
        , _enabled(false)
        , _bufferSize(0)
        , _mmapSize(0)
        , _fd(-1)
        , _mem(MAP_FAILED)
    {
        memset(&_attr, 0, sizeof(perf_event_attr));
        _attr.size = sizeof(perf_event_attr);
        _attr.disabled = 1;
        _attr.watermark = 1;
        _attr.wakeup_watermark = 1;
        _attr.exclude_callchain_kernel = 1;
        _attr.exclude_guest = 1;
        _attr.sample_id_all = 1;
    }
    virtual ~NaviPerfWorkerBase()
    {
        if (-1 != _fd) {
            ::close(_fd);
        }
        if (MAP_FAILED != _mem) {
            munmap(_mem, _mmapSize);
        }
    }

public:
    bool init(const NaviLoggerPtr &logger);
    int getFd() const { return _fd; }
    virtual bool enable() { return doEnable(); }
    bool isEnabled() const { return _enabled; }
    bool disable();

public:
    virtual void doInit() = 0;
    virtual void sample() = 0;

protected:
    bool doEnable();
    void doSample(bool branchStack, int pageFaultType);

protected:
    virtual int perf_event_open(int cpu, int groupFd, unsigned long flags)
    {
        return syscall(__NR_perf_event_open, &_attr, _pid, cpu, groupFd, flags);
    }

protected:
    DECLARE_LOGGER();
    pid_t _pid;
    NaviPerfThread *_thread = nullptr;
    volatile bool _enabled = false;
    perf_event_attr _attr;
    size_t _bufferSize;
    size_t _mmapSize;
    int _fd;
    void *_mem;
};

class NaviPerfWorker : public NaviPerfWorkerBase {
public:
    NaviPerfWorker(pid_t pid, NaviPerfThread *thread);
    void doInit() override;
    bool enable() override;
    void sample() override;

private:
    int perf_event_open(int cpu, int groupFd, unsigned long flags) override;

private:
    void __attribute__((noinline)) flushLBR(int32_t depth);

private:
    bool _lbrEnabled = false;
};

class NaviContextSwitchWorker : public NaviPerfWorkerBase {
public:
    NaviContextSwitchWorker(pid_t pid, NaviPerfThread *thread);
    void doInit() override;
    void sample() override;
};

class NaviMajorPageFaultWorker : public NaviPerfWorkerBase {
public:
    NaviMajorPageFaultWorker(pid_t pid, NaviPerfThread *thread);
    void doInit() override;
    void sample() override;
};

class NaviMinorPageFaultWorker : public NaviPerfWorkerBase {
public:
    NaviMinorPageFaultWorker(pid_t pid, NaviPerfThread *thread);
    void doInit() override;
    void sample() override;
};

}
