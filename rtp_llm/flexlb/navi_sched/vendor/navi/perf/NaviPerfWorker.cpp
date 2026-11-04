#include "navi/perf/NaviPerfWorker.h"
#include "navi/perf/NaviPerfThread.h"
#include "navi/proto/GraphVis.pb.h"

#define PERF_RECORD_MISC_SWITCH_OUT_PREEMPT       (1 << 14)

namespace navi {

constexpr uint32_t PAGE_SIZE = 4096;

#ifdef  __aarch64__
#define mb()    asm volatile("dmb ish":::"memory")
#else
#define mb()    asm volatile("mfence":::"memory")
#endif

bool NaviPerfWorkerBase::init(const NaviLoggerPtr &logger) {
    _logger.object = this;
    _logger.logger = logger;
    doInit();
    _fd = perf_event_open(-1, -1, PERF_FLAG_FD_CLOEXEC);
    if (_fd < 0) {
        NAVI_LOG(ERROR, "open perf fd for pid [%d] failed, error [%s]", _pid,
                 strerror(errno));
        return false;
    }
    NAVI_LOG(INFO, "perf fd [%d] for pid [%d]", _fd, _pid);
    _mem = mmap(0, _mmapSize, PROT_READ | PROT_WRITE, MAP_SHARED, _fd, 0);
    if (MAP_FAILED == _mem) {
        NAVI_LOG(ERROR,
                 "mmap perf mem for pid [%d] failed, fd [%d], err [%s]", _pid,
                 _fd, strerror(errno));
        return false;
    }
    ioctl(_fd, PERF_EVENT_IOC_RESET, 0);
    return true;
}

bool NaviPerfWorkerBase::doEnable() {
    if (_enabled) {
        return true;
    }
    _enabled = (0 == ioctl(_fd, PERF_EVENT_IOC_ENABLE, 0));
    return _enabled;
}

bool NaviPerfWorkerBase::disable() {
    if (!_enabled) {
        return true;
    }
    _enabled = (0 != ioctl(_fd, PERF_EVENT_IOC_DISABLE, 0));
    return !_enabled;
}

#define NEW_SWITCH_SAMPLE (21 + 256)
void NaviPerfWorkerBase::doSample(bool branchStack, int pageFaultType) {
    perf_event_mmap_page *header = (perf_event_mmap_page *)_mem;
    auto ptr = &header->data_head;
    uint64_t data_head = *(volatile uint64_t *)ptr;
    mb();
    uint64_t data_tail = header->data_tail;
    uint8_t *bufferStart = ((uint8_t *)_mem) + PAGE_SIZE;
    uint8_t *bufferEnd = ((uint8_t *)_mem) + _mmapSize;
    while (data_head != data_tail) {
        uint8_t *sampleBase = bufferStart + (data_tail & (_bufferSize - 1));
        auto sample = (perf_event_header *)sampleBase;
        auto sampleSize = sample->size;
        uint32_t naviEventSize = sampleSize - sizeof(perf_event_header);
        uint8_t *copyBase =
            bufferStart +
            ((data_tail + sizeof(perf_event_header)) & (_bufferSize - 1));
        auto entryType = PET_UNKNOWN;
        if (sample->type == PERF_RECORD_SAMPLE) {
            if (pageFaultType == 1) {
                entryType = PET_PAGE_FAULT_MAJOR;
            } else if (pageFaultType == 2) {
                entryType = PET_PAGE_FAULT_MINOR;
            } else {
                entryType = PET_SAMPLE;
            }
        } else if (sample->type == PERF_RECORD_THROTTLE) {
            entryType = PET_THROTTLE;
        } else if (sample->type == PERF_RECORD_UNTHROTTLE) {
            entryType = PET_UNTHROTTLE;
        } else if (sample->type == PERF_RECORD_SWITCH) {
            if (sample->misc & PERF_RECORD_MISC_SWITCH_OUT) {
                entryType = (sample->misc & PERF_RECORD_MISC_SWITCH_OUT_PREEMPT) ? PET_SWITCH_OUT_PREEMPT : PET_SWITCH_OUT;
            } else {
                entryType = PET_SWITCH_IN;
            }
        } else if (sample->type == NEW_SWITCH_SAMPLE) {
            if (sample->misc & PERF_RECORD_MISC_SWITCH_OUT) {
                entryType = (sample->misc & PERF_RECORD_MISC_SWITCH_OUT_PREEMPT) ? PET_SWITCH_OUT_PREEMPT_NEW : PET_SWITCH_OUT_NEW;
            } else {
                entryType = PET_SWITCH_IN_NEW;
            }
        } else {
            NAVI_LOG(ERROR, "unknown sample [%d]", sample->type);
            data_tail += sampleSize;
            continue;
        }
        auto entry = (NaviPerfEventEntry *)malloc(
            naviEventSize + offsetof(NaviPerfEventEntry, data));
        new (entry) NaviPerfEventEntry();
        entry->sampleType = entryType;
        uint8_t *dataStart = sampleBase + sizeof(perf_event_header);
        uint8_t *dataEnd = dataStart + naviEventSize;
        if (dataEnd < bufferEnd || dataStart >= bufferEnd) {
            memcpy(&entry->data, copyBase, naviEventSize);
        } else {
            uint8_t *dest = (uint8_t *)&entry->data;
            size_t len1 = bufferEnd - copyBase;
            memcpy(dest, dataStart, len1);
            size_t len2 = naviEventSize - len1;
            memcpy(dest + len1, bufferStart, len2);
        }
        if (sample->type == PERF_RECORD_SAMPLE) {
            size_t callchainSize = 0;
            if (_attr.sample_type & NaviPerfCallchainEntry::SAMPLE_TYPE) {
                entry->callchainEntry = (NaviPerfCallchainEntry *)(((char *)&entry->data) + sizeof(NaviPerfEvent));
                callchainSize = 8 * (entry->callchainEntry->nr + 1);
            }
            if (branchStack) {
                entry->branchEntry =
                    (NaviPerfBranchEntry *)(((char *)&entry->data) + sizeof(NaviPerfEvent) + callchainSize);
            }
        } else if (sample->type == NEW_SWITCH_SAMPLE) {
            if (_attr.sample_type & NaviPerfCallchainEntry::SAMPLE_TYPE) {
                entry->callchainEntry = (NaviPerfCallchainEntry *)(((char *)&entry->data) + sizeof(NaviPerfEventSwitchNew));
            }
        }
        _thread->collectResultEntry(entry);
        data_tail += sampleSize;
    }
    mb();
    ptr = &header->data_tail;
    *(volatile uint64_t *)ptr = data_tail;
}

NaviPerfWorker::NaviPerfWorker(pid_t pid, NaviPerfThread *thread)
    : NaviPerfWorkerBase(pid, thread)
{
}

void NaviPerfWorker::doInit() {
    _logger.addPrefix("navi perf worker [%d]", _pid);
    constexpr uint32_t BUFFER_SIZE = 256 * PAGE_SIZE;
    static_assert(0 == (BUFFER_SIZE & (BUFFER_SIZE - 1)),
                  "buffer size should be power of two");
    _bufferSize = BUFFER_SIZE;
    _mmapSize = _bufferSize + PAGE_SIZE;

    _attr.type = PERF_TYPE_HARDWARE;
    _attr.config = PERF_COUNT_HW_REF_CPU_CYCLES;
    _attr.sample_period = 60000;
    _attr.sample_type = NaviPerfEvent::SAMPLE_TYPE |
                        NaviPerfBranchEntry::SAMPLE_TYPE |
                        NaviPerfCallchainEntry::SAMPLE_TYPE;
    _attr.branch_sample_type = NaviPerfBranchEntry::BRANCH_SAMPLE_TYPE;
    _attr.context_switch = 0;
}

int NaviPerfWorker::perf_event_open(int cpu, int groupFd, unsigned long flags) {
    auto fd = NaviPerfWorkerBase::perf_event_open(cpu, groupFd, flags);
    if (fd < 0) {
        NAVI_LOG(ERROR, "open perf fd for pid [%d] failed, error [%s], retry with lbr disabled", _pid, strerror(errno));
        _attr.sample_type &= ~(NaviPerfBranchEntry::SAMPLE_TYPE);
        fd = NaviPerfWorkerBase::perf_event_open(cpu, groupFd, flags);
    }
    _lbrEnabled = (_attr.sample_type & NaviPerfBranchEntry::SAMPLE_TYPE);
    NAVI_LOG(INFO, "lbr enable [%d]", _lbrEnabled);
    return fd;
}

void NaviPerfWorker::flushLBR(int32_t depth) {
    if (++depth < 32) {
        flushLBR(depth);
    }
    NAVI_MEMORY_BARRIER();
}

bool NaviPerfWorker::enable() {
    if (doEnable()) {
        if (_lbrEnabled) {
            flushLBR(0);
        }
        return true;
    } else {
        return false;
    }
}

#define NEW_SWITCH_SAMPLE (21 + 256)
void NaviPerfWorker::sample() {
    doSample(_lbrEnabled, 0);
}

NaviContextSwitchWorker::NaviContextSwitchWorker(pid_t pid, NaviPerfThread *thread)
    : NaviPerfWorkerBase(pid, thread)
{
}

void NaviContextSwitchWorker::doInit() {
    _logger.addPrefix("navi perf context switch worker [%d]", _pid);
    constexpr uint32_t BUFFER_SIZE = 16 * PAGE_SIZE;
    static_assert(0 == (BUFFER_SIZE & (BUFFER_SIZE - 1)),
                  "buffer size should be power of two");
    _bufferSize = BUFFER_SIZE;
    _mmapSize = _bufferSize + PAGE_SIZE;

    _attr.type = PERF_TYPE_SOFTWARE;
    _attr.config = PERF_COUNT_SW_CONTEXT_SWITCHES;
    _attr.sample_period = 1;
    _attr.sample_type = NaviPerfEvent::SAMPLE_TYPE | NaviPerfCallchainEntry::SAMPLE_TYPE;
    _attr.context_switch = 1;
}

void NaviContextSwitchWorker::sample() {
    doSample(false, 0);
}

NaviMajorPageFaultWorker::NaviMajorPageFaultWorker(pid_t pid, NaviPerfThread *thread)
    : NaviPerfWorkerBase(pid, thread)
{
}

void NaviMajorPageFaultWorker::doInit() {
    _logger.addPrefix("navi perf major page fault worker [%d]", _pid);
    constexpr uint32_t BUFFER_SIZE = 16 * PAGE_SIZE;
    static_assert(0 == (BUFFER_SIZE & (BUFFER_SIZE - 1)),
                  "buffer size should be power of two");
    _bufferSize = BUFFER_SIZE;
    _mmapSize = _bufferSize + PAGE_SIZE;

    _attr.type = PERF_TYPE_SOFTWARE;
    _attr.config = PERF_COUNT_SW_PAGE_FAULTS_MAJ;
    _attr.sample_period = 1;
    _attr.sample_type = NaviPerfEvent::SAMPLE_TYPE | NaviPerfCallchainEntry::SAMPLE_TYPE;
}

void NaviMajorPageFaultWorker::sample() {
    doSample(false, 1);
}

NaviMinorPageFaultWorker::NaviMinorPageFaultWorker(pid_t pid, NaviPerfThread *thread)
    : NaviPerfWorkerBase(pid, thread)
{
}

void NaviMinorPageFaultWorker::doInit() {
    _logger.addPrefix("navi perf minor page fault worker [%d]", _pid);
    constexpr uint32_t BUFFER_SIZE = 16 * PAGE_SIZE;
    static_assert(0 == (BUFFER_SIZE & (BUFFER_SIZE - 1)),
                  "buffer size should be power of two");
    _bufferSize = BUFFER_SIZE;
    _mmapSize = _bufferSize + PAGE_SIZE;

    _attr.type = PERF_TYPE_SOFTWARE;
    _attr.config = PERF_COUNT_SW_PAGE_FAULTS_MIN;
    _attr.sample_period = 1;
    _attr.sample_type = NaviPerfEvent::SAMPLE_TYPE | NaviPerfCallchainEntry::SAMPLE_TYPE;
}

void NaviMinorPageFaultWorker::sample() {
    doSample(false, 2);
}

}
