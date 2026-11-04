#pragma once

#include "navi/common.h"
#include "navi/engine/StackTable.h"
#include <linux/perf_event.h>

namespace navi {

class PerfEventDef;
class KernelPerfDef;
struct NaviPerfCallchainEntry;
struct NaviPerfBranchEntry;

// see man perf_event_open
struct NaviPerfEvent {
public:
    void fillProto(PerfEventDef *perfEvent,
                   bool pageFault,
                   NaviPerfCallchainEntry *callchainEntry,
                   NaviPerfBranchEntry *branchEntry,
                   StackTable &stackTable) const;
public:
    uint64_t ip;       /* if PERF_SAMPLE_IP */
    uint32_t pid, tid; /* if PERF_SAMPLE_TID */
    uint64_t time;     /* if PERF_SAMPLE_TIME */
    uint64_t addr;     /* if PERF_SAMPLE_ADDR */
    uint32_t cpu, res; /* if PERF_SAMPLE_CPU */
public:
    static constexpr uint32_t SAMPLE_TYPE =
        PERF_SAMPLE_IP | PERF_SAMPLE_TID | PERF_SAMPLE_TIME | PERF_SAMPLE_CPU | PERF_SAMPLE_ADDR;
};

struct NaviPerfCallchainEntry {
    uint64_t nr;
    uint64_t ips[0];
public:
    static constexpr uint64_t SAMPLE_TYPE = PERF_SAMPLE_CALLCHAIN;
};

struct NaviPerfBranchEntry {
    uint64_t bnr;                    /* if PERF_SAMPLE_BRANCH_STACK */
    struct perf_branch_entry lbr[0]; /* if PERF_SAMPLE_BRANCH_STACK */
public:
    static constexpr uint64_t SAMPLE_TYPE = PERF_SAMPLE_BRANCH_STACK;
    static constexpr uint64_t BRANCH_SAMPLE_TYPE =
        PERF_SAMPLE_BRANCH_CALL_STACK | PERF_SAMPLE_BRANCH_USER |
        PERF_SAMPLE_BRANCH_NO_FLAGS | PERF_SAMPLE_BRANCH_NO_CYCLES;
};

struct NaviPerfEventThrottle {
public:
    void fillProto(PerfEventDef *perfEvent) const;
public:
    uint64_t time;
    uint64_t id;
    uint64_t stream_id;
};

struct NaviPerfEventSwitch {
public:
    void fillProto(PerfEventDef *perfEvent) const;
public:
    uint32_t pid, tid; /* if PERF_SAMPLE_TID */
    uint64_t time;     /* if PERF_SAMPLE_TIME */
    uint32_t cpu, res; /* if PERF_SAMPLE_CPU */
};

struct NaviPerfEventSwitchNew {
public:
    void fillProto(PerfEventDef *perfEvent,
                   NaviPerfCallchainEntry *callchainEntry,
                   StackTable &stackTable) const;
public:
    uint32_t next_prev_pid;
    uint32_t next_prev_tid;
    uint32_t pid, tid; /* if PERF_SAMPLE_TID */
    uint64_t time;     /* if PERF_SAMPLE_TIME */
    uint32_t cpu, res; /* if PERF_SAMPLE_CPU */
};

struct NaviPerfEventEntry {
public:
    NaviPerfEventEntry()
        : next(nullptr)
        , sampleType(0)
        , branchEntry(nullptr)
        , callchainEntry(nullptr)
    {
    }
public:
    NaviPerfEventEntry *next;
    int32_t sampleType;
    NaviPerfBranchEntry *branchEntry;
    NaviPerfCallchainEntry *callchainEntry;
    union {
        char data[0];
        NaviPerfEvent normalEvent;
        NaviPerfEventThrottle throttleEvent;
        NaviPerfEventSwitch switchEvent;
        NaviPerfEventSwitchNew switchEventNew;
    };
};

class NaviPerfThread;
class NaviPerfResult;
NAVI_TYPEDEF_PTR(NaviPerfResult);

class NaviPerfResult
{
public:
    NaviPerfResult();
    ~NaviPerfResult();
private:
    NaviPerfResult(const NaviPerfResult &);
    NaviPerfResult &operator=(const NaviPerfResult &);
public:
    void setPrev(const NaviPerfResultPtr &prev);
    NaviPerfResultPtr stealPrev();
    bool addSubResult(const NaviPerfResultPtr &sub);
    int addEntry(NaviPerfEventEntry *entry);
    void stop();
    size_t getEventCount() const;
    void fillProto(KernelPerfDef *perf, StackTable &stackTable) const;
public:
    static uint64_t getSampleTime(NaviPerfEventEntry *entry);
private:
    void addToEntryList(NaviPerfEventEntry *entry);    
private:
    friend class NaviPerfThread;
private:
    int64_t _beginTime;
    int64_t _endTime;
    NaviPerfResultPtr _sub;
    NaviPerfResultPtr _prev;
    size_t _eventCount;
    NaviPerfEventEntry *_head;
    NaviPerfEventEntry *_tail;
};

}

