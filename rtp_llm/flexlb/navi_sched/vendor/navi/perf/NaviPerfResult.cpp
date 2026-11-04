#include "navi/perf/NaviPerfResult.h"
#include "navi/proto/GraphVis.pb.h"
#include "navi/util/CommonUtil.h"

namespace navi {

void NaviPerfEventThrottle::fillProto(PerfEventDef *perfEvent) const {
    perfEvent->set_time(time);
}

void NaviPerfEvent::fillProto(PerfEventDef *perfEvent,
                              bool pageFault,
                              NaviPerfCallchainEntry *callchainEntry,
                              NaviPerfBranchEntry *branchEntry,
                              StackTable &stackTable) const
{
    perfEvent->set_pid(pid);
    perfEvent->set_tid(tid);
    perfEvent->set_time(time);
    perfEvent->set_cpu(cpu);
    perfEvent->set_event_addr(addr);
    int64_t branchNum = branchEntry ? branchEntry->bnr : 0;
    int64_t matchIndex = -1;
    int64_t callchainNum = callchainEntry ? callchainEntry->nr : 0;
    int32_t count = 0;
    uint64_t addrs[branchNum + callchainNum + 2];
    if (pageFault) {
        addrs[count++] = addr;
    }
    if (branchNum == 0) {
        matchIndex = 0;
    } else {
        uint64_t matchIp = branchEntry->lbr[branchNum - 1].from;
        for (int64_t i = 1; i < callchainNum; i++) {
            int64_t diff = matchIp - callchainEntry->ips[i];
            if (abs(diff) <= 5) {
                matchIndex = i;
                break;
            }
        }
        auto ip0 = branchEntry->lbr[0].to;
        addrs[count++] = ip0;
        for (int64_t i = 0; i < branchNum; i++) {
            addrs[count++] = branchEntry->lbr[i].from;
        }
    }
    if (matchIndex >= 0) {
        for (int64_t i = matchIndex + 1; i < callchainNum; ++i) {
            addrs[count++] = callchainEntry->ips[i];
        }
    }
    auto hash = stackTable.addStack(count, addrs);
    perfEvent->set_stack_hash(hash);
}

void NaviPerfEventSwitch::fillProto(PerfEventDef *perfEvent) const {
    perfEvent->set_pid(pid);
    perfEvent->set_tid(tid);
    perfEvent->set_time(time);
    perfEvent->set_cpu(cpu);
}

void NaviPerfEventSwitchNew::fillProto(PerfEventDef *perfEvent,
                                       NaviPerfCallchainEntry *callchainEntry,
                                       StackTable &stackTable) const {
    perfEvent->set_pid(pid);
    perfEvent->set_tid(tid);
    perfEvent->set_time(time);
    perfEvent->set_cpu(cpu);
    perfEvent->set_next_prev_pid(next_prev_pid);
    perfEvent->set_next_prev_tid(next_prev_tid);
    if (!callchainEntry || callchainEntry->nr <= 1) {
        return;
    }
    auto hash = stackTable.addStack(callchainEntry->nr - 1, &callchainEntry->ips[1]);
    perfEvent->set_stack_hash(hash);
}

NaviPerfResult::NaviPerfResult()
    : _beginTime(CommonUtil::getTimelineTimeNs())
    , _endTime(_beginTime)
    , _eventCount(0)
    , _head(nullptr)
    , _tail(nullptr)
{
}

NaviPerfResult::~NaviPerfResult() {
    auto entry = _head;
    while (entry) {
        auto next = entry->next;
        free(entry);
        entry = next;
    }
}

void NaviPerfResult::stop() {
    _endTime = CommonUtil::getTimelineTimeNs();
}

void NaviPerfResult::fillProto(KernelPerfDef *perf, StackTable &stackTable) const {
    auto entry = _head;
    while (entry) {
        auto perfEvent = perf->add_events();
        switch (entry->sampleType) {
        case PET_SAMPLE:
            entry->normalEvent.fillProto(perfEvent, false, entry->callchainEntry,
                                         entry->branchEntry, stackTable);
            break;
        case PET_PAGE_FAULT_MAJOR:
        case PET_PAGE_FAULT_MINOR:
            entry->normalEvent.fillProto(perfEvent, true, entry->callchainEntry,
                                         entry->branchEntry, stackTable);
            break;
        case PET_THROTTLE:
        case PET_UNTHROTTLE:
            entry->throttleEvent.fillProto(perfEvent);
            break;
        case PET_SWITCH_IN:
        case PET_SWITCH_OUT:
        case PET_SWITCH_OUT_PREEMPT:
            entry->switchEvent.fillProto(perfEvent);
            break;
        case PET_SWITCH_IN_NEW:
        case PET_SWITCH_OUT_NEW:
        case PET_SWITCH_OUT_PREEMPT_NEW:
            entry->switchEventNew.fillProto(perfEvent, entry->callchainEntry, stackTable);
            break;
        case PET_UNKNOWN:
        default:
            assert(false);
            break;
        }
        perfEvent->set_sample_type((PerfEventType)entry->sampleType);
        entry = entry->next;
    }
}

bool NaviPerfResult::addSubResult(const NaviPerfResultPtr &sub) {
    if (_sub) {
        return _sub->addSubResult(sub);
    }
    if (_endTime == _beginTime) {
        if (sub->_beginTime > _beginTime) {
            _sub = sub;
            return true;
        }
    } else {
        if (sub->_beginTime > _beginTime && sub->_endTime < _endTime) {
            _sub = sub;
            return true;
        }
    }
    return false;
}

int NaviPerfResult::addEntry(NaviPerfEventEntry *entry) {
    if (_sub) {
        auto ret = _sub->addEntry(entry);
        switch (ret) {
        case -1:
            // entry too old
            break;
        case 0:
            // fail
            _sub.reset();
            break;
        case 1:
            // success
            return 1;
        default:
            assert(false);
            break;
        }
    }
    auto sampleTime = getSampleTime(entry);
    if (sampleTime < _beginTime) {
        return -1;
    } else if (_beginTime == _endTime || sampleTime < _endTime) {
        addToEntryList(entry);
        return 1;
    } else {
        return 0;
    }
}

void NaviPerfResult::addToEntryList(NaviPerfEventEntry *entry) {
    if (!_head) {
        _head = entry;
        _tail  = entry;
    } else {
        _tail->next = entry;
        _tail = entry;
    }
    _eventCount++;
}

void NaviPerfResult::setPrev(const NaviPerfResultPtr &prev) {
    _prev = prev;
}

NaviPerfResultPtr NaviPerfResult::stealPrev() {
    return std::move(_prev);
}

size_t NaviPerfResult::getEventCount() const {
    return _eventCount;
}

uint64_t NaviPerfResult::getSampleTime(NaviPerfEventEntry *entry) {
    switch (entry->sampleType) {
    case PET_SAMPLE:
    case PET_PAGE_FAULT_MAJOR:
    case PET_PAGE_FAULT_MINOR:
        return entry->normalEvent.time;
    case PET_THROTTLE:
    case PET_UNTHROTTLE:
        return entry->throttleEvent.time;
    case PET_SWITCH_IN:
    case PET_SWITCH_OUT:
    case PET_SWITCH_OUT_PREEMPT:
        return entry->switchEvent.time;
    case PET_SWITCH_IN_NEW:
    case PET_SWITCH_OUT_NEW:
    case PET_SWITCH_OUT_PREEMPT_NEW:
        return entry->switchEventNew.time;
    case PET_UNKNOWN:
    default:
        assert(false);
        return 0;
    }
}

}

