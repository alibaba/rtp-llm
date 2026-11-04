#pragma once

#include "navi/common.h"

namespace navi {

#define COLLECT_METRIC_FLAG (1)
#define COLLECT_PERF_FLAG (1u << 1)
#define COLLECT_CONTEXT_SWITCH_FLAG (1u << 2)
#define COLLECT_PAGE_FAULT_FLAG (1u << 3)
#define COLLECT_MALLOC_FLAG (1u << 4)
#define NEED_PERF_SCOPE_FLAG (1u << 5)
#define COLLECT_EDGE_DATA_FLAG (1u << 6)
#define COLLECT_SNAPSHOT_FLAG (1u << 7)
#define PERF_COLLECT_FLAGS (COLLECT_PERF_FLAG | COLLECT_CONTEXT_SWITCH_FLAG | COLLECT_PAGE_FAULT_FLAG)

class CollectFlags {
public:
    explicit CollectFlags(uint64_t flags = 0)
        : _flags(flags)
    {
    }
public:
    bool needCollect() const {
        return ((_flags & ~(NEED_PERF_SCOPE_FLAG | COLLECT_SNAPSHOT_FLAG)) != 0);
    }
    bool needHostInfo() const {
        return ((_flags & ~NEED_PERF_SCOPE_FLAG) != 0);
    }
    uint64_t getPerfFlags() const {
        return _flags & PERF_COLLECT_FLAGS;
    }
    void setPerfFlags(uint64_t flags) {
        _flags = ((_flags & (~PERF_COLLECT_FLAGS)) | flags);
    }
    bool needPerfCollect() const {
        return _flags & (COLLECT_PERF_FLAG | COLLECT_CONTEXT_SWITCH_FLAG | COLLECT_PAGE_FAULT_FLAG);
    }
    void setFlags(uint64_t flags) {
        _flags = flags;
    }
    uint64_t getFlags() const {
        return _flags;
    }
    void setCollectEdgeData(bool collect) {
        if (collect) {
            _flags |= COLLECT_EDGE_DATA_FLAG;
        } else {
            _flags &= ~COLLECT_EDGE_DATA_FLAG;
        }
    }
    void setCollectMetric(bool collect) {
        if (collect) {
            _flags |= COLLECT_METRIC_FLAG;
        } else {
            _flags &= ~COLLECT_METRIC_FLAG;
        }
    }
    bool collectMetric() const {
        return _flags & COLLECT_METRIC_FLAG;
    }
    void setCollectPerf(bool collect) {
        if (collect) {
            _flags |= COLLECT_PERF_FLAG;
        } else {
            _flags &= ~COLLECT_PERF_FLAG;
        }
    }
    bool collectPerf() const {
        return _flags & COLLECT_PERF_FLAG;
    }
    void setCollectContextSwitch(bool collect) {
        if (collect) {
            _flags |= COLLECT_CONTEXT_SWITCH_FLAG;
        } else {
            _flags &= ~COLLECT_CONTEXT_SWITCH_FLAG;
        }
    }
    bool collectContextSwitch() const {
        return _flags & COLLECT_CONTEXT_SWITCH_FLAG;
    }
    void setCollectPageFault(bool collect) {
        if (collect) {
            _flags |= COLLECT_PAGE_FAULT_FLAG;
        } else {
            _flags &= ~COLLECT_PAGE_FAULT_FLAG;
        }
    }
    bool collectPageFault() const {
        return _flags & COLLECT_PAGE_FAULT_FLAG;
    }
    void setCollectMalloc(bool collect) {
        if (collect) {
            _flags |= COLLECT_MALLOC_FLAG;
        } else {
            _flags &= ~COLLECT_MALLOC_FLAG;
        }
    }
    bool collectMalloc() const {
        return _flags & COLLECT_MALLOC_FLAG;
    }
    void setCollectSnapshot(bool collect) {
        if (collect) {
            _flags |= COLLECT_SNAPSHOT_FLAG;
        } else {
            _flags &= ~COLLECT_SNAPSHOT_FLAG;
        }
    }
    bool collectSnapshot() const {
        return _flags & COLLECT_SNAPSHOT_FLAG;
    }
private:
    uint64_t _flags = 0;
};

thread_local extern CollectFlags current_collect_flags;

#define NAVI_COLLECT_MALLOC() (navi::current_collect_flags.collectMalloc())

}
