#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/atomic.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"

namespace multi_call {

class DiffCounter
{
public:
    DiffCounter();
    ~DiffCounter();

private:
    DiffCounter(const DiffCounter &);
    DiffCounter &operator=(const DiffCounter &);

public:
    void inc();
    int64_t snapshot();
    int64_t current() const;
    void setCurrent(int64_t value);
    std::string snapshotStr();

private:
    int64_t _snapshot;
    atomic64_t _counter;
};

} // namespace multi_call
