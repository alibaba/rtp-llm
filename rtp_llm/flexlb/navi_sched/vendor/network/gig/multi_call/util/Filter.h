#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/atomic.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/ControllerParam.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/util/CompareFlag.h"

namespace multi_call {

class Filter
{
public:
    Filter(int32_t windowSize);
    ~Filter();

private:
    Filter(const Filter &);
    Filter &operator=(const Filter &);

public:
    bool update(float input, float truncate = INVALID_FILTER_VALUE);
    bool isValid() const;
    float output() const;
    int64_t count() const;
    void truncate(float value);
    void setCurrent(float current);
    void setCounter(int64_t counter);
    int32_t getWindowSize() const;
    CompareFlag compare(const Filter &rhs) const;
    float getAlpha() const;
    void reset();
    std::string toString() const;
    void setMinValue(float minValue);

private:
    int32_t _windowSize;
    float _alpha;
    float _minValue;
    volatile FloatIntUnion _current;
    atomic64_t _filterCounter;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(Filter);

} // namespace multi_call

