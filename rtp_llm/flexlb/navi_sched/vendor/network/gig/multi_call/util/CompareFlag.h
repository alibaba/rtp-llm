#pragma once

#include <assert.h>

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"

namespace multi_call {

enum CompareFlagEnum { CF_BETTER = 1, CF_WORSE = 0, CF_UNKNOWN = -1 };

class CompareFlag
{
public:
    CompareFlag(CompareFlagEnum flag) : _flag(flag) {
    }
    CompareFlag(bool flag) : _flag(CompareFlagEnum(flag)) {
    }
    bool isValid() const {
        return CF_UNKNOWN != _flag;
    }
    operator bool() const {
        assert(_flag != CF_UNKNOWN);
        return _flag == CF_BETTER;
    }
    bool operator==(const CompareFlag &rhs) const {
        return _flag == rhs._flag;
    }
    bool operator!=(const CompareFlag &rhs) const {
        return _flag != rhs._flag;
    }

private:
    CompareFlagEnum _flag;
};

extern const CompareFlag BetterCompareFlag;
extern const CompareFlag WorseCompareFlag;
extern const CompareFlag UnknownCompareFlag;

} // namespace multi_call
