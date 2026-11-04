#pragma once

#include "kmonitor/client/common/Common.h"

BEGIN_KMONITOR_NAMESPACE(kmonitor);

class StatisticsType {
public:
    using type = int;
    static bool NeedSummary(int flag);
    static bool NeedMinMax(int flag);
    static bool NeedP50(int flag);
    static bool NeedP75(int flag);
    static bool NeedP90(int flag);
    static bool NeedP95(int flag);
    static bool NeedP99(int flag);
    static bool NeedPercentile(int flag);

private:
    StatisticsType(const StatisticsType &);
    StatisticsType &operator=(const StatisticsType &);

public:
    static constexpr int SUMMARY = 1 << 0;
    static constexpr int MIN_MAX = 1 << 1;
    static constexpr int PERCENTILE_50 = 1 << 2;
    static constexpr int PERCENTILE_75 = 1 << 3;
    static constexpr int PERCENTILE_90 = 1 << 4;
    static constexpr int PERCENTILE_95 = 1 << 5;
    static constexpr int PERCENTILE_99 = 1 << 6;
    static constexpr int DEFAULT = SUMMARY | MIN_MAX;
};

END_KMONITOR_NAMESPACE(kmonitor);
