#include "kmonitor/client/StatisticsType.h"

BEGIN_KMONITOR_NAMESPACE(kmonitor);

bool StatisticsType::NeedSummary(int flag) { return (flag & StatisticsType::SUMMARY) > 0; }

bool StatisticsType::NeedMinMax(int flag) { return (flag & StatisticsType::MIN_MAX) > 0; }

bool StatisticsType::NeedP50(int flag) { return (flag & StatisticsType::PERCENTILE_50) > 0; }

bool StatisticsType::NeedP75(int flag) { return (flag & StatisticsType::PERCENTILE_75) > 0; }

bool StatisticsType::NeedP90(int flag) { return (flag & StatisticsType::PERCENTILE_90) > 0; }

bool StatisticsType::NeedP95(int flag) { return (flag & StatisticsType::PERCENTILE_95) > 0; }

bool StatisticsType::NeedP99(int flag) { return (flag & StatisticsType::PERCENTILE_99) > 0; }

bool StatisticsType::NeedPercentile(int flag) {
    return NeedP50(flag) || NeedP75(flag) || NeedP90(flag) || NeedP95(flag) || NeedP99(flag);
}

END_KMONITOR_NAMESPACE(kmonitor);
