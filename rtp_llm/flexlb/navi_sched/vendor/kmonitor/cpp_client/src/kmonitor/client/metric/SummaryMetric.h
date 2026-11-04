#pragma once
/*
 * Copyright 2014-2020 Alibaba Inc. All rights reserved.
 * Created on: 2019-07-11 19:00
 * Author Name: yixuan.ly
 * Author Email: yixuan.ly@alibaba-inc.com
 */

#include <string>

#include "kmonitor/client/StatisticsType.h"
#include "kmonitor/client/common/Common.h"
#include "kmonitor/client/common/MinMaxCalculator.h"
#include "kmonitor/client/common/SimpleQuantileCalculator.h"
#include "kmonitor/client/metric/DDSketch.h"
#include "kmonitor/client/metric/Metric.h"

BEGIN_KMONITOR_NAMESPACE(kmonitor);

class SummaryMetric : public Metric {
public:
    ~SummaryMetric();
    explicit SummaryMetric(const std::string &name, StatisticsType::type flag = StatisticsType::DEFAULT);
    void doUpdate(double value) override;
    void doSnapshot(MetricsRecord *record, int64_t period) override;

private:
    SummaryMetric(const SummaryMetric &);
    SummaryMetric &operator=(const SummaryMetric &);
    void flushSimpleQuantile();

private:
    MetricsInfoPtr summary_info_;
    std::vector<std::pair<MetricsInfoPtr, double>> quantile_info_;
    DDSketch *ddsketch_;
    MinMaxCalculator calculator_;
    std::optional<SimpleQuantileCalculator> quantile_calculator_; // 缓存原始值用于直接计算分位数
    static const double RELATIVE_ACCURACY;

private:
    friend class SummaryMetricTest_TestUpdate_Test;
    friend class SummaryMetricTest_TestSnapshot_Test;
};

TYPEDEF_PTR(SummaryMetric);

END_KMONITOR_NAMESPACE(kmonitor);
