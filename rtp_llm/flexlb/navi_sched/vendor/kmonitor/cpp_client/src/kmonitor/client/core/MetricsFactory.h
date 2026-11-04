#pragma once
/*
 * Copyright 2014-2020 Alibaba Inc. All rights reserved.
 * Created on: 2017-02-27 10:05
 * Author Name: xsank.mz
 * Author Email: xsank.mz@alibaba-inc.com
 * */

#include <string>

#include "kmonitor/client/MetricType.h"
#include "kmonitor/client/StatisticsType.h"
#include "kmonitor/client/common/Common.h"

BEGIN_KMONITOR_NAMESPACE(kmonitor);

class Metric;

class MetricsFactory {
public:
    static Metric *
    CreateMetric(const std::string &name, MetricType metric_type, StatisticsType::type flag = StatisticsType::DEFAULT);

private:
    MetricsFactory(const MetricsFactory &);
    MetricsFactory &operator=(const MetricsFactory &);
};

END_KMONITOR_NAMESPACE(kmonitor);
