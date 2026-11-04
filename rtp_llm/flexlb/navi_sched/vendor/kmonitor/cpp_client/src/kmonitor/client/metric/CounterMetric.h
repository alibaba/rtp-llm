#pragma once
/*
 * Copyright 2014-2020 Alibaba Inc. All rights reserved.
 * Created on: 2017-02-24 10:47
 * Author Name: xsank.mz
 * Author Email: xsank.mz@alibaba-inc.com
 */

#include <string>

#include "kmonitor/client/common/Common.h"
#include "kmonitor/client/metric/Metric.h"

BEGIN_KMONITOR_NAMESPACE(kmonitor);

class CounterMetric : public Metric {
public:
    virtual ~CounterMetric();
    explicit CounterMetric(const std::string &name);
    virtual void doUpdate(double value) override;
    virtual void doSnapshot(MetricsRecord *record, int64_t period) override;

    void clear() override { value_ = 0; }

private:
    CounterMetric(const CounterMetric &);
    CounterMetric &operator=(const CounterMetric &);

private:
    double value_;

private:
    friend class CounterMetricTest_TestUpdate_Test;
    friend class CounterMetricTest_TestSnapshot_Test;
};

TYPEDEF_PTR(CounterMetric);

END_KMONITOR_NAMESPACE(kmonitor);
