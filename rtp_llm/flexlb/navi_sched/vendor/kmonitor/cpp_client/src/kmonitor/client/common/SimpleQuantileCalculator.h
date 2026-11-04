#pragma once
/*
 * Copyright 2014-2020 Alibaba Inc. All rights reserved.
 * Created on: 2017-09-06 10:00
 * Author Name: beifei
 * Author Email: beifei@taobao.com
 */

#include <vector>

#include "kmonitor/client/common/Common.h"
BEGIN_KMONITOR_NAMESPACE(kmonitor);

class UniformSnapshot;

class SimpleQuantileCalculator {
public:
    explicit SimpleQuantileCalculator(int64_t maxCount = 100);
    ~SimpleQuantileCalculator();

    size_t Size() { return measurements_.size(); }

    bool Valid() { return valid_; }

    bool Add(double value);
    double GetValueAtQuantile(double quantile);
    std::vector<double> ExtractAll();

private:
    SimpleQuantileCalculator(const SimpleQuantileCalculator &);
    SimpleQuantileCalculator &operator=(const SimpleQuantileCalculator &);

private:
    std::vector<double> measurements_;
    int64_t maxCount_;
    bool valid_;
    bool ordered_;
};

END_KMONITOR_NAMESPACE(kmonitor);
