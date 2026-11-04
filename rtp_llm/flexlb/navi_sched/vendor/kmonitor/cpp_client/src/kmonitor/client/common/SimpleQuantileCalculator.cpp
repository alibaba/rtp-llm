/*
 * Copyright 2014-2020 Alibaba Inc. All rights reserved.
 * Created on: 2017-09-06 10:00
 * Author Name: beifei
 * Author Email: beifei@taobao.com
 */

#include "kmonitor/client/common/SimpleQuantileCalculator.h"

#include <algorithm>
#include <math.h>

BEGIN_KMONITOR_NAMESPACE(kmonitor);

SimpleQuantileCalculator::SimpleQuantileCalculator(int64_t maxCount)
    : maxCount_(maxCount), valid_(true), ordered_(false) {
    measurements_.reserve(maxCount);
}

SimpleQuantileCalculator::~SimpleQuantileCalculator() {}

bool SimpleQuantileCalculator::Add(double value) {
    measurements_.push_back(value);
    ordered_ = false;
    return measurements_.size() < maxCount_;
}

double SimpleQuantileCalculator::GetValueAtQuantile(double quantile) {
    if (quantile < 0 || quantile > 1 || measurements_.empty()) {
        return 0;
    }

    double pos = quantile * (measurements_.size() + 1);
    size_t index = static_cast<size_t>(pos);
    if (!ordered_) {
        std::sort(measurements_.begin(), measurements_.end());
        ordered_ = true;
    }
    if (index < 1) {
        return measurements_.front();
    } else if (index >= measurements_.size()) {
        return measurements_.back();
    }
    double lower = measurements_.at(index - 1);
    double upper = measurements_.at(index);
    return lower + (pos - floor(pos)) * (upper - lower);
}

std::vector<double> SimpleQuantileCalculator::ExtractAll() {
    valid_ = false;
    return std::move(measurements_);
}

END_KMONITOR_NAMESPACE(kmonitor);
