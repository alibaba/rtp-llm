/*
 * Copyright 2014-2020 Alibaba Inc. All rights reserved.
 * Created on: 2017-02-24 10:47
 * Author Name: xsank.mz
 * Author Email: xsank.mz@alibaba-inc.com
 */

#include "kmonitor/client/metric/SummaryMetric.h"

#include <memory>
#include <string>

#include "kmonitor/client/StatisticsType.h"
#include "kmonitor/client/core/MetricsInfo.h"
#include "kmonitor/client/metric/SummaryMetric.h"

BEGIN_KMONITOR_NAMESPACE(kmonitor);

using namespace std;

const double SummaryMetric::RELATIVE_ACCURACY = 0.01;

SummaryMetric::SummaryMetric(const string &name, StatisticsType::type flag) : Metric(name) {
    const string fullMetric = name + ".summary";
    ddsketch_ = new DDSketch(RELATIVE_ACCURACY);
    if (StatisticsType::NeedSummary(flag)) {
        summary_info_ = MetricsInfoPtr(new MetricsInfo(fullMetric, fullMetric, {{Metric::HEADER_FORMAT, "ddsketch"}}));
    }
    if (!StatisticsType::NeedMinMax(flag)) {
        info_ = nullptr;
    }
    if (StatisticsType::NeedP50(flag)) {
        quantile_info_.emplace_back(std::make_shared<MetricsInfo>(name + ".p50"), 0.5);
    }
    if (StatisticsType::NeedP75(flag)) {
        quantile_info_.emplace_back(std::make_shared<MetricsInfo>(name + ".p75"), 0.75);
    }
    if (StatisticsType::NeedP90(flag)) {
        quantile_info_.emplace_back(std::make_shared<MetricsInfo>(name + ".p90"), 0.90);
    }
    if (StatisticsType::NeedP95(flag)) {
        quantile_info_.emplace_back(std::make_shared<MetricsInfo>(name + ".p95"), 0.95);
    }
    if (StatisticsType::NeedP99(flag)) {
        quantile_info_.emplace_back(std::make_shared<MetricsInfo>(name + ".p99"), 0.99);
    }
    if (StatisticsType::NeedPercentile(flag)) {
        quantile_calculator_.emplace();
    }
}

void SummaryMetric::flushSimpleQuantile() {
    auto items = quantile_calculator_->ExtractAll();
    for (auto &item : items) {
        ddsketch_->accept(item);
    }
}

void SummaryMetric::doUpdate(double value) {
    if (::isnan(value)) {
        return;
    }
    if (info_) {
        calculator_.Add(value);
    }
    // 如果需要单机分位数，考虑到ddsketch在数据量小时的准确度可能不够，优先缓存起来使用排序算法。
    if (quantile_calculator_ && quantile_calculator_->Valid()) {
        bool isFull = !quantile_calculator_->Add(value);
        if (isFull) {
            flushSimpleQuantile();
        }
    } else {
        ddsketch_->accept(value);
    }
}

void SummaryMetric::doSnapshot(MetricsRecord *record, int64_t period) {
    if (ddsketch_ == nullptr) {
        ddsketch_ = new DDSketch(RELATIVE_ACCURACY);
        return;
    }
    if (info_ && calculator_.Count() > 0) {
        record->AddValue(info_, calculator_.ToString());
        calculator_.Reset();
    }
    ddsketch_->Flush();
    // 如果需要直接写出分位数的值
    for (auto &item : quantile_info_) {
        double value;
        if (quantile_calculator_ && quantile_calculator_->Valid() && quantile_calculator_->Size() > 0) {
            value = quantile_calculator_->GetValueAtQuantile(item.second);
            record->AddValue(item.first, value);
        } else if (ddsketch_->getCount() > 0) {
            value = ddsketch_->getValueAtQuantile(item.second);
            record->AddValue(item.first, value);
        }
    }
    // 汇报ddsketch的原始值，不进行计算
    if (summary_info_) {
        if (quantile_calculator_ && quantile_calculator_->Valid() && quantile_calculator_->Size() > 0) {
            flushSimpleQuantile();
        }
        ddsketch_->Flush();
        if (ddsketch_->getCount() > 0) {
            record->AddValue(summary_info_, autil::legacy::FastToJsonString(ddsketch_, false));
        }
    }
    if (quantile_calculator_) {
        quantile_calculator_.emplace();
    }
    delete ddsketch_;
    ddsketch_ = new DDSketch(RELATIVE_ACCURACY);
}

SummaryMetric::~SummaryMetric() {
    summary_info_.reset();
    delete ddsketch_;
}

END_KMONITOR_NAMESPACE(kmonitor);
