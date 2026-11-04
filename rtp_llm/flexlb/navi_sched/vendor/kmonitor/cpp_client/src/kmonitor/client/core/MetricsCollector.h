#pragma once
/*
 * Copyright 2014-2020 Alibaba Inc. All rights reserved.
 * Created on: 2017-02-27 11:00
 * Author Name: xsank.mz
 * Author Email: xsank.mz@alibaba-inc.com
 * */

#include <string>
#include <vector>

#include "kmonitor/client/common/Common.h"
#include "kmonitor/client/core/MetricsRecord.h"

BEGIN_KMONITOR_NAMESPACE(kmonitor);

class MetricsRecord;

class MetricsCollector {
public:
    MetricsCollector();
    ~MetricsCollector();

    MetricsRecord *AddRecord(const std::string &name, const MetricsTagsPtr &tags, int64_t curTime);

    void Clear();
    MetricsRecords StealRecords();
    const MetricsRecords &GetRecords() const;

private:
    MetricsCollector(const MetricsCollector &);
    MetricsCollector &operator=(const MetricsCollector &);

private:
    MetricsRecords records_;
};

END_KMONITOR_NAMESPACE(kmonitor);
