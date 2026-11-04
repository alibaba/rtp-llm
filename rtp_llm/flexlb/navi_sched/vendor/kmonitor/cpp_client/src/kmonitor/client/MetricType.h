#pragma once
/*
 * Copyright 2014-2020 Alibaba Inc. All rights reserved.
 * Created on: 2017-02-27 11:32
 * Author Name: xsank.mz
 * Author Email: xsank.mz@alibaba-inc.com
 * */

#include <strings.h> // strcasecmp

#include "kmonitor/client/common/Common.h"

BEGIN_KMONITOR_NAMESPACE(kmonitor);

enum MetricType : unsigned int {
    GAUGE,
    SUMMARY,
    COUNTER,
    QPS,
    RAW,
    STATUS
};

inline static MetricType MetricTypeFromString(const std::string &s) {
    MetricType type = GAUGE;
    if (strcasecmp("GAUGE", s.c_str()) == 0) {
        type = GAUGE;
    } else if (strcasecmp("SUMMARY", s.c_str()) == 0) {
        type = SUMMARY;
    } else if (strcasecmp("COUNTER", s.c_str()) == 0) {
        type = COUNTER;
    } else if (strcasecmp("QPS", s.c_str()) == 0) {
        type = QPS;
    } else if (strcasecmp("RAW", s.c_str()) == 0) {
        type = RAW;
    } else if (strcasecmp("STATUS", s.c_str()) == 0) {
        type = STATUS;
    }
    return type;
}

inline static std::string MetricTypeToString(MetricType type) {
    switch (type) {
    case GAUGE:
        return "GAUGE";
    case SUMMARY:
        return "SUMMARY";
    case COUNTER:
        return "COUNTER";
    case QPS:
        return "QPS";
    case RAW:
        return "RAW";
    case STATUS:
        return "STATUS";
    default:
        return "unknown";
    }
}

END_KMONITOR_NAMESPACE(kmonitor);
