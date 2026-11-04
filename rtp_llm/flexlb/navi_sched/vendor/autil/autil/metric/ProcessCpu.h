/*
 * Copyright 2014-2020 Alibaba Inc. All rights reserved.
 * Created on: 2019-07-24 10:47
 * Author Name: yixuan.ly
 * Author Email: yixuan.ly@alibaba-inc.com
 */

#pragma once

#include <memory>
#include <string>

#include "autil/metric/Cpu.h"

namespace autil {
namespace metric {

struct ProcessCpuStat {
    jiffies_t utime = 0;
    jiffies_t stime = 0;
    jiffies_t cutime = 0;
    jiffies_t cstime = 0;

    ProcessCpuStat() = default;

    void reset() {
        utime = 0;
        stime = 0;
        cutime = 0;
        cstime = 0;
    }
};

class ProcessCpu {
public:
    ProcessCpu();
    ~ProcessCpu();

private:
    ProcessCpu(const ProcessCpu &);
    ProcessCpu &operator=(const ProcessCpu &);

public:
    double getUsage();

private:
    jiffies_t getProcessTime();
    void parseProcCPUStat(std::string line, jiffies_t &processCpuTotTime);
    void getProcessCpuStat();
    friend class ProcessCpuTest;

private:
    Cpu _cpu;
    jiffies_t _curProcessCpuTotTime;
    jiffies_t _preProcessCpuTotTime;
};

typedef std::shared_ptr<ProcessCpu> ProcessCpuPtr;

} // namespace metric
} // namespace autil
