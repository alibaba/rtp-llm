#include "autil/metric/ProcessCpu.h"

#include <fstream> // IWYU pragma: keep
#include <sstream> // IWYU pragma: keep
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

using namespace std;

namespace autil {
namespace metric {

ProcessCpu::ProcessCpu() {
    _curProcessCpuTotTime = 0;
    _preProcessCpuTotTime = 0;
}

ProcessCpu::~ProcessCpu() {}

double ProcessCpu::getUsage() {
    _cpu.update();
    _preProcessCpuTotTime = _curProcessCpuTotTime;
    getProcessCpuStat();
    jiffies_t totalTime = _cpu.getTotalDiff();
    jiffies_t processTime = getProcessTime();
    if (processTime >= totalTime) {
        return 100.0;
    }
    return processTime / (double)totalTime * 100.0;
}

void ProcessCpu::getProcessCpuStat() {
    char filePath[1024];
    pid_t pid = getpid();
    snprintf(filePath, sizeof(filePath), "/proc/%d/stat", pid);

    ifstream in(filePath);
    if (!in) {
        return;
    }

    string line;
    std::getline(in, line);
    return parseProcCPUStat(line, _curProcessCpuTotTime);
}

void ProcessCpu::parseProcCPUStat(string line, jiffies_t &processCpuTotTime) {
    /*
     * the line string in /proc/[pid]/stat is:
     * pid %d, comm %s, state %c, ppid %d, pgrp %d, session %d,
     * tty_nr %d, tpgid %d, flags %u, minflt %lu, cminflt %lu,
     * majflt %lu, cmajflt %lu,
     * utime %lu, stime %lu, cutime %ld, cstime %ld,
     * ...
     */
    struct ProcessCpuStat processCpuStat;
    size_t pos = 0;
    for (int32_t i = 0; i < 13 && pos != string::npos; i++) {
        pos = line.find(" ", pos) + 1;
    }

    if (pos == string::npos) {
        return;
    }

    line = line.substr(pos);

    istringstream iss(line);
    iss >> processCpuStat.utime >> processCpuStat.stime >> processCpuStat.cutime >> processCpuStat.cstime;
    processCpuTotTime = processCpuStat.utime + processCpuStat.stime + processCpuStat.cutime + processCpuStat.cstime;
}

jiffies_t ProcessCpu::getProcessTime() {
    if (_curProcessCpuTotTime <= _preProcessCpuTotTime) {
        return 0;
    }
    return _curProcessCpuTotTime - _preProcessCpuTotTime;
}

} // namespace metric
} // namespace autil
