#include "autil/metric/Cpu.h"

#include <fstream>
#include <stdlib.h>

#include "autil/metric/MetricUtil.h"

using namespace std;

namespace autil {
namespace metric {

const string Cpu::CPU_PROC_STAT("/proc/stat");

Cpu::Cpu() {
    _diffTotal = 0;
    _procStatFile = CPU_PROC_STAT;
}

Cpu::~Cpu() {}

void Cpu::update() {
    ifstream fin(_procStatFile.c_str());
    string cpuLine;
    std::getline(fin, cpuLine);

    _prevStat = _curStat;
    parseCpuStat(cpuLine.c_str(), _curStat);
    _diffTotal = (_curStat.user - _prevStat.user + _curStat.nice - _prevStat.nice + _curStat.sys - _prevStat.sys +
                  _curStat.idle - _prevStat.idle + _curStat.iow - _prevStat.iow + _curStat.irq - _prevStat.irq +
                  _curStat.sirq - _prevStat.sirq + _curStat.steal - _prevStat.steal + _curStat.guest - _prevStat.guest);
}

void Cpu::parseCpuStat(const char *statStr, CpuStat &cpuStat) {
    //      user    nice  sys     idle       iow     irq   softirq steal guest
    // cpu  5177561 1668 5818435 8657470941 2376067 64343 2502729   0     0

    // skip the "cpu"
    const char *p = MetricUtil::skipToken(statStr);

#define EXTRACT_FIELD(field)                                                                                           \
    {                                                                                                                  \
        cpuStat.field = strtoll(p, NULL, 10);                                                                          \
        p = MetricUtil::skipToken(p);                                                                                  \
    }

    EXTRACT_FIELD(user);
    EXTRACT_FIELD(nice);
    EXTRACT_FIELD(sys);
    EXTRACT_FIELD(idle);
    EXTRACT_FIELD(iow);
    EXTRACT_FIELD(irq);
    EXTRACT_FIELD(sirq);
    EXTRACT_FIELD(steal);
    EXTRACT_FIELD(guest);

#undef EXTRACT_FIELD
}

void Cpu::setProcStatFile(const string &file) { _procStatFile = file; }

double Cpu::getIoWait() const {
    if (_prevStat.iow > _curStat.iow) {
        return 0;
    }
    return (_curStat.iow - _prevStat.iow) / (double)_diffTotal * 100;
}

} // namespace metric
} // namespace autil
