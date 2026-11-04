#include "navi/engine/CreatorStats.h"

namespace navi {

CreatorStats::CreatorStats()
    : _createLatencyFilter(FILTER_WINDOW_SIZE)
    , _initLatencyFilter(FILTER_WINDOW_SIZE)
{
    _createLatencyFilter.setCurrent(2000);
    _initLatencyFilter.setCurrent(2000);
}

CreatorStats::~CreatorStats() {
}

void CreatorStats::updateCreateLatency(float latencyUs) {
    _createLatencyFilter.update(latencyUs);
}

void CreatorStats::updateInitLatency(float latencyUs) {
    _initLatencyFilter.update(latencyUs);
}

float CreatorStats::getCreateAvgLatency() {
    return _createLatencyFilter.output();
}

float CreatorStats::getInitAvgLatency() {
    return _initLatencyFilter.output();
}

}

