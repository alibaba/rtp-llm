#include "navi/log/TraceAppender.h"
#include "navi/log/LoggingEvent.h"
#include "navi/log/NaviLogManager.h"
#include "navi/engine/Node.h"
#include "navi/engine/Edge.h"
#include <iostream>

namespace navi {

TraceAppender::TraceAppender(const std::string &levelStr)
    : TraceAppender(getLevelByString(levelStr))
{
}

TraceAppender::TraceAppender(LogLevel level)
    : Appender(level, nullptr)
{
}

TraceAppender::~TraceAppender() {
}

void TraceAppender::addBtFilter(const LogBtFilterParam &param) {
    _btFilter.addParam(param);
}

void TraceAppender::setLogManager(NaviLogManager *logManager) {
    _manager = logManager;
}

void TraceAppender::fillObjectPrefix(LoggingEvent &event) {
    if (LOT_UNKNOWN == event.objectType || !event.object || !event.prefix.empty()) {
        return;
    }
    auto it = _prefixMap.find(event.object);
    if (_prefixMap.end() != it) {
        event.prefix = it->second;
        return;
    }
    std::string prefix;
    switch (event.objectType) {
    case LOT_NODE: {
        prefix = ((Node *)event.object)->getLogPrefix();
        break;
    }
    case LOT_EDGE: {
        prefix = ((Edge *)event.object)->getLogPrefix();
        break;
    }
    default:
        break;
    }
    _prefixMap[event.object] = prefix;
    event.prefix = std::move(prefix);
}

int TraceAppender::append(LoggingEvent& event) {
    if (_stopped) {
        return 0;
    }
    auto traceEvent = event;
    if (unlikely(_btFilter.pass(traceEvent))) {
        if (_manager) {
            traceEvent.bt = _manager->getCurrentBtString();
        }
    }

    if (_subAppender) {
        _subAppender->append(traceEvent);
    }

    autil::ScopedLock lock(_lock);
    fillObjectPrefix(traceEvent);
    _traceCollector.append(std::move(traceEvent));

    return 0;
}

void TraceAppender::setFirstErrorEvent(const LoggingEventPtr &event) {
    autil::ScopedLock lock(_lock);
    if (!_firstErrorEvent) {
        _firstErrorEvent = event;
    }
}

LoggingEventPtr TraceAppender::firstErrorEvent() const {
    autil::ScopedLock lock(_lock);
    return _firstErrorEvent;
}

void TraceAppender::clearFirstErrorEvent() {
    LoggingEventPtr errorEvent;
    {
        autil::ScopedLock lock(_lock);
        errorEvent = std::move(_firstErrorEvent);
    }
}

void TraceAppender::collectTrace(TraceCollector &collector) {
    autil::ScopedLock lock(_lock);
    collector.merge(_traceCollector);
}

void TraceAppender::stop() {
    _stopped = true;
}

}
