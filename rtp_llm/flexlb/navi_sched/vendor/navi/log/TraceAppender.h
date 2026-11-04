#pragma once

#include "autil/Lock.h"
#include "navi/log/AlogAppender.h"
#include "navi/log/LogBtFilter.h"
#include "navi/log/TraceCollector.h"

namespace navi {

class NaviLogManager;

typedef NaviUnorderedMap<void *, std::string> ObjectPrefixMap;

class TraceAppender : public Appender
{
public:
    TraceAppender(const std::string &levelStr);
    TraceAppender(LogLevel level);
    ~TraceAppender();
private:
    TraceAppender &operator=(const TraceAppender &);
public:
    void setSubAppender(std::shared_ptr<Appender> subAppender) { _subAppender = std::move(subAppender); }
    void addBtFilter(const LogBtFilterParam &param);
    void setLogManager(NaviLogManager *logManager);
    int append(LoggingEvent &event) override;
    void setFirstErrorEvent(const LoggingEventPtr &event);
    LoggingEventPtr firstErrorEvent() const;
    void clearFirstErrorEvent();
    void flush() override {
    }
    void collectTrace(TraceCollector &collector);
    void stop();
private:
    bool logBt(const LoggingEvent &event);
    void fillObjectPrefix(LoggingEvent &event);
private:
    mutable autil::ThreadMutex _lock;
    LoggingEventPtr _firstErrorEvent;
    volatile bool _stopped = false;
    ObjectPrefixMap _prefixMap;
    TraceCollector _traceCollector;
    LogBtFilter _btFilter;
    std::shared_ptr<Appender> _subAppender;
};

}
