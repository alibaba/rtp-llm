#include "navi/log/NaviLogger.h"
#include "navi/engine/CollectFlags.h"
#include "navi/log/AlogAppender.h"
#include "navi/log/LoggingEvent.h"
#include "navi/log/NaviLogManager.h"
#include "navi/log/TraceAppender.h"
#include "navi/log/ConsoleAppender.h"
#include "navi/util/CommonUtil.h"
#include <stdarg.h>

namespace navi {

thread_local const NaviObjectLogger *NAVI_TLS_LOGGER = nullptr;
thread_local LogObjectType NAVI_TLS_LOGGER_OBJECT_TYPE = LOT_UNKNOWN;
thread_local void *NAVI_TLS_LOGGER_OBJECT = nullptr;

NaviLogger::NaviLogger(size_t maxMessageLength,
                       const std::shared_ptr<NaviLogManager> &logManager)
    : _level(LOG_LEVEL_DISABLE)
    , _stopped(false)
    , _maxMessageLength(maxMessageLength)
    , _logManager(logManager)
    , _hasError(false)
{
}

NaviLogger::NaviLogger(const NaviLogger &other, bool reuseTraceAppender)
    : _id(other._id)
    , _name(other._name)
    , _level(other._level)
    , _stopped(false)
    , _maxMessageLength(other._maxMessageLength)
    , _logManager(other._logManager)
    , _hasError(false)
{
    copyAppenders(other, reuseTraceAppender);
}

NaviLogger::~NaviLogger() {
    auto _logger = this;
    NAVI_LOG(SCHEDULE2, "destruct: %p", this);
}

void NaviLogger::copyAppenders(const NaviLogger &other, bool reuseTraceAppender) {
    _appenders = _logManager->getFileAppenders();
    _ownAppenders = other._ownAppenders;
    if (reuseTraceAppender) {
        _traceAppender = other._traceAppender;
        _traceAppenderPtr = _traceAppender.get();
    }
    updateLogLevel();
}

void NaviLogger::init(SessionId id) {
    setLoggerId(id);
    _appenders = _logManager->getFileAppenders();
    updateLogLevel();
    auto _logger = this;
    NAVI_LOG(SCHEDULE2, "construct: %p", this);
}

void NaviLogger::stop() {
    bool expect = false;
    _stopped.compare_exchange_weak(expect, true);
}

bool NaviLogger::stopped() {
    return _stopped.load(std::memory_order_relaxed);
}

void NaviLogger::setLoggerId(SessionId id) {
    _id = id;
    updateName();
}

SessionId NaviLogger::getLoggerId() const {
    return _id;
}

const std::string &NaviLogger::getName() const {
    return _name;
}

void NaviLogger::addAppender(const std::shared_ptr<Appender> &appender) {
    if (!appender) {
        return;
    }
    _ownAppenders.push_back(appender);
    _appenders.push_back(appender.get());
    updateLogLevel();
}

LogLevel NaviLogger::getTraceLevel() const {
    return _traceAppender ? _traceAppender->getLevel() : LOG_LEVEL_DISABLE;
}

bool NaviLogger::hasTraceAppender() const {
    return _traceAppender != nullptr;
}

void NaviLogger::stopTraceAppender() {
    if (_traceAppenderPtr) {
        _traceAppenderPtr->stop();
    }
}

std::shared_ptr<TraceAppender> NaviLogger::getTraceAppender() const {
    return _traceAppender;
}

std::shared_ptr<ConsoleAppender> NaviLogger::getConsoleAppender() const {
    for (const auto &appender : _ownAppenders) {
        auto consoleAppender = std::dynamic_pointer_cast<ConsoleAppender>(appender);
        if (consoleAppender) {
            return consoleAppender;
        }
    }
    return nullptr;
}

void NaviLogger::setTraceAppender(const std::shared_ptr<TraceAppender> &appender) {
    if (!appender) {
        return;
    }
    appender->setLogManager(_logManager.get());
    NAVI_MEMORY_BARRIER();
    _traceAppenderPtr = appender.get();
    _traceAppender = appender;
    updateLogLevel();
}

void NaviLogger::collectTrace(TraceCollector &collector) {
    if (_traceAppender) {
        _traceAppender->collectTrace(collector);
    }
}

void NaviLogger::updateLogLevel() {
    LogLevel maxLevel = LOG_LEVEL_DISABLE;
    for (auto appender : _appenders) {
        maxLevel = std::max(maxLevel, appender->getLevel());
    }
    if (_traceAppenderPtr) {
        maxLevel = std::max(maxLevel, _traceAppenderPtr->getLevel());
    }
    _level = maxLevel;
}

void NaviLogger::updateTraceLevel(const std::string &levelStr) {
    if (_traceAppender) {
        _traceAppender->setLevel(levelStr);
        updateLogLevel();
    }
}

void NaviLogger::updateName() {
    _name = CommonUtil::formatSessionId(_id);
}

char *NaviLogger::getTlsBuffer(size_t maxMessageLength) {
    static thread_local size_t bufferSize = maxMessageLength;
    static thread_local std::unique_ptr<char[]> bufferStorage(new char[bufferSize]);
    if (unlikely(bufferSize != maxMessageLength)) {
        bufferSize = maxMessageLength;
        bufferStorage.reset(new char[bufferSize]);
    }
    return bufferStorage.get();
}

void NaviLogger::log(LogLevel level,
                     AppenderTarget target,
                     void *object,
                     LogObjectType objectType,
                     const std::string &prefix,
                     const char *file,
                     int line,
                     const char *func,
                     const char *fmt,
                     va_list ap)
{
    if (__builtin_expect(stopped(), 0)) {
        return;
    }
    char *buffer = getTlsBuffer(_maxMessageLength);
    vsnprintf(buffer, _maxMessageLength, fmt, ap);
    LoggingEvent event(_name, object, objectType, prefix, buffer, level, file, line, func);
    _log(event, target);
}

void NaviLogger::_log(LoggingEvent& event, AppenderTarget target)
{
    if (event.level <= LOG_LEVEL_ERROR && !_hasError) {
        auto traceLevel = _traceAppenderPtr ? _traceAppenderPtr->getLevel() : LOG_LEVEL_DISABLE;
        std::string btStr;
        if (_traceAppenderPtr && LOG_LEVEL_DISABLE != traceLevel) {
            btStr = getCurrentBtString();
        }
        LoggingEventPtr eventPtr;
        {
            autil::ScopedLock lock(_lock);
            if (!_hasError) {
                _hasError = true;
                eventPtr = std::make_shared<LoggingEvent>(event);
                eventPtr->bt = std::move(btStr);
                _firstErrorEvent = eventPtr;
            }
        }
        if (_traceAppenderPtr && LOG_LEVEL_DISABLE != traceLevel && eventPtr) {
            _traceAppenderPtr->setFirstErrorEvent(eventPtr);
        }
    }
    switch (target) {
    case AppenderTarget::All: {
        for (const auto &appender : _appenders) {
            if (event.level <= appender->getLevel()) {
                appender->append(event);
            }
        }
        if (_traceAppenderPtr && event.level <= _traceAppenderPtr->getLevel()) {
            _traceAppenderPtr->append(event);
        }
        break;
    }
    case AppenderTarget::Trace: {
        if (_traceAppenderPtr && event.level <= _traceAppenderPtr->getLevel()) {
            _traceAppenderPtr->append(event);
        }
        break;
    }
    }
}

void NaviLogger::clearFirstErrorEvent() {
    LoggingEventPtr errorEvent;
    {
        autil::ScopedLock lock(_lock);
        _hasError = false;
        errorEvent = std::move(_firstErrorEvent);
    }
    if (unlikely(_traceAppenderPtr != nullptr)) {
        _traceAppenderPtr->clearFirstErrorEvent();
    }
}

LoggingEventPtr NaviLogger::firstErrorEvent() {
    if (unlikely(_traceAppenderPtr != nullptr)) {
        auto event = _traceAppenderPtr->firstErrorEvent();
        if (event) {
            return event;
        }
    }
    {
        autil::ScopedLock lock(_lock);
        return _firstErrorEvent;
    }
}

std::string NaviLogger::getCurrentBtString() const {
    return _logManager->getCurrentBtString();
}

std::string NaviLogger::getExceptionBt(const autil::legacy::ExceptionBase &e) const {
    return _logManager->getExceptionBt(e);
}

const std::shared_ptr<NaviLogManager> &NaviLogger::getLogManager() const {
    return _logManager;
}

}
