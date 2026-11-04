#pragma once

/*
we copy design and code from alog for navi's log
*/

#include <list>
#include <set>
#include <vector>

#include "autil/Lock.h"
#include "navi/common.h"
#include "navi/log/NaviLoggerProvider.h"
#include "navi/log/TraceCollector.h"

namespace navi {

class LoggingEvent;
class Appender;
class FileAppender;
class TraceAppender;
class ConsoleAppender;
class NaviLogManager;

class NaviLogger;
NAVI_TYPEDEF_PTR(NaviLogger);
class NaviObjectLogger;

enum class AppenderTarget {
    All,
    Trace,
};

class NaviLogger {
public:
    NaviLogger(size_t maxMessageLength, const std::shared_ptr<NaviLogManager> &logManager);
    NaviLogger(const NaviLogger &other, bool reuseTraceAppender);
    ~NaviLogger();
public:
    void init(SessionId id);
    void stop();
    bool stopped();
    void setLoggerId(SessionId id);
    SessionId getLoggerId() const;
    const std::string &getName() const;
public:
    void log(LogLevel level,
             AppenderTarget target,
             void *object,
             LogObjectType objectType,
             const std::string &prefix,
             const char *filename,
             int line,
             const char *function,
             const char *format,
             va_list ap);
    inline bool isLevelEnabled(LogLevel level) { return (level <= _level) ? true : false; }
    LogLevel getTraceLevel() const;
    void addAppender(const std::shared_ptr<Appender> &appender);
    bool hasTraceAppender() const;
    void collectTrace(TraceCollector &collector);
    LoggingEventPtr firstErrorEvent();
    void clearFirstErrorEvent();
    std::string getCurrentBtString() const;
    std::string getExceptionBt(const autil::legacy::ExceptionBase &e) const;
    void updateTraceLevel(const std::string &levelStr);
private:
    void stopTraceAppender();
    std::shared_ptr<TraceAppender> getTraceAppender() const;
    std::shared_ptr<ConsoleAppender> getConsoleAppender() const;
    void setTraceAppender(const std::shared_ptr<TraceAppender> &appender);
    void copyAppenders(const NaviLogger &other, bool reuseTraceAppender);
    void updateLogLevel();
    void updateName();
    void _log(LoggingEvent &event, AppenderTarget target);
    const std::shared_ptr<NaviLogManager> &getLogManager() const;
private:
    static char *getTlsBuffer(size_t maxMessageLength);
private:
    friend class Navi;
    friend class NaviSnapshot;
    friend class NaviR;
    friend class NaviLoggerProvider;
    friend class NaviWorkerBase;
    friend struct GraphParam;
    friend class ThreadLocalInit;
private:
    SessionId _id;
    std::string _name;
    LogLevel _level;
    std::atomic<bool> _stopped;
    size_t _maxMessageLength;
    std::vector<Appender *> _appenders;
    TraceAppender *_traceAppenderPtr = nullptr;
    std::vector<std::shared_ptr<Appender>> _ownAppenders;
    std::shared_ptr<TraceAppender> _traceAppender;
    std::shared_ptr<NaviLogManager> _logManager;
    mutable autil::ThreadMutex _lock;
    bool _hasError;
    LoggingEventPtr _firstErrorEvent;
};

thread_local extern const NaviObjectLogger *NAVI_TLS_LOGGER;
thread_local extern LogObjectType NAVI_TLS_LOGGER_OBJECT_TYPE;
thread_local extern void *NAVI_TLS_LOGGER_OBJECT;

class NaviObjectLogger : public alog::LoggerInterceptor {
public:
    NaviObjectLogger() : object(nullptr) {}
    explicit NaviObjectLogger(void *object_,
                              const char *prefix_,
                              const NaviLoggerPtr &logger_ = NAVI_TLS_LOGGER ? NAVI_TLS_LOGGER->logger : nullptr)
        : object(object_), prefixCstr(prefix_), logger(logger_) {}
    ~NaviObjectLogger() = default;
    NaviObjectLogger(const NaviObjectLogger &) = default;
    NaviObjectLogger(NaviObjectLogger &&) = default;
    NaviObjectLogger &operator=(const NaviObjectLogger &) = default;
    NaviObjectLogger &operator=(NaviObjectLogger &&) = default;
    // for alog interceptor
    bool hasTraceAppender() const { return logger && logger->hasTraceAppender(); }
    bool isLevelEnabled(uint32_t level, uint32_t originalLevel) const override {
        return hasTraceAppender() && logger->getTraceLevel() >= ((LogLevel)level);
    }
    void log(uint32_t level, const char *file, int line, const char *func, const char *fmt, va_list ap) const override {
        if (logger) {
            logger->log((LogLevel)level,
                        AppenderTarget::Trace,
                        object ? object : NAVI_TLS_LOGGER_OBJECT,
                        object ? LOT_UNKNOWN : NAVI_TLS_LOGGER_OBJECT_TYPE,
                        prefix,
                        file,
                        line,
                        func,
                        fmt,
                        ap);
        }
    }
  public:
    static NaviObjectLogger capture() {
        return NAVI_TLS_LOGGER ? *NAVI_TLS_LOGGER : NaviObjectLogger{};
    }

public:
    void addPrefix(const char *fmt, ...) __attribute__((format(printf, 2, 3)));

public:
    void *object;
    const char *prefixCstr = nullptr;
    std::string prefix;
    NaviLoggerPtr logger;
};

inline void NaviObjectLogger::addPrefix(const char *fmt, ...) {

    constexpr int32_t MAX_PREFIX_LENGTH = 512;
    char buffer[MAX_PREFIX_LENGTH];
    va_list ap;
    va_start(ap, fmt);
    auto n = vsnprintf(buffer, MAX_PREFIX_LENGTH, fmt, ap);
    va_end(ap);
    n = std::max(0, std::min(n, MAX_PREFIX_LENGTH - 1));
    if (prefix.empty() && prefixCstr) {
        prefix = std::string(prefixCstr);
    }
    if (!prefix.empty()) {
        prefix.append(" ", 1);
    }
    prefix.append(buffer, n);
}

class NaviLoggerObjectScope {
public:
    NaviLoggerObjectScope(LogObjectType objectType, void *object)
        : _oldObjectType(NAVI_TLS_LOGGER_OBJECT_TYPE)
        , _oldObject(NAVI_TLS_LOGGER_OBJECT)
    {
        NAVI_TLS_LOGGER_OBJECT_TYPE = objectType;
        NAVI_TLS_LOGGER_OBJECT = object;
    }
    ~NaviLoggerObjectScope() {
        NAVI_TLS_LOGGER_OBJECT_TYPE = _oldObjectType;
        NAVI_TLS_LOGGER_OBJECT = _oldObject;
    }
private:
    LogObjectType _oldObjectType = LOT_UNKNOWN;
    void *_oldObject = nullptr;
};

class NaviLoggerScope {
public:
    NaviLoggerScope(const NaviObjectLogger &logger)
        : _oldLogger(NAVI_TLS_LOGGER), _oldInterceptor(alog::ALOG_LOG_INTERCEPTOR) {
        NAVI_TLS_LOGGER = &logger;
        alog::ALOG_LOG_INTERCEPTOR = logger.hasTraceAppender() ? &logger : nullptr;
    }
    NaviLoggerScope(const NaviObjectLogger *logger)
        : _oldLogger(NAVI_TLS_LOGGER), _oldInterceptor(alog::ALOG_LOG_INTERCEPTOR) {
        NAVI_TLS_LOGGER = logger;
        alog::ALOG_LOG_INTERCEPTOR = (logger && logger->hasTraceAppender()) ? logger : nullptr;
    }
    ~NaviLoggerScope() {
        NAVI_TLS_LOGGER = _oldLogger;
        alog::ALOG_LOG_INTERCEPTOR = _oldInterceptor;
    }

private:
    const NaviObjectLogger *_oldLogger;
    const alog::LoggerInterceptor *_oldInterceptor;
};

#define DECLARE_LOGGER() navi::NaviObjectLogger _logger;

inline void NAVI_EVENT_LOG(const NaviObjectLogger &objectLogger,
                           LogLevel level,
                           const char *file,
                           int line,
                           const char *func,
                           const char *fmt,
                           ...) __attribute__((format(printf, 6, 7)));
inline void NAVI_EVENT_LOG(const NaviObjectLogger *objectLogger,
                           LogLevel level,
                           const char *file,
                           int line,
                           const char *func,
                           const char *fmt,
                           ...) __attribute__((format(printf, 6, 7)));
inline void NAVI_EVENT_LOG(
    const NaviLoggerPtr &logger, LogLevel level, const char *file, int line, const char *func, const char *fmt, ...)
    __attribute__((format(printf, 6, 7)));
inline void
NAVI_EVENT_LOG(NaviLogger *logger, LogLevel level, const char *file, int line, const char *func, const char *fmt, ...)
    __attribute__((format(printf, 6, 7)));

inline bool logEnable(const NaviObjectLogger &objectLogger, LogLevel level) {
    auto logger = objectLogger.logger.get();
    return logger && logger->isLevelEnabled(level);
}

inline void NAVI_EVENT_LOG(const NaviObjectLogger &objectLogger,
                           LogLevel level,
                           const char *file,
                           int line,
                           const char *func,
                           const char *fmt,
                           ...) {
    va_list ap;
    va_start(ap, fmt);
    objectLogger.logger->log(level,
                             AppenderTarget::All,
                             objectLogger.object ? objectLogger.object : NAVI_TLS_LOGGER_OBJECT,
                             objectLogger.object ? LOT_UNKNOWN : NAVI_TLS_LOGGER_OBJECT_TYPE,
                             objectLogger.prefix,
                             file,
                             line,
                             func,
                             fmt,
                             ap);
    va_end(ap);
}

inline bool logEnable(const NaviObjectLogger *objectLogger, LogLevel level) {
    if (!objectLogger) {
        return false;
    }
    auto logger = objectLogger->logger.get();
    return logger && logger->isLevelEnabled(level);
}

inline void NAVI_EVENT_LOG(const NaviObjectLogger *objectLogger,
                           LogLevel level,
                           const char *file,
                           int line,
                           const char *func,
                           const char *fmt,
                           ...) {
    va_list ap;
    va_start(ap, fmt);
    objectLogger->logger->log(level,
                              AppenderTarget::All,
                              objectLogger->object ? objectLogger->object : NAVI_TLS_LOGGER_OBJECT,
                              objectLogger->object ? LOT_UNKNOWN : NAVI_TLS_LOGGER_OBJECT_TYPE,
                              objectLogger->prefix,
                              file,
                              line,
                              func,
                              fmt,
                              ap);
    va_end(ap);
}

inline void NAVI_OBJECT_EVENT_LOG(const NaviObjectLogger *objectLogger,
                                  LogLevel level,
                                  void *object,
                                  LogObjectType objectType,
                                  const char *file,
                                  int line,
                                  const char *func,
                                  const char *fmt,
                                  ...) {
    va_list ap;
    va_start(ap, fmt);
    objectLogger->logger->log(level, AppenderTarget::All, object, objectType, std::string(), file, line, func, fmt, ap);
    va_end(ap);
}

inline bool logEnable(const NaviLoggerPtr &logger, LogLevel level) { return logger && logger->isLevelEnabled(level); }

inline void NAVI_EVENT_LOG(
    const NaviLoggerPtr &logger, LogLevel level, const char *file, int line, const char *func, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    logger->log(level,
                AppenderTarget::All,
                NAVI_TLS_LOGGER_OBJECT,
                NAVI_TLS_LOGGER_OBJECT_TYPE,
                std::string(),
                file,
                line,
                func,
                fmt,
                ap);
    va_end(ap);
}

inline bool logEnable(NaviLogger *logger, LogLevel level) { return logger && logger->isLevelEnabled(level); }

inline void
NAVI_EVENT_LOG(NaviLogger *logger, LogLevel level, const char *file, int line, const char *func, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    logger->log(level,
                AppenderTarget::All,
                NAVI_TLS_LOGGER_OBJECT,
                NAVI_TLS_LOGGER_OBJECT_TYPE,
                std::string(),
                file,
                line,
                func,
                fmt,
                ap);
    va_end(ap);
}

#define NAVI_LOG_INTERNAL(logger, level, format, args...)                      \
    {                                                                          \
        if (__builtin_expect(logEnable(logger, navi::LOG_LEVEL_##level), 0)) { \
            NAVI_EVENT_LOG(logger,                                             \
                           navi::LOG_LEVEL_##level,                            \
                           __FILE__,                                           \
                           __LINE__,                                           \
                           __FUNCTION__,                                       \
                           format,                                             \
                           ##args);                                            \
        }                                                                      \
    }

#define NAVI_OBJECT_LOG_INTERNAL(logger, level, object, objectType, format, args...)                                   \
    {                                                                                                                  \
        if (__builtin_expect(logEnable(logger, navi::LOG_LEVEL_##level), 0)) {                                         \
            NAVI_OBJECT_EVENT_LOG(logger,                                                                              \
                                  navi::LOG_LEVEL_##level,                                                             \
                                  (void *)object,                                                                      \
                                  objectType,                                                                          \
                                  __FILE__,                                                                            \
                                  __LINE__,                                                                            \
                                  __FUNCTION__,                                                                        \
                                  format,                                                                              \
                                  ##args);                                                                             \
        }                                                                                                              \
    }

#define NAVI_LOG(level, format, args...) NAVI_LOG_INTERNAL(_logger, level, format, ##args)

#define NAVI_KERNEL_LOG(level, format, args...) NAVI_LOG_INTERNAL(navi::NAVI_TLS_LOGGER, level, format, ##args)

#define NAVI_OBJECT_LOG(object, objectType, level, format, args...)                                                    \
    NAVI_OBJECT_LOG_INTERNAL(navi::NAVI_TLS_LOGGER, level, object, objectType, format, ##args)

#define NAVI_TLS_LOG(level, format, args...) NAVI_LOG_INTERNAL(navi::NAVI_TLS_LOGGER, level, format, ##args)

#define NAVI_INTERVAL_LOG_IMPL(logInterval, logger, level, format, args...)                                            \
    do {                                                                                                               \
        static int logCounter;                                                                                         \
        if (((logger) && (logger)->hasTraceAppender()) || 0 <= logCounter) {                                           \
            NAVI_LOG_INTERNAL((logger), level, format, ##args);                                                        \
            logCounter = logInterval;                                                                                  \
        }                                                                                                              \
        logCounter--;                                                                                                  \
    } while (0)

#define NAVI_TLS_INTERVAL_LOG(logInterval, level, format, args...)                                                     \
    NAVI_INTERVAL_LOG_IMPL(logInterval, navi::NAVI_TLS_LOGGER, level, format, ##args)

#define NAVI_INTERVAL_LOG(logInterval, level, format, args...)                                                         \
    NAVI_INTERVAL_LOG_IMPL(logInterval, &_logger, level, format, args)

#define NAVI_KERNEL_FAIL_AS_ERROR_LOG(failAsError, level, format, args...)                                             \
    {                                                                                                                  \
        auto logLevel = (failAsError) ? LOG_LEVEL_ERROR : navi::LOG_LEVEL_##level;                                     \
        if (__builtin_expect(logEnable(navi::NAVI_TLS_LOGGER, logLevel), 0)) {                                         \
            NAVI_EVENT_LOG(navi::NAVI_TLS_LOGGER, logLevel, __FILE__, __LINE__, __FUNCTION__, format, ##args);         \
        }                                                                                                              \
    }

} // namespace navi
