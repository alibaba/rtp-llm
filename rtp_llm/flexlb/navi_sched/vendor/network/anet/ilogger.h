#pragma once

#include <string>

/**
 * Interface to define how log system is used in ANET/ARPC.
 */
class ILogger {
public:
    virtual ~ILogger() {}

    virtual void setLogLevel(const int level) = 0;
    virtual void setLogLevel(const char *level) = 0;
    virtual void logPureMessage(int level, const char *file, int line, const char *function, const char *buf) = 0;
    virtual void log(int level, const char *file, int line, const char *function, const char *fmt, ...)
        __attribute__((format(printf, 6, 7))) = 0;
    virtual bool isLevelEnabled(const int level) = 0;

    /* The interfaces below are for the purpose of search app compliance. */
    virtual void logSetup() = 0;
    virtual void logSetup(const std::string &configFile) = 0;
    virtual void logTearDown() = 0;
};

