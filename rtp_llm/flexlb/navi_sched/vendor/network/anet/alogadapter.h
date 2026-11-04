#pragma once
#include <string>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/ilogger.h"
#include "alog/Logger.h"

class AlogAdapter : public ILogger {
public:
    AlogAdapter();
    AlogAdapter(const std::string &loggerName);
    AlogAdapter(const char *loggerName);
    ~AlogAdapter();
    void log(int level, const char *file, int line, const char *function, const char *fmt, ...)
        __attribute__((format(printf, 6, 7)));
    void logPureMessage(int level, const char *file, int line, const char *function, const char *);
    void setLogLevel(const char *level);
    void setLogLevel(const int level);
    int getLogLevel();
    bool isLevelEnabled(const int level) { return logger->isLevelEnabled(level); }

    void logSetup() { AlogAdapter::logSetupStatic(); }
    void logSetup(const std::string &configFile) { AlogAdapter::logSetupStatic(configFile); }
    void logTearDown() { AlogAdapter::logTearDownStatic(); }
    static void logSetupStatic();
    static void logSetupStatic(const std::string &configFile);
    static void logTearDownStatic();

    alog::Logger *getAlogLogger(void) { return logger; }

private:
    alog::Logger *logger;
};

