#pragma once

#include "navi/common.h"
#include "autil/legacy/jsonizable.h"
#include "navi/log/LogBtFilter.h"

namespace navi {

class LogConfig;
class LogConfigDef;
class FileAppenderDef;

class FileLogConfig : public autil::legacy::Jsonizable
{
public:
    FileLogConfig();
    ~FileLogConfig();
private:
    FileLogConfig(const LogConfig &);
    FileLogConfig &operator=(const LogConfig &);
public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper &json) override;
    void fillProto(FileAppenderDef &def) const;
public:
    uint32_t translateDelayTime() const;
private:
    static std::string transformPattern(const std::string &inputPattern);
    static std::string getProgressName();
    static std::string getLastDirectory(uint32_t layers);
public:
    std::string fileName;
    std::string levelStr;
    std::string logPattern;
    bool autoFlush;
    size_t maxFileSize;
    std::string delayTimeStr;
    bool compress;
    size_t cacheLimit;
    size_t logKeepCount;
    bool asyncFlush;
    size_t flushThreshold;
    size_t flushInterval;
    std::vector<LogBtFilterParam> btFilters;
private:
    static const size_t DEFAULT_MAX_FILE_SIZE_IN_MB = 1024; // 1024MB
    static const size_t DEFAULT_CACHE_LIMIT_IN_MB = 128; // 128MB
    static const size_t DEFAULT_LOG_KEEP_COUNT = 20;
    static const size_t DEFAULT_FLUSH_THRESHOLD_IN_KB = 1024;
    static const size_t DEFAULT_FLUSH_INTERVAL_IN_MS = 1000;
};

class LogConfig : public autil::legacy::Jsonizable
{
public:
    LogConfig()
        : maxMessageLength(DEFAULT_MAX_MESSAGE_LENGTH)
    {
    }
    ~LogConfig() {
    }
public:
    void fillProto(LogConfigDef &def) const;
private:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper &json) override;
public:
    static LogConfig getDefaultLogConfig(const std::string &name);
public:
    size_t maxMessageLength;
    std::vector<FileLogConfig> fileAppenders;
private:
    static const size_t DEFAULT_MAX_MESSAGE_LENGTH = 1024 * 1024;
};

}
