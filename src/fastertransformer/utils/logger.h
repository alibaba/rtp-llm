/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstdint>
#include <exception>
#include <cstdlib>
#include <iomanip>
#include <map>
#include <pthread.h>
#include <string>
#include <string>
#include <unistd.h>

#include "alog/Appender.h"
#include "alog/Logger.h"

#include "src/fastertransformer/utils/exception.h"
#include "src/fastertransformer/utils/string_utils.h"
#include "EnvUtils.h"

namespace fastertransformer {

class Logger {
public:
    Logger(const std::string& submodule_name) {
        logger_ = alog::Logger::getLogger(submodule_name.c_str());
        if (logger_ == nullptr) {
            throw std::runtime_error("getLogger should not be nullptr");
        }
        alog::Logger::MAX_MESSAGE_LENGTH = 102400;

        bool use_console_append = getEnvWithDefault("FT_SERVER_TEST", "0") == "1";
        if (use_console_append) {
            console_appender_ = (alog::ConsoleAppender*)alog::ConsoleAppender::getAppender();
            console_appender_->setAutoFlush(true);
            logger_->setAppender(console_appender_);
        } else {
            std::string file_appender_path = getEnvWithDefault("LOG_PATH", "logs") + "/" + submodule_name + ".log";
            file_appender_ = (alog::FileAppender*)alog::FileAppender::getAppender(file_appender_path.c_str());
            file_appender_->setCacheLimit(1024);
            file_appender_->setHistoryLogKeepCount(5);
            file_appender_->setFlushIntervalInMS(1000);
            file_appender_->setFlushThreshold(1000);
            logger_->setAppender(file_appender_);
        }

        logger_->setInheritFlag(false);
        base_log_level_ = getLevelfromstr("LOG_LEVEL");
        logger_->setLevel(base_log_level_);
    }

    Logger(Logger const&) = delete;
    void operator=(Logger const&) = delete;


    static Logger& getEngineLogger() {
        static Logger engine_logger_("engine");
        return engine_logger_;
    }

    static Logger& getAccessLogger() {
        static Logger access_logger_("access");
        return access_logger_;
    }

    static Logger& getStackTraceLogger() {
        static Logger stack_trace_logger_("stack_trace");
        return stack_trace_logger_;
    }

    void setBaseLevel(const uint32_t base_level) {
        base_log_level_ = base_level;
        logger_->setLevel(base_log_level_);
        log(alog::LOG_LEVEL_INFO,
            __FILE__,
            __LINE__,
            __PRETTY_FUNCTION__,
            "Set logger level to: [%s]",
            getLevelName(base_level).c_str());
    }

    template<typename... Args>
    void log(uint32_t          level,
             const std::string file,
             int               line,
             const std::string func,
             const std::string format,
             const Args&... args) {
        std::string fmt    = getPrefix(file, line, func) + format;
        std::string logstr = fmtstr(fmt, args...);
        logger_->log(level, "%s", logstr.c_str());
        tryFlush(level);
    }

    void log(std::exception const& ex, uint32_t level = alog::LOG_LEVEL_ERROR) {
        log(level, __FILE__, __LINE__, __PRETTY_FUNCTION__,"%s: %s", FTException::demangle(typeid(ex).name()).c_str(), ex.what());
    }

    void setRank(int32_t rank) {
        rank_ = rank;
    }

    bool isDebugMode() {
        return base_log_level_ >= alog::LOG_LEVEL_DEBUG;
    }

    bool isTraceMode() {
        return base_log_level_ >= alog::LOG_LEVEL_TRACE1;
    }

    void flush() {
        fflush(stdout);
        fflush(stderr);
        logger_->flush();
    }

    bool isLevelEnabled(int32_t level) {
        return logger_->isLevelEnabled(level);
    }

private:
    void tryFlush(int32_t level) {
        if (base_log_level_ >= alog::LOG_LEVEL_DEBUG || level <= alog::LOG_LEVEL_ERROR) {
            flush();
        }
    }

    uint32_t getLevelfromstr(const char* s);

    inline const std::string getPrefix(const std::string& file, int line, const std::string& func) {
        return "[" + std::to_string(getpid()) + ":" + std::to_string(gettid()) + "][RANK " + std::to_string(rank_) + "][" + file + ":"
               + std::to_string(line) + "][" + func + "]";
    }

    inline const std::string getLevelName(const uint32_t level) {
        return level_name_.at(level);
    }

private:
    alog::Logger*          logger_;
    alog::FileAppender*    file_appender_;
    alog::ConsoleAppender* console_appender_;

    uint32_t base_log_level_ = alog::LOG_LEVEL_INFO;
    const std::map<const uint32_t, const std::string> level_name_ = {{alog::LOG_LEVEL_TRACE1, "TRACE"},
                                                                     {alog::LOG_LEVEL_DEBUG, "DEBUG"},
                                                                     {alog::LOG_LEVEL_INFO, "INFO"},
                                                                     {alog::LOG_LEVEL_WARN, "WARNING"},
                                                                     {alog::LOG_LEVEL_ERROR, "ERROR"}};

    int32_t rank_ = 0;
};

#define FT_LOG(level, ...)                                                                                             \
    do {                                                                                                               \
        auto& logger = fastertransformer::Logger::getEngineLogger();                                                   \
        if (!logger.isLevelEnabled(level)) {                                                                           \
            break;                                                                                                     \
        }                                                                                                              \
        logger.log(level, __FILE__, __LINE__, __PRETTY_FUNCTION__, __VA_ARGS__);                                       \
    } while (0)

#define FT_LOG_TRACE(...) FT_LOG(alog::LOG_LEVEL_TRACE1, __VA_ARGS__)
#define FT_LOG_DEBUG(...) FT_LOG(alog::LOG_LEVEL_DEBUG, __VA_ARGS__)
#define FT_LOG_INFO(...) FT_LOG(alog::LOG_LEVEL_INFO, __VA_ARGS__)
#define FT_LOG_WARNING(...) FT_LOG(alog::LOG_LEVEL_WARN, __VA_ARGS__)
#define FT_LOG_ERROR(...) FT_LOG(alog::LOG_LEVEL_ERROR, __VA_ARGS__)
#define FT_LOG_EXCEPTION(ex, ...) fastertransformer::Logger::getEngineLogger().log(ex, ##__VA_ARGS__)

#define FT_ACCESS_LOG(level, ...)                                                                                      \
    do {                                                                                                               \
        auto& logger = fastertransformer::Logger::getAccessLogger();                                                   \
        if (!logger.isLevelEnabled(level)) {                                                                           \
            break;                                                                                                     \
        }                                                                                                              \
        logger.log(level, __FILE__, __LINE__, __PRETTY_FUNCTION__, __VA_ARGS__);                                       \
    } while (0)

#define FT_ACCESS_LOG_TRACE(...) FT_ACCESS_LOG(alog::LOG_LEVEL_TRACE1, __VA_ARGS__)
#define FT_ACCESS_LOG_DEBUG(...) FT_ACCESS_LOG(alog::LOG_LEVEL_DEBUG, __VA_ARGS__)
#define FT_ACCESS_LOG_INFO(...) FT_ACCESS_LOG(alog::LOG_LEVEL_INFO, __VA_ARGS__)
#define FT_ACCESS_LOG_WARNING(...) FT_ACCESS_LOG(alog::LOG_LEVEL_WARN, __VA_ARGS__)
#define FT_ACCESS_LOG_ERROR(...) FT_ACCESS_LOG(alog::LOG_LEVEL_ERROR, __VA_ARGS__)
#define FT_ACCESS_LOG_EXCEPTION(ex, ...) fastertransformer::Logger::getAccessLogger().log(ex, ##__VA_ARGS__)


#define FT_STACKTRACE_LOG(level, ...)                                                                                  \
    do {                                                                                                               \
        auto& logger = fastertransformer::Logger::getStackTraceLogger();                                               \
        if (!logger.isLevelEnabled(level)) {                                                                           \
            break;                                                                                                     \
        }                                                                                                              \
        logger.log(level, __FILE__, __LINE__, __PRETTY_FUNCTION__, __VA_ARGS__);                                       \
    } while (0)

#define FT_STACKTRACE_LOG_TRACE(...) FT_STACKTRACE_LOG(alog::LOG_LEVEL_TRACE1, __VA_ARGS__)
#define FT_STACKTRACE_LOG_DEBUG(...) FT_STACKTRACE_LOG(alog::LOG_LEVEL_DEBUG, __VA_ARGS__)
#define FT_STACKTRACE_LOG_INFO(...) FT_STACKTRACE_LOG(alog::LOG_LEVEL_INFO, __VA_ARGS__)
#define FT_STACKTRACE_LOG_WARNING(...) FT_STACKTRACE_LOG(alog::LOG_LEVEL_WARN, __VA_ARGS__)
#define FT_STACKTRACE_LOG_ERROR(...) FT_STACKTRACE_LOG(alog::LOG_LEVEL_ERROR, __VA_ARGS__)
#define FT_STACKTRACE_LOG_EXCEPTION(ex, ...) fastertransformer::Logger::getStackTraceLogger().log(ex, ##__VA_ARGS__)
}  // namespace fastertransformer
