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
#include <stdint.h>

#include "alog/Appender.h"
#include "alog/Logger.h"
#include "autil/EnvUtil.h"
#include "autil/TimeUtility.h"

#include "rtp_llm/cpp/utils/Exception.h"
#include "rtp_llm/cpp/utils/StringUtil.h"

namespace rtp_llm {

bool initLogger(std::string log_file_path = "");

class Logger {
public:
    Logger(const std::string& submodule_name);

    Logger(Logger const&)         = delete;
    void operator=(Logger const&) = delete;

    static Logger& getEngineLogger() {
        static Logger engine_logger_("engine");
        return engine_logger_;
    }

    static Logger& getAccessLogger() {
        static Logger access_logger_("access");
        return access_logger_;
    }

    static Logger& getQueryAccessLogger() {
        static Logger query_access_logger_("query_access");
        return query_access_logger_;
    }

    static Logger& getStackTraceLogger() {
        static Logger stack_trace_logger_("stack_trace");
        return stack_trace_logger_;
    }

    void setBaseLevel(const uint32_t base_level);

    template<typename... Args>
    void log(uint32_t          level,
             const std::string file,
             int               line,
             const std::string func,
             const std::string format,
             const Args&... args) {
        std::string fmt;
        if (isTraceMode()) {
            fmt = getTracePrefix() + format;
        } else {
            fmt = getPrefix(file, line, func) + format;
        }
        std::string logstr = rtp_llm::fmtstr(fmt, args...);
        logger_->log(level, "%s", logstr.c_str());
        tryFlush(level);
    }

    template<typename... Args>
    void log_access(uint32_t level, const std::string format, const Args&... args) {
        std::string logstr = rtp_llm::fmtstr(format, args...);
        logger_->log(level, "%s", logstr.c_str());
        tryFlush(level);
    }

    void log(std::exception const& ex, uint32_t level = alog::LOG_LEVEL_ERROR) {
        log(level,
            __FILE__,
            __LINE__,
            __PRETTY_FUNCTION__,
            "%s: %s",
            RTPException::demangle(typeid(ex).name()).c_str(),
            ex.what());
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
        return "[RANK " + std::to_string(rank_) + "][" + ip_ + "][" + file + ":" + std::to_string(line) + "][" + func
               + "] ";
    }

    inline const std::string getTracePrefix() {
        return "[RANK " + std::to_string(rank_) + "][" + ip_ + "] ";
    }

    inline const std::string getLevelName(const uint32_t level) {
        return level_name_.at(level);
    }

private:
    alog::Logger*          logger_;
    alog::FileAppender*    file_appender_;
    alog::ConsoleAppender* console_appender_;

    uint32_t                                          base_log_level_ = alog::LOG_LEVEL_INFO;
    const std::map<const uint32_t, const std::string> level_name_     = {{alog::LOG_LEVEL_TRACE1, "TRACE"},
                                                                         {alog::LOG_LEVEL_DEBUG, "DEBUG"},
                                                                         {alog::LOG_LEVEL_INFO, "INFO"},
                                                                         {alog::LOG_LEVEL_WARN, "WARNING"},
                                                                         {alog::LOG_LEVEL_ERROR, "ERROR"}};

    int32_t     rank_ = 0;
    std::string ip_;
};

}  // namespace rtp_llm

#define RTP_LLM_INTERVAL_LOG(logInterval, level, format, args...)                                                      \
    do {                                                                                                               \
        static int64_t logTimestamp;                                                                                   \
        int64_t        now = autil::TimeUtility::currentTimeInSeconds();                                               \
        if (now - logTimestamp > logInterval) {                                                                        \
            RTP_LLM_LOG(alog::LOG_LEVEL_##level, format, ##args);                                                      \
            logTimestamp = now;                                                                                        \
        }                                                                                                              \
    } while (0)

#define RTP_LLM_LOG(level, ...)                                                                                        \
    do {                                                                                                               \
        auto& logger = rtp_llm::Logger::getEngineLogger();                                                             \
        if (!logger.isLevelEnabled(level)) {                                                                           \
            break;                                                                                                     \
        }                                                                                                              \
        logger.log(level, __FILE__, __LINE__, __PRETTY_FUNCTION__, __VA_ARGS__);                                       \
    } while (0)

#define RTP_LLM_LOG_TRACE(...) RTP_LLM_LOG(alog::LOG_LEVEL_TRACE1, __VA_ARGS__)
#define RTP_LLM_LOG_DEBUG(...) RTP_LLM_LOG(alog::LOG_LEVEL_DEBUG, __VA_ARGS__)
#define RTP_LLM_LOG_INFO(...) RTP_LLM_LOG(alog::LOG_LEVEL_INFO, __VA_ARGS__)
#define RTP_LLM_LOG_WARNING(...) RTP_LLM_LOG(alog::LOG_LEVEL_WARN, __VA_ARGS__)
#define RTP_LLM_LOG_ERROR(...) RTP_LLM_LOG(alog::LOG_LEVEL_ERROR, __VA_ARGS__)
#define RTP_LLM_LOG_EXCEPTION(ex, ...) rtp_llm::Logger::getEngineLogger().log(ex, ##__VA_ARGS__)

#define RTP_LLM_ACCESS_LOG(level, ...)                                                                                 \
    do {                                                                                                               \
        auto& logger = rtp_llm::Logger::getAccessLogger();                                                             \
        if (!logger.isLevelEnabled(level)) {                                                                           \
            break;                                                                                                     \
        }                                                                                                              \
        logger.log_access(level, __VA_ARGS__);                                                                         \
    } while (0)

#define RTP_LLM_ACCESS_LOG_TRACE(...) RTP_LLM_ACCESS_LOG(alog::LOG_LEVEL_TRACE1, __VA_ARGS__)
#define RTP_LLM_ACCESS_LOG_DEBUG(...) RTP_LLM_ACCESS_LOG(alog::LOG_LEVEL_DEBUG, __VA_ARGS__)
#define RTP_LLM_ACCESS_LOG_INFO(...) RTP_LLM_ACCESS_LOG(alog::LOG_LEVEL_INFO, __VA_ARGS__)
#define RTP_LLM_ACCESS_LOG_WARNING(...) RTP_LLM_ACCESS_LOG(alog::LOG_LEVEL_WARN, __VA_ARGS__)
#define RTP_LLM_ACCESS_LOG_ERROR(...) RTP_LLM_ACCESS_LOG(alog::LOG_LEVEL_ERROR, __VA_ARGS__)
#define RTP_LLM_ACCESS_LOG_EXCEPTION(ex, ...) rtp_llm::Logger::getAccessLogger().log(ex, ##__VA_ARGS__)

#define RTP_LLM_QUERY_ACCESS_LOG(level, ...)                                                                           \
    do {                                                                                                               \
        auto& logger = rtp_llm::Logger::getQueryAccessLogger();                                                        \
        logger.log(level, __FILE__, __LINE__, __PRETTY_FUNCTION__, __VA_ARGS__);                                       \
    } while (0)
#define RTP_LLM_QUERY_ACCESS_LOG_TRACE(...) RTP_LLM_QUERY_ACCESS_LOG(alog::LOG_LEVEL_TRACE1, __VA_ARGS__)
#define RTP_LLM_QUERY_ACCESS_LOG_DEBUG(...) RTP_LLM_QUERY_ACCESS_LOG(alog::LOG_LEVEL_DEBUG, __VA_ARGS__)
#define RTP_LLM_QUERY_ACCESS_LOG_INFO(...) RTP_LLM_QUERY_ACCESS_LOG(alog::LOG_LEVEL_INFO, __VA_ARGS__)
#define RTP_LLM_QUERY_ACCESS_LOG_WARNING(...) RTP_LLM_QUERY_ACCESS_LOG(alog::LOG_LEVEL_WARN, __VA_ARGS__)
#define RTP_LLM_QUERY_ACCESS_LOG_ERROR(...) RTP_LLM_QUERY_ACCESS_LOG(alog::LOG_LEVEL_ERROR, __VA_ARGS__)
#define RTP_LLM_QUERY_ACCESS_LOG_EXCEPTION(ex, ...) rtp_llm::Logger::getQueryAccessLogger().log(ex, ##__VA_ARGS__)

#define RTP_LLM_STACKTRACE_LOG(level, ...)                                                                             \
    do {                                                                                                               \
        auto& logger = rtp_llm::Logger::getStackTraceLogger();                                                         \
        if (!logger.isLevelEnabled(level)) {                                                                           \
            break;                                                                                                     \
        }                                                                                                              \
        logger.log(level, __FILE__, __LINE__, __PRETTY_FUNCTION__, __VA_ARGS__);                                       \
    } while (0)

#define RTP_LLM_STACKTRACE_LOG_TRACE(...) RTP_LLM_STACKTRACE_LOG(alog::LOG_LEVEL_TRACE1, __VA_ARGS__)
#define RTP_LLM_STACKTRACE_LOG_DEBUG(...) RTP_LLM_STACKTRACE_LOG(alog::LOG_LEVEL_DEBUG, __VA_ARGS__)
#define RTP_LLM_STACKTRACE_LOG_INFO(...) RTP_LLM_STACKTRACE_LOG(alog::LOG_LEVEL_INFO, __VA_ARGS__)
#define RTP_LLM_STACKTRACE_LOG_WARNING(...) RTP_LLM_STACKTRACE_LOG(alog::LOG_LEVEL_WARN, __VA_ARGS__)
#define RTP_LLM_STACKTRACE_LOG_ERROR(...) RTP_LLM_STACKTRACE_LOG(alog::LOG_LEVEL_ERROR, __VA_ARGS__)
#define RTP_LLM_STACKTRACE_LOG_EXCEPTION(ex, ...) rtp_llm::Logger::getStackTraceLogger().log(ex, ##__VA_ARGS__)
