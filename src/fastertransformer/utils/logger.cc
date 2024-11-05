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

#include "src/fastertransformer/utils/logger.h"

namespace fastertransformer {

Logger::Logger(const std::string& submodule_name) {
    logger_ = alog::Logger::getLogger(submodule_name.c_str());
    if (logger_ == nullptr) {
        throw std::runtime_error("getLogger should not be nullptr");
    }
    alog::Logger::MAX_MESSAGE_LENGTH = 1024000;

    int use_console_append = autil::EnvUtil::getEnv("FT_SERVER_TEST", 0) == 1;
    if (use_console_append) {
        console_appender_ = (alog::ConsoleAppender*)alog::ConsoleAppender::getAppender();
        console_appender_->setAutoFlush(true);
        logger_->setAppender(console_appender_);
    } else {
        std::string file_appender_path = autil::EnvUtil::getEnv("LOG_PATH", "logs") + "/" + submodule_name + ".log";
        file_appender_ = (alog::FileAppender*)alog::FileAppender::getAppender(file_appender_path.c_str());
        file_appender_->setAutoFlush(true);
        file_appender_->setAsyncFlush(false);
        file_appender_->setFlushIntervalInMS(100);
        file_appender_->setFlushThreshold(1);
        logger_->setAppender(file_appender_);
    }

    logger_->setInheritFlag(false);
    base_log_level_ = getLevelfromstr("LOG_LEVEL");
    logger_->setLevel(base_log_level_);
}

void Logger::setBaseLevel(const uint32_t base_level) {
    base_log_level_ = base_level;
    logger_->setLevel(base_log_level_);
    log(alog::LOG_LEVEL_INFO,
        __FILE__,
        __LINE__,
        __PRETTY_FUNCTION__,
        "Set logger level to: [%s]",
        getLevelName(base_level).c_str());
}

uint32_t Logger::getLevelfromstr(const char* s) {
    char* level_name = std::getenv(s);
    if (level_name != nullptr) {
        std::map<std::string, uint32_t> name_to_level = {
            {"TRACE", alog::LOG_LEVEL_TRACE1},
            {"DEBUG", alog::LOG_LEVEL_DEBUG},
            {"INFO", alog::LOG_LEVEL_INFO},
            {"WARNING", alog::LOG_LEVEL_WARN},
            {"ERROR", alog::LOG_LEVEL_ERROR},
        };
        auto level = name_to_level.find(level_name);
        if (level != name_to_level.end()) {
            return level->second;
        } else {
            throw std::runtime_error("[FT][WARNING] Invalid logger level for env: " + std::string(s)
                                     + " with value: " + std::string(level_name));
            level_name = nullptr;
        }
    }
    return base_log_level_;
}

}  // namespace fastertransformer
