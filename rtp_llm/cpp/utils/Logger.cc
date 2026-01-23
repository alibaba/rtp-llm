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

#include <filesystem>
#include <iostream>
#include <stdexcept>

#include "rtp_llm/cpp/utils/Logger.h"
#include "autil/NetUtil.h"

namespace rtp_llm {

bool initLogger(std::string log_file_path) {
    std::cerr << "initLogger log_file_path: " << log_file_path << std::endl;
    if (log_file_path == "") {
        std::string alog_conf_full_path = std::filesystem::current_path().string() + "/rtp_llm/config/alog.conf";
        bool        exist               = std::filesystem::exists(alog_conf_full_path);
        if (!exist) {
            AUTIL_ROOT_LOG_CONFIG();
            AUTIL_ROOT_LOG_SETLEVEL(INFO);
            return true;
        }
        log_file_path = alog_conf_full_path;
    }

    bool exist = std::filesystem::exists(log_file_path);

    if (exist) {
        try {
            alog::Configurator::configureLogger(log_file_path.c_str());
        } catch (std::exception& e) {
            std::cerr << "Failed to configure logger. Logger config file [" << log_file_path << "], errorMsg ["
                      << e.what() << "]." << std::endl;
            return false;
        }
    } else {
        std::cerr << "log config file [" << log_file_path << "] doesn't exist. errorCode: [" << std::endl;
        return false;
    }

    return true;
}

Logger::Logger(const std::string& submodule_name) {
    int use_console_append = autil::EnvUtil::getEnv("FT_SERVER_TEST", 0) == 1;
    if (use_console_append) {
        logger_ = alog::Logger::getLogger("console");
    } else {
        logger_ = alog::Logger::getLogger(submodule_name.c_str());
    }
    if (logger_ == nullptr) {
        throw std::runtime_error("getLogger should not be nullptr");
    }
    if (std::getenv("LOG_LEVEL") != nullptr) {
        uint32_t log_level = getLevelfromstr("LOG_LEVEL");
        logger_->setLevel(log_level);
        base_log_level_ = logger_->getLevel();
    }
    auto success = autil::NetUtil::GetDefaultIp(ip_);
    if (!success) {
        printf("Logger failed to get default ip\n");
    }
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
            throw std::runtime_error("[WARNING] Invalid logger level for env: " + std::string(s)
                                     + " with value: " + std::string(level_name));
            level_name = nullptr;
        }
    }
    return base_log_level_;
}

}  // namespace rtp_llm
