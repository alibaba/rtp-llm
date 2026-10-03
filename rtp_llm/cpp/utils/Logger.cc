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

namespace {

struct CachedLogLevelEnv {
    bool        has_value;
    std::string value;
};

const bool kUseConsoleAppender = autil::EnvUtil::getEnv("FT_SERVER_TEST", 0) == 1;

const CachedLogLevelEnv kLogLevelEnv = []() {
    const char* env = std::getenv("LOG_LEVEL");
    return CachedLogLevelEnv{env != nullptr, env ? env : ""};
}();

uint32_t getLevelfromstr(const std::string& level_name) {
    const std::map<std::string, uint32_t> name_to_level = {
        {"TRACE", alog::LOG_LEVEL_TRACE1},
        {"DEBUG", alog::LOG_LEVEL_DEBUG},
        {"INFO", alog::LOG_LEVEL_INFO},
        {"WARNING", alog::LOG_LEVEL_WARN},
        {"ERROR", alog::LOG_LEVEL_ERROR},
    };
    auto level = name_to_level.find(level_name);
    if (level != name_to_level.end()) {
        return level->second;
    }
    throw std::runtime_error("[WARNING] Invalid logger level for env: LOG_LEVEL with value: " + level_name);
}

alog::Logger* configuredBackend(const char* name) {
    auto* backend = alog::Logger::getLogger(kUseConsoleAppender ? "console" : name);
    if (backend == nullptr) {
        throw std::runtime_error("getLogger should not be nullptr");
    }
    if (kLogLevelEnv.has_value) {
        backend->setLevel(getLevelfromstr(kLogLevelEnv.value));
    }
    return backend;
}

void bindLoggersForStartup() {
    // RTP singletons can be shared across DSOs while alog registries are not.
    // Resolve backends here, in the DSO that just configured the appenders.
    const auto bind = [](Logger& logger, const char* name) {
        auto* backend = configuredBackend(name);
        logger.bindBackendForStartup(backend, backend->getLevel());
    };
    bind(Logger::getEngineLogger(), "engine");
    bind(Logger::getAccessLogger(), "access");
    bind(Logger::getQueryAccessLogger(), "query_access");
    bind(Logger::getStackTraceLogger(), "stack_trace");
}

}  // namespace

bool initLogger(std::string log_file_path) {
    std::cerr << "initLogger log_file_path: " << log_file_path << std::endl;
    if (log_file_path == "") {
        std::string alog_conf_full_path = std::filesystem::current_path().string() + "/rtp_llm/config/alog.conf";
        bool        exist               = std::filesystem::exists(alog_conf_full_path);
        if (!exist) {
            AUTIL_ROOT_LOG_CONFIG();
            AUTIL_ROOT_LOG_SETLEVEL(INFO);
            bindLoggersForStartup();
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

    bindLoggersForStartup();
    return true;
}

Logger::Logger(const std::string& submodule_name) {
    auto* backend = configuredBackend(submodule_name.c_str());
    bindBackendForStartup(backend, backend->getLevel());
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

}  // namespace rtp_llm
