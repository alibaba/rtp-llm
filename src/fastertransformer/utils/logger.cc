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

Logger::Logger() {
    level_ = getLevelfromstr("FT_DEBUG_LEVEL");
    print_level_ = getLevelfromstr("FT_DEBUG_PRINT_LEVEL");
}

Logger::Level Logger::getLevelfromstr(const char* s) {
    char* level_name = std::getenv(s);
    if (level_name != nullptr) {
        std::map<std::string, Level> name_to_level = {
            {"TRACE", TRACE},
            {"DEBUG", DEBUG},
            {"INFO", INFO},
            {"WARNING", WARNING},
            {"ERROR", ERROR},
        };
        auto level = name_to_level.find(level_name);
        if (level != name_to_level.end()) {
            return level->second;
        } else {
            throw std::runtime_error(
                    "[FT][WARNING] Invalid logger level for env: " + std::string(s) + " with value: " + std::string(level_name));
            level_name = nullptr;
        }
    }    
    return Logger::DEFAULT_LOG_LEVEL;
}

void Logger::log(std::exception const& ex, Logger::Level level)
{
    log(level, "%s: %s", FTException::demangle(typeid(ex).name()).c_str(), ex.what());
}

}  // namespace fastertransformer
