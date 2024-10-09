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
            throw std::runtime_error(
                    "[FT][WARNING] Invalid logger level for env: " + std::string(s) + " with value: " + std::string(level_name));
            level_name = nullptr;
        }
    }    
    return base_log_level_;
}

}  // namespace fastertransformer
