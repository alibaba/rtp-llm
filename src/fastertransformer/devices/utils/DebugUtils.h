#pragma once

#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/devices/DeviceBase.h"

namespace fastertransformer {

inline bool isDebugMode() {
    static char* level_name = std::getenv("FT_DEBUG_LEVEL");
    return level_name && (strcmp(level_name, "DEBUG") == 0);
}

inline bool enableDebugPrint() {
    static char* level_name = std::getenv("FT_DEBUG_PRINT_LEVEL");
    return level_name && (strcmp(level_name, "DEBUG") == 0);
}

void printBufferData(const Buffer& buffer, const std::string& hint, DeviceBase* device = nullptr);

}
