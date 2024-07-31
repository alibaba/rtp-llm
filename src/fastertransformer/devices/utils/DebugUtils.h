#pragma once

#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/devices/DeviceBase.h"

namespace fastertransformer {

inline bool isDebugMode() {
    return Logger::getLogger().getLevel() == Logger::DEBUG;
}

inline bool enableDebugPrint() {
    return Logger::getLogger().getPrintLevel() == Logger::DEBUG;
}

void printBufferData(const Buffer& buffer, const std::string& hint, DeviceBase* device = nullptr, bool force_print = false);

}
