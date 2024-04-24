#pragma once

#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/devices/DeviceBase.h"

namespace fastertransformer {
    void printBufferData(const Buffer& buffer, const std::string& hint, DeviceBase* device = nullptr);
}
