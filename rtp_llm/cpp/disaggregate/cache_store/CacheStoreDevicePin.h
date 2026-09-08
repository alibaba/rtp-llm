#pragma once

#include "rtp_llm/cpp/utils/DevicePin.h"
#include "rtp_llm/cpp/utils/Logger.h"

#include <exception>

namespace rtp_llm {

inline bool tryPinThreadDevice(int device_id, const char* context) {
    try {
        setCurrentThreadDeviceIfNeeded(device_id);
        return true;
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("%s device pin failed, error is %s", context, e.what());
        return false;
    } catch (...) {
        RTP_LLM_LOG_WARNING("%s device pin failed with unknown error", context);
        return false;
    }
}

}  // namespace rtp_llm
