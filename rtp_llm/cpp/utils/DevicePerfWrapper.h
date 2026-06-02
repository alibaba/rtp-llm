#pragma once
#include "rtp_llm/cpp/utils/StringUtil.h"
#include <string>

namespace rtp_llm {

template<typename... Args>
class DevicePerfWrapper {
public:
    DevicePerfWrapper(bool enable_device_perf, const std::string& format, const Args&... args) {
        if (enable_device_perf) {
            name_    = rtp_llm::fmtstr(format, args...);
            perfing_ = true;
        }
    }

    void stop() {
        if (!perfing_) {
            return;
        }
        // perfRangePop is a no-op
        perfing_ = false;
    }

    ~DevicePerfWrapper() {
        stop();
    }

private:
    bool        perfing_ = false;
    std::string name_;
};

}  // namespace rtp_llm
