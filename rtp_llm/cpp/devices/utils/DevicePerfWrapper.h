#pragma once
#include "rtp_llm/cpp/devices/DeviceBase.h"

namespace rtp_llm {

template<typename... Args>
class DevicePerfWrapper {
public:
    DevicePerfWrapper(const DeviceBase* device, const std::string& format, const Args&... args) {
        device_ = device;
        if (device_->enableDevicePerf()) {
            name_ = rtp_llm::fmtstr(format, args...);
            device_->perfRangePush(name_);
            perfing_ = true;
        }
    }

    void stop() {
        if (!perfing_) {
            return;
        }
        device_->perfRangePop();
        perfing_ = false;
    }

    ~DevicePerfWrapper() {
        stop();
    }

private:
    bool              perfing_ = false;
    const DeviceBase* device_;
    std::string       name_;
};

}  // namespace rtp_llm