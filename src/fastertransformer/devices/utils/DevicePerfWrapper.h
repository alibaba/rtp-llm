#pragma once
#include "src/fastertransformer/devices/DeviceBase.h"

namespace fastertransformer {

class DevicePerfWrapper {
public:
    DevicePerfWrapper(const DeviceBase* device, const std::string& name) {
        device_ = device;   
        name_ = name;
        device_->perfRangePush(name_);
        is_stopped_ = false;
    }

    void stop() {
        if (is_stopped_) {
            return;
        }
        device_->perfRangePop();
        is_stopped_ = true;
    }

    ~DevicePerfWrapper() {
        stop();
    }
private:
    bool is_stopped_ = false;
    const DeviceBase* device_;
    std::string name_;
};

} // namespace fastertransformer