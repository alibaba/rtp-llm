#pragma once

#include "src/fastertransformer/devices/DeviceBase.h"
// #include "src/fastertransformer/devices/DeviceFactory.h"

#include "torch/extension.h"
#include <cstdio>
#include <iostream>
#include <memory>

namespace torch_ext {

namespace ft = fastertransformer;

// For now the DeviceExporter only export single device as there is no pipeline parallelism
// It may need to hold multiple devices in the future.
class DeviceExporter {
public:
    DeviceExporter(const fastertransformer::DeviceInitParams& params) : device_params_(params) {};
    virtual ~DeviceExporter() {};

    ft::DeviceType getDeviceType();
    int64_t getDeviceId();

protected:
    fastertransformer::DeviceInitParams device_params_;
};

template <class Device>
class DeviceExporterImpl : public DeviceExporter {
public:
    DeviceExporterImpl(const fastertransformer::DeviceInitParams& params) : DeviceExporter(params) {};
    ~DeviceExporterImpl() {};
};

} // namespace torch_ext

