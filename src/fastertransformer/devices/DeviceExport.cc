#include "src/fastertransformer/devices/DeviceFactory.h"

#include "torch/extension.h"
#include <cstdio>
#include <iostream>
#include <memory>

using namespace fastertransformer;

namespace torch_ext {
}

// static auto devices_init = torch::RegisterOperators("devices::init_devices", &DeviceFactory::initDevices);

