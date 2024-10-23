#pragma once

#include "src/fastertransformer/devices/DeviceFactory.h"

#include "torch/extension.h"
#include <cstdio>
#include <iostream>
#include <memory>

namespace torch_ext {

void registerDeviceOps(py::module& m);

} // namespace torch_ext

