#pragma once

#include <torch/torch.h>

#include "src/fastertransformer/devices/testing/TestBase.h"

namespace fastertransformer {

template <DeviceType device>
class FfnTest : public DeviceTestBase<device> {
};

} // namespace fastertransformer
