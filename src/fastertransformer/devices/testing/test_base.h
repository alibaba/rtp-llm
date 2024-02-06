#pragma once

#include <gtest/gtest.h>

#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/core/Tensor.h"
#include "src/fastertransformer/core/allocator_cuda.h"
#include "src/fastertransformer/utils/logger.h"

using namespace fastertransformer;

template <DeviceType device_type>
class DeviceTestBase : public ::testing::Test {
public:
    void SetUp() override {
        device_ = DeviceFactory::getDevice(device_type);
    }
    void TearDown() override {}

protected:
    DeviceBase* device_;
};
