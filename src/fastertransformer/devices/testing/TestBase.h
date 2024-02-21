#pragma once

#include <gtest/gtest.h>

#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/core/Buffer.h"
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
    template <typename T>
    std::shared_ptr<Buffer> createHostBuffer(const std::vector<size_t>& shape, const std::vector<T>& data) {
        auto buffer = device_->allocateBuffer({getTensorType<T>(), shape, AllocationType::HOST}, {});
        memcpy(buffer->data(), data.data(), data.size() * sizeof(T));
        return buffer;
    }

    template<typename T>
    void assertBufferValueEqual(const Buffer& buffer, const std::vector<T>& expected) {
        ASSERT_EQ(buffer.size(), expected.size());
        for (size_t i = 0; i < buffer.size(); i++) {
            printf("i=%ld, buffer[i] = %f, expected[i] = %f\n", i, ((T*)buffer.data())[i], expected[i]);
            ASSERT_EQ(((T*)buffer.data())[i], expected[i]);
        }
    }

protected:
    DeviceBase* device_;
};
