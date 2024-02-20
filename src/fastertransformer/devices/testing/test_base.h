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
    template <typename T>
    std::shared_ptr<Tensor> createHostTensor(const std::vector<size_t>& shape, const std::vector<T>& data) {
        auto tensor = device_->allocateBuffer({getTensorType<T>(), shape, AllocationType::HOST}, {});
        assert(tensor->size() == data.size());
        memcpy(tensor->data(), data.data(), data.size() * sizeof(T));
        return tensor;
    }

    void assertOpSuccess(OpStatus status) {
        assert(status.ok());
    }

    template<typename T>
    void assertTensorValueEqual(const Tensor& tensor, const std::vector<T>& expected) {
        assert(tensor.size() == expected.size());
        for (size_t i = 0; i < tensor.size(); i++) {
            printf("i=%ld, tensor[i] = %f, expected[i] = %f\n", i, ((T*)tensor.data())[i], expected[i]);
            assert(((T*)tensor.data())[i] == expected[i]);
        }
    }

protected:
    DeviceBase* device_;
};
