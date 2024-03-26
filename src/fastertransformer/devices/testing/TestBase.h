#pragma once

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/devices/utils/BufferUtils.h"
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/utils/logger.h"

#include <numeric>
#include <stdlib.h>

using namespace fastertransformer;

template <DeviceType device_type>
class DeviceTestBase : public ::testing::Test {
public:
    void SetUp() override {
        setenv("FT_DEBUG_LEVEL", "DEBUG", 1);
        device_ = DeviceFactory::getDevice(device_type);
        const char* test_src_dir = std::getenv("TEST_SRCDIR");
        const char* test_work_space = std::getenv("TEST_WORKSPACE");
        const char* test_binary = getenv("TEST_BINARY");

        if (!(test_src_dir && test_work_space && test_binary)) {
            std::cerr << "Unable to retrieve TEST_SRCDIR / TEST_WORKSPACE / TEST_BINARY env!" << std::endl;
            abort();
        }

        std::string test_binary_str = std::string(test_binary);
        assert(*test_binary_str.rbegin() != '/');
        size_t filePos = test_binary_str.rfind('/');
        test_data_path_ = std::string(test_src_dir) + "/" + std::string(test_work_space) + "/"
                        + test_binary_str.substr(0, filePos) + "/";

        std::cout << "test_src_dir [" << test_src_dir << "]" << std::endl;
        std::cout << "test_work_space [" << test_work_space << "]" << std::endl;
        std::cout << "test_binary [" << test_binary << "]" << std::endl;
        std::cout << "test using data path [" << test_data_path_ << "]" << std::endl;
    }

    void TearDown() override {}

protected:
    double rtol_ = 1e-03;
    double atol_ = 1e-03;

protected:
    template <typename T>
    void printBuffer(const Buffer& buffer, const std::string& hint = "") {
        auto values = getBufferValues<T>(buffer);
        for (size_t i = 0; i < values.size(); i++) {
            std::cout << values[i] << " ";
        }
        std::cout << " " << hint << std::endl;
    }

    template <typename T>
    BufferPtr createBuffer(const std::vector<size_t>& shape, const std::vector<T>& data,
                           AllocationType alloc_type = AllocationType::DEVICE)
    {
        const auto num_elements = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
        assert(num_elements == data.size());
        if (alloc_type == AllocationType::DEVICE) {
            return createDeviceBuffer<T>(shape, data.data());
        } else {
            return createHostBuffer<T>(shape, data.data());
        }
    }

    template <typename T>
    BufferPtr createHostBuffer(const std::vector<size_t>& shape, const T* data) {
        return createHostBuffer<T>(shape, static_cast<const void*>(data));
    }

    template <typename T>
    BufferPtr createHostBuffer(const std::vector<size_t>& shape, const void* data) {
        auto buffer = device_->allocateBuffer({getTensorType<T>(), shape, AllocationType::HOST}, {});
        if (data && (buffer->size() > 0)) {
            memcpy(buffer->data(), data, sizeof(T) * buffer->size());
        }
        return buffer;
    }

    template <typename T>
    BufferPtr createDeviceBuffer(const std::vector<size_t>& shape, const void* data) {
        auto host_buffer = createHostBuffer<T>(shape, data);
        auto buffer = device_->allocateBuffer({getTensorType<T>(), shape, AllocationType::DEVICE}, {});
        device_->copy(CopyParams(*buffer, *host_buffer));
        return move(buffer);
    }

    template<typename T>
    void assertBufferValueEqual(const Buffer& buffer, const std::vector<T>& expected) {
        ASSERT_EQ(buffer.size(), expected.size());
        for (size_t i = 0; i < buffer.size(); i++) {
            printf("i=%ld, buffer[i] = %f, expected[i] = %f\n", i, ((T*)buffer.data())[i], expected[i]);
            ASSERT_EQ(((T*)buffer.data())[i], expected[i]);
        }
    }

    template<typename T>
    std::vector<T> getBufferValues(const Buffer& buffer) {
        std::vector<T> values(buffer.size());
        if (buffer.where() == MemoryType::MEMORY_GPU) {
            auto host_buffer = createHostBuffer<T>(buffer.shape(), nullptr);
            device_->copy(CopyParams(*host_buffer, buffer));
            memcpy(values.data(), host_buffer->data(), sizeof(T) * buffer.size());
        } else {
            memcpy(values.data(), buffer.data(), sizeof(T) * buffer.size());
        }
        return values;
    }

    torch::Tensor bufferToTensor(const Buffer& buffer) {
        auto host_buffer = device_->allocateBuffer(
            {buffer.type(), buffer.shape(), AllocationType::HOST}
        );
        device_->copy(CopyParams(*host_buffer, buffer));

        return torch::from_blob(
            host_buffer->data(), bufferShapeToTorchShape(buffer),
            c10::TensorOptions().device(torch::Device(torch::kCPU))
                                .dtype(dataTypeToTorchType(buffer.type()))
        ).clone();
    }

    void assertTensorClose(const torch::Tensor& a, const torch::Tensor& b) {
        auto a_cmp = a;
        auto b_cmp = b;
        ASSERT_TRUE(a.is_floating_point() == b.is_floating_point());

        if (a_cmp.dtype() != b_cmp.dtype()) {
            auto cmp_type = (a_cmp.dtype().itemsize() > b_cmp.dtype().itemsize()) ?
                            a_cmp.dtype() : b_cmp.dtype();
            a_cmp = a_cmp.to(cmp_type);
            b_cmp = b_cmp.to(cmp_type);
        }

        ASSERT_TRUE(torch::allclose(a_cmp, b_cmp, rtol_, atol_));
    }

protected:
    const std::string &getTestDataPath() const {
        return test_data_path_;
    }

protected:
    DeviceBase* device_;
    std::string test_data_path_;
};
