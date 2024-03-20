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
    template <typename T>
    std::unique_ptr<Buffer> createHostBuffer(const std::vector<size_t>& shape, const T* data) {
        return createHostBuffer<T>(shape, static_cast<const void*>(data));
    }

    template <typename T>
    std::unique_ptr<Buffer> createHostBuffer(const std::vector<size_t>& shape, const void* data) {
        auto buffer = device_->allocateBuffer({getTensorType<T>(), shape, AllocationType::HOST}, {});
        memcpy(buffer->data(), data, sizeof(T) * buffer->size());
        return buffer;
    }

    template <typename T>
    std::unique_ptr<Buffer> createDeviceBuffer(const std::vector<size_t>& shape, const void* data) {
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

protected:
    const std::string &getTestDataPath() const {
        return test_data_path_;
    }

protected:
    DeviceBase* device_;
    std::string test_data_path_;
};
