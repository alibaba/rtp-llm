#pragma once

#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/utils/logger.h"
#include "src/fastertransformer/cuda/cuda_utils.h"

#include <torch/torch.h>
#include <gtest/gtest.h>

#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <vector>

namespace fastertransformer {

template <DeviceType device_type>
class DeviceTestBase : public ::testing::Test {
public:
    void SetUp() override {
        device_ = DeviceFactory::getDevice(device_type);
    }
    void TearDown() override {}

protected:

    std::vector<size_t> convert_shape(torch::Tensor tensor) {
        std::vector<size_t> v_shape;
        for (int i = 0; i < tensor.dim(); i++) {
            v_shape.push_back(tensor.size(i));
        }
        return v_shape;
    }

    template <typename T>
    std::shared_ptr<Buffer> createHostBuffer(const std::vector<size_t>& shape, const std::vector<T>& data) {
        auto buffer = device_->allocateBuffer({getTensorType<T>(), shape, AllocationType::HOST}, {});
        memcpy(buffer->data(), data.data(), data.size() * sizeof(T));
        return buffer;
    }

    template <typename T>
    std::shared_ptr<Buffer>  createDeviceBuffer(const std::vector<size_t>& shape, const void* data) {
        auto buffer = device_->allocateBuffer({getTensorType<T>(), shape, AllocationType::DEVICE}, {});
        check_cuda_error(cudaMemcpy(buffer->data(), data, sizeof(T) * buffer->size(), cudaMemcpyHostToDevice));
        return buffer;
    }

    size_t TorchTensorBytesSize(torch::Tensor tensor) {
        return tensor.numel() * torch::elementSize(torch::typeMetaToScalarType(tensor.dtype()));
    }

    std::vector<int64_t> TorchShapeConvert(const Buffer& buffer) {
        std::vector<int64_t> tensor_shape(buffer.shape().size());
        std::transform(
            buffer.shape().begin(), buffer.shape().end(), tensor_shape.begin(), [](size_t x) { return (int64_t)x;});
        return tensor_shape;
    }

    template <typename T>
    std::shared_ptr<Buffer> CreateDeviceBuffer(torch::Tensor& tensor) {
        EXPECT_EQ(tensor.device(), torch::kCPU);
        if constexpr (std::is_same<T, half>::value || std::is_same<T, float>::value) {
            EXPECT_EQ(tensor.dtype(), torch::kFloat);
        } else if constexpr (std::is_same<T, int>::value) {
            EXPECT_EQ(tensor.dtype(), torch::kInt);
        }
        
        size_t bytes_size = tensor.numel() * sizeof(T);
        T* tmp = reinterpret_cast<T*>(malloc(bytes_size));
        EXPECT_NE(tmp, nullptr);
        float* data = reinterpret_cast<float*>(tensor.data_ptr());
        if constexpr (std::is_same<T, half>::value) {
            auto float2half = [](float x) { return half(x); };
            std::transform(data, data + tensor.numel(), tmp, float2half);
        } else if constexpr (std::is_same<T, float>::value || std::is_same<T, int>::value) {
            std::memcpy(tmp, data, bytes_size);
        } else {
            free(tmp);
            FAIL();
        }

        auto buffer = device_->allocateBuffer({getTensorType<T>(), convert_shape(tensor), AllocationType::DEVICE}, {});
        check_cuda_error(cudaMemcpy(buffer->data(), (void*)tmp, bytes_size, cudaMemcpyHostToDevice));
        cudaDeviceSynchronize();
        free(tmp);
        return buffer;
    }

    torch::Tensor CreateTensor(const Buffer& buffer) {
        EXPECT_EQ(buffer.where(), MemoryType::MEMORY_GPU);
        size_t bytes_size = buffer.size() * sizeof(float);
        float* result = reinterpret_cast<float*>(malloc(bytes_size));
        if (buffer.type() == DataType::TYPE_FP16) {
            half* data = reinterpret_cast<half*>(buffer.data());
            half* tmp = reinterpret_cast<half*>(malloc(buffer.size() * sizeof(half)));
            check_cuda_error(cudaMemcpy(tmp, data, sizeof(half) * buffer.size(), cudaMemcpyDeviceToHost));
            cudaDeviceSynchronize();
            auto half2float = [](half x) { return float(x); };
            std::transform(tmp, tmp + buffer.size(), result, half2float);
            free(tmp);
        } else if (buffer.type() == DataType::TYPE_FP32) {
            float* data = reinterpret_cast<float*>(buffer.data());
            check_cuda_error(cudaMemcpy(result, data, sizeof(float) * buffer.size(), cudaMemcpyDeviceToHost));
            cudaDeviceSynchronize();
        }
        auto tensor = torch::from_blob(
            (void*)result, TorchShapeConvert(buffer), torch::Device(torch::kCPU)).to(torch::kFloat);
        return tensor;
    }

    template <typename T>
    std::vector<T> CopyToHost(const Buffer& buffer) {
        std::vector<T> host_buffer(buffer.size());
        check_cuda_error(cudaMemcpy(host_buffer.data(), buffer.data(), sizeof(T) * buffer.size(), cudaMemcpyDeviceToHost));
        return host_buffer;
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

} // namespace fastertransformer