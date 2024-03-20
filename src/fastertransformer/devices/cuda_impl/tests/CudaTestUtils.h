#pragma once

#include <torch/torch.h>

#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/devices/testing/TestBase.h"
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/utils/logger.h"
#include "src/fastertransformer/utils/BufferUtils.h"
#include "src/fastertransformer/cuda/cuda_utils.h"

#include <gtest/gtest.h>

#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <vector>

template <>
struct c10::CppTypeToScalarType<half>
    : std::integral_constant<c10::ScalarType, c10::ScalarType::Half> {};

namespace fastertransformer {

class CudaDeviceTestBase : public DeviceTestBase<DeviceType::Cuda> {
public:
    void SetUp() override {
        DeviceTestBase<DeviceType::Cuda>::SetUp();
    }
    void TearDown() override {
        DeviceTestBase<DeviceType::Cuda>::TearDown();
    }

protected:

    template <typename T>
    std::unique_ptr<Buffer> CreateDeviceBuffer(torch::Tensor& tensor) {
        EXPECT_EQ(tensor.device(), torch::kCPU);
        if constexpr (std::is_same<T, int>::value) {
            EXPECT_EQ(tensor.dtype(), torch::kInt);
        }

        size_t bytes_size = tensor.numel() * sizeof(T);
        T* tmp = reinterpret_cast<T*>(malloc(bytes_size));
        EXPECT_NE(tmp, nullptr);
        float* data = reinterpret_cast<float*>(tensor.data_ptr());
        if (std::is_same<T, half>::value && (tensor.dtype() == torch::kFloat32)) {
            auto float2half = [](float x) { return half(x); };
            std::transform(data, data + tensor.numel(), tmp, float2half);
        } else if (c10::CppTypeToScalarType<T>::value == tensor.dtype()) {
            std::memcpy(tmp, data, bytes_size);
        } else {
            free(tmp);
            assert(false);
        }

        auto buffer = device_->allocateBuffer(
            {getTensorType<T>(), torchShapeToBufferShape(tensor.sizes()), AllocationType::DEVICE}, {});
        check_cuda_error(cudaMemcpy(buffer->data(), (void*)tmp, bytes_size, cudaMemcpyHostToDevice));
        cudaDeviceSynchronize();
        free(tmp);
        return move(buffer);
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
            (void*)result, bufferShapeToTorchShape(buffer), torch::Device(torch::kCPU)).to(torch::kFloat);
        return tensor;
    }

};

} // namespace fastertransformer