#pragma once

#include <torch/torch.h>

#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/devices/testing/TestBase.h"
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/utils/logger.h"
#include "src/fastertransformer/devices/utils/BufferUtils.h"
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

    template <typename T>
    std::unique_ptr<Buffer> CreateKVBlockArray(torch::Tensor& k_cache,
                                               torch::Tensor& v_cache,
                                               size_t seq_len,
                                               size_t maxBlocksPerSeq,
                                               size_t tokensPerBlock) {
        // k, v tensor shape is [batch_size, head_kv_size, kv_seq_len, head_dim].
        // split tensor to small tensor which shape is [head_size, tokensPerBlock, head_dim].
        // and the tensor map is [block_size, 2, block_num]

        auto batch_size     = k_cache.size(0);
        auto head_kv_size   = k_cache.size(1);
        auto kv_seq_len     = k_cache.size(2);
        auto head_dim       = k_cache.size(3);

        EXPECT_GE(maxBlocksPerSeq * tokensPerBlock, kv_seq_len + seq_len);
        EXPECT_EQ(kv_seq_len % tokensPerBlock, 0);
        auto k_tensor = k_cache.view({(int)batch_size,
                                      (int)head_kv_size,
                                      (int)(kv_seq_len / tokensPerBlock),
                                      (int)tokensPerBlock,
                                      (int)head_dim});
        k_tensor = k_tensor.transpose(1, 2);

        auto v_tensor = v_cache.view({(int)batch_size,
                                      (int)head_kv_size,
                                      (int)(kv_seq_len / tokensPerBlock),
                                      (int)tokensPerBlock,
                                      (int)head_dim});
        v_tensor = v_tensor.transpose(1, 2);

        std::vector<void*> block_pointers(batch_size * 2 * maxBlocksPerSeq, nullptr);

        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < maxBlocksPerSeq; j++) {
                if (j < (int)(kv_seq_len / tokensPerBlock)) {
                    auto k_tmp = k_tensor.index({i, j, "..."});
                    auto v_tmp = v_tensor.index({i, j, "..."});
                    auto k_buffer = CreateDeviceBuffer<T>(k_tmp);
                    auto v_buffer = CreateDeviceBuffer<T>(v_tmp);
                    block_pointers[i * maxBlocksPerSeq * 2 + j] = k_buffer->data();
                    block_pointers[i * maxBlocksPerSeq * 2 + maxBlocksPerSeq + j] = v_buffer->data();
                } else {
                    auto k_tmp = torch::zeros({1, 1, (int)head_kv_size, (int)tokensPerBlock, (int)head_dim});
                    auto v_tmp = torch::zeros({1, 1, (int)head_kv_size, (int)tokensPerBlock, (int)head_dim});
                    auto k_buffer = CreateDeviceBuffer<T>(k_tmp);
                    auto v_buffer = CreateDeviceBuffer<T>(v_tmp);
                    block_pointers[i * maxBlocksPerSeq * 2 + j] = k_buffer->data();
                    block_pointers[i * maxBlocksPerSeq * 2 + maxBlocksPerSeq + j] = v_buffer->data();
                }
            }
        }
        for (auto ptr : block_pointers) {
            EXPECT_NE(ptr, nullptr);
        }

        
        auto buffer = device_->allocateBuffer(
            {DataType::TYPE_UINT64, {(size_t)batch_size, maxBlocksPerSeq}, AllocationType::HOST}, {});

        std::memcpy(buffer->data(), block_pointers.data(), block_pointers.size() * sizeof(void*));
        return std::move(buffer);
    }
    

};

} // namespace fastertransformer
