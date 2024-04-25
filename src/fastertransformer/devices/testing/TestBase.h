#pragma once

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/devices/utils/BufferTorchUtils.h"
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/utils/logger.h"
#include "maga_transformer/cpp/cache/CacheManager.h"
#include "maga_transformer/cpp/utils/KvCacheUtils.h"

#include <numeric>
#include <stdlib.h>

using namespace fastertransformer;

static const std::string DEFAULT_DEVICE = "CPU";

class DeviceTestBase : public ::testing::Test {
public:
    void SetUp() override {
        setenv("FT_DEBUG_LEVEL", "DEBUG", 1);
        const auto test_src_dir = getenv("TEST_SRCDIR");
        const auto test_work_space = getenv("TEST_WORKSPACE");
        const auto test_binary = getenv("TEST_BINARY");

        auto device_name = getenv("TEST_USING_DEVICE");
        DeviceType device_type = getDeviceType(device_name ? device_name : DEFAULT_DEVICE);
        device_ = DeviceFactory::getDevice(device_type);

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
    void printBuffer(const Buffer& buffer, const std::string& hint = "") {
        auto values = getBufferValues<T>(buffer);
        for (size_t i = 0; i < values.size(); i++) {
            std::cout << values[i] << " ";
        }
        std::cout << " " << hint << std::endl;
    }

    BufferPtr createBuffer(const std::vector<size_t>& shape, DataType type,
                           AllocationType alloc_type = AllocationType::DEVICE)
    {
        if (alloc_type == AllocationType::DEVICE) {
            return device_->allocateBuffer({type, shape, AllocationType::DEVICE}, {});
        } else {
            return device_->allocateBuffer({type, shape, AllocationType::HOST}, {});
        }
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
        if (data && (buffer->size() > 0)) {
            device_->copy(CopyParams(*buffer, *host_buffer));
        }
        return move(buffer);
    }

    template <typename T>
    BufferPtr createDeviceBuffer(torch::Tensor tensor) {
        const auto targetType = c10::CppTypeToScalarType<T>::value;
        if (tensor.scalar_type() != targetType) {
            tensor = tensor.to(targetType);
        }
        return tensorToBuffer(tensor);
    }

    template<typename T>
    void assertBufferValueEqual(const Buffer& buffer, const std::vector<T>& expected) {
        ASSERT_EQ(buffer.size(), expected.size());
        auto comp_buffer = device_->allocateBuffer(
            {buffer.type(), buffer.shape(), AllocationType::HOST}
        );
        device_->copy(CopyParams(*comp_buffer, buffer));
        for (size_t i = 0; i < buffer.size(); i++) {
            printf("i=%ld, buffer[i] = %f, expected[i] = %f\n", i,
                    (comp_buffer->data<T>())[i], expected[i]);
            ASSERT_EQ((comp_buffer->data<T>())[i], expected[i]);
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

    std::unique_ptr<Buffer> tensorToBuffer(const torch::Tensor& tensor,
                                           AllocationType alloc_type = AllocationType::DEVICE)
    {
        assert(tensor.is_cpu());
        auto buffer = torchTensor2Buffer(tensor);
        if (alloc_type == AllocationType::DEVICE) {
            auto device_buffer = device_->allocateBuffer(
                {buffer->type(), buffer->shape(), AllocationType::DEVICE}
            );
            device_->copy(CopyParams(*device_buffer, *buffer));
            printf("created device buffer from tensor at %p with data=%p\n", device_buffer.get(), device_buffer->data());
            return std::move(device_buffer);
        } else {
            return std::move(buffer);
        }
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
                    auto k_buffer = createDeviceBuffer<T>(k_tmp);
                    auto v_buffer = createDeviceBuffer<T>(v_tmp);
                    block_pointers[i * maxBlocksPerSeq * 2 + j] = k_buffer->data();
                    block_pointers[i * maxBlocksPerSeq * 2 + maxBlocksPerSeq + j] = v_buffer->data();
                } else {
                    auto k_tmp = torch::zeros({1, 1, (int)head_kv_size, (int)tokensPerBlock, (int)head_dim});
                    auto v_tmp = torch::zeros({1, 1, (int)head_kv_size, (int)tokensPerBlock, (int)head_dim});
                    auto k_buffer = createDeviceBuffer<T>(k_tmp);
                    auto v_buffer = createDeviceBuffer<T>(v_tmp);
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

    BufferPtr allocateKVBlocks(const rtp_llm::CacheConfig& cache_config,
                               const std::vector<int32_t>& input_lengths,
                               const std::vector<int32_t>& sequence_lengths)
    {
        cache_manager_ = std::make_shared<rtp_llm::CacheManager>(cache_config, device_);
        const auto max_seq_len = *std::max_element(input_lengths.begin(), input_lengths.end());
        const auto batch_layer_kv_block_num =
            ((max_seq_len / cache_config.seq_size_per_block) + 2) * input_lengths.size();
        const auto batch_size = input_lengths.size();
        const auto num_layers = 1;

        auto kv_blocks_buf = device_->allocateBuffer({
            DataType::TYPE_INT64, {num_layers, 2, batch_size, batch_layer_kv_block_num}, AllocationType::HOST
        });
        rtp_llm::BatchKVCacheBlockAddr batch_kv_cache;

        for (auto i = 0; i < batch_size; i++) {
            auto [success, kv_cache] = cache_manager_->malloc(batch_layer_kv_block_num);
            EXPECT_TRUE(success);
            batch_kv_cache.pushBack(kv_cache);
        }
        for (auto i = 0; i < batch_size; i++) {
            rtp_llm::memcpyKvCache(
                kv_blocks_buf->data<uint64_t>(),
                batch_kv_cache.k_ptr[i],
                batch_kv_cache.v_ptr[i],
                1,
                kv_blocks_buf->shape().back(),
                batch_size,
                i
            );
        }

        return move(kv_blocks_buf);
    }

    void assertTensorClose(const torch::Tensor& a, const torch::Tensor& b,
                           double rtol = 0, double atol = 0) {
        auto a_cmp = a;
        auto b_cmp = b;
        rtol = rtol ? rtol : rtol_;
        atol = atol ? atol : rtol_;
        ASSERT_TRUE(a.is_floating_point() == b.is_floating_point());

        if (a_cmp.dtype() != b_cmp.dtype()) {
            auto cmp_type = (a_cmp.dtype().itemsize() > b_cmp.dtype().itemsize()) ?
                            a_cmp.dtype() : b_cmp.dtype();
            a_cmp = a_cmp.to(cmp_type);
            b_cmp = b_cmp.to(cmp_type);
        }

        const auto close = torch::allclose(a_cmp, b_cmp, rtol, atol);
        if (!close) {
            std::cout << "assert tensor close failed!" << std::endl;
            std::cout << "rtol: " << rtol << std::endl;
            std::cout << "atol: " << atol << std::endl;
            std::cout << "a: " << a << std::endl;
            std::cout << "b: " << b << std::endl;
            std::cout << "cmp diff: " << a_cmp - b_cmp << std::endl;
            std::cout << "abs diff: " << torch::abs(a_cmp - b_cmp) / torch::abs(a_cmp) << std::endl;
            ASSERT_TRUE(false);
        }
    }

    const std::string &getTestDataPath() const {
        return test_data_path_;
    }

protected:
    DeviceBase* device_;
    std::string test_data_path_;
    double rtol_ = 1e-03;
    double atol_ = 1e-03;
    rtp_llm::CacheManagerPtr cache_manager_;
};

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
