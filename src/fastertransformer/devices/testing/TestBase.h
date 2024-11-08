#pragma once

#include <arpa/inet.h>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <numeric>
#include <stdlib.h>
#include <sys/socket.h>
#include <torch/torch.h>

#ifdef ENABLE_FP8
#include <cuda_fp8.h>
#endif

#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"
#include "src/fastertransformer/core/Buffer.h"
#include "maga_transformer/cpp/utils/Logger.h"
#include "src/fastertransformer/th_op/GptInitParameter.h"
#include "maga_transformer/cpp/cache/CacheManager.h"
#include "maga_transformer/cpp/stream/StreamCacheResource.h"
#include "maga_transformer/cpp/utils/KVCacheUtils.h"
#include "autil/EnvUtil.h"


using namespace fastertransformer;

static const std::string DEFAULT_DEVICE = "CPU";

#define ASSERT_VECTOR_EQUAL(x, y)                                                                                       \
    ASSERT_EQ(x.size(), y.size()) << "Vectors x and y are of unequal length";                                          \
    for (int i = 0; i < x.size(); ++i) {                                                                               \
        ASSERT_EQ(x[i], y[i]) << "Vectors x and y differ at index " << i;                                              \
    }

#define ASSERT_VECTOR_NEAR(x, y, abs_error)                                                                            \
    ASSERT_EQ(x.size(), y.size()) << "Vectors x and y are of unequal length";                                          \
    for (int i = 0; i < x.size(); ++i) {                                                                               \
        ASSERT_NEAR(x[i], y[i], abs_error) << "Vectors x and y differ at index " << i;                                 \
    }

class DeviceTestBase: public ::testing::Test {
public:
    void SetUp() override {
        initTestDevices();
        initTestDataDir();
        torch::manual_seed(114514);
    }

    virtual void initTestDevices() {
        autil::EnvUtil::setEnv("DEVICE_RESERVE_MEMORY_BYTES", std::to_string(device_reserve_memory_size_));
        autil::EnvUtil::setEnv("HOST_RESERVE_MEMORY_BYTES", std::to_string(host_reserve_memory_size_));
        ft::DeviceFactory::initDevices(GptInitParameter());
        device_ = ft::DeviceFactory::getDefaultDevice();
    }

    void initTestDataDir() {
        const auto test_src_dir = getenv("TEST_SRCDIR");
        const auto test_work_space = getenv("TEST_WORKSPACE");
        const auto test_binary = getenv("TEST_BINARY");
        if (!(test_src_dir && test_work_space && test_binary)) {
            std::cerr << "Unable to retrieve TEST_SRCDIR / TEST_WORKSPACE / TEST_BINARY env!" << std::endl;
            abort();
        }

        std::string test_binary_str = std::string(test_binary);
        FT_CHECK(*test_binary_str.rbegin() != '/');
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
    void printBuffer(const ft::Buffer& buffer, const std::string& hint = "") {
        auto values = getBufferValues<T>(buffer);
        for (size_t i = 0; i < values.size(); i++) {
            std::cout << values[i] << " ";
        }
        std::cout << " " << hint << std::endl;
    }

    ft::BufferPtr createBuffer(const std::vector<size_t>& shape, ft::DataType type,
                               ft::AllocationType alloc_type = ft::AllocationType::DEVICE)
    {
        if (alloc_type == ft::AllocationType::DEVICE) {
            return device_->allocateBuffer({type, shape, ft::AllocationType::DEVICE}, {});
        } else {
            return device_->allocateBuffer({type, shape, ft::AllocationType::HOST}, {});
        }
    }

    template <typename T>
    ft::BufferPtr createBuffer(const std::vector<size_t>& shape, const std::vector<T>& data,
                               ft::AllocationType alloc_type = ft::AllocationType::DEVICE)
    {
        const auto num_elements = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
        FT_CHECK(num_elements == data.size());
        if (alloc_type == ft::AllocationType::DEVICE) {
            return createDeviceBuffer<T>(shape, data.data());
        } else {
            return createHostBuffer<T>(shape, data.data());
        }
    }

    template <typename T>
    ft::BufferPtr createHostBuffer(const std::vector<size_t>& shape, const T* data) {
        return createHostBuffer<T>(shape, static_cast<const void*>(data));
    }

    template <typename T>
    ft::BufferPtr createHostBuffer(const std::vector<size_t>& shape, const void* data) {
        auto buffer = device_->allocateBuffer({ft::getTensorType<T>(), shape, ft::AllocationType::HOST}, {});
        if (data && (buffer->size() > 0)) {
            memcpy(buffer->data(), data, sizeof(T) * buffer->size());
        }
        device_->syncAndCheck();
        return buffer;
    }

    template <typename T>
    ft::BufferPtr createDeviceBuffer(const std::vector<size_t>& shape, const void* data) {
        auto host_buffer = createHostBuffer<T>(shape, data);
        auto buffer = device_->allocateBuffer({ft::getTensorType<T>(), shape, ft::AllocationType::DEVICE}, {});
        if (data && (buffer->size() > 0)) {
            device_->copy({*buffer, *host_buffer});
        }
        device_->syncAndCheck();
        return buffer;
    }

    template <typename T>
    ft::BufferPtr createDeviceBuffer(torch::Tensor tensor) {
        const auto targetType = c10::CppTypeToScalarType<T>::value;
        if (tensor.scalar_type() != targetType) {
            tensor = tensor.to(targetType);
        }
        return tensorToBuffer(tensor);
    }

    template<typename T>
    void assertBufferValueEqual(const ft::Buffer& buffer, const std::vector<T>& expected) {
        ASSERT_EQ(buffer.size(), expected.size());
        auto comp_buffer = device_->allocateBuffer(
            {buffer.type(), buffer.shape(), ft::AllocationType::HOST}
        );
        device_->copy({*comp_buffer, buffer});
        device_->syncAndCheck();
        for (size_t i = 0; i < buffer.size(); i++) {
            printf("i=%ld, buffer[i] = %f, expected[i] = %f\n", i,
                   float((comp_buffer->data<T>())[i]), float(expected[i]));
            ASSERT_EQ((comp_buffer->data<T>())[i], expected[i]);
        }
    }

    template<typename T>
    std::vector<T> getBufferValues(const ft::Buffer& buffer) {
        std::vector<T> values(buffer.size());
        device_->syncAndCheck();
        if (buffer.where() == ft::MemoryType::MEMORY_GPU) {
            auto host_buffer = createHostBuffer<T>(buffer.shape(), nullptr);
            device_->copy({*host_buffer, buffer});
            device_->syncAndCheck();
            memcpy(values.data(), host_buffer->data(), sizeof(T) * buffer.size());
        } else {
            memcpy(values.data(), buffer.data(), sizeof(T) * buffer.size());
        }
        return values;
    }

    ft::BufferPtr tensorToBuffer(const torch::Tensor& tensor,
                                 ft::AllocationType alloc_type = ft::AllocationType::DEVICE)
    {
        if (tensor.is_quantized()) {
            return tensorToBuffer(tensor,
                                  tensor.q_per_channel_scales().to(torch::kHalf),
                                  tensor.q_per_channel_zero_points().to(torch::kHalf));
        }
        FT_CHECK(tensor.is_cpu());
        auto buffer = ft::torchTensor2Buffer(tensor);
        if (alloc_type == ft::AllocationType::DEVICE) {
            auto device_buffer = device_->allocateBuffer(
                    {buffer->type(), buffer->shape(), ft::AllocationType::DEVICE}
            );
            device_->copy({*device_buffer, *buffer});
            device_->syncAndCheck();
            printf("created device buffer from tensor at %p with data=%p\n", device_buffer.get(), device_buffer->data());
            return device_buffer;
        } else {
            return buffer;
        }
    }

    ft::BufferPtr tensorToBuffer(const torch::Tensor& tensor,
                                 const torch::Tensor& scales,
                                 const torch::Tensor& zeros,
                                 ft::AllocationType alloc_type = ft::AllocationType::DEVICE)
    {
        auto buffer = ft::torchTensor2Buffer(tensor, scales, zeros);
        if (alloc_type == ft::AllocationType::DEVICE) {
            auto device_buffer = device_->allocateBufferLike(*buffer);
            device_->copy({*device_buffer, *buffer});
            device_->syncAndCheck();
            printf("created device buffer from tensor at %p with data=%p\n", device_buffer.get(), device_buffer->data());
            return device_buffer;
        } else {
            return buffer;
        }
    }

    torch::Tensor bufferToTensor(const ft::Buffer& buffer, ft::DeviceBase* device = nullptr) {
        if (!device) {
            device = device_;
        }
        auto host_buffer = device->allocateBuffer(
            {buffer.type(), buffer.shape(), ft::AllocationType::HOST}
        );
        device->copy({*host_buffer, buffer});
        device->syncAndCheck();

        return torch::from_blob(
            host_buffer->data(), bufferShapeToTorchShape(buffer),
            c10::TensorOptions().device(torch::Device(torch::kCPU))
                                .dtype(dataTypeToTorchType(buffer.type()))
        ).clone();
    }



    ft::BufferPtr allocateKVBlocks(const rtp_llm::CacheConfig& cache_config,
                               const std::vector<int32_t>& input_lengths,
                               torch::Tensor& kvCache)
    {
        if (!cache_manager_) {
            cache_manager_ = std::make_shared<rtp_llm::CacheManager>(cache_config, device_);
        }

        auto max_seq_len = *std::max_element(input_lengths.begin(), input_lengths.end());
        max_seq_len = (max_seq_len == 0) ? 1 : max_seq_len;
        const auto tokensPerBlock = cache_config.seq_size_per_block;
        const auto batch_layer_kv_block_num = ((max_seq_len + tokensPerBlock - 1) / tokensPerBlock + 1);
        const auto batch_size = input_lengths.size();

        auto kv_cache_offset = device_->allocateBuffer({
            ft::DataType::TYPE_INT32, {batch_size, batch_layer_kv_block_num}, ft::AllocationType::HOST
        });

        rtp_llm::BatchKVCacheBlockAddr batch_kv_cache;

        for (auto i = 0; i < batch_size; i++) {
            auto [success, kv_cache] = cache_manager_->malloc(batch_layer_kv_block_num);
            EXPECT_TRUE(success);
            batch_kv_cache.pushBack(kv_cache);
        }
        for (auto i = 0; i < batch_size; i++) {
            std::memcpy((*kv_cache_offset)[i].data(),
                        batch_kv_cache.batch_offset[i].data(),
                        batch_kv_cache.batch_offset[i].size() * sizeof(int));
            // [batch(i), layer_num(j), ...]
            if (kvCache.dim() == 5) {
                // [layernum, batch, 2, max_pad_seq, dim]
                auto max_pad_seq = kvCache.sizes()[3];
                auto k_indexs = batch_kv_cache.batch_offset[i];
                for (auto k = 0; k < (max_pad_seq / cache_config.seq_size_per_block); k++) {
                    auto block_start = k * cache_config.seq_size_per_block;
                    auto block_end   = block_start + cache_config.seq_size_per_block;
                    auto kblock = kvCache.index(
                        {torch::indexing::Slice(),
                         i,
                         0,
                         torch::indexing::Slice(block_start, block_end),
                         torch::indexing::Slice()}).contiguous();
                    auto vblock = kvCache.index(
                        {torch::indexing::Slice(),
                         i,
                         1,
                         torch::indexing::Slice(block_start, block_end),
                         torch::indexing::Slice()}).contiguous();
                    auto kblock_buffer = ft::torchTensor2Buffer(kblock);
                    auto vblock_buffer = ft::torchTensor2Buffer(vblock);
                    cache_manager_->setKVBlockValue(k_indexs[k],
                                                    *kblock_buffer,
                                                    *vblock_buffer);
                }
            }

        }
        auto kv_cache_offset_gpu_buf = device_->allocateBuffer({kv_cache_offset->type(), kv_cache_offset->shape()});
        device_->copy({*kv_cache_offset_gpu_buf, *kv_cache_offset});
        return kv_cache_offset_gpu_buf;
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
        a_cmp = a_cmp.squeeze();
        b_cmp = b_cmp.squeeze();

        const auto close = torch::allclose(a_cmp, b_cmp, rtol, atol);
        if (!close) {
            std::cout << "assert tensor close failed!" << std::endl;
            std::cout << "rtol: " << rtol << std::endl;
            std::cout << "atol: " << atol << std::endl;
            std::cout << "a: " << a << std::endl;
            std::cout << "b: " << b << std::endl;
            std::cout << "abs diff: " << torch::abs(a_cmp - b_cmp) << std::endl;
            std::cout << "rel diff: " << torch::abs(a_cmp - b_cmp) / torch::abs(a_cmp) << std::endl;
            ASSERT_TRUE(false);
        }
    }

    size_t getFreePort() {
        int sockfd = socket(AF_INET, SOCK_STREAM, 0);
        EXPECT_TRUE(sockfd >= 0);

        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
        addr.sin_port = 0;

        if (bind(sockfd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            EXPECT_TRUE(false);
        }

        socklen_t addr_len = sizeof(addr);
        if (getsockname(sockfd, (struct sockaddr*)&addr, &addr_len) < 0) {
            EXPECT_TRUE(false);
        }
        close(sockfd);
        return ntohs(addr.sin_port);
    }

    const std::string &getTestDataPath() const {
        return test_data_path_;
    }

    torch::Tensor randTensor(at::IntArrayRef shape,
                             torch::Dtype dtype,
                             int64_t seed = 0) {
        torch::TensorOptions float_options =
            torch::TensorOptions(torch::kFloat).device(torch::Device(torch::kCPU));
        torch::TensorOptions half_tensor_options =
            torch::TensorOptions(torch::kFloat16).device(torch::Device(torch::kCPU));
        auto generator = at::detail::createCPUGenerator();
        if (seed != 0) {
            generator = at::detail::createCPUGenerator(seed);
        }
        auto output = torch::rand(shape, generator, float_options);
        if (c10::isQIntType(dtype)) {
            int axis = output.dim()-1;
            auto scales = torch::rand(output.sizes()[axis], half_tensor_options);
            auto zeros  = torch::zeros(output.sizes()[axis]);
            output = at::quantize_per_channel(output, scales, zeros, axis, dtype);
        } else {
            output = output.to(dtype);
        }
        return output;
    }

protected:
    ft::DeviceBase* device_ = nullptr;
    std::string test_data_path_;
    double rtol_ = 1e-03;
    double atol_ = 1e-03;
    rtp_llm::CacheManagerPtr cache_manager_;
    size_t device_reserve_memory_size_ = 1L * 1024 * 1024; // 1MB;
    size_t host_reserve_memory_size_ = 1L * 1024 * 1024 * 1024; // 1GB;
};

#define RTP_LLM_RUN_DEVICE_TEST(test_class, case_name, ...) \
    TEST_F(test_class, case_name) {                         \
        case_name(__VA_ARGS__);                             \
    }

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
