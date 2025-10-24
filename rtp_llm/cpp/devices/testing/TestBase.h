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

#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/config/GptInitParameter.h"
#include "rtp_llm/cpp/cache/CacheManager.h"
#include "rtp_llm/cpp/utils/KVCacheUtils.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "autil/EnvUtil.h"

using namespace rtp_llm;

static const std::string DEFAULT_DEVICE = "CPU";

#define ASSERT_VECTOR_EQUAL(x, y)                                                                                      \
    ASSERT_EQ(x.size(), y.size()) << "Vectors x and y are of unequal length";                                          \
    for (int i = 0; i < x.size(); ++i) {                                                                               \
        ASSERT_EQ(x[i], y[i]) << "Vectors x and y differ at index " << i;                                              \
    }

#define ASSERT_VECTOR_NEAR(x, y, abs_error)                                                                            \
    ASSERT_EQ(x.size(), y.size()) << "Vectors x and y are of unequal length";                                          \
    for (int i = 0; i < x.size(); ++i) {                                                                               \
        ASSERT_NEAR(x[i], y[i], abs_error) << "Vectors x and y differ at index " << i;                                 \
    }

class EngineBaseTest: public ::testing::Test {
public:
    void SetUp() override {
        rtp_llm::initLogger();
        initTestDataDir();
        torch::manual_seed(114514);
        setenv("SAMPLE_TEST", "1", 1);
    }

    virtual void initTestDevices() {}

    void initTestDataDir() {
        const auto test_src_dir    = getenv("TEST_SRCDIR");
        const auto test_work_space = getenv("TEST_WORKSPACE");
        const auto test_binary     = getenv("TEST_BINARY");
        if (!(test_src_dir && test_work_space && test_binary)) {
            std::cerr << "Unable to retrieve TEST_SRCDIR / TEST_WORKSPACE / TEST_BINARY env!" << std::endl;
            abort();
        }

        std::string test_binary_str = std::string(test_binary);
        RTP_LLM_CHECK(*test_binary_str.rbegin() != '/');
        size_t filePos  = test_binary_str.rfind('/');
        test_data_path_ = std::string(test_src_dir) + "/" + std::string(test_work_space) + "/"
                          + test_binary_str.substr(0, filePos) + "/";

        std::cout << "test_src_dir [" << test_src_dir << "]" << std::endl;
        std::cout << "test_work_space [" << test_work_space << "]" << std::endl;
        std::cout << "test_binary [" << test_binary << "]" << std::endl;
        std::cout << "test using data path [" << test_data_path_ << "]" << std::endl;
    }

    void TearDown() override {}

protected:
    std::string test_data_path_;
};

class DeviceTestBase: public EngineBaseTest {
public:
    void SetUp() override {
        EngineBaseTest::SetUp();
        initTestDevices();
        torch::manual_seed(114514);
    }

    virtual void initTestDevices() {
        rtp_llm::GptInitParameter gpt_init_params;
        gpt_init_params.device_resource_config.device_reserve_memory_bytes = device_reserve_memory_size_;
        gpt_init_params.device_resource_config.host_reserve_memory_bytes   = host_reserve_memory_size_;
        rtp_llm::DeviceFactory::initDevices(gpt_init_params);
        device_ = rtp_llm::DeviceFactory::getDefaultDevice();
    }

    void TearDown() override {}

protected:
    template<typename T>
    void printBuffer(const rtp_llm::Buffer& buffer, const std::string& hint = "") {
        auto values = getBufferValues<T>(buffer);
        for (size_t i = 0; i < values.size(); i++) {
            std::cout << values[i] << " ";
        }
        std::cout << " " << hint << std::endl;
    }

    rtp_llm::BufferPtr createBuffer(const std::vector<size_t>& shape,
                                    rtp_llm::DataType          type,
                                    rtp_llm::AllocationType    alloc_type = rtp_llm::AllocationType::DEVICE) {
        if (alloc_type == rtp_llm::AllocationType::DEVICE) {
            return device_->allocateBuffer({type, shape, rtp_llm::AllocationType::DEVICE}, {});
        } else {
            return device_->allocateBuffer({type, shape, rtp_llm::AllocationType::HOST}, {});
        }
    }

    template<typename T>
    rtp_llm::BufferPtr createBuffer(const std::vector<size_t>& shape,
                                    const std::vector<T>&      data,
                                    rtp_llm::AllocationType    alloc_type = rtp_llm::AllocationType::DEVICE) {
        const auto num_elements = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
        RTP_LLM_CHECK(num_elements == data.size());
        if (alloc_type == rtp_llm::AllocationType::DEVICE) {
            return createDeviceBuffer<T>(shape, data.data());
        } else {
            return createHostBuffer<T>(shape, data.data());
        }
    }

    template<typename T>
    rtp_llm::BufferPtr createHostBuffer(const std::vector<size_t>& shape, const T* data) {
        return createHostBuffer<T>(shape, static_cast<const void*>(data));
    }

    template<typename T>
    rtp_llm::BufferPtr createHostBuffer(const std::vector<size_t>& shape, const void* data) {
        auto buffer = device_->allocateBuffer({rtp_llm::getTensorType<T>(), shape, rtp_llm::AllocationType::HOST}, {});
        if (data && (buffer->size() > 0)) {
            memcpy(buffer->data(), data, sizeof(T) * buffer->size());
        }
        device_->syncAndCheck();
        return buffer;
    }

    template<typename T>
    rtp_llm::BufferPtr createDeviceBuffer(const std::vector<size_t>& shape, const void* data) {
        auto host_buffer = createHostBuffer<T>(shape, data);
        auto buffer =
            device_->allocateBuffer({rtp_llm::getTensorType<T>(), shape, rtp_llm::AllocationType::DEVICE}, {});
        if (data && (buffer->size() > 0)) {
            device_->copy({*buffer, *host_buffer});
        }
        device_->syncAndCheck();
        return buffer;
    }

    template<typename T>
    rtp_llm::BufferPtr createDeviceBuffer(torch::Tensor tensor) {
        const auto targetType = c10::CppTypeToScalarType<T>::value;
        if (tensor.scalar_type() != targetType) {
            tensor = tensor.to(targetType);
        }
        return tensorToBuffer(tensor);
    }

    template<typename T>
    void assertBufferValueEqual(const rtp_llm::Buffer& buffer, const std::vector<T>& expected) {
        ASSERT_EQ(buffer.size(), expected.size());
        auto comp_buffer = device_->allocateBuffer({buffer.type(), buffer.shape(), rtp_llm::AllocationType::HOST});
        device_->copy({*comp_buffer, buffer});
        device_->syncAndCheck();
        for (size_t i = 0; i < buffer.size(); i++) {
            printf(
                "i=%ld, buffer[i] = %f, expected[i] = %f\n", i, float((comp_buffer->data<T>())[i]), float(expected[i]));
            ASSERT_EQ((comp_buffer->data<T>())[i], expected[i]);
        }
    }

    template<typename T>
    std::vector<T> getBufferValues(const rtp_llm::Buffer& buffer) {
        std::vector<T> values(buffer.size());
        device_->syncAndCheck();
        if (buffer.where() == rtp_llm::MemoryType::MEMORY_GPU) {
            auto host_buffer = createHostBuffer<T>(buffer.shape(), nullptr);
            device_->copy({*host_buffer, buffer});
            device_->syncAndCheck();
            memcpy(values.data(), host_buffer->data(), sizeof(T) * buffer.size());
        } else {
            memcpy(values.data(), buffer.data(), sizeof(T) * buffer.size());
        }
        return values;
    }

    rtp_llm::BufferPtr tensorToBuffer(const torch::Tensor&    tensor,
                                      rtp_llm::AllocationType alloc_type = rtp_llm::AllocationType::DEVICE) {
        if (tensor.is_quantized()) {
            return tensorToBuffer(tensor,
                                  tensor.q_per_channel_scales().to(torch::kHalf),
                                  tensor.q_per_channel_zero_points().to(torch::kHalf));
        }
        RTP_LLM_CHECK(tensor.is_cpu());
        auto buffer = rtp_llm::torchTensor2Buffer(tensor);
        if (alloc_type == rtp_llm::AllocationType::DEVICE) {
            auto device_buffer =
                device_->allocateBuffer({buffer->type(), buffer->shape(), rtp_llm::AllocationType::DEVICE});
            device_->copy({*device_buffer, *buffer});
            device_->syncAndCheck();
            printf(
                "created device buffer from tensor at %p with data=%p\n", device_buffer.get(), device_buffer->data());
            return device_buffer;
        } else {
            return buffer;
        }
    }

    rtp_llm::BufferPtr tensorToBuffer(const torch::Tensor&    tensor,
                                      const torch::Tensor&    scales,
                                      const torch::Tensor&    zeros,
                                      rtp_llm::AllocationType alloc_type = rtp_llm::AllocationType::DEVICE) {
        auto buffer = rtp_llm::torchTensor2Buffer(tensor, scales, zeros);
        if (alloc_type == rtp_llm::AllocationType::DEVICE) {
            auto device_buffer = device_->allocateBufferLike(*buffer);
            device_->copy({*device_buffer, *buffer});
            device_->syncAndCheck();
            printf(
                "created device buffer from tensor at %p with data=%p\n", device_buffer.get(), device_buffer->data());
            return device_buffer;
        } else {
            return buffer;
        }
    }

    torch::Tensor bufferToTensor(const rtp_llm::Buffer& buffer, rtp_llm::DeviceBase* device = nullptr) {
        if (!device) {
            device = device_;
        }
        auto host_buffer = device->allocateBuffer({buffer.type(), buffer.shape(), rtp_llm::AllocationType::HOST});
        device->copy({*host_buffer, buffer});
        device->syncAndCheck();

        return torch::from_blob(
                   host_buffer->data(),
                   bufferShapeToTorchShape(buffer),
                   c10::TensorOptions().device(torch::Device(torch::kCPU)).dtype(dataTypeToTorchType(buffer.type())))
            .clone();
    }

    rtp_llm::BufferPtr allocateKVBlocks(const rtp_llm::CacheConfig& cache_config,
                                        const std::vector<int32_t>& input_lengths,
                                        torch::Tensor&              kvCache,
                                        bool                        need_padding = true) {
        if (!cache_manager_) {
            cache_manager_ = std::make_shared<rtp_llm::CacheManager>(cache_config, device_);
        }

        auto max_seq_len                    = *std::max_element(input_lengths.begin(), input_lengths.end());
        max_seq_len                         = (max_seq_len == 0) ? 1 : max_seq_len;
        const auto tokensPerBlock           = cache_config.seq_size_per_block;
        const auto batch_layer_kv_block_num = need_padding ? ((max_seq_len + tokensPerBlock - 1) / tokensPerBlock) + 1 :
                                                             ((max_seq_len + tokensPerBlock - 1) / tokensPerBlock);
        const auto batch_size               = input_lengths.size();

        auto kv_cache_block_id = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_INT32, {batch_size, batch_layer_kv_block_num}, rtp_llm::AllocationType::HOST});

        rtp_llm::BatchKVCacheResource batch_kv_cache;

        for (auto i = 0; i < batch_size; i++) {
            auto [success, kv_cache] = cache_manager_->malloc({0, batch_layer_kv_block_num, true});
            EXPECT_TRUE(success);
            batch_kv_cache.pushBack(kv_cache);
        }
        for (auto i = 0; i < batch_size; i++) {
            std::memcpy((*kv_cache_block_id)[i].data(),
                        batch_kv_cache.batch_block_id[i].data(),
                        batch_kv_cache.batch_block_id[i].size() * sizeof(int));
            // [batch(i), layer_num(j), ...]
            if (kvCache.dim() == 5) {
                // [layernum, batch, 2, max_pad_seq, dim]
                auto max_pad_seq = kvCache.sizes()[3];
                auto k_indexs    = batch_kv_cache.batch_block_id[i];
                for (auto k = 0; k < (max_pad_seq / cache_config.seq_size_per_block); k++) {
                    auto          block_start = k * cache_config.seq_size_per_block;
                    auto          block_end   = block_start + cache_config.seq_size_per_block;
                    torch::Tensor kblock, vblock;
                    if (need_padding) {
                        kblock = kvCache
                                     .index({torch::indexing::Slice(),
                                             i,
                                             0,
                                             torch::indexing::Slice(block_start, block_end),
                                             torch::indexing::Slice()})
                                     .contiguous();
                        vblock = kvCache
                                     .index({torch::indexing::Slice(),
                                             i,
                                             1,
                                             torch::indexing::Slice(block_start, block_end),
                                             torch::indexing::Slice()})
                                     .contiguous();
                    } else {
                        kblock = kvCache
                                     .index({torch::indexing::Slice(),
                                             i,
                                             torch::indexing::Slice(),
                                             torch::indexing::Slice(block_start, block_end),
                                             torch::indexing::Slice()})
                                     .reshape({2,
                                               cache_config.seq_size_per_block,
                                               cache_config.local_head_num_kv,
                                               cache_config.size_per_head})
                                     .transpose(2, 1)
                                     .contiguous();
                        // vblock is not used in setKVBlockValue in this case
                        vblock = kvCache
                                     .index({torch::indexing::Slice(),
                                             i,
                                             1,
                                             torch::indexing::Slice(block_start, block_end),
                                             torch::indexing::Slice()})
                                     .reshape({cache_config.seq_size_per_block,
                                               cache_config.local_head_num_kv,
                                               cache_config.size_per_head})
                                     .transpose(1, 0)
                                     .contiguous();
                    }
                    auto kblock_buffer = rtp_llm::torchTensor2Buffer(kblock);
                    auto vblock_buffer = rtp_llm::torchTensor2Buffer(vblock);
                    // std::cout << "index: " << k << " start: " << block_start << " end: " << block_end << std::endl;
                    // std::cout << "block index: " << k_indexs[k] << std::endl;
                    cache_manager_->setKVBlockValue(k_indexs[k], *kblock_buffer, *vblock_buffer);
                }
            }
        }
        return kv_cache_block_id;
    }

    void assertTensorClose(const torch::Tensor& a, const torch::Tensor& b, double rtol = 0, double atol = 0) {
        auto a_cmp = a;
        auto b_cmp = b;
        rtol       = rtol ? rtol : rtol_;
        atol       = atol ? atol : rtol_;
        ASSERT_TRUE(a.is_floating_point() == b.is_floating_point());

        if (a_cmp.dtype() != b_cmp.dtype()) {
            auto cmp_type = (a_cmp.dtype().itemsize() > b_cmp.dtype().itemsize()) ? a_cmp.dtype() : b_cmp.dtype();
            a_cmp         = a_cmp.to(cmp_type);
            b_cmp         = b_cmp.to(cmp_type);
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
        addr.sin_family      = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
        addr.sin_port        = 0;

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

    const std::string& getTestDataPath() const {
        return test_data_path_;
    }

    torch::Tensor randTensor(at::IntArrayRef shape, torch::Dtype dtype, int64_t seed = 0) {
        torch::TensorOptions float_options = torch::TensorOptions(torch::kFloat).device(torch::Device(torch::kCPU));
        torch::TensorOptions half_tensor_options =
            torch::TensorOptions(torch::kFloat16).device(torch::Device(torch::kCPU));
        auto generator = at::detail::createCPUGenerator();
        if (seed != 0) {
            generator = at::detail::createCPUGenerator(seed);
        }
        auto output = torch::rand(shape, generator, float_options);
        if (c10::isQIntType(dtype)) {
            int  axis   = output.dim() - 1;
            auto scales = torch::rand(output.sizes()[axis], half_tensor_options);
            auto zeros  = torch::zeros(output.sizes()[axis]);
            output      = at::quantize_per_channel(output, scales, zeros, axis, dtype);
        } else {
            output = output.to(dtype);
        }
        return output;
    }

protected:
    rtp_llm::DeviceBase*     device_ = nullptr;
    double                   rtol_   = 1e-03;
    double                   atol_   = 1e-03;
    rtp_llm::CacheManagerPtr cache_manager_;
    size_t                   device_reserve_memory_size_ = 1024L * 1024 * 1024;      // 1MB;
    size_t                   host_reserve_memory_size_   = 1L * 1024 * 1024 * 1024;  // 1GB;
};

#define RTP_LLM_RUN_DEVICE_TEST(test_class, case_name, ...)                                                            \
    TEST_F(test_class, case_name) {                                                                                    \
        case_name(__VA_ARGS__);                                                                                        \
    }
