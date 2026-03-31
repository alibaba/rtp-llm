#pragma once

#include <arpa/inet.h>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <numeric>
#include <algorithm>
#include <stdlib.h>
#include <sys/socket.h>
#include <torch/torch.h>

#ifdef ENABLE_FP8
#include <cuda_fp8.h>
#endif

#include "rtp_llm/cpp/core/ExecOps.h"
#include "rtp_llm/cpp/core/torch_utils/TypeConvert.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/utils/KVCacheUtils.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateConfig.h"
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
        rtp_llm::ParallelismConfig parallelism_config;
        rtp_llm::ModelConfig       model_config;
        model_config.max_seq_len = max_seq_len_;
        rtp_llm::EPLBConfig                  eplb_config;
        rtp_llm::FMHAConfig                  fmha_config;
        rtp_llm::DeviceResourceConfig        device_resource_config;
        rtp_llm::MoeConfig                   moe_config;
        rtp_llm::SpeculativeExecutionConfig  sp_config;
        rtp_llm::MiscellaneousConfig         misc_config;
        rtp_llm::ProfilingDebugLoggingConfig profiling_debug_logging_config;
        rtp_llm::HWKernelConfig              hw_kernel_config;
        rtp_llm::ConcurrencyConfig           concurrency_config;
        rtp_llm::FfnDisAggregateConfig       ffn_disaggregate_config;
        rtp_llm::RuntimeConfig               runtime_config;
        rtp_llm::ModelSpecificConfig         model_specific_config;

        rtp_llm::initExecCtx(parallelism_config,
                             model_config,
                             eplb_config,
                             fmha_config,
                             device_resource_config,
                             moe_config,
                             sp_config,
                             misc_config,
                             profiling_debug_logging_config,
                             hw_kernel_config,
                             concurrency_config,
                             ffn_disaggregate_config,
                             runtime_config,
                             model_specific_config,
                             rtp_llm::NcclCommConfig{});
    }

    void TearDown() override {}

protected:
    // Build new-style CacheConfig for MHA tests
    static rtp_llm::CacheConfig makeMhaCacheConfig(uint              layer_num,
                                                   uint              block_num,
                                                   uint              local_head_num_kv,
                                                   uint              size_per_head,
                                                   uint              tokens_per_block,
                                                   rtp_llm::DataType dtype) {
        // Delegate to the unified cache/test helper so derived fields (stride/bytes, layer_all_num, etc.)
        // are always initialized consistently.
        return rtp_llm::test::makeSimpleMhaCacheConfig(/*layer_num=*/static_cast<int>(layer_num),
                                                       /*block_num=*/static_cast<int>(block_num),
                                                       /*tokens_per_block=*/static_cast<size_t>(tokens_per_block),
                                                       dtype,
                                                       /*local_head_num_kv=*/static_cast<uint32_t>(local_head_num_kv),
                                                       /*size_per_head=*/static_cast<uint32_t>(size_per_head));
    }

    // ---- torch-native test helpers (no Buffer dependency) ----

    // Create a CUDA tensor from CPU data vector
    template<typename T>
    torch::Tensor createDeviceTensor(const std::vector<int64_t>& shape, const std::vector<T>& data) {
        auto options = torch::dtype(c10::CppTypeToScalarType<T>::value);
        auto tensor  = torch::from_blob(const_cast<T*>(data.data()), shape, options).clone();
        return tensor.to(torch::kCUDA);
    }

    // Create a CUDA tensor from raw CPU data pointer
    template<typename T>
    torch::Tensor createDeviceTensor(const std::vector<int64_t>& shape, const T* data) {
        auto options = torch::dtype(c10::CppTypeToScalarType<T>::value);
        if (data) {
            auto tensor = torch::from_blob(const_cast<T*>(data), shape, options).clone();
            return tensor.to(torch::kCUDA);
        }
        return torch::empty(shape, options).to(torch::kCUDA);
    }

    // Create an empty CUDA tensor
    torch::Tensor createDeviceTensor(const std::vector<int64_t>& shape, torch::Dtype dtype) {
        return torch::empty(shape, torch::dtype(dtype).device(torch::kCUDA));
    }

    // Create a CPU tensor from data vector
    template<typename T>
    torch::Tensor createHostTensor(const std::vector<int64_t>& shape, const std::vector<T>& data) {
        auto options = torch::dtype(c10::CppTypeToScalarType<T>::value);
        return torch::from_blob(const_cast<T*>(data.data()), shape, options).clone();
    }

    // Get values from a tensor as a vector (handles GPU→CPU copy)
    template<typename T>
    std::vector<T> getTensorValues(const torch::Tensor& tensor) {
        auto           cpu_tensor = tensor.cpu().contiguous();
        std::vector<T> values(cpu_tensor.numel());
        memcpy(values.data(), cpu_tensor.data_ptr<T>(), sizeof(T) * cpu_tensor.numel());
        return values;
    }

    // Assert tensor values equal to expected vector
    template<typename T>
    void assertTensorValueEqual(const torch::Tensor& tensor, const std::vector<T>& expected) {
        ASSERT_EQ(tensor.numel(), (int64_t)expected.size());
        auto values = getTensorValues<T>(tensor);
        for (size_t i = 0; i < expected.size(); i++) {
            printf("i=%ld, tensor[i] = %f, expected[i] = %f\n", i, float(values[i]), float(expected[i]));
            ASSERT_EQ(values[i], expected[i]);
        }
    }

    // Print tensor values
    template<typename T>
    void printTensor(const torch::Tensor& tensor, const std::string& hint = "") {
        auto values = getTensorValues<T>(tensor);
        for (size_t i = 0; i < values.size(); i++) {
            std::cout << values[i] << " ";
        }
        std::cout << " " << hint << std::endl;
    }

    torch::Tensor allocateKVBlocks(const rtp_llm::CacheConfig& cache_config,
                                   const std::vector<int32_t>& input_lengths,
                                   torch::Tensor&              kvCache,
                                   bool                        need_padding = true) {
        if (!cache_manager_) {
            cache_manager_ = std::make_shared<rtp_llm::KVCacheManager>(cache_config);
            bool inited    = cache_manager_->init();
            EXPECT_TRUE(inited);
            if (!inited) {
                return torch::Tensor();
            }
        }

        auto max_seq_len                    = *std::max_element(input_lengths.begin(), input_lengths.end());
        max_seq_len                         = (max_seq_len == 0) ? 1 : max_seq_len;
        const auto tokensPerBlock           = cache_config.seq_size_per_block;
        const auto batch_layer_kv_block_num = need_padding ? ((max_seq_len + tokensPerBlock - 1) / tokensPerBlock) + 1 :
                                                             ((max_seq_len + tokensPerBlock - 1) / tokensPerBlock);
        const auto batch_size               = input_lengths.size();

        auto kv_cache_block_id = torch::empty(
            {1, static_cast<int64_t>(batch_size), static_cast<int64_t>(batch_layer_kv_block_num)}, torch::kInt32);

        auto batch_kv_cache = std::make_shared<rtp_llm::BatchKVCacheResource>();
        batch_kv_cache->resetBatchSize(batch_size);
        batch_kv_cache->initGroups(1, cache_config.layer_all_num, cache_config.layer_to_group_id);

        auto complete_token_ids =
            std::make_shared<rtp_llm::CompleteTokenIds>(static_cast<int>(batch_size),
                                                        static_cast<int>(batch_size),
                                                        static_cast<int>(batch_layer_kv_block_num * tokensPerBlock),
                                                        static_cast<int>(tokensPerBlock));
        // Initialize to avoid nullptr access in CompleteTokenIds::data()
        {
            auto generate_input             = std::make_shared<rtp_llm::GenerateInput>();
            generate_input->generate_config = std::make_shared<rtp_llm::GenerateConfig>();
            generate_input->input_ids       = torch::empty({0}, torch::kInt32);
            complete_token_ids->init(generate_input);
            complete_token_ids->setSeqLength(batch_layer_kv_block_num * tokensPerBlock);
        }

        rtp_llm::MallocInfo malloc_info{batch_kv_cache, complete_token_ids};
        auto                malloc_result = cache_manager_->malloc(malloc_info);
        EXPECT_TRUE(malloc_result.success);

        for (size_t i = 0; i < batch_size; i++) {
            const auto& indices = batch_kv_cache->blocks(static_cast<int>(i));
            auto        row_ptr = kv_cache_block_id.data_ptr<int32_t>() + i * batch_layer_kv_block_num;
            std::memcpy(row_ptr, indices.data(), indices.size() * sizeof(int));
            if (kvCache.dim() == 5) {
                // [layernum, batch, 2, max_pad_seq, dim]
                auto       max_pad_seq    = kvCache.sizes()[3];
                auto       k_indexs       = indices;
                const auto max_k_blocks   = max_pad_seq / cache_config.seq_size_per_block;
                const auto blocks_to_fill = std::min<size_t>(max_k_blocks, k_indexs.size());
                for (size_t k = 0; k < blocks_to_fill; ++k) {
                    auto          block_start = k * cache_config.seq_size_per_block;
                    auto          block_end   = block_start + cache_config.seq_size_per_block;
                    torch::Tensor kblock, vblock;
                    if (need_padding) {
                        kblock = kvCache
                                     .index({torch::indexing::Slice(),
                                             static_cast<int64_t>(i),
                                             0,
                                             torch::indexing::Slice(block_start, block_end),
                                             torch::indexing::Slice()})
                                     .contiguous();
                        vblock = kvCache
                                     .index({torch::indexing::Slice(),
                                             static_cast<int64_t>(i),
                                             1,
                                             torch::indexing::Slice(block_start, block_end),
                                             torch::indexing::Slice()})
                                     .contiguous();
                    } else {
                        kblock = kvCache
                                     .index({torch::indexing::Slice(),
                                             static_cast<int64_t>(i),
                                             torch::indexing::Slice(),
                                             torch::indexing::Slice(block_start, block_end),
                                             torch::indexing::Slice()})
                                     .reshape({2,
                                               static_cast<int64_t>(cache_config.seq_size_per_block),
                                               static_cast<int64_t>(cache_config.cache_specs[0]->local_head_num_kv),
                                               static_cast<int64_t>(
                                                   static_cast<rtp_llm::MHAKVCacheSpec&>(*cache_config.cache_specs[0])
                                                       .size_per_head)})
                                     .transpose(2, 1)
                                     .contiguous();
                        // vblock is not used in setKVBlockValue in this case
                        vblock = kvCache
                                     .index({torch::indexing::Slice(),
                                             static_cast<int64_t>(i),
                                             1,
                                             torch::indexing::Slice(block_start, block_end),
                                             torch::indexing::Slice()})
                                     .reshape({static_cast<int64_t>(cache_config.seq_size_per_block),
                                               static_cast<int64_t>(cache_config.cache_specs[0]->local_head_num_kv),
                                               static_cast<int64_t>(
                                                   static_cast<rtp_llm::MHAKVCacheSpec&>(*cache_config.cache_specs[0])
                                                       .size_per_head)})
                                     .transpose(1, 0)
                                     .contiguous();
                    }
                    // std::cout << "index: " << k << " start: " << block_start << " end: " << block_end << std::endl;
                    // std::cout << "block index: " << k_indexs[k] << std::endl;
                    if (!cache_manager_->setKVBlockValue(k_indexs[k], kblock, vblock)) {
                        std::cout << "setKVBlockValue failed for block index: " << k_indexs[k] << std::endl;
                        return torch::Tensor();
                    }
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
    double                                   rtol_ = 1e-03;
    double                                   atol_ = 1e-03;
    std::shared_ptr<rtp_llm::KVCacheManager> cache_manager_;
    int64_t                                  max_seq_len_ = 8192;
};

#define RTP_LLM_RUN_DEVICE_TEST(test_class, case_name, ...)                                                            \
    TEST_F(test_class, case_name) {                                                                                    \
        case_name(__VA_ARGS__);                                                                                        \
    }
