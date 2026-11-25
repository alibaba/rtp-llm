#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/DeviceBlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStore.h"

#if USING_CUDA
#include <cuda_runtime.h>
#elif USING_ROCM
#include <hip/hip_runtime.h>
#endif

namespace rtp_llm {
namespace test {
namespace {

#if USING_CUDA || USING_ROCM
class RecordingMemoryUtil: public MemoryUtil {
public:
    bool regLoadDestinationMr(void* buf, uint64_t size, bool gpu, uint64_t aligned_size) override {
        if (++load_registrations == fail_registration) {
            throw std::runtime_error("test registration failure");
        }
        return MemoryUtil::regLoadDestinationMr(buf, size, gpu, aligned_size);
    }

    bool regUserMr(void* buf, uint64_t size, bool gpu, uint64_t aligned_size) override {
        reg_calls.push_back({buf, size, gpu, aligned_size});
        return true;
    }

    bool deregUserMr(void* buf, bool gpu) override {
        dereg_calls.push_back({buf, gpu});
        return true;
    }

    bool isMemoryMr(void*, uint64_t, bool, bool) override {
        return false;
    }

    bool findMemoryMr(void*, void*, uint64_t, bool, bool) override {
        return false;
    }

    bool isRdmaMode() override {
        return true;
    }

    struct RegCall {
        void*    buf;
        uint64_t size;
        bool     gpu;
        uint64_t aligned_size;
    };

    struct DeregCall {
        void* buf;
        bool  gpu;
    };

    std::vector<RegCall>   reg_calls;
    std::vector<DeregCall> dereg_calls;
    size_t                 load_registrations = 0;
    size_t                 fail_registration  = 0;
};

class RecordingCacheStore: public CacheStore {
public:
    explicit RecordingCacheStore(std::shared_ptr<MemoryUtil> memory_util): memory_util_(std::move(memory_util)) {}

    void store(const std::shared_ptr<RequestBlockBuffer>&, CacheStoreStoreDoneCallback callback) override {
        if (callback) {
            callback(false, CacheStoreErrorCode::InvalidParams);
        }
    }

    void load(const std::shared_ptr<RequestBlockBuffer>&,
              CacheStoreLoadDoneCallback callback,
              const std::string&,
              uint32_t,
              uint32_t,
              uint32_t,
              int,
              int) override {
        if (callback) {
            callback(false, CacheStoreErrorCode::InvalidParams);
        }
    }

    std::shared_ptr<LoadContext> loadBuffers(const std::vector<std::shared_ptr<RequestBlockBuffer>>&,
                                             const std::string&,
                                             uint32_t,
                                             uint32_t,
                                             int64_t,
                                             LoadContext::CheckCancelFunc,
                                             int,
                                             int) override {
        return nullptr;
    }

    std::shared_ptr<StoreContext> storeBuffers(const std::vector<std::shared_ptr<RequestBlockBuffer>>&,
                                               int64_t) override {
        return nullptr;
    }

    std::shared_ptr<RemoteStoreTask>
    submitRemoteStoreTask(const std::shared_ptr<RemoteStoreRequest>&,
                          const std::shared_ptr<CacheStoreRemoteStoreMetricsCollector>&,
                          RemoteStoreTask::CheckCancelFunc) override {
        return nullptr;
    }

    void releaseRemoteStoreTask(const std::shared_ptr<RemoteStoreTask>&) override {}

    bool regUserBuffers(const std::vector<std::shared_ptr<BlockBuffer>>&) override {
        return true;
    }

    std::shared_ptr<BlockBuffer> findUserBuffer(const std::string&) override {
        return nullptr;
    }

    const std::shared_ptr<MemoryUtil>& getMemoryUtil() const override {
        return memory_util_;
    }

    void debugInfo() override {}

private:
    std::shared_ptr<MemoryUtil> memory_util_;
};

int deviceCount() {
#if USING_CUDA
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess ? count : 0;
#elif USING_ROCM
    int count = 0;
    return hipGetDeviceCount(&count) == hipSuccess ? count : 0;
#else
    return 0;
#endif
}

bool getDevice(int* device) {
#if USING_CUDA
    return cudaGetDevice(device) == cudaSuccess;
#elif USING_ROCM
    return hipGetDevice(device) == hipSuccess;
#else
    (void)device;
    return false;
#endif
}

bool setDevice(int device) {
#if USING_CUDA
    return cudaSetDevice(device) == cudaSuccess;
#elif USING_ROCM
    return hipSetDevice(device) == hipSuccess;
#else
    (void)device;
    return false;
#endif
}

bool synchronizeDevice() {
#if USING_CUDA
    return cudaDeviceSynchronize() == cudaSuccess;
#elif USING_ROCM
    return hipDeviceSynchronize() == hipSuccess;
#else
    return false;
#endif
}

DeviceBlockPoolConfig makeSmallBlockPoolConfig() {
    auto cache_config = makeSimpleMhaCacheConfig(1, 4, 1, DataType::TYPE_FP16, 1, 64);
    auto config =
        DeviceBlockPoolConfigHelper::createConfigForGroup(cache_config, cache_config.topology().groups().front());
    config.pool_name                 = "raw_device_malloc_test";
    config.use_device_malloc_backing = true;
    return config;
}
#endif

}  // namespace

TEST(BlockPoolDeviceMallocTest, AllocatesUsableGpuBacking) {
#if USING_CUDA || USING_ROCM
    if (deviceCount() == 0) {
        GTEST_SKIP() << "No GPU is visible";
    }

    auto config = makeSmallBlockPoolConfig();
    auto pool   = std::make_shared<DeviceBlockPool>(std::make_shared<const DeviceBlockPoolConfig>(config));
    ASSERT_TRUE(pool->init());
    ASSERT_NE(pool->getBaseAddress(), nullptr);
    EXPECT_EQ(pool->where(), MemoryType::MEMORY_GPU);

    auto layer_tensors = pool->allLayerCacheBase();
    ASSERT_EQ(layer_tensors.size(), 1u);
    EXPECT_TRUE(layer_tensors[0].is_cuda());
    EXPECT_EQ(static_cast<size_t>(layer_tensors[0].nbytes()), config.total_size_bytes);
    EXPECT_NO_THROW(layer_tensors[0].fill_(7));
    EXPECT_TRUE(synchronizeDevice());

    auto block = pool->convertIndexToBuffer(/*layer_id=*/0, /*block_id=*/1);
    ASSERT_FALSE(block.empty());
    size_t total_bytes = 0;
    for (const auto& part : block) {
        EXPECT_TRUE(part.is_cuda);
        EXPECT_NE(part.addr, nullptr);
        total_bytes += part.size_bytes;
    }
    EXPECT_EQ(total_bytes, config.memory_layouts[0].kv_block_stride_bytes);
#else
    GTEST_SKIP() << "Raw device allocation is only supported in CUDA and ROCm builds";
#endif
}

TEST(BlockPoolDeviceMallocTest, PassesGpuBackingToMemoryRegistrationBoundary) {
#if USING_CUDA || USING_ROCM
    if (deviceCount() == 0) {
        GTEST_SKIP() << "No GPU is visible";
    }

    auto config = makeSmallBlockPoolConfig();
    auto pool   = std::make_shared<DeviceBlockPool>(std::make_shared<const DeviceBlockPoolConfig>(config));
    ASSERT_TRUE(pool->init());

    const auto& layout      = config.memory_layouts[0];
    auto        memory_util = std::make_shared<RecordingMemoryUtil>();
    auto        cache_store = std::make_shared<RecordingCacheStore>(memory_util);
    pool->regUserMr(/*model_id=*/0, cache_store);

    ASSERT_EQ(memory_util->load_registrations, 1u);
    ASSERT_EQ(memory_util->reg_calls.size(), 1u);
    EXPECT_EQ(memory_util->reg_calls[0].buf, pool->getBaseAddress());
    EXPECT_EQ(memory_util->reg_calls[0].size, layout.kv_block_pool_size_bytes);
    EXPECT_TRUE(memory_util->reg_calls[0].gpu);
    EXPECT_EQ(memory_util->reg_calls[0].aligned_size, layout.kv_block_stride_bytes);

    pool->deregUserMr();
    ASSERT_EQ(memory_util->dereg_calls.size(), 1u);
    EXPECT_EQ(memory_util->dereg_calls[0].buf, pool->getBaseAddress());
    EXPECT_TRUE(memory_util->dereg_calls[0].gpu);
    pool.reset();
    EXPECT_EQ(memory_util->dereg_calls.size(), 1u);
#else
    GTEST_SKIP() << "Raw device allocation is only supported in CUDA and ROCm builds";
#endif
}

TEST(BlockPoolDeviceMallocTest, RollsBackPartialRegistrationAndRetiresOnDestruction) {
#if USING_CUDA || USING_ROCM
    if (deviceCount() == 0) {
        GTEST_SKIP() << "No GPU is visible";
    }
    auto config = makeSmallBlockPoolConfig();
    auto second = config.memory_layouts.front();
    second.kv_cache_offset_bytes += config.total_size_bytes;
    config.memory_layouts.push_back(second);
    config.total_size_bytes *= 2;
    auto pool = std::make_shared<DeviceBlockPool>(std::make_shared<const DeviceBlockPoolConfig>(config));
    ASSERT_TRUE(pool->init());
    auto memory               = std::make_shared<RecordingMemoryUtil>();
    auto store                = std::make_shared<RecordingCacheStore>(memory);
    memory->fail_registration = 2;
    EXPECT_THROW(pool->regUserMr(0, store), std::runtime_error);
    ASSERT_EQ(memory->reg_calls.size(), 1u);
    ASSERT_EQ(memory->dereg_calls.size(), 1u);
    EXPECT_EQ(memory->dereg_calls.front().buf, memory->reg_calls.front().buf);

    pool->regUserMr(0, store);
    ASSERT_EQ(memory->reg_calls.size(), 3u);
    auto other = std::make_shared<RecordingMemoryUtil>();
    pool->regUserMr(0, std::make_shared<RecordingCacheStore>(other));
    pool.reset();
    ASSERT_EQ(memory->dereg_calls.size(), 3u);
    EXPECT_EQ(memory->dereg_calls[1].buf, memory->reg_calls[2].buf);
    EXPECT_EQ(memory->dereg_calls[2].buf, memory->reg_calls[1].buf);
    EXPECT_TRUE(other->reg_calls.empty());
    EXPECT_TRUE(other->dereg_calls.empty());
#else
    GTEST_SKIP() << "Raw device allocation is only supported in CUDA and ROCm builds";
#endif
}

TEST(BlockPoolDeviceMallocTest, DestructionRestoresCurrentDevice) {
#if USING_CUDA || USING_ROCM
    if (deviceCount() < 2) {
        GTEST_SKIP() << "At least two visible GPUs are required";
    }

    int original_device = -1;
    ASSERT_TRUE(getDevice(&original_device));
    ASSERT_TRUE(setDevice(0));

    auto pool =
        std::make_shared<DeviceBlockPool>(std::make_shared<const DeviceBlockPoolConfig>(makeSmallBlockPoolConfig()));
    ASSERT_TRUE(pool->init());

    ASSERT_TRUE(setDevice(1));
    pool.reset();

    int current_device = -1;
    EXPECT_TRUE(getDevice(&current_device));
    EXPECT_EQ(current_device, 1);
    EXPECT_TRUE(setDevice(original_device));
#else
    GTEST_SKIP() << "Raw device allocation is only supported in CUDA and ROCm builds";
#endif
}

}  // namespace test
}  // namespace rtp_llm
