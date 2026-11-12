#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <functional>
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
    bool regUserMr(void* buf, uint64_t size, bool gpu, uint64_t aligned_size) override {
        reg_calls.push_back({buf, size, gpu, aligned_size});
        return true;
    }

    bool deregUserMr(void* buf, bool gpu) override {
        if (on_deregister) {
            on_deregister();
        }
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
    std::function<void()>  on_deregister;
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

DeviceBlockPoolConfig makeSmallBlockPoolConfig(DataType dtype = DataType::TYPE_FP16) {
    auto cache_config = makeSimpleMhaCacheConfig(1, 4, 1, dtype, 1, 64);
    auto config       = DeviceBlockPoolConfigHelper::createConfigForGroup(cache_config.main_config,
                                                                    cache_config.topology().groups().front());
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

    ASSERT_EQ(memory_util->reg_calls.size(), 1u);
    EXPECT_EQ(memory_util->reg_calls[0].buf, pool->getBaseAddress());
    EXPECT_EQ(memory_util->reg_calls[0].size, layout.kv_block_pool_size_bytes);
    EXPECT_TRUE(memory_util->reg_calls[0].gpu);
    EXPECT_EQ(memory_util->reg_calls[0].aligned_size, layout.kv_block_stride_bytes);

    pool->deregUserMr();
    ASSERT_EQ(memory_util->dereg_calls.size(), 1u);
    EXPECT_EQ(memory_util->dereg_calls[0].buf, pool->getBaseAddress());
    EXPECT_TRUE(memory_util->dereg_calls[0].gpu);
#else
    GTEST_SKIP() << "Raw device allocation is only supported in CUDA and ROCm builds";
#endif
}

TEST(BlockPoolDeviceMallocTest, DeregistersBeforeReleasingBackingAndDoesNotDeregisterTwice) {
#if USING_CUDA || USING_ROCM
    if (deviceCount() == 0) {
        GTEST_SKIP() << "No GPU is visible";
    }
    for (auto dtype : {DataType::TYPE_FP16, DataType::TYPE_INT8}) {
        for (bool raw_backing : {false, true}) {
            for (bool explicit_deregister : {false, true}) {
                auto config                      = makeSmallBlockPoolConfig(dtype);
                config.use_device_malloc_backing = raw_backing;
                auto pool = std::make_shared<DeviceBlockPool>(std::make_shared<const DeviceBlockPoolConfig>(config));
                ASSERT_TRUE(pool->init());
                auto memory = std::make_shared<RecordingMemoryUtil>();
                auto store  = std::make_shared<RecordingCacheStore>(memory);
                pool->regUserMr(0, store);
                auto storage = pool->cache_aligned_buffer_.storage().getWeakStorageImpl();
                memory->on_deregister = [storage]() { EXPECT_FALSE(storage.expired()); };
                const auto registered = memory->reg_calls;
                ASSERT_EQ(registered.size(), dtype == DataType::TYPE_INT8 ? 2u : 1u);
                if (explicit_deregister) {
                    pool->deregUserMr();
                    pool->deregUserMr();
                }
                pool.reset();
                ASSERT_EQ(memory->dereg_calls.size(), registered.size());
                for (size_t index = 0; index < registered.size(); ++index) {
                    EXPECT_EQ(memory->dereg_calls[index].buf, registered[index].buf);
                    EXPECT_EQ(memory->dereg_calls[index].gpu, registered[index].gpu);
                }
                EXPECT_TRUE(storage.expired());
                memory->on_deregister = {};
            }
        }
    }
#else
    GTEST_SKIP() << "Device allocation is only supported in CUDA and ROCm builds";
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
