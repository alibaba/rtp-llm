#include <gtest/gtest.h>

#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/BlockPool.h"
#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"

#if USING_CUDA
#include <cuda_runtime.h>
#endif

namespace {
struct HostAllocationState {
    size_t allocations      = 0;
    size_t frees            = 0;
    size_t requested_bytes  = 0;
    size_t allocation_bytes = 0;
    bool   fail             = false;
} host_allocation;
}  // namespace

#if USING_CUDA
// Linker wrapping is confined to this test binary; production has no injection interface.
extern "C" cudaError_t __wrap_cudaHostAlloc(void** ptr, size_t bytes, unsigned int flags) {
    ++host_allocation.allocations;
    host_allocation.requested_bytes = bytes;
    EXPECT_EQ(flags, cudaHostAllocDefault);
    *ptr = nullptr;
    if (!host_allocation.fail) {
        *ptr = std::malloc(host_allocation.allocation_bytes == 0 ? bytes : host_allocation.allocation_bytes);
    }
    return *ptr == nullptr ? cudaErrorMemoryAllocation : cudaSuccess;
}

extern "C" cudaError_t __wrap_cudaFreeHost(void* ptr) {
    ++host_allocation.frees;
    std::free(ptr);
    return cudaSuccess;
}
#endif

namespace rtp_llm {
namespace {

class BlockPoolHostAllocationTest: public ::testing::Test {
protected:
    void SetUp() override {
        host_allocation = {};
        if (const char* value = std::getenv(kPinEnv)) {
            old_pin_ = value;
        }
        setPin("1");
    }
    void TearDown() override {
        setPin(old_pin_ ? old_pin_->c_str() : nullptr);
    }
    static void setPin(const char* value) {
        if (value) {
            setenv(kPinEnv, value, 1);
        } else {
            unsetenv(kPinEnv);
        }
    }

private:
    static constexpr const char* kPinEnv = "RTP_LLM_PIN_HOST_BLOCK_POOL";
    std::optional<std::string> old_pin_;
};

BlockPoolConfig configFor(size_t bytes) {
    return BlockPoolConfigHelper::createConfig(1, 1, bytes, TYPE_FP16);
}

#if USING_CUDA
class ExactHostAllocationTest: public BlockPoolHostAllocationTest,
                               public ::testing::WithParamInterface<const char*> {};

TEST_P(ExactHostAllocationTest, PassesExactNonPowerOfTwoSizeAndFreesOnce) {
    constexpr size_t kBytes = 20 * 1024 * 1024;
    setPin(GetParam());
    {
        BlockPool pool(configFor(kBytes), AllocationType::HOST);
        ASSERT_TRUE(pool.init());
    }
    EXPECT_EQ(host_allocation.requested_bytes, kBytes);
    EXPECT_EQ(host_allocation.allocations, 1u);
    EXPECT_EQ(host_allocation.frees, 1u);
}
INSTANTIATE_TEST_SUITE_P(PinModes, ExactHostAllocationTest, ::testing::Values(static_cast<const char*>(nullptr), "1"));

TEST_F(BlockPoolHostAllocationTest, ExportedViewRetainsBackingAfterPoolDestruction) {
    torch::Tensor view;
    {
        BlockPool pool(configFor(4096), AllocationType::HOST);
        ASSERT_TRUE(pool.init());
        view = pool.allLayerCacheBase().at(0);
        view.fill_(7);
    }
    EXPECT_EQ(host_allocation.frees, 0u);
    EXPECT_EQ(view.flatten()[0].item<float>(), 7.0f);
    view = torch::Tensor();
    EXPECT_EQ(host_allocation.frees, 1u);
}

TEST_F(BlockPoolHostAllocationTest, CudaFailureFallsBackToPageable) {
    host_allocation.fail = true;
    BlockPool pool(configFor(4096), AllocationType::HOST);
    ASSERT_TRUE(pool.init());
    EXPECT_EQ(host_allocation.allocations, 1u);
    EXPECT_EQ(host_allocation.frees, 0u);
    EXPECT_FALSE(pool.allLayerCacheBase().at(0).is_pinned());
    auto view = pool.allLayerCacheBase().at(0);
    view.fill_(7);
    EXPECT_EQ(view.flatten()[0].item<float>(), 7.0f);
}

TEST_F(BlockPoolHostAllocationTest, TensorConstructionFailureReleasesCudaAllocationOnce) {
    host_allocation.allocation_bytes = 1;
    const size_t invalid_bytes  = static_cast<size_t>(std::numeric_limits<int64_t>::max()) + 1;

    BlockPool pool(configFor(invalid_bytes), AllocationType::HOST);
    EXPECT_THROW(pool.init(), std::exception);
    EXPECT_EQ(host_allocation.allocations, 1u);
    EXPECT_EQ(host_allocation.frees, 1u);
}
#endif

TEST_F(BlockPoolHostAllocationTest, DisabledPinningBypassesCudaAllocation) {
    setPin("false");
    BlockPool pool(configFor(4096), AllocationType::HOST);
    ASSERT_TRUE(pool.init());
    EXPECT_EQ(host_allocation.allocations, 0u);
    EXPECT_FALSE(pool.allLayerCacheBase().at(0).is_pinned());
}

#if !USING_CUDA
TEST_F(BlockPoolHostAllocationTest, NonCudaBuildKeepsTorchPinningPath) {
    BlockPool pool(configFor(4096), AllocationType::HOST);
    ASSERT_TRUE(pool.init());
    EXPECT_EQ(host_allocation.allocations, 0u);
    auto view = pool.allLayerCacheBase().at(0);
    view.fill_(7);
    EXPECT_EQ(view.flatten()[0].item<float>(), 7.0f);
}
#endif

torch::Tensor owningBytes(size_t bytes, const std::shared_ptr<size_t>& frees) {
    void* ptr = std::malloc(bytes);
    return torch::from_blob(ptr,
                            {static_cast<int64_t>(bytes)},
                            [frees](void* p) {
                                ++*frees;
                                std::free(p);
                            },
                            torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
}

class MemoryLayoutStrategyOwnershipTest: public ::testing::TestWithParam<bool> {};

TEST_P(MemoryLayoutStrategyOwnershipTest, ExportedKvAndScaleViewsRetainBothSourceOwners) {
    auto kv_frees    = std::make_shared<size_t>(0);
    auto scale_frees = std::make_shared<size_t>(0);
    auto kv         = owningBytes(32, kv_frees).narrow(0, 16, 16);
    auto scale      = owningBytes(24, scale_frees).narrow(0, 16, 8);
    auto config     = configFor(16).memory_layouts.front();
    config.enable_kv_scale = true;
    config.is_mla = GetParam();
    config.seq_size_per_block = GetParam() ? 2 : 1;
    config.kv_scale_pool_size_bytes = config.kv_scale_stride_bytes = 8;
    std::vector<torch::Tensor> kv_views;
    std::vector<torch::Tensor> scale_views;
    {
        MemoryLayoutStrategy strategy;
        ASSERT_TRUE(strategy.init(config, kv, scale, kv.data_ptr()));
        kv_views    = strategy.getLayerCacheTensors();
        scale_views = strategy.getLayerScaleCacheTensors();
        kv          = torch::Tensor();
        scale       = torch::Tensor();
    }
    EXPECT_EQ(*kv_frees, 0u);
    EXPECT_EQ(*scale_frees, 0u);
    kv_views.at(0).fill_(3);
    scale_views.at(0).fill_(4);
    kv_views.clear();
    scale_views.clear();
    EXPECT_EQ(*kv_frees, 1u);
    EXPECT_EQ(*scale_frees, 1u);
}

INSTANTIATE_TEST_SUITE_P(MhaAndMla, MemoryLayoutStrategyOwnershipTest, ::testing::Bool());

}  // namespace
}  // namespace rtp_llm
