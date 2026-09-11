#include <gtest/gtest.h>

#include <optional>

#include "rtp_llm/cpp/cache/MemoryEvaluationHelper.h"

namespace rtp_llm {
namespace {

constexpr size_t MiB = 1024 * 1024;
constexpr size_t GiB = 1024 * MiB;

class MemoryEvaluationHelperTest: public ::testing::Test {
protected:
    MemoryEvaluationHelperTest() {
        runtime_config_.reserve_runtime_mem_mb    = 1024;
        runtime_config_.max_generate_batch_size   = 1;
        model_config_.vocab_size                  = 1024;
        kv_cache_config_.kv_cache_mem_mb          = 0;
        kv_cache_config_.runtime_mem_safety_ratio = 0.05;
        status_.available_bytes                   = 40 * GiB;
        status_.total_bytes                       = 80 * GiB;
    }

    size_t size(const std::optional<WarmUpResult>& warm_up_result = std::nullopt) {
        return MemoryEvaluationHelper::getKVCacheMemorySize(
            runtime_config_, kv_cache_config_, model_config_, status_, warm_up_result);
    }

    RuntimeConfig runtime_config_;
    KVCacheConfig kv_cache_config_;
    ModelConfig   model_config_;
    MemoryStatus  status_;
};

TEST_F(MemoryEvaluationHelperTest, ExplicitKvCacheSizeTakesPrecedence) {
    kv_cache_config_.kv_cache_mem_mb = 4096;
    EXPECT_EQ(size(), 4096 * MiB);
}

TEST_F(MemoryEvaluationHelperTest, NoWarmupUsesFixedFloorAndRatio) {
    EXPECT_EQ(size(), 36 * GiB);
}

TEST_F(MemoryEvaluationHelperTest, TrustedWarmupDeductsTransientCudaGraphAndSafetyFromLatestFreeMemory) {
    WarmUpResult warm_up;
    warm_up.forward_measurement_trusted    = true;
    warm_up.cuda_graph_measurement_trusted = true;
    warm_up.init_free_memory_bytes         = 42 * GiB;
    warm_up.transient_peak_headroom_bytes  = 4 * GiB;
    warm_up.cuda_graph_memory_bytes        = 2 * GiB;
    status_.available_bytes                = 38 * GiB;

    EXPECT_EQ(size(warm_up), 28 * GiB);
}

TEST_F(MemoryEvaluationHelperTest, ConfiguredReserveUsesLatestFreeMemoryAfterPersistentGrowth) {
    runtime_config_.reserve_runtime_mem_mb = 20 * 1024;
    WarmUpResult warm_up;
    warm_up.forward_measurement_trusted   = true;
    warm_up.init_free_memory_bytes        = 42 * GiB;
    warm_up.transient_peak_headroom_bytes = 4 * GiB;
    status_.available_bytes               = 38 * GiB;

    // The 4 GiB persistent growth is already absent from current free memory.
    // Allocate 18 GiB of KV so the configured 20 GiB remains available.
    EXPECT_EQ(size(warm_up), 18 * GiB);
}

TEST_F(MemoryEvaluationHelperTest, ForwardOnlyMeasurementUsesForwardGrowthAndSafety) {
    WarmUpResult warm_up;
    warm_up.forward_measurement_trusted   = true;
    warm_up.init_free_memory_bytes        = 42 * GiB;
    warm_up.transient_peak_headroom_bytes = 4 * GiB;

    EXPECT_EQ(size(warm_up), 32 * GiB);
}

TEST_F(MemoryEvaluationHelperTest, GraphOnlyMeasurementUsesGraphGrowthAndSafety) {
    WarmUpResult warm_up;
    warm_up.cuda_graph_measurement_trusted = true;
    warm_up.init_free_memory_bytes         = 42 * GiB;
    warm_up.cuda_graph_memory_bytes        = 2 * GiB;

    EXPECT_EQ(size(warm_up), 34 * GiB);
}

TEST_F(MemoryEvaluationHelperTest, ZeroForwardGrowthDoesNotDiscardTrustedGraphMeasurement) {
    WarmUpResult warm_up;
    warm_up.forward_measurement_trusted    = true;
    warm_up.cuda_graph_measurement_trusted = true;
    warm_up.init_free_memory_bytes         = 42 * GiB;
    warm_up.transient_peak_headroom_bytes  = 0;
    warm_up.cuda_graph_memory_bytes        = 2 * GiB;
    status_.available_bytes                = 42 * GiB;

    EXPECT_EQ(size(warm_up), 36 * GiB);
}

TEST_F(MemoryEvaluationHelperTest, UntrustedWarmupUsesFallbackSizing) {
    WarmUpResult warm_up;
    warm_up.forward_measurement_trusted   = false;
    warm_up.init_free_memory_bytes        = 60 * GiB;
    warm_up.transient_peak_headroom_bytes = 6 * GiB;
    warm_up.cuda_graph_memory_bytes       = 20 * GiB;

    EXPECT_EQ(size(warm_up), 36 * GiB);
}

}  // namespace
}  // namespace rtp_llm
