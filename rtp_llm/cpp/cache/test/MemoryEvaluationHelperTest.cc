#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

#include "rtp_llm/cpp/cache/MemoryEvaluationHelper.h"
#include "rtp_llm/cpp/config/StaticConfig.h"
#include "rtp_llm/cpp/testing/TestLogCapture.h"

namespace rtp_llm {
namespace {

constexpr size_t MiB = 1024 * 1024;
constexpr size_t GiB = 1024 * MiB;

// Drives the sizing-assembly decision points through the MemoryStatus overload of
// getKVCacheMemorySize, which exists precisely so these branches are testable without a
// device: formula selection (warmup vs no-warmup), measurement degradation paths, which pool
// each path divides (pre-warmup vs post-teardown) and the min() reduction that only the
// fallback path applies, the trusted path's cap against real free memory (and the abort when
// even the runtime reserve no longer fits), the explicit kv_cache_mem_mb short circuit, and the
// MiB-knob validation. The pure max()/additive math itself is pinned separately in
// RuntimeMemorySizingTest.cc.
class MemoryEvaluationHelperTest: public ::testing::Test {
protected:
    void SetUp() override {
        // Several cases assert that a bad knob aborts sizing, and myAssert consults this
        // process-wide flag: with core dumps enabled the expected throw becomes a SIGABRT
        // with no failure diagnostics. Save/restore convention shared with the sibling
        // tests in this directory (see BlockPoolTest).
        old_core_dump_on_exception_                  = StaticConfig::user_ft_core_dump_on_exception;
        StaticConfig::user_ft_core_dump_on_exception = false;
    }

    void TearDown() override {
        StaticConfig::user_ft_core_dump_on_exception = old_core_dump_on_exception_;
    }

    MemoryEvaluationHelperTest() {
        runtime_config_.reserve_runtime_mem_mb          = 1024;  // configured = 1 GiB
        runtime_config_.max_generate_batch_size         = 1;
        model_config_.vocab_size                        = 1024;  // sampler estimate = 32 KiB, never binds
        kv_cache_config_.kv_cache_mem_mb                = 0;
        kv_cache_config_.runtime_mem_safety_ratio       = 0.05;
        kv_cache_config_.runtime_mem_no_warmup_floor_mb = 2048;

        status_.available_bytes = 40 * GiB;
        status_.total_bytes     = 80 * GiB;  // safety term = 4 GiB
    }

    size_t size(const std::optional<WarmUpResult>&               warm_up_result = std::nullopt,
                const std::optional<SpeculativeExecutionConfig>& sp_config      = std::nullopt) {
        return MemoryEvaluationHelper::getKVCacheMemorySize(
            runtime_config_, kv_cache_config_, model_config_, status_, warm_up_result, sp_config);
    }

    RuntimeConfig runtime_config_;
    KVCacheConfig kv_cache_config_;
    ModelConfig   model_config_;
    MemoryStatus  status_;
    bool          old_core_dump_on_exception_ = false;
};

TEST_F(MemoryEvaluationHelperTest, ExplicitKvCacheMemShortCircuitsEverything) {
    test::TestLogCapture capture("kv_alloc_explicit");
    kv_cache_config_.kv_cache_mem_mb = 4096;

    WarmUpResult warm_up;
    warm_up.available_bytes_pre_warmup  = 1 * GiB;  // would otherwise shrink the budget hard
    warm_up.device_reserved_bytes       = 1 * GiB;
    warm_up.measured_total_growth_bytes = 30 * GiB;

    EXPECT_EQ(size(warm_up), 4096 * MiB);
    EXPECT_EQ(size(), 4096 * MiB);
    EXPECT_NE(capture.content().find("[KV_ALLOC_EXPLICIT] source=kv_cache_mem_mb kv_cache_free=4096 MiB"),
              std::string::npos);
    EXPECT_EQ(capture.content().find("[KV_ALLOC]"), std::string::npos);
}

TEST_F(MemoryEvaluationHelperTest, NoWarmupFormulaAppliesFloorAndRatioAsMaxTerms) {
    test::TestLogCapture capture("kv_alloc_no_warmup_base");
    // max(configured 1 GiB, sampler ~0, floor 2 GiB, ratio 4 GiB) = 4 GiB.
    EXPECT_EQ(size(), 40 * GiB - 4 * GiB);
    EXPECT_NE(capture.content().find("[KV_ALLOC] warm_up=0 base=40960 MiB (no_warmup)"), std::string::npos);
}

// The trusted path pairs the pre-warmup pool with the total growth, so the later samples must not
// pull the base down: they are smaller by exactly what the warmup left resident, and those bytes are
// already inside measured_total_growth_bytes. Applying a min() here would subtract them twice, which
// is the over-reservation this pairing exists to remove. The cap does not bind in this case --
// pool_shrink (2 GiB) stays within max(configured, growth) -- so the contract is observable in the
// returned value.
TEST_F(MemoryEvaluationHelperTest, TrustedMeasurementSizesAgainstThePreWarmupPoolWithoutMinReduction) {
    WarmUpResult warm_up;
    warm_up.measurement_trusted = true;
    // Physically consistent: device_reserved_bytes and status_.available_bytes are both "free after
    // teardown" and must agree; the pre-warmup pool is larger by the 2 GiB the warmup left resident.
    warm_up.available_bytes_pre_warmup  = 42 * GiB;
    warm_up.device_reserved_bytes       = status_.available_bytes;  // 40 GiB
    warm_up.measured_total_growth_bytes = 6 * GiB;

    // max(configured 1, growth 6, sampler ~0) + safety 4 = 10 GiB; the no-warmup floor must not apply.
    // Sizing against the post-teardown pool with the transient share (6 - 2 = 4 GiB) would give the
    // same 32 GiB -- that equivalence is the point of the pairing.
    EXPECT_EQ(size(warm_up), 42 * GiB - 10 * GiB);

    // Neither the config-time sample nor a lower post-teardown sample may cap the base.
    EXPECT_LT(status_.available_bytes, warm_up.available_bytes_pre_warmup);
    warm_up.device_reserved_bytes = 38 * GiB;
    EXPECT_EQ(size(warm_up), 42 * GiB - 10 * GiB);
}

TEST_F(MemoryEvaluationHelperTest, ZeroMeasurementDegradesToTheNoWarmupFormulaExactly) {
    test::TestLogCapture capture("zero_measurement_degraded");
    WarmUpResult         warm_up;
    warm_up.measurement_trusted = true;
    // Deliberately generous, to prove a degraded measurement cannot reach for the larger pool: with
    // no growth term, nothing would account for what the warmup left resident.
    warm_up.available_bytes_pre_warmup  = 60 * GiB;
    warm_up.device_reserved_bytes       = status_.available_bytes;
    warm_up.measured_total_growth_bytes = 0;  // broken measurement pipeline

    // Byte-for-byte the no-warmup result, floor included: degrading to the additive
    // formula with 0 would reserve less than the untraced path.
    EXPECT_EQ(size(warm_up), size());
    EXPECT_NE(capture.content().find("measured_total_growth_bytes is 0"), std::string::npos);
    EXPECT_NE(capture.content().find("[KV_ALLOC] warm_up=0"), std::string::npos);
}

TEST_F(MemoryEvaluationHelperTest, UnspecifiedTrustUsesNoWarmupFormulaAndKeepsTheMinReduction) {
    WarmUpResult warm_up;
    EXPECT_FALSE(warm_up.measurement_trusted);  // fail closed until a producer explicitly opts in
    // Same reason as the degraded case: the pre-warmup pool must be unreachable without a growth
    // term to cover the resident bytes.
    warm_up.available_bytes_pre_warmup  = 60 * GiB;
    warm_up.device_reserved_bytes       = status_.available_bytes;
    warm_up.measured_total_growth_bytes = 6 * GiB;  // must be ignored by the formula

    EXPECT_EQ(size(warm_up), size());

    // The post-forward available_bytes sample still participates via min(): that timing
    // is the behaviour-compatibility point of keeping the PDFUSION warmup forward.
    warm_up.device_reserved_bytes = 38 * GiB;
    EXPECT_EQ(size(warm_up), size() - 2 * GiB);
}

TEST_F(MemoryEvaluationHelperTest, RejectsInvalidMiBKnobs) {
    const auto expect_rejected = [this](const char* knob, const char* reason) {
        try {
            (void)size();
            FAIL() << "expected invalid " << knob << " to be rejected";
        } catch (const std::runtime_error& error) {
            const std::string message = error.what();
            EXPECT_NE(message.find(knob), std::string::npos) << message;
            EXPECT_NE(message.find(reason), std::string::npos) << message;
        }
    };

    kv_cache_config_.runtime_mem_no_warmup_floor_mb = -1;
    expect_rejected("runtime_mem_no_warmup_floor_mb", "must be non-negative");

    kv_cache_config_.runtime_mem_no_warmup_floor_mb = 2048;
    runtime_config_.reserve_runtime_mem_mb          = -1;
    expect_rejected("reserve_runtime_mem_mb", "must be non-negative");

    runtime_config_.reserve_runtime_mem_mb = 1024;
    kv_cache_config_.kv_cache_mem_mb       = -1;
    // Negative means "not explicitly set" for the short circuit (only > 0 takes it), so
    // this falls through to the formula rather than throwing: pin that reading.
    EXPECT_EQ(size(), 40 * GiB - 4 * GiB);

    // The other half of checkedMiBToBytes: a value whose byte count would overflow size_t.
    // The MiB knobs come from argparse with an upper bound, but nothing stops an entrypoint
    // that builds RuntimeConfig directly from setting this.
    kv_cache_config_.kv_cache_mem_mb                = 0;
    kv_cache_config_.runtime_mem_no_warmup_floor_mb = std::numeric_limits<int64_t>::max();
    expect_rejected("runtime_mem_no_warmup_floor_mb", "is too large");
}

TEST_F(MemoryEvaluationHelperTest, AbortsWhenReserveExceedsAvailable) {
    status_.available_bytes = 3 * GiB;  // below the 4 GiB no-warmup reserve
    EXPECT_THROW(size(), std::runtime_error);
}

TEST_F(MemoryEvaluationHelperTest, TrustedMeasurementAbortNamesTheMeasuredFormula) {
    test::TestLogCapture capture("trusted_measurement_reserve_exceeds_base");
    status_.available_bytes = 8 * GiB;
    WarmUpResult warm_up;
    warm_up.measurement_trusted         = true;
    warm_up.available_bytes_pre_warmup  = status_.available_bytes;
    warm_up.device_reserved_bytes       = status_.available_bytes;
    warm_up.measured_total_growth_bytes = 8 * GiB;

    // runtime_required = measured growth 8 GiB + safety 4 GiB, which cannot fit in the trusted
    // path's 8 GiB pre-warmup base. This is the first reserve check, before the config-time cap.
    EXPECT_THROW(size(warm_up), std::runtime_error);
    EXPECT_NE(capture.content().find("measured_growth"), std::string::npos);
}

TEST_F(MemoryEvaluationHelperTest, TrustedMeasurementAbortsWhenRuntimeRequirementDoesNotFitCurrentFreeMemory) {
    test::TestLogCapture capture("trusted_measurement_current_free_too_small");
    status_.available_bytes = 8 * GiB;
    WarmUpResult warm_up;
    warm_up.measurement_trusted         = true;
    warm_up.available_bytes_pre_warmup  = 40 * GiB;
    warm_up.device_reserved_bytes       = 40 * GiB;
    warm_up.measured_total_growth_bytes = 6 * GiB;

    // The pre-warmup base can hold the 10 GiB requirement, but current free memory cannot.
    EXPECT_THROW(size(warm_up), std::runtime_error);
    EXPECT_NE(capture.content().find("actually free at cache-config time"), std::string::npos);
    EXPECT_NE(capture.content().find("runtime_mem_safety_ratio"), std::string::npos);
}

// The guard is a strict >, so equality must abort too: a zero-byte KV cache cannot hold a
// single block and the block_num > 0 check downstream would fail with a less useful message.
TEST_F(MemoryEvaluationHelperTest, AbortsWhenAvailableExactlyEqualsReserve) {
    status_.available_bytes = 4 * GiB;  // exactly the no-warmup reserve (ratio term binds)
    EXPECT_THROW(size(), std::runtime_error);

    status_.available_bytes = 4 * GiB + 1;
    EXPECT_EQ(size(), 1u);
}

// Scenario floors raise the *configured* term, so they must keep behaving as one input of
// the max() rather than replacing the measured peak or the additive safety headroom.
TEST_F(MemoryEvaluationHelperTest, ScenarioFloorRaisesConfiguredTermAndStillTakesSafetyHeadroom) {
    model_config_.mm_model_config.is_multimodal = true;  // raises configured to 2 GiB

    WarmUpResult warm_up;
    warm_up.measurement_trusted         = true;
    warm_up.available_bytes_pre_warmup  = status_.available_bytes;
    warm_up.device_reserved_bytes       = status_.available_bytes;
    warm_up.measured_total_growth_bytes = 1 * GiB;  // below the raised floor on purpose

    // max(configured 2 GiB, growth 1 GiB, sampler ~0) + safety 4 GiB = 6 GiB.
    EXPECT_EQ(size(warm_up), 40 * GiB - 6 * GiB);

    // With a growth above the floor the measurement wins the max() again, proving the floor
    // is a term and not a replacement.
    warm_up.measured_total_growth_bytes = 8 * GiB;
    EXPECT_EQ(size(warm_up), 40 * GiB - 12 * GiB);
}

// The std::invalid_argument thrown by the dependency-free sizing layer must surface through
// this layer's RTP_LLM_FAIL conversion -- that is the entire reason the catch exists -- and
// the message must still name the knob so the abort is actionable.
TEST_F(MemoryEvaluationHelperTest, ConvertsSizingLayerExceptionAndNamesTheKnob) {
    for (double ratio : {1.5, std::numeric_limits<double>::quiet_NaN()}) {
        SCOPED_TRACE(ratio);
        kv_cache_config_.runtime_mem_safety_ratio = ratio;
        EXPECT_THROW(
            {
                try {
                    size();
                } catch (const std::runtime_error& error) {
                    EXPECT_NE(std::string(error.what()).find("runtime_mem_safety_ratio"), std::string::npos);
                    throw;
                }
            },
            std::runtime_error);
    }
}

// The [KV_ALLOC] warm_up=<0|1> field is a smoke contract; pin it in this no-GPU gate so a
// regression fails here instead of only in an H20 smoke run. warm_up=1 iff the measurement was
// trusted and non-zero; both degradation paths report warm_up=0.
TEST_F(MemoryEvaluationHelperTest, KvAllocWarmUpFlagReflectsTheFormulaActuallyUsed) {
    {
        test::TestLogCapture capture("kv_alloc_trusted");
        WarmUpResult         warm_up;
        warm_up.measurement_trusted         = true;
        warm_up.available_bytes_pre_warmup  = 42 * GiB;
        warm_up.device_reserved_bytes       = 40 * GiB;
        warm_up.measured_total_growth_bytes = 6 * GiB;
        size(warm_up);
        EXPECT_NE(capture.content().find("[KV_ALLOC] warm_up=1"), std::string::npos);
        EXPECT_NE(capture.content().find("(pre_warmup)"), std::string::npos);
    }
    {
        test::TestLogCapture capture("kv_alloc_discarded");
        WarmUpResult         warm_up;
        warm_up.measurement_trusted         = false;  // PDFUSION
        warm_up.available_bytes_pre_warmup  = 42 * GiB;
        warm_up.device_reserved_bytes       = 40 * GiB;
        warm_up.measured_total_growth_bytes = 6 * GiB;
        size(warm_up);
        EXPECT_NE(capture.content().find("[KV_ALLOC] warm_up=0"), std::string::npos);
        EXPECT_NE(capture.content().find("(post_teardown)"), std::string::npos);
    }
    {
        test::TestLogCapture capture("kv_alloc_degraded");
        WarmUpResult         warm_up;
        warm_up.measurement_trusted         = true;
        warm_up.available_bytes_pre_warmup  = 42 * GiB;
        warm_up.device_reserved_bytes       = 40 * GiB;
        warm_up.measured_total_growth_bytes = 0;  // broken pipeline
        size(warm_up);
        EXPECT_NE(capture.content().find("[KV_ALLOC] warm_up=0"), std::string::npos);
    }
}

// pool_shrink exceeding measured growth has no measurement basis. It must WARN, and only the
// measured share may offset the runtime requirement when the config-time cap is applied.
TEST_F(MemoryEvaluationHelperTest, UnaccountedPoolShrinkWarnsAndDoesNotReduceTheReserve) {
    test::TestLogCapture capture("unaccounted_pool_shrink");
    WarmUpResult         warm_up;
    warm_up.measurement_trusted         = true;
    warm_up.available_bytes_pre_warmup  = 40 * GiB + 512 * MiB;
    warm_up.device_reserved_bytes       = status_.available_bytes;  // 40 GiB
    warm_up.measured_total_growth_bytes = 256 * MiB;

    // runtime_required = max(configured 1 GiB, growth 256 MiB, sampler ~0) + safety 4 GiB = 5 GiB.
    // Only 256 MiB of the 512 MiB shrink is explained, leaving 4 GiB + 768 MiB reserved.
    const size_t budget = size(warm_up);
    EXPECT_EQ(budget, 40 * GiB - (5 * GiB - 256 * MiB));
    EXPECT_LE(budget + 4 * GiB, status_.available_bytes);
    EXPECT_NE(capture.content().find("[KV_ALLOC_POOL_SHRINK]"), std::string::npos);
    EXPECT_NE(capture.content().find("[KV_ALLOC_CAPPED]"), std::string::npos);
}

TEST_F(MemoryEvaluationHelperTest, BudgetIsCappedToTheRuntimeRequirementNotCoveredByRetainedMemory) {
    test::TestLogCapture capture("kv_budget_capped");
    WarmUpResult         warm_up;
    warm_up.measurement_trusted = true;
    // pool_shrink 20 GiB vastly outruns the 1 GiB measured growth: the resident bytes were never
    // reserved, so the pre-warmup base overstates what is actually available.
    warm_up.available_bytes_pre_warmup  = 60 * GiB;
    warm_up.device_reserved_bytes       = status_.available_bytes;  // 40 GiB
    warm_up.measured_total_growth_bytes = 1 * GiB;

    // runtime_required = max(configured 1, growth 1, sampler ~0) + safety 4 = 5 GiB, so the uncapped
    // budget would be 60 - 5 = 55 GiB, far past the 40 GiB actually free.
    const size_t capped = size(warm_up);
    // Only the 1 GiB measured growth may be treated as retained. The unexplained 19 GiB shrink
    // must not consume the 4 GiB safety margin.
    EXPECT_EQ(capped, 36 * GiB);
    EXPECT_NE(capture.content().find("[KV_ALLOC_CAPPED]"), std::string::npos);
    EXPECT_NE(capture.content().find("capped=1"), std::string::npos);
    EXPECT_LE(capped + warm_up.measured_total_growth_bytes, status_.available_bytes);
    EXPECT_LE(capped + 4 * GiB, status_.available_bytes);
}

TEST_F(MemoryEvaluationHelperTest, ConfigTimeDriftPreservesUncoveredRuntimeRequirement) {
    test::TestLogCapture capture("config_time_drift");
    status_.available_bytes = 35 * GiB;
    WarmUpResult warm_up;
    warm_up.measurement_trusted         = true;
    warm_up.available_bytes_pre_warmup  = 40 * GiB;
    warm_up.device_reserved_bytes       = 40 * GiB;
    warm_up.measured_total_growth_bytes = 6 * GiB;

    const size_t budget = size(warm_up);
    EXPECT_EQ(budget, 25 * GiB);  // 35 GiB current free - (6 GiB growth + 4 GiB safety)
    EXPECT_LE(budget + 6 * GiB, status_.available_bytes);
    EXPECT_NE(capture.content().find("[KV_ALLOC_CONFIG_DRIFT]"), std::string::npos);
}

TEST_F(MemoryEvaluationHelperTest, ZeroSafetyRatioStillPreservesMeasuredGrowth) {
    kv_cache_config_.runtime_mem_safety_ratio = 0.0;
    status_.available_bytes                   = 35 * GiB;
    WarmUpResult warm_up;
    warm_up.measurement_trusted         = true;
    warm_up.available_bytes_pre_warmup  = 40 * GiB;
    warm_up.device_reserved_bytes       = 40 * GiB;
    warm_up.measured_total_growth_bytes = 6 * GiB;

    const size_t budget = size(warm_up);
    EXPECT_EQ(budget, 29 * GiB);
    EXPECT_LE(budget + warm_up.measured_total_growth_bytes, status_.available_bytes);
}

// Small GPU + small measured peak: the additive reserve legitimately drops below the no-warmup
// floor, which the warmup path deliberately does not apply. It must be greppable.
TEST_F(MemoryEvaluationHelperTest, SmallGpuBelowNoWarmupFloorWarns) {
    test::TestLogCapture capture("below_no_warmup_floor");
    status_.total_bytes     = 16 * GiB;  // safety term = 0.8 GiB
    status_.available_bytes = 14 * GiB;
    WarmUpResult warm_up;
    warm_up.measurement_trusted         = true;
    warm_up.available_bytes_pre_warmup  = 14 * GiB;
    warm_up.device_reserved_bytes       = 14 * GiB;
    warm_up.measured_total_growth_bytes = 256 * MiB;  // tiny model

    size(warm_up);
    EXPECT_NE(capture.content().find("[KV_ALLOC_BELOW_FLOOR]"), std::string::npos);
}

// Speculative decoding raises the configured term to a 2 GiB minimum; it stays one input of the
// max() rather than replacing the measured growth.
TEST_F(MemoryEvaluationHelperTest, SpeculativeConfigRaisesConfiguredTermAndStaysAMaxInput) {
    SpeculativeExecutionConfig sp_config;
    sp_config.type = SP_TYPE_MTP;  // any non-NONE type raises configured to 2 GiB

    WarmUpResult warm_up;
    warm_up.measurement_trusted         = true;
    warm_up.available_bytes_pre_warmup  = 40 * GiB;
    warm_up.device_reserved_bytes       = 40 * GiB;
    warm_up.measured_total_growth_bytes = 1 * GiB;  // below the raised configured floor

    // max(configured 2 GiB, growth 1 GiB, sampler ~0) + safety 4 GiB = 6 GiB.
    EXPECT_EQ(size(warm_up, sp_config), 40 * GiB - 6 * GiB);

    // With growth above the raised floor the measurement wins the max() again.
    warm_up.measured_total_growth_bytes = 8 * GiB;
    EXPECT_EQ(size(warm_up, sp_config), 40 * GiB - 12 * GiB);
}

}  // namespace
}  // namespace rtp_llm
