#include "gtest/gtest.h"
#include "rtp_llm/cpp/engine_base/schedulers/BatchDecodeScheduler.h"

namespace rtp_llm {

TEST(BatchDecodeSchedulerModeTest, LegacyConstructorDefaultsToDecode) {
    RuntimeConfig runtime;
    runtime.batch_decode_scheduler_config.batch_decode_scheduler_batch_size  = 1;
    runtime.batch_decode_scheduler_config.batch_decode_scheduler_warmup_type = 1;
    BatchDecodeScheduler scheduler(runtime, nullptr, nullptr);
    EXPECT_EQ(scheduler.scheduler_type_, BatchDecodeScheduler::kBatchDecode);
    EXPECT_EQ(scheduler.batch_size_, 1);
}

TEST(BatchDecodeSchedulerModeTest, PrefillRoleDoesNotDependOnSyntheticWarmupType) {
    for (int warmup_type : {0, 1}) {
        RuntimeConfig runtime;
        runtime.batch_decode_scheduler_config.batch_decode_scheduler_batch_size  = 1;
        runtime.batch_decode_scheduler_config.batch_decode_scheduler_warmup_type = warmup_type;
        BatchDecodeScheduler scheduler(runtime, nullptr, nullptr, 0, RoleType::PREFILL);
        EXPECT_EQ(scheduler.scheduler_type_, BatchDecodeScheduler::kBatchPrefill);
        EXPECT_EQ(scheduler.batch_size_, 1);
    }
}

TEST(BatchDecodeSchedulerModeTest, OtherRolesRetainDecodeMode) {
    for (RoleType role : {RoleType::DECODE, RoleType::PDFUSION}) {
        for (int warmup_type : {0, 1}) {
            RuntimeConfig runtime;
            runtime.batch_decode_scheduler_config.batch_decode_scheduler_warmup_type = warmup_type;
            BatchDecodeScheduler scheduler(runtime, nullptr, nullptr, 0, role);
            EXPECT_EQ(scheduler.scheduler_type_, BatchDecodeScheduler::kBatchDecode);
        }
    }
}

TEST(BatchDecodeSchedulerModeTest, RuntimeControlStillOverridesInitialModeAndBatchSize) {
    RuntimeConfig runtime;
    runtime.batch_decode_scheduler_config.batch_decode_scheduler_batch_size = 1;
    BatchDecodeScheduler scheduler(runtime, nullptr, nullptr, 3, RoleType::PREFILL);
    scheduler.updateSchedulerInfo(R"({"batch_size":16,"mode":"decode"})");
    EXPECT_EQ(scheduler.scheduler_type_, BatchDecodeScheduler::kBatchDecode);
    EXPECT_EQ(scheduler.batch_size_, 16);
    EXPECT_EQ(scheduler.dp_rank_, 3);
    scheduler.updateSchedulerInfo(R"({"batch_size":32,"mode":"prefill"})");
    EXPECT_EQ(scheduler.scheduler_type_, BatchDecodeScheduler::kBatchPrefill);
    EXPECT_EQ(scheduler.batch_size_, 32);
    EXPECT_EQ(scheduler.dp_rank_, 3);
}

}  // namespace rtp_llm
