#include "gtest/gtest.h"

#include "rtp_llm/cpp/cache/events/KVCacheEventPublisherAssembly.h"

namespace rtp_llm::test {

// These tests pin the gating and derivation rules used by
// KVCacheManager::initCacheEventPublisher, which cannot be constructed in a
// GPU-free unit test itself. Any behavior change here must be mirrored there.

TEST(KVCacheEventPublisherAssemblyTest, GateDisablesInactiveConfigurations) {
    for (const auto& type : {std::string(""), std::string("none")}) {
        EXPECT_EQ(KVCacheEventPublisherGate::DISABLED_INACTIVE,
                  evaluateKVCacheEventPublisherGate(type, /*warmup=*/false, /*tp_rank=*/0, /*pp_size=*/1, /*cp_sharded=*/false, true));
    }
    // Warmup wins even over a valid type.
    EXPECT_EQ(KVCacheEventPublisherGate::DISABLED_INACTIVE,
              evaluateKVCacheEventPublisherGate("kvcm", /*warmup=*/true, /*tp_rank=*/0, /*pp_size=*/1, /*cp_sharded=*/false, true));
}

TEST(KVCacheEventPublisherAssemblyTest, GateWarnsOnUnknownType) {
    EXPECT_EQ(KVCacheEventPublisherGate::DISABLED_UNKNOWN_TYPE,
              evaluateKVCacheEventPublisherGate("KVCM", /*warmup=*/false, /*tp_rank=*/0, /*pp_size=*/1, /*cp_sharded=*/false, true));
    EXPECT_EQ(KVCacheEventPublisherGate::DISABLED_UNKNOWN_TYPE,
              evaluateKVCacheEventPublisherGate("http", /*warmup=*/false, /*tp_rank=*/0, /*pp_size=*/1, /*cp_sharded=*/false, true));
}

TEST(KVCacheEventPublisherAssemblyTest, GateRejectsNonOwnerTopologies) {
    EXPECT_EQ(KVCacheEventPublisherGate::DISABLED_PIPELINE_PARALLEL,
              evaluateKVCacheEventPublisherGate("kvcm", /*warmup=*/false, /*tp_rank=*/0, /*pp_size=*/2, /*cp_sharded=*/false, true));
    EXPECT_EQ(KVCacheEventPublisherGate::DISABLED_NON_OWNER_RANK,
              evaluateKVCacheEventPublisherGate("kvcm", /*warmup=*/false, /*tp_rank=*/1, /*pp_size=*/1, /*cp_sharded=*/false, true));
}

TEST(KVCacheEventPublisherAssemblyTest, GateRejectsTopologiesWithoutReuseGroups) {
    EXPECT_EQ(KVCacheEventPublisherGate::DISABLED_NO_REUSE_GROUP,
              evaluateKVCacheEventPublisherGate("kvcm", /*warmup=*/false, /*tp_rank=*/0, /*pp_size=*/1, /*cp_sharded=*/false, false));
}

TEST(KVCacheEventPublisherAssemblyTest, GateRejectsCpShardedKVCache) {
    // CP sharded KV cache publishes per-rank keys whose token granularity
    // differs from the external logical block size, so the publisher must be
    // disabled even on the owner rank.
    EXPECT_EQ(KVCacheEventPublisherGate::DISABLED_CP_SHARDED,
              evaluateKVCacheEventPublisherGate("kvcm", /*warmup=*/false, /*tp_rank=*/0, /*pp_size=*/1, /*cp_sharded=*/true, true));
    EXPECT_EQ(KVCacheEventPublisherGate::DISABLED_CP_SHARDED,
              evaluateKVCacheEventPublisherGate("log", /*warmup=*/false, /*tp_rank=*/0, /*pp_size=*/1, /*cp_sharded=*/true, true));
    // Pipeline parallelism is reported first when both limitations apply.
    EXPECT_EQ(KVCacheEventPublisherGate::DISABLED_PIPELINE_PARALLEL,
              evaluateKVCacheEventPublisherGate("kvcm", /*warmup=*/false, /*tp_rank=*/0, /*pp_size=*/2, /*cp_sharded=*/true, true));
}

TEST(KVCacheEventPublisherAssemblyTest, GateEnablesSupportedTypesOnOwnerRank) {
    EXPECT_EQ(KVCacheEventPublisherGate::ENABLED,
              evaluateKVCacheEventPublisherGate("log", /*warmup=*/false, /*tp_rank=*/0, /*pp_size=*/1, /*cp_sharded=*/false, true));
    EXPECT_EQ(KVCacheEventPublisherGate::ENABLED,
              evaluateKVCacheEventPublisherGate("kvcm", /*warmup=*/false, /*tp_rank=*/0, /*pp_size=*/1, /*cp_sharded=*/false, true));
}

TEST(KVCacheEventPublisherAssemblyTest, DeriveConfigClampsNonPositiveValues) {
    KVCacheEventPublisherRawSettings raw;
    raw.type                  = "kvcm";
    raw.manager_endpoint      = "http://127.0.0.1:8080";
    raw.queue_capacity        = 0;
    raw.report_batch_size     = -5;
    raw.flush_interval_ms     = 0;
    raw.heartbeat_interval_ms = -1;
    raw.request_timeout_ms    = 0;
    raw.snapshot_timeout_ms   = -100;
    raw.retry_interval_ms     = 0;
    raw.snapshot_interval_ms  = 0;
    raw.log_max_keys          = -3;

    const auto config = deriveKVCacheEventPublisherConfig(raw);
    EXPECT_EQ("kvcm", config.type);
    EXPECT_EQ("http://127.0.0.1:8080", config.manager_endpoint);
    EXPECT_EQ(1u, config.queue_capacity);
    EXPECT_EQ(1u, config.report_batch_size);
    EXPECT_EQ(1, config.flush_interval_ms);
    EXPECT_EQ(1, config.heartbeat_interval_ms);
    EXPECT_EQ(1, config.request_timeout_ms);
    EXPECT_EQ(1, config.snapshot_timeout_ms);
    EXPECT_EQ(1, config.retry_interval_ms);
    EXPECT_EQ(1, config.snapshot_interval_ms);
    // log_max_keys=0 is a valid "no keys in log lines" setting, so negative
    // values clamp to 0 rather than 1.
    EXPECT_EQ(0u, config.log_max_keys_per_batch);
}

TEST(KVCacheEventPublisherAssemblyTest, DeriveConfigKeepsPositiveValues) {
    KVCacheEventPublisherRawSettings raw;
    raw.type                  = "log";
    raw.queue_capacity        = 4096;
    raw.report_batch_size     = 128;
    raw.flush_interval_ms     = 25;
    raw.heartbeat_interval_ms = 2000;
    raw.request_timeout_ms    = 1500;
    raw.snapshot_timeout_ms   = 30000;
    raw.retry_interval_ms     = 500;
    raw.snapshot_interval_ms  = 300000;
    raw.log_max_keys          = 8;

    const auto config = deriveKVCacheEventPublisherConfig(raw);
    EXPECT_EQ(4096u, config.queue_capacity);
    EXPECT_EQ(128u, config.report_batch_size);
    EXPECT_EQ(25, config.flush_interval_ms);
    EXPECT_EQ(2000, config.heartbeat_interval_ms);
    EXPECT_EQ(1500, config.request_timeout_ms);
    EXPECT_EQ(30000, config.snapshot_timeout_ms);
    EXPECT_EQ(500, config.retry_interval_ms);
    EXPECT_EQ(300000, config.snapshot_interval_ms);
    EXPECT_EQ(8u, config.log_max_keys_per_batch);
}

TEST(KVCacheEventPublisherAssemblyTest, InstanceGroupFallsBackToRecoGroup) {
    EXPECT_EQ("event-group", resolveKVCacheEventInstanceGroup("event-group", "reco-group"));
    EXPECT_EQ("reco-group", resolveKVCacheEventInstanceGroup("", "reco-group"));
    EXPECT_EQ("", resolveKVCacheEventInstanceGroup("", ""));
}

TEST(KVCacheEventPublisherAssemblyTest, SpecSizeAggregatesAllGroupsAcrossTpRanks) {
    EXPECT_EQ(0, aggregateKVCacheEventSpecSizeBytes({}, 4));
    EXPECT_EQ(300, aggregateKVCacheEventSpecSizeBytes({100, 200}, 1));
    EXPECT_EQ(1200, aggregateKVCacheEventSpecSizeBytes({100, 200}, 4));
    // Non-positive tp_size clamps to 1 rather than zeroing the spec.
    EXPECT_EQ(300, aggregateKVCacheEventSpecSizeBytes({100, 200}, 0));
}

}  // namespace rtp_llm::test
