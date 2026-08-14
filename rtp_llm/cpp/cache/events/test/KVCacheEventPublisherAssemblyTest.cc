#include "gtest/gtest.h"

#include <limits>
#include <memory>

#include "rtp_llm/cpp/cache/events/KVCacheEventPublisherAssembly.h"

namespace rtp_llm::test {

namespace {

class DiagnosticPublisher final: public KVCacheEventPublisher {
public:
    explicit DiagnosticPublisher(PublisherStatus status): status_(status) {}

    bool start() noexcept override {
        return false;
    }

    PublishResult tryPublish(KVCacheEvent) noexcept override {
        return PublishResult::NOT_RUNNING;
    }

    void stop() noexcept override {}

    PublisherStatus status() const noexcept override {
        return status_;
    }

    bool enabled() const noexcept override {
        return false;
    }

private:
    PublisherStatus status_;
};

KVCacheEventPublisherRawSettings validRawSettings() {
    KVCacheEventPublisherRawSettings raw;
    raw.type                  = "kvcm";
    raw.manager_endpoint      = "http://127.0.0.1:8080";
    raw.queue_capacity        = 100;
    raw.report_batch_size     = 10;
    raw.flush_interval_ms     = 20;
    raw.heartbeat_interval_ms = 1000;
    raw.request_timeout_ms    = 1500;
    raw.snapshot_timeout_ms   = 30000;
    raw.retry_interval_ms     = 500;
    raw.snapshot_interval_ms  = 300000;
    raw.log_max_keys          = 8;
    raw.snapshot_max_keys     = 1000;
    raw.snapshot_max_bytes    = 1024 * 1024;
    return raw;
}

}  // namespace

// These tests pin the gating and derivation rules used by
// KVCacheManager::initCacheEventPublisher, which cannot be constructed in a
// GPU-free unit test itself. Any behavior change here must be mirrored there.

TEST(KVCacheEventPublisherAssemblyTest, GateDisablesInactiveConfigurations) {
    for (const auto& type : {std::string(""), std::string("none")}) {
        EXPECT_EQ(KVCacheEventPublisherGate::DISABLED_INACTIVE,
                  evaluateKVCacheEventPublisherGate(
                      type, /*warmup=*/false, /*tp_rank=*/0, /*pp_size=*/1, /*cp_sharded=*/false, true));
    }
    // Warmup wins even over a valid type.
    EXPECT_EQ(KVCacheEventPublisherGate::DISABLED_INACTIVE,
              evaluateKVCacheEventPublisherGate(
                  "kvcm", /*warmup=*/true, /*tp_rank=*/0, /*pp_size=*/1, /*cp_sharded=*/false, true));
}

TEST(KVCacheEventPublisherAssemblyTest, GateWarnsOnUnknownType) {
    EXPECT_EQ(KVCacheEventPublisherGate::DISABLED_UNKNOWN_TYPE,
              evaluateKVCacheEventPublisherGate(
                  "KVCM", /*warmup=*/false, /*tp_rank=*/0, /*pp_size=*/1, /*cp_sharded=*/false, true));
    EXPECT_EQ(KVCacheEventPublisherGate::DISABLED_UNKNOWN_TYPE,
              evaluateKVCacheEventPublisherGate(
                  "http", /*warmup=*/false, /*tp_rank=*/0, /*pp_size=*/1, /*cp_sharded=*/false, true));
}

TEST(KVCacheEventPublisherAssemblyTest, NonOwnerRankDoesNotRepeatConfigurationWarnings) {
    EXPECT_EQ(KVCacheEventPublisherGate::DISABLED_NON_OWNER_RANK,
              evaluateKVCacheEventPublisherGate(
                  "KVCM", /*warmup=*/false, /*tp_rank=*/1, /*pp_size=*/1, /*cp_sharded=*/false, true));
    EXPECT_EQ(KVCacheEventPublisherGate::DISABLED_NON_OWNER_RANK,
              evaluateKVCacheEventPublisherGate(
                  "KVCM", /*warmup=*/false, /*tp_rank=*/1, /*pp_size=*/2, /*cp_sharded=*/true, true));
    EXPECT_EQ(KVCacheEventPublisherGate::DISABLED_NON_OWNER_RANK,
              evaluateKVCacheEventPublisherGate(
                  "kvcm", /*warmup=*/false, /*tp_rank=*/1, /*pp_size=*/2, /*cp_sharded=*/true, true));
}

TEST(KVCacheEventPublisherAssemblyTest, GateRejectsNonOwnerTopologies) {
    EXPECT_EQ(KVCacheEventPublisherGate::DISABLED_PIPELINE_PARALLEL,
              evaluateKVCacheEventPublisherGate(
                  "kvcm", /*warmup=*/false, /*tp_rank=*/0, /*pp_size=*/2, /*cp_sharded=*/false, true));
    EXPECT_EQ(KVCacheEventPublisherGate::DISABLED_NON_OWNER_RANK,
              evaluateKVCacheEventPublisherGate(
                  "kvcm", /*warmup=*/false, /*tp_rank=*/1, /*pp_size=*/1, /*cp_sharded=*/false, true));
}

TEST(KVCacheEventPublisherAssemblyTest, GateRejectsTopologiesWithoutReuseGroups) {
    EXPECT_EQ(KVCacheEventPublisherGate::DISABLED_NO_REUSE_GROUP,
              evaluateKVCacheEventPublisherGate(
                  "kvcm", /*warmup=*/false, /*tp_rank=*/0, /*pp_size=*/1, /*cp_sharded=*/false, false));
}

TEST(KVCacheEventPublisherAssemblyTest, GateRejectsHostPlacedReuseGroupsForHbmProtocol) {
    EXPECT_EQ(KVCacheEventPublisherGate::DISABLED_NON_DEVICE_GROUP,
              evaluateKVCacheEventPublisherGate("kvcm",
                                                /*warmup=*/false,
                                                /*tp_rank=*/0,
                                                /*pp_size=*/1,
                                                /*cp_sharded=*/false,
                                                /*has_reuse_group=*/true,
                                                /*all_reuse_groups_on_device=*/false));
}

TEST(KVCacheEventPublisherAssemblyTest, GateRejectsCpShardedKVCache) {
    // CP sharded KV cache publishes per-rank keys whose token granularity
    // differs from the external logical block size, so the publisher must be
    // disabled even on the owner rank.
    EXPECT_EQ(KVCacheEventPublisherGate::DISABLED_CP_SHARDED,
              evaluateKVCacheEventPublisherGate(
                  "kvcm", /*warmup=*/false, /*tp_rank=*/0, /*pp_size=*/1, /*cp_sharded=*/true, true));
    EXPECT_EQ(KVCacheEventPublisherGate::DISABLED_CP_SHARDED,
              evaluateKVCacheEventPublisherGate(
                  "log", /*warmup=*/false, /*tp_rank=*/0, /*pp_size=*/1, /*cp_sharded=*/true, true));
    // Pipeline parallelism is reported first when both limitations apply.
    EXPECT_EQ(KVCacheEventPublisherGate::DISABLED_PIPELINE_PARALLEL,
              evaluateKVCacheEventPublisherGate(
                  "kvcm", /*warmup=*/false, /*tp_rank=*/0, /*pp_size=*/2, /*cp_sharded=*/true, true));
}

TEST(KVCacheEventPublisherAssemblyTest, GateEnablesSupportedTypesOnOwnerRank) {
    EXPECT_EQ(KVCacheEventPublisherGate::ENABLED,
              evaluateKVCacheEventPublisherGate(
                  "log", /*warmup=*/false, /*tp_rank=*/0, /*pp_size=*/1, /*cp_sharded=*/false, true));
    EXPECT_EQ(KVCacheEventPublisherGate::ENABLED,
              evaluateKVCacheEventPublisherGate(
                  "kvcm", /*warmup=*/false, /*tp_rank=*/0, /*pp_size=*/1, /*cp_sharded=*/false, true));
}

TEST(KVCacheEventPublisherAssemblyTest, PublisherTypeRegistryMatchesGateAndLifecycleRules) {
    for (const auto& type : {kKVCacheEventPublisherLog, kKVCacheEventPublisherKVCM}) {
        EXPECT_TRUE(isSupportedKVCacheEventPublisherType(type));
        EXPECT_FALSE(isInactiveKVCacheEventPublisherType(type));
        EXPECT_EQ(KVCacheEventPublisherGate::ENABLED,
                  evaluateKVCacheEventPublisherGate(
                      type, /*warmup=*/false, /*tp_rank=*/0, /*pp_size=*/1, /*cp_sharded=*/false, true));
    }
    EXPECT_FALSE(installKVCacheEventPublisherBeforeStart(kKVCacheEventPublisherLog));
    EXPECT_TRUE(installKVCacheEventPublisherBeforeStart(kKVCacheEventPublisherKVCM));
    EXPECT_TRUE(isInactiveKVCacheEventPublisherType(kKVCacheEventPublisherNone));
    EXPECT_TRUE(isInactiveKVCacheEventPublisherType(""));
    EXPECT_FALSE(isSupportedKVCacheEventPublisherType("unsupported"));
}

TEST(KVCacheEventPublisherAssemblyTest, OnlySnapshotPublisherInstallsBeforeStart) {
    EXPECT_TRUE(installKVCacheEventPublisherBeforeStart("kvcm"));
    EXPECT_FALSE(installKVCacheEventPublisherBeforeStart("log"));
    EXPECT_FALSE(installKVCacheEventPublisherBeforeStart("none"));
    EXPECT_FALSE(installKVCacheEventPublisherBeforeStart("unsupported"));
}

TEST(KVCacheEventPublisherAssemblyTest, OnlyTpOwnerReportsPublisherMetrics) {
    EXPECT_TRUE(shouldReportKVCacheEventMetrics(0));
    EXPECT_FALSE(shouldReportKVCacheEventMetrics(1));
    EXPECT_FALSE(shouldReportKVCacheEventMetrics(-1));
}

TEST(KVCacheEventPublisherAssemblyTest, StartupFailurePreservesDiagnosticsAndIsTerminal) {
    PublisherStatus publisher_status;
    publisher_status.state                 = PublisherState::DEGRADED;
    publisher_status.queue_size            = 7;
    publisher_status.accepted_count        = 11;
    publisher_status.dropped_count         = 13;
    publisher_status.queue_high_watermark  = 9;
    publisher_status.request_failure_count = 17;
    auto publisher                         = std::make_shared<DiagnosticPublisher>(publisher_status);

    const auto status = publisherInitializationFailureStatus(publisher);
    EXPECT_EQ(PublisherState::CIRCUIT_OPEN, status.state);
    EXPECT_EQ(0u, status.queue_size);
    EXPECT_EQ(9u, status.queue_high_watermark);
    EXPECT_EQ(11u, status.accepted_count);
    EXPECT_EQ(13u, status.dropped_count);
    EXPECT_EQ(17u, status.request_failure_count);

    const auto missing_status = publisherInitializationFailureStatus(nullptr);
    EXPECT_EQ(PublisherState::CIRCUIT_OPEN, missing_status.state);
    EXPECT_EQ(0u, missing_status.queue_size);
    EXPECT_EQ(0u, missing_status.queue_high_watermark);
    EXPECT_EQ(0u, missing_status.accepted_count);
    EXPECT_EQ(0u, missing_status.dropped_count);
}

TEST(KVCacheEventPublisherAssemblyTest, RequestedButRejectedPublisherIsObservableAsGated) {
    const auto status = publisherGateFailureStatus();
    EXPECT_EQ(PublisherState::GATED, status.state);
    EXPECT_EQ(0u, status.queue_size);
    EXPECT_EQ(0u, status.request_failure_count);
}

TEST(KVCacheEventPublisherAssemblyTest, DeriveConfigRejectsInvalidLowerBounds) {
    auto expect_invalid = [](auto member, auto value) {
        auto raw    = validRawSettings();
        raw.*member = value;
        EXPECT_THROW(deriveKVCacheEventPublisherConfig(raw), std::runtime_error);
    };

    expect_invalid(&KVCacheEventPublisherRawSettings::queue_capacity, int64_t{0});
    expect_invalid(&KVCacheEventPublisherRawSettings::report_batch_size, int64_t{-1});
    expect_invalid(&KVCacheEventPublisherRawSettings::flush_interval_ms, 0);
    expect_invalid(&KVCacheEventPublisherRawSettings::heartbeat_interval_ms, -1);
    expect_invalid(&KVCacheEventPublisherRawSettings::request_timeout_ms, 0);
    expect_invalid(&KVCacheEventPublisherRawSettings::snapshot_timeout_ms, -1);
    expect_invalid(&KVCacheEventPublisherRawSettings::retry_interval_ms, 0);
    expect_invalid(&KVCacheEventPublisherRawSettings::snapshot_interval_ms, -1);
    expect_invalid(&KVCacheEventPublisherRawSettings::log_max_keys, int64_t{-1});
    expect_invalid(&KVCacheEventPublisherRawSettings::snapshot_max_keys, int64_t{0});
    expect_invalid(&KVCacheEventPublisherRawSettings::snapshot_max_bytes, int64_t{-1});

    auto raw         = validRawSettings();
    raw.log_max_keys = 0;
    EXPECT_EQ(0u, deriveKVCacheEventPublisherConfig(raw).log_max_keys_per_batch);
}

TEST(KVCacheEventPublisherAssemblyTest, DeriveConfigKeepsPositiveValues) {
    KVCacheEventPublisherRawSettings raw;
    raw.type                  = "log";
    raw.manager_endpoint      = "https://kvcm.example.test:8443/base";
    raw.queue_capacity        = 4096;
    raw.report_batch_size     = 128;
    raw.flush_interval_ms     = 25;
    raw.heartbeat_interval_ms = 2000;
    raw.request_timeout_ms    = 1500;
    raw.snapshot_timeout_ms   = 30000;
    raw.retry_interval_ms     = 500;
    raw.snapshot_interval_ms  = 300000;
    raw.log_max_keys          = 8;
    raw.snapshot_max_keys     = 1000000;
    raw.snapshot_max_bytes    = 268435456;

    const auto config = deriveKVCacheEventPublisherConfig(raw);
    EXPECT_EQ("log", config.type);
    EXPECT_EQ("https://kvcm.example.test:8443/base", config.manager_endpoint);
    EXPECT_EQ(4096u, config.queue_capacity);
    EXPECT_EQ(128u, config.report_batch_size);
    EXPECT_EQ(25, config.flush_interval_ms);
    EXPECT_EQ(2000, config.heartbeat_interval_ms);
    EXPECT_EQ(1500, config.request_timeout_ms);
    EXPECT_EQ(30000, config.snapshot_timeout_ms);
    EXPECT_EQ(500, config.retry_interval_ms);
    EXPECT_EQ(300000, config.snapshot_interval_ms);
    EXPECT_EQ(8u, config.log_max_keys_per_batch);
    EXPECT_EQ(1000000u, config.snapshot_max_keys);
    EXPECT_EQ(268435456u, config.snapshot_max_bytes);
}

TEST(KVCacheEventPublisherAssemblyTest, RawSettingsMapEveryKVCacheConfigField) {
    KVCacheConfig source;
    source.kv_cache_event_publisher_type        = "kvcm";
    source.kv_cache_event_manager_endpoint      = "http://manager:56020";
    source.kv_cache_event_queue_capacity        = 101;
    source.kv_cache_event_report_batch_size     = 102;
    source.kv_cache_event_flush_interval_ms     = 103;
    source.kv_cache_event_heartbeat_interval_ms = 104;
    source.kv_cache_event_request_timeout_ms    = 105;
    source.kv_cache_event_snapshot_timeout_ms   = 106;
    source.kv_cache_event_retry_interval_ms     = 107;
    source.kv_cache_event_snapshot_interval_ms  = 108;
    source.kv_cache_event_log_max_keys          = 109;
    source.kv_cache_event_snapshot_max_keys     = 110;
    source.kv_cache_event_snapshot_max_bytes    = 111;

    const auto raw = makeKVCacheEventPublisherRawSettings(source);
    EXPECT_EQ(source.kv_cache_event_publisher_type, raw.type);
    EXPECT_EQ(source.kv_cache_event_manager_endpoint, raw.manager_endpoint);
    EXPECT_EQ(source.kv_cache_event_queue_capacity, raw.queue_capacity);
    EXPECT_EQ(source.kv_cache_event_report_batch_size, raw.report_batch_size);
    EXPECT_EQ(source.kv_cache_event_flush_interval_ms, raw.flush_interval_ms);
    EXPECT_EQ(source.kv_cache_event_heartbeat_interval_ms, raw.heartbeat_interval_ms);
    EXPECT_EQ(source.kv_cache_event_request_timeout_ms, raw.request_timeout_ms);
    EXPECT_EQ(source.kv_cache_event_snapshot_timeout_ms, raw.snapshot_timeout_ms);
    EXPECT_EQ(source.kv_cache_event_retry_interval_ms, raw.retry_interval_ms);
    EXPECT_EQ(source.kv_cache_event_snapshot_interval_ms, raw.snapshot_interval_ms);
    EXPECT_EQ(source.kv_cache_event_log_max_keys, raw.log_max_keys);
    EXPECT_EQ(source.kv_cache_event_snapshot_max_keys, raw.snapshot_max_keys);
    EXPECT_EQ(source.kv_cache_event_snapshot_max_bytes, raw.snapshot_max_bytes);
}

TEST(KVCacheEventPublisherAssemblyTest, ContextIdentitySettingsMapEveryKVCacheConfigField) {
    KVCacheConfig source;
    source.kv_cache_event_instance_group = "event-group";
    source.reco_instance_group           = "reco-group";
    source.kv_cache_event_instance_id    = "instance-1";
    source.kv_cache_event_host_ip_port   = "10.0.0.8:18000";

    const auto settings = makeKVCacheEventPublisherContextSettings(source);
    EXPECT_EQ(source.kv_cache_event_instance_group, settings.event_instance_group);
    EXPECT_EQ(source.reco_instance_group, settings.reco_instance_group);
    EXPECT_EQ(source.kv_cache_event_instance_id, settings.instance_id);
    EXPECT_EQ(source.kv_cache_event_host_ip_port, settings.host_ip_port);
}

TEST(KVCacheEventPublisherAssemblyTest, InstanceGroupFallsBackToRecoGroup) {
    EXPECT_EQ("event-group", resolveKVCacheEventInstanceGroup("event-group", "reco-group"));
    EXPECT_EQ("reco-group", resolveKVCacheEventInstanceGroup("", "reco-group"));
    EXPECT_EQ("", resolveKVCacheEventInstanceGroup("", ""));
}

TEST(KVCacheEventPublisherAssemblyTest, ContextBuildsStableNamesAndSizedURI) {
    KVCacheEventPublisherContextSettings settings;
    settings.reco_instance_group = "reco-group";
    settings.instance_id         = "instance-1";
    settings.host_ip_port        = "10.0.0.8:18000";
    settings.model_name          = "model-a";
    settings.dtype               = "BF16";
    settings.block_size_tokens   = 64;
    settings.spec_size_bytes     = 4096;
    settings.tp_size             = 4;
    settings.dp_size             = 2;
    settings.pp_size             = 1;
    settings.dp_rank             = 1;
    settings.use_mla             = true;

    const auto context = makeKVCacheEventPublisherContext(settings);
    EXPECT_EQ("reco-group", context.instance_group);
    EXPECT_EQ("instance-1", context.instance_id);
    EXPECT_EQ("10.0.0.8:18000", context.host_ip_port);
    EXPECT_EQ("model-a", context.model_name);
    EXPECT_EQ("BF16", context.dtype);
    EXPECT_EQ("rtp_llm_hbm_64", context.spec_name);
    EXPECT_EQ("rtp-llm://10.0.0.8:18000/hbm?size=4096", context.location_uri);
    EXPECT_EQ(64, context.block_size_tokens);
    EXPECT_EQ(4096, context.spec_size_bytes);
    EXPECT_EQ(4, context.tp_size);
    EXPECT_EQ(2, context.dp_size);
    EXPECT_EQ(1, context.pp_size);
    EXPECT_EQ(1, context.dp_rank);
    EXPECT_TRUE(context.use_mla);
}

TEST(KVCacheEventPublisherAssemblyTest, ContextBuilderLeavesInvalidIdentityForPublisherValidation) {
    KVCacheEventPublisherContextSettings settings;
    settings.block_size_tokens = 0;
    settings.spec_size_bytes   = 1;

    const auto context = makeKVCacheEventPublisherContext(settings);
    EXPECT_TRUE(context.host_ip_port.empty());
    EXPECT_EQ("rtp_llm_hbm_0", context.spec_name);
    EXPECT_EQ("rtp-llm:///hbm?size=1", context.location_uri);
    EXPECT_EQ(0, context.block_size_tokens);
}

TEST(KVCacheEventPublisherAssemblyTest, DeriveConfigAcceptsExactResourceCeilings) {
    auto raw               = validRawSettings();
    raw.queue_capacity     = static_cast<int64_t>(kKVCacheEventMaxQueueCapacity);
    raw.report_batch_size  = static_cast<int64_t>(kKVCacheEventMaxReportBatchSize);
    raw.snapshot_max_keys  = static_cast<int64_t>(kKVCacheEventMaxSnapshotKeys);
    raw.snapshot_max_bytes = static_cast<int64_t>(kKVCacheEventMaxSnapshotBytes);

    const auto config = deriveKVCacheEventPublisherConfig(raw);
    EXPECT_EQ(kKVCacheEventMaxQueueCapacity, config.queue_capacity);
    EXPECT_EQ(kKVCacheEventMaxReportBatchSize, config.report_batch_size);
    EXPECT_EQ(kKVCacheEventMaxSnapshotKeys, config.snapshot_max_keys);
    EXPECT_EQ(kKVCacheEventMaxSnapshotBytes, config.snapshot_max_bytes);
}

TEST(KVCacheEventPublisherAssemblyTest, DeriveConfigRejectsEveryOversizedResourceLimit) {
    auto raw           = validRawSettings();
    raw.queue_capacity = static_cast<int64_t>(kKVCacheEventMaxQueueCapacity) + 1;
    EXPECT_THROW(deriveKVCacheEventPublisherConfig(raw), std::runtime_error);
    raw.queue_capacity    = 1;
    raw.report_batch_size = static_cast<int64_t>(kKVCacheEventMaxReportBatchSize) + 1;
    EXPECT_THROW(deriveKVCacheEventPublisherConfig(raw), std::runtime_error);
    raw.report_batch_size = 1;
    raw.snapshot_max_keys = static_cast<int64_t>(kKVCacheEventMaxSnapshotKeys) + 1;
    EXPECT_THROW(deriveKVCacheEventPublisherConfig(raw), std::runtime_error);
    raw.snapshot_max_keys  = 1;
    raw.snapshot_max_bytes = static_cast<int64_t>(kKVCacheEventMaxSnapshotBytes) + 1;
    EXPECT_THROW(deriveKVCacheEventPublisherConfig(raw), std::runtime_error);

    raw.snapshot_max_bytes = std::numeric_limits<int64_t>::max();
    EXPECT_THROW(deriveKVCacheEventPublisherConfig(raw), std::runtime_error);
}

TEST(KVCacheEventPublisherAssemblyTest, SpecSizeAggregatesAllGroupsAcrossTpRanks) {
    EXPECT_EQ(std::nullopt, aggregateKVCacheEventSpecSizeBytes({}, 4));
    EXPECT_EQ(std::optional<int64_t>(300), aggregateKVCacheEventSpecSizeBytes({100, 200}, 1));
    EXPECT_EQ(std::optional<int64_t>(1200), aggregateKVCacheEventSpecSizeBytes({100, 200}, 4));
    // Invalid parallelism must not silently produce a plausible registration.
    EXPECT_EQ(std::nullopt, aggregateKVCacheEventSpecSizeBytes({100, 200}, 0));
    EXPECT_EQ(std::nullopt, aggregateKVCacheEventSpecSizeBytes({100, 200}, -1));
}

TEST(KVCacheEventPublisherAssemblyTest, SpecSizeRejectsInvalidOrOverflowingInput) {
    constexpr auto kMax = std::numeric_limits<int64_t>::max();
    EXPECT_EQ(std::nullopt, aggregateKVCacheEventSpecSizeBytes({100, 0}, 1));
    EXPECT_EQ(std::nullopt, aggregateKVCacheEventSpecSizeBytes({100, -1}, 1));
    EXPECT_EQ(std::nullopt, aggregateKVCacheEventSpecSizeBytes({kMax, 1}, 1));
    EXPECT_EQ(std::nullopt, aggregateKVCacheEventSpecSizeBytes({kMax / 2 + 1}, 2));
}

TEST(KVCacheEventPublisherAssemblyTest, Int32ContextFieldsRejectNarrowing) {
    EXPECT_EQ(std::numeric_limits<int32_t>::min(),
              checkedKVCacheEventInt32(std::numeric_limits<int32_t>::min(), "dp_rank"));
    EXPECT_EQ(std::numeric_limits<int32_t>::max(),
              checkedKVCacheEventInt32(std::numeric_limits<int32_t>::max(), "tp_size"));
    EXPECT_THROW(checkedKVCacheEventInt32(static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1, "tp_size"),
                 std::runtime_error);
    EXPECT_THROW(checkedKVCacheEventInt32(static_cast<int64_t>(std::numeric_limits<int32_t>::min()) - 1, "dp_rank"),
                 std::runtime_error);
}

}  // namespace rtp_llm::test
