#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test/KVCMMockTestBase.h"

#include <array>
#include <atomic>
#include <csignal>
#include <condition_variable>
#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <unistd.h>

#include "kmonitor/client/MetricsReporter.h"
#include "kmonitor/client/KMonitorFactory.h"
#include "kmonitor/client/core/MetricsData.h"
#include "rtp_llm/cpp/cache/DeviceBlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/events/KVCacheEventPublisher.h"
#include "rtp_llm/cpp/cache/events/KVCacheEventReporter.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/SdkBufferSafety.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/DecodePublicationAccounting.h"

namespace rtp_llm {
namespace {

class ControlOnlyReporter final: public KVCacheEventReporter {
public:
    std::atomic<bool> enabled{false};

    bool post(const std::string& route, const std::string& body, std::string& response) noexcept override {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            requests_.emplace_back(route, body);
        }
        response = std::string(R"({"header":{"status":{"code":"OK"}},"snapshot_required":true,
            "extra_info":"{\"rtp_remote_cache_enabled\":)") + (enabled.load() ? "true" : "false") + R"(}"})";
        cv_.notify_all();
        return true;
    }

    bool waitFor(const std::string& text) {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, std::chrono::seconds(10), [&] {
            return std::any_of(requests_.begin(), requests_.end(), [&](const auto& request) {
                return request.first == text || request.second.find(text) != std::string::npos;
            });
        });
    }

    std::vector<std::pair<std::string, std::string>> requests() {
        std::lock_guard<std::mutex> lock(mutex_);
        return requests_;
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::vector<std::pair<std::string, std::string>> requests_;
};

TEST(KVCMLocalTest, ManagerKeepsRemoteControlIndependentOfHbmPublicationSettings) {
    for (int mode = 0; mode < 8; ++mode) {
        SCOPED_TRACE(mode);
        auto environment = makeBackendEnvironment("kvcm_control_only_" + std::to_string(mode));
        auto client = std::make_shared<MockClientWrapper>();
        std::string expected_registration;
        EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Invoke([&](const auto& configs, auto, auto&, auto&) {
            expected_registration = configs.at("")->registerInstanceRequest();
            return true;
        }));
        EXPECT_CALL(*client, shutdown()).Times(1);
        KVCacheConfig config;
        config.enable_remote_cache = true;
        config.kvcm_server_address = "outer-rpc";
        config.kv_cache_event_host_ip_port = "127.0.0.1:19999";
        config.kv_cache_event_snapshot_interval_ms = mode == 4 ? 0 : 600000;
        config.kv_cache_event_publisher_type = mode == 0 ? "none" : (mode == 6 ? "unknown" : "kvcm");
        config.reuse_cache = mode != 1;
        config.enable_device_cache = mode != 2;
        if (mode != 0 && mode != 3) {
            config.kv_cache_event_instance_id = "hbm-event-instance";
        }
        RuntimeConfig runtime;
        runtime.model_name = "control_fixture";
        if (mode == 3) {
            auto infos = std::make_shared<kvcm::KVCMConfig::LocationSpecInfoMap>();
            infos->emplace("tp0_Fdefault", environment.cache_config.blockSizeBytesForGroup("default"));
            auto groups = std::make_shared<kvcm::KVCMConfig::LocationSpecGroups>();
            groups->emplace("Fdefault", std::vector<std::string>{"tp0_Fdefault"});
            auto custom = std::make_shared<kvcm::KVCMConfig>(false, "", 8, "custom-group", "custom-remote",
                std::vector<std::string>{"rpc-a", "rpc-b"}, infos, std::make_shared<kvcm::MetaChannelConfig>(),
                std::make_shared<kvcm::SdkWrapperConfig>(), groups,
                kvcm::ModelDeployment(runtime.model_name, "fp16", false, 1, 1, 1, "extra", "user-data"));
            config.kvcm_client_config = autil::legacy::ToJsonString(kvcm::ClientWrapper::ConfigMap{{"alias", custom}});
        }
        auto backend = std::make_shared<KVCMStorageBackend>(
            environment.cache_config, config, runtime, singleRankConfig(),
            SpeculativeExecutionConfig{}, nullptr, false, nullptr, client);
        ASSERT_FALSE(backend->controlConfig().has_value());
        ASSERT_TRUE(initSingleRank(*backend, environment));
        ASSERT_TRUE(backend->controlConfig().has_value());
        const auto control = *backend->controlConfig();
        EXPECT_EQ(control.register_instance_request, expected_registration);
        if (mode == 3) {
            EXPECT_EQ(control.instance_group, "custom-group");
            EXPECT_EQ(control.instance_id, "custom-remote");
            EXPECT_EQ(control.addresses, (std::vector<std::string>{"rpc-a", "rpc-b"}));
        }

        auto manager_config = environment.cache_config;
        if (mode == 5) {
            auto groups = manager_config.topology().groups();
            groups.front().policy.enable_prefix_reuse = false;
            auto layers = manager_config.topology().layers();
            manager_config.setTopology(std::move(groups), std::move(layers));
        }
        KVCacheManager manager(manager_config, false, nullptr, config, singleRankConfig(), runtime);
        auto full = std::make_shared<FullGroupSet>(
            std::vector<DeviceBlockPoolPtr>{environment.device_pool}, nullptr, nullptr);
        full->initialize(0, manager_config.topologyPtr(), {"default"});
        manager.block_tree_cache_ = block_tree_cache_test::makeBlockTreeCacheForTest({full});
        manager.block_tree_cache_->storage_backend_ = backend;
        auto reporter = std::make_shared<ControlOnlyReporter>();
        manager.initCacheEventPublisher(reporter);
        ASSERT_NE(manager.cache_event_publisher_, nullptr);
        ASSERT_TRUE(reporter->waitFor("EVENT_NODE_REGISTER"));
        const auto waitForRemoteState = [&](bool expected) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            while (std::chrono::steady_clock::now() < deadline) {
                if (backend->remoteCacheEnabled() == expected) {
                    return true;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            return false;
        };
        ASSERT_TRUE(waitForRemoteState(false));
        EXPECT_EQ(manager.block_tree_cache_->tree()->event_publisher_ != nullptr, mode == 7);
        reporter->enabled.store(true);
        ASSERT_TRUE(waitForRemoteState(true));
        ASSERT_TRUE(reporter->waitFor("EVENT_HEARTBEAT"));
        EXPECT_EQ(manager.cache_event_publisher_->tryPublish({KVCacheEventType::BLOCK_ADD, 42, 0}),
                  mode == 7 ? PublishResult::ACCEPTED : PublishResult::DISABLED);
        if (mode == 7) {
            ASSERT_TRUE(reporter->waitFor("EVENT_BLOCK_SNAPSHOT"));
            ASSERT_TRUE(reporter->waitFor("EVENT_BLOCK_ADD"));
        }
        manager.stopCacheEventPublisher();
        const auto requests = reporter->requests();
        ASSERT_FALSE(requests.empty());
        EXPECT_NE(requests.back().second.find("EVENT_HOST_DOWN"), std::string::npos);
        size_t registration_count = 0;
        for (const auto& request : requests) {
            if (mode != 7) {
                EXPECT_EQ(request.second.find("EVENT_BLOCK_"), std::string::npos);
            }
            if (request.first == "/api/registerInstance") {
                ++registration_count;
                autil::legacy::json::JsonMap actual;
                autil::legacy::FromJsonString(actual, request.second);
                if (mode == 7) {
                    EXPECT_EQ(autil::legacy::AnyCast<std::string>(actual.at("instance_id")), "hbm-event-instance");
                    continue;
                }
                EXPECT_EQ(actual.erase("trace_id"), 1u);
                autil::legacy::json::JsonMap expected;
                autil::legacy::FromJsonString(expected, expected_registration);
                EXPECT_EQ(autil::legacy::ToJsonString(actual), autil::legacy::ToJsonString(expected));
            }
        }
        EXPECT_EQ(registration_count, 1u);
    }
}

double snapshotGauge(kmonitor::MutableMetric* metric) {
    if (metric == nullptr) {
        ADD_FAILURE() << "metric is null";
        return -1;
    }
    kmonitor::MetricsTags tags;
    kmonitor::Metric*     gauge = metric->DeclareMetric(&tags);
    if (gauge == nullptr) {
        ADD_FAILURE() << "metric series is missing";
        return -1;
    }
    kmonitor::MetricsRecord record(nullptr, nullptr, 0);
    gauge->Snapshot(&record, 1000);
    EXPECT_TRUE(metric->UndeclareMetric(gauge));
    if (record.Values().size() != 1) {
        ADD_FAILURE() << "unexpected metric value count=" << record.Values().size();
        return -1;
    }
    return std::stod(record.Values().front()->Value());
}

kvcm::KVCMConfigPtr makeLocalConfig() {
    auto location_infos  = std::make_shared<kvcm::KVCMConfig::LocationSpecInfoMap>();
    auto location_groups = std::make_shared<kvcm::KVCMConfig::LocationSpecGroups>();
    auto channel         = std::make_shared<kvcm::MetaChannelConfig>(/*retry_time=*/1,
                                                             /*connection_timeout=*/1000,
                                                             /*call_timeout=*/100);
    auto sdk             = std::make_shared<kvcm::SdkWrapperConfig>();
    return std::make_shared<kvcm::KVCMConfig>(/*enable_vipserver=*/false,
                                              /*vipserver_domain=*/"",
                                              /*block_size=*/8,
                                              "instance_group",
                                              "instance_id",
                                              std::vector<std::string>{"direct"},
                                              location_infos,
                                              channel,
                                              sdk,
                                              location_groups,
                                              kvcm::ModelDeployment());
}

TEST(KVCMLocalTest, PublicationStateMetricsKeepQuarantineInPendingTotals) {
    auto reporter = std::make_shared<kmonitor::MetricsReporter>("", "", kmonitor::MetricsTags{});
    RtpLLMDecodePublicationStateMetricsCollector collector;
    collector.pending_bytes    = 100;
    collector.pending_jobs     = 3;
    collector.queued_jobs      = 1;
    collector.running_jobs     = 1;
    collector.quarantined_jobs = 1;
    collector.byte_limit       = 512;
    collector.job_limit        = 4;
    collector.oldest_age_us    = 7000000;
    reporter->report<RtpLLMDecodePublicationMetrics, RtpLLMDecodePublicationStateMetricsCollector>(nullptr, &collector);
    auto* metrics = reporter->getMetricsGroup<RtpLLMDecodePublicationMetrics>();
    ASSERT_NE(metrics, nullptr);
    EXPECT_DOUBLE_EQ(snapshotGauge(metrics->pending_bytes_metric), 100);
    EXPECT_DOUBLE_EQ(snapshotGauge(metrics->pending_jobs_metric), 3);
    EXPECT_DOUBLE_EQ(snapshotGauge(metrics->queued_jobs_metric), 1);
    EXPECT_DOUBLE_EQ(snapshotGauge(metrics->running_jobs_metric), 1);
    EXPECT_DOUBLE_EQ(snapshotGauge(metrics->quarantined_jobs_metric), 1);
    EXPECT_DOUBLE_EQ(snapshotGauge(metrics->byte_limit_metric), 512);
    EXPECT_DOUBLE_EQ(snapshotGauge(metrics->job_limit_metric), 4);
    EXPECT_DOUBLE_EQ(snapshotGauge(metrics->oldest_age_us_metric), 7000000);
}

TEST(KVCMLocalTest, DirectClientRoutesMetadataAndPayload) {
    auto  factory           = std::make_unique<kvcm::MockClientFactory>();
    auto* factory_ptr       = factory.get();
    auto  subscriber        = std::make_unique<kvcm::MockSubscriber>();
    auto* subscriber_ptr    = subscriber.get();
    auto  meta_client       = std::make_unique<kv_cache_manager::MockMetaClient>();
    auto* meta_ptr          = meta_client.get();
    auto  destruction_count = std::make_shared<int>(0);
    auto  transfer_client   = std::make_unique<kv_cache_manager::MockTransferClient>(destruction_count);
    auto* transfer_ptr      = transfer_client.get();

    EXPECT_CALL(*factory_ptr, createSubscriber(false)).WillOnce(Invoke([&subscriber](bool) {
        return std::move(subscriber);
    }));
    EXPECT_CALL(*subscriber_ptr, init(std::vector<std::string>{"direct"})).WillOnce(Return(true));
    EXPECT_CALL(*subscriber_ptr, getAddresses(_)).Times(0);
    EXPECT_CALL(*factory_ptr, createMetaClient(_, _)).WillOnce(Invoke([&meta_client](const auto&, const auto&) {
        return std::move(meta_client);
    }));
    static const std::string storage_config = R"({"sdk_backend_configs":[]})";
    EXPECT_CALL(*meta_ptr, GetStorageConfig()).WillOnce(::testing::ReturnRef(storage_config));
    EXPECT_CALL(*factory_ptr, createTransferClient(_, _)).WillOnce(Invoke([&transfer_client](const auto&, const auto&) {
        return std::move(transfer_client);
    }));

    kvcm::ClientWrapper          wrapper(std::move(factory));
    std::array<char, 64>         registration{};
    kv_cache_manager::RegistSpan span{registration.data(), registration.size()};
    ASSERT_TRUE(
        wrapper.init({{"", makeLocalConfig()}}, {kv_cache_manager::RoleType::HYBRID, &span, "tp0_Ffull"}, "default"));

    const std::vector<int64_t> keys{1, 2};
    kv_cache_manager::Location location;
    location.emplace_back(kv_cache_manager::LocationSpecUnit{"tp0_Ffull", "uri"});
    kv_cache_manager::Locations expected_locations{std::move(location)};
    EXPECT_CALL(*meta_ptr, MatchLocation("match", kv_cache_manager::QueryType::QT_PREFIX_MATCH, keys, _, _, _, _))
        .WillOnce(Return(std::make_pair(kv_cache_manager::ClientErrorCode::ER_OK, expected_locations)));
    const auto [match_ok, locations] = wrapper.match(
        "", "match", kv_cache_manager::QueryType::QT_PREFIX_MATCH, keys, kv_cache_manager::BlockMaskOffset{0}, {});
    EXPECT_TRUE(match_ok);
    EXPECT_EQ(locations.size(), expected_locations.size());

    kv_cache_manager::UriStrVec    uris{"uri"};
    kv_cache_manager::BlockBuffers buffers(1);
    buffers.front().iovs.push_back(
        {kv_cache_manager::MemoryType::CPU, registration.data(), registration.size(), false});
    EXPECT_CALL(*transfer_ptr, LoadKvCaches(uris, _, _)).WillOnce(Return(kv_cache_manager::ClientErrorCode::ER_OK));
    EXPECT_TRUE(wrapper.loadKvCachesForTag("default", uris, buffers));

    kv_cache_manager::WriteLocation write_location;
    write_location.write_session_id = "session";
    EXPECT_CALL(*meta_ptr, StartWrite("start", keys, std::vector<int64_t>{}, std::vector<std::string>{"Ffull"}, 9, 0))
        .WillOnce(Return(std::make_pair(kv_cache_manager::ClientErrorCode::ER_OK, write_location)));
    const auto [start_ok, actual_write_location] =
        wrapper.getWriteLocation("", "start", keys, {}, {"Ffull"}, /*write_timeout_seconds=*/9);
    EXPECT_TRUE(start_ok);
    EXPECT_EQ(actual_write_location.write_session_id, "session");

    EXPECT_CALL(*transfer_ptr, SaveKvCaches(uris, _, _))
        .WillOnce(Return(
            std::make_pair(kv_cache_manager::ClientErrorCode::ER_OK, kv_cache_manager::UriStrVec{"actual_uri"})));
    const auto [save_ok, actual_uris] = wrapper.saveKvCachesForTag("default", uris, buffers);
    EXPECT_TRUE(save_ok);
    EXPECT_EQ(actual_uris, (kv_cache_manager::UriStrVec{"actual_uri"}));

    const kv_cache_manager::UriStrVec failed_uris{"failed_uri"};
    EXPECT_CALL(*transfer_ptr, SaveKvCaches(failed_uris, _, _))
        .WillOnce(Return(std::make_pair(kv_cache_manager::ClientErrorCode::ER_SDK_TIMEOUT,
                                        kv_cache_manager::UriStrVec{"partial_uri"})));
    const auto [failed_save_ok, failed_actual_uris] =
        wrapper.saveKvCachesForTag("default", failed_uris, buffers);
    EXPECT_FALSE(failed_save_ok);
    EXPECT_TRUE(failed_actual_uris.empty());

    EXPECT_CALL(*meta_ptr,
                FinishWrite("finish",
                            "session",
                            ::testing::VariantWith<kv_cache_manager::BlockMaskOffset>(::testing::Eq(0u)),
                            ::testing::IsEmpty()))
        .WillOnce(Return(kv_cache_manager::ClientErrorCode::ER_OK));
    EXPECT_TRUE(wrapper.finishWrite("", "finish", "session", kv_cache_manager::BlockMaskOffset{0}, {}));
    wrapper.shutdown();
    EXPECT_EQ(*destruction_count, 1);
}

TEST(KVCMLocalTest, InitRejectsMissingTopologyAndInvalidPoolShape) {
    auto environment    = makeBackendEnvironment("kvcm_storage_backend_invalid_init_shape");
    auto client_wrapper = std::make_shared<MockClientWrapper>();
    EXPECT_CALL(*client_wrapper, initForPools(_, _, _, _)).Times(0);
    EXPECT_CALL(*client_wrapper, shutdown()).Times(0);
    auto backend  = makeBackend(environment, singleRankConfig(), client_wrapper);
    auto resolver = [&](int layer_id, const std::string&, int block_id) {
        return environment.device_pool->convertIndexToBuffer(layer_id, block_id);
    };

    EXPECT_ANY_THROW(backend->init(nullptr, {}, resolver));
    EXPECT_ANY_THROW(backend->init(environment.cache_config.topologyPtr(), {}, resolver));
    EXPECT_ANY_THROW(backend->init(environment.cache_config.topologyPtr(), {{"default", nullptr}}, resolver));
}

TEST(KVCMLocalTest, DisabledPublicationIgnoresDecodeOnlyBackendAndBudgetRestrictions) {
    autil::EnvGuard off("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "0");
    autil::EnvGuard budget("RTP_LLM_REMOTE_CACHE_DECODE_PENDING_MIB", "invalid");
    autil::EnvGuard jobs("RTP_LLM_REMOTE_CACHE_DECODE_PENDING_JOBS", "invalid");
    auto            environment = makeBackendEnvironment("kvcm_disabled_decode_preserves_prefill");
    auto            client      = std::make_shared<MockClientWrapper>();
    EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client, shutdown()).Times(1);
    KVCacheConfig config;
    config.kvcm_server_address   = "unused-test-address";
    config.kvcm_model_sdk_config = R"([{"type":"pace_ssd"}])";
    // Deliberately outside the new Decode SSD/GDR pairing restrictions: off
    // must delegate the ordinary configuration to its original SDK path.
    config.kvcm_read_backend_type = 0;
    RuntimeConfig runtime;
    runtime.model_name = "kvcm_disabled_decode";
    BackendHandle backend(std::make_unique<KVCMStorageBackend>(environment.cache_config,
                                                               config,
                                                               runtime,
                                                               singleRankConfig(),
                                                               SpeculativeExecutionConfig{},
                                                               nullptr,
                                                               /*gdr_enabled=*/false,
                                                               nullptr,
                                                               client));
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));
    kv_cache_manager::Locations locations{
        kv_cache_manager::Location{kv_cache_manager::LocationSpecUnit{"tp0_Fdefault", "read_uri"}}};
    EXPECT_CALL(*client, match(_, _, _, std::vector<int64_t>{101}, _, _))
        .WillOnce(Return(std::make_pair(true, locations)));
    auto observation = match(*backend.backend, makeStorageRequest(environment));
    ASSERT_TRUE(observation.success);
    ASSERT_EQ(observation.matched_blocks_num, 1u);
    EXPECT_CALL(*client, loadKvCachesForTag("default", kv_cache_manager::UriStrVec{"read_uri"}, _, _))
        .WillOnce(Return(true));
    EXPECT_TRUE(read(*backend.backend, makeStorageRequest(environment), std::move(observation.match_meta)));
}

TEST(KVCMLocalTest, InitRejectsInvalidQuarantineAbortRatio) {
    autil::EnvGuard quarantine_abort_ratio("RTP_LLM_REMOTE_CACHE_QUARANTINE_ABORT_RATIO", "invalid");
    auto            environment    = makeBackendEnvironment("kvcm_storage_backend_invalid_quarantine_ratio");
    auto            client_wrapper = std::make_shared<MockClientWrapper>();
    EXPECT_CALL(*client_wrapper, initForPools(_, _, _, _)).Times(0);
    // StorageBackend::init unwinds the implementation even when validation
    // fails before SDK registration. The cleanup must remain idempotent.
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), client_wrapper);
    EXPECT_FALSE(initSingleRank(*backend.backend, environment));
}

TEST(KVCMLocalTest, GdrRegistrationUsesActualAllocationSpan) {
    auto             environment    = makeBackendEnvironment("kvcm_storage_backend_gdr_span");
    auto             client_wrapper = std::make_shared<MockClientWrapper>();

    EXPECT_CALL(*client_wrapper, initForPools(_, _, _, _))
        .WillOnce(Invoke([&](const kvcm::ClientWrapper::ConfigMap&,
                             kv_cache_manager::RoleType,
                             const std::vector<kvcm::ClientWrapper::PoolRegistration>& registrations,
                             const std::vector<std::string>&) {
            if (registrations.size() != 1u) {
                ADD_FAILURE() << "expected one GDR pool registration";
                return false;
            }
            const auto& registration = registrations.front();
            EXPECT_EQ(registration.span.base, environment.device_pool->getBaseAddress());
            EXPECT_EQ(registration.span.size, environment.device_pool->getAllocationSizeBytes());
            const long page_size = sysconf(_SC_PAGESIZE);
            EXPECT_GT(page_size, 0);
            if (page_size <= 0) {
                return false;
            }
            EXPECT_EQ(reinterpret_cast<uintptr_t>(registration.span.base) % static_cast<size_t>(page_size), 0u);
            EXPECT_EQ(registration.span.size % static_cast<size_t>(page_size), 0u);
            EXPECT_TRUE(registration.memory_registrations.has_value());
            if (!registration.memory_registrations.has_value()) {
                return false;
            }
            EXPECT_EQ(registration.memory_registrations->gpu.size(), 1u);
            if (registration.memory_registrations->gpu.size() != 1u) {
                return false;
            }
            EXPECT_EQ(registration.memory_registrations->gpu.front().size,
                      environment.device_pool->getAllocationSizeBytes());
            return true;
        }));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment,
                               singleRankConfig(),
                               client_wrapper,
                               /*broadcast_manager=*/nullptr,
                               /*metrics_reporter=*/nullptr,
                               /*gdr_enabled=*/true);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));
}

TEST(KVCMLocalTest, PublicationSwitchPreservesLegacyHostPoolFallbackAndRequiresGpuWhenEnabled) {
    autil::EnvGuard async("RTP_LLM_STREAM_ASYNC", "0");
    for (const bool enabled : {false, true}) {
        SCOPED_TRACE(enabled);
        autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", enabled ? "1" : "0");
        auto            environment = makeHybridBackendEnvironment("kvcm_storage_backend_mixed_gdr_span");
        ASSERT_EQ(environment.cache_config.groupTags().size(), 2u);

        const auto& host_tag = environment.cache_config.groupTags().back();
        ASSERT_NE(host_tag, environment.cache_config.groupTags().front());
        auto old_pool = environment.pools_by_tag.at(host_tag);
        old_pool->decRef(environment.block_id);

        auto host_config = std::make_shared<DeviceBlockPoolConfig>(DeviceBlockPoolConfigHelper::createConfigForGroup(
            environment.cache_config, environment.cache_config.topology().group(host_tag)));
        host_config->pool_name                 = "mixed_gdr_host_pinned";
        host_config->use_pinned_cpu_backing    = true;
        host_config->use_device_malloc_backing = false;
        auto host_pool                         = std::make_shared<DeviceBlockPool>(host_config);
        ASSERT_TRUE(host_pool->init());
        const auto host_block = host_pool->malloc();
        ASSERT_TRUE(host_block.has_value());
        ASSERT_EQ(*host_block, environment.block_id);
        host_pool->incRef(*host_block);
        environment.pools_by_tag[host_tag] = host_pool;

        auto client_wrapper = std::make_shared<MockClientWrapper>();
        if (enabled) {
            EXPECT_CALL(*client_wrapper, initForPools(_, _, _, _)).Times(0);
        } else {
            EXPECT_CALL(*client_wrapper, initForPools(_, _, _, _))
                .WillOnce(Invoke([&](const auto&, auto, const auto& registrations, const auto& tags) {
                    EXPECT_EQ(registrations.size(), 2u);
                    EXPECT_EQ(registrations.size(), tags.size());
                    bool saw_host = false;
                    bool saw_gpu  = false;
                    for (size_t i = 0; i < tags.size(); ++i) {
                        if (tags[i] == host_tag) {
                            saw_host = true;
                            EXPECT_EQ(registrations[i].span.base, host_pool->getBaseAddress());
                            EXPECT_EQ(registrations[i].span.size, host_pool->getTotalSizeBytes());
                            EXPECT_FALSE(registrations[i].memory_registrations.has_value());
                        } else {
                            saw_gpu              = true;
                            const auto& gpu_pool = environment.pools_by_tag.at(tags[i]);
                            EXPECT_EQ(registrations[i].span.base, gpu_pool->getBaseAddress());
                            EXPECT_EQ(registrations[i].span.size, gpu_pool->getAllocationSizeBytes());
                            EXPECT_TRUE(registrations[i].memory_registrations.has_value());
                            if (!registrations[i].memory_registrations.has_value()) {
                                return false;
                            }
                            const auto& gpu_spans = registrations[i].memory_registrations->gpu;
                            EXPECT_EQ(gpu_spans.size(), 1u);
                            if (gpu_spans.size() != 1u) {
                                return false;
                            }
                            EXPECT_EQ(gpu_spans.front().base, gpu_pool->getBaseAddress());
                            EXPECT_EQ(gpu_spans.front().size, gpu_pool->getAllocationSizeBytes());
                            EXPECT_EQ(gpu_spans.front().device_id, gpu_pool->deviceIndex());
                        }
                    }
                    EXPECT_TRUE(saw_host);
                    EXPECT_TRUE(saw_gpu);
                    return true;
                }));
        }
        EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
        // Both branches use a valid P CP1 configuration. The enabled negative must
        // reach actual pool-registration validation, not fail earlier on role/SDK.
        KVCacheConfig config;
        config.kvcm_server_address                 = "unused-test-address";
        config.kvcm_instance_id_salt               = "host-pool-fallback-contract";
        config.kvcm_model_sdk_config               = R"([{"type":"pace"}])";
        config.kvcm_read_backend_type              = 3;
        auto parallel                              = singleRankConfig();
        parallel.role_type                         = RoleType::PREFILL;
        parallel.prefill_cp_config.prefill_cp_size = 1;
        RuntimeConfig runtime;
        runtime.model_name = "host_pool_fallback";
        BackendHandle backend(std::make_unique<KVCMStorageBackend>(environment.cache_config,
                                                                   config,
                                                                   runtime,
                                                                   parallel,
                                                                   SpeculativeExecutionConfig{},
                                                                   nullptr,
                                                                   /*gdr_enabled=*/true,
                                                                   nullptr,
                                                                   client_wrapper));
        EXPECT_EQ(initSingleRank(*backend.backend, environment), !enabled);
    }
}

TEST(KVCMLocalTest, LegacyEmptyRemoteResponsePreservesOneofPresence) {
    FunctionResponsePB response;
    response.mutable_remote_response();
    std::string serialized;
    ASSERT_TRUE(response.SerializeToString(&serialized));

    FunctionResponsePB parsed;
    ASSERT_TRUE(parsed.ParseFromString(serialized));
    ASSERT_TRUE(parsed.has_remote_response());
    EXPECT_EQ(parsed.remote_response().transfer_status(), REMOTE_TRANSFER_STATUS_UNSPECIFIED);
}

TEST(KVCMLocalTest, ExecuteFunctionPropagatesMetadataErrorsAndPreservesTransferStatus) {
    rtp_llm::initLogger();
    createDevice();
    auto environment    = makeBackendEnvironment("kvcm_rpc_result");
    auto client_wrapper = std::make_shared<MockClientWrapper>();
    EXPECT_CALL(*client_wrapper, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), client_wrapper);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));

    KVCacheManager manager(environment.cache_config);
    ASSERT_TRUE(manager.init());
    manager.blockTreeCache()->storage_backend_ = std::move(backend.backend);

    for (const auto operation : {REMOTE_OPERATION_REMOVE_CACHE,
                                 REMOTE_OPERATION_GET_LOCATIONS_BY_BACKEND,
                                 REMOTE_OPERATION_GET_HOST_CACHE_STATE}) {
        FunctionRequestPB request;
        request.mutable_remote_request()->set_op(operation);
        request.mutable_remote_request()->mutable_metadata()->set_query_type(999);
        FunctionResponsePB response;
        EXPECT_FALSE(manager.executeFunction(request, response));
    }

    for (const auto ec :
         {kv_cache_manager::ClientErrorCode::ER_CLIENT_NOT_EXISTS, kv_cache_manager::ClientErrorCode::ER_SDK_TIMEOUT}) {
        client_wrapper->load_status = ec;
        client_wrapper->save_status = ec;
        for (const auto operation : {REMOTE_OPERATION_READ, REMOTE_OPERATION_WRITE}) {
            FunctionRequestPB request;
            auto*             remote = request.mutable_remote_request();
            remote->set_op(operation);
            remote->add_group_tags("default");
            remote->add_block_ids(environment.block_id);
            remote->add_uris("transfer_uri");
            FunctionResponsePB response;
            EXPECT_TRUE(manager.executeFunction(request, response));
            ASSERT_TRUE(response.has_remote_response());
            EXPECT_EQ(response.remote_response().transfer_status(),
                      ec == kv_cache_manager::ClientErrorCode::ER_SDK_TIMEOUT ? REMOTE_TRANSFER_STATUS_TIMEOUT :
                                                                                REMOTE_TRANSFER_STATUS_FAILED);
        }
    }
}

BackendHandle makeCompletionProofBackend(const BackendEnvironment&                 environment,
                                         const std::shared_ptr<MockClientWrapper>& client) {
    KVCacheConfig config;
    config.kvcm_server_address                 = "unused-test-address";
    config.kvcm_instance_id_salt               = "completion-proof";
    config.kvcm_model_sdk_config               = R"([{"type":"pace"}])";
    config.kvcm_read_backend_type              = 0;
    auto parallel                              = singleRankConfig();
    parallel.role_type                         = RoleType::PREFILL;
    parallel.prefill_cp_config.prefill_cp_size = 1;
    RuntimeConfig runtime;
    runtime.model_name = "kvcm_test_model";
    return BackendHandle(std::make_unique<MockQuiescentKVCMBackend>(environment.cache_config,
                                                              config,
                                                              runtime,
                                                              parallel,
                                                              SpeculativeExecutionConfig{},
                                                              nullptr,
                                                              true,
                                                              nullptr,
                                                              client));
}

TEST(KVCMLocalTest, ProductionNonrootBackendInstallsPayloadObserverWithoutGeneralReporter) {
    autil::EnvGuard async("RTP_LLM_STREAM_ASYNC", "0");
    autil::EnvGuard manual("kmonitorManuallyMode", "true");
    for (const bool enabled : {false, true}) {
        autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", enabled ? "1" : "0");
        for (const bool sharded : {false, true}) {
            for (int cp : {2, 4, 8}) {
                for (int rank = 1; rank < cp; ++rank) {
                    SCOPED_TRACE(::testing::PrintToString(std::make_tuple(enabled, sharded, cp, rank)));
                    auto environment = makeBackendEnvironment("payload_observer_nonroot");
                    auto client      = std::make_shared<MockClientWrapper>();
                    EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
                    EXPECT_CALL(*client, shutdown()).WillOnce(Invoke([raw = client.get()] {
                        raw->kvcm::ClientWrapper::shutdown();
                    }));
                    KVCacheConfig config;
                    config.kvcm_server_address                  = "unused-test-address";
                    config.kvcm_instance_id_salt                = "payload-observer";
                    config.kvcm_model_sdk_config                = R"([{"type":"pace"}])";
                    auto parallel                               = singleRankConfig();
                    parallel.role_type                          = RoleType::PREFILL;
                    parallel.tp_size                            = cp;
                    parallel.tp_rank                            = rank;
                    parallel.prefill_cp_config.prefill_cp_size  = cp;
                    parallel.prefill_cp_config.kv_cache_sharded = sharded;
                    RuntimeConfig runtime;
                    runtime.model_name = "payload_observer_nonroot";
                    BackendHandle backend(std::make_unique<KVCMStorageBackend>(environment.cache_config,
                                                                               config,
                                                                               runtime,
                                                                               parallel,
                                                                               SpeculativeExecutionConfig{},
                                                                               nullptr,
                                                                               /*gdr_enabled=*/true,
                                                                               /*general_reporter=*/nullptr,
                                                                               client));
                    ASSERT_TRUE(initSingleRank(*backend.backend, environment));
                    EXPECT_EQ(static_cast<bool>(client->transfer_observer_), enabled);
                    if (enabled) {
                        EXPECT_TRUE(kmonitor::KMonitorFactory::IsStarted());
                        // Exercise the observer installed by the actual backend,
                        // not a reporter or callback injected by this fixture.
                        client->observeTransfer(/*read=*/true, 13 + rank, false);
                        client->observeTransfer(/*read=*/false, 37 + rank, false);
                        client->observeTransfer(/*read=*/false, 0, true);
                    }
                    backend.backend->shutdown();
                    EXPECT_FALSE(client->transfer_observer_);
                    if (enabled) {
                        EXPECT_TRUE(kmonitor::KMonitorFactory::IsStarted());
                    }
                }
            }
        }
    }
}

TEST(KVCMLocalTest, SuccessfulReadBindsItsRequiredSourceButLegacyOverwriteClearsIt) {
    for (const bool known_source : {true, false}) {
        SCOPED_TRACE(known_source);
        autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1");
        autil::EnvGuard async("RTP_LLM_STREAM_ASYNC", "0");
        auto            environment = makeBackendEnvironment("kvcm_physical_source_binding");
        auto            client      = std::make_shared<MockClientWrapper>();
        EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
        EXPECT_CALL(*client, shutdown()).Times(1);
        auto backend = makeCompletionProofBackend(environment, client);
        ASSERT_TRUE(initSingleRank(*backend.backend, environment));
        auto source = DecodeCacheOrigin::parse("v1." + std::string(32, 'a') + ".4.0.8.8.9");
        ASSERT_TRUE(source);
        auto old_value        = *source;
        old_value.session     = std::string(32, 'b');
        const auto old        = std::make_shared<const DecodeCacheOrigin>(old_value);
        const auto generation = environment.device_pool->decodeOriginIncarnation(environment.block_id);
        ASSERT_NE(generation, uint64_t{0});
        ASSERT_TRUE(environment.device_pool->bindDecodeOrigin(environment.block_id, generation, "default", old));
        const std::string           canonical = "pace://allocated/1?media_type=2&node_id=1&size=32";
        const auto                  uri       = known_source ? *attachDecodeCacheOrigin(canonical, *source) : canonical;
        kv_cache_manager::Locations locations{{{"tp0_Fdefault", uri}}};
        EXPECT_CALL(*client, match(_, _, _, _, _, _)).WillOnce(Return(std::make_pair(true, locations)));
        auto observation = match(*backend.backend, makeStorageRequest(environment));
        ASSERT_TRUE(observation.success);
        ASSERT_NE(observation.match_meta, nullptr);
        EXPECT_CALL(*client, loadKvCachesForTag("default", kv_cache_manager::UriStrVec{uri}, _, _))
            .WillOnce(Invoke([&](const auto&, const auto&, auto&, const auto&) {
                // Previous provenance is invalid before a destination write.
                EXPECT_FALSE(environment.device_pool->decodeOrigin(environment.block_id, "default"));
                return true;
            }));
        auto source_meta = observation.match_meta;
        EXPECT_TRUE(read(*backend.backend, makeStorageRequest(environment), std::move(observation.match_meta)));
        EXPECT_EQ(static_cast<bool>(source_meta->decode_sources), known_source);
        if (known_source) {
            ASSERT_TRUE(source_meta->decode_sources);
            EXPECT_EQ(countDecodeReuse(*source_meta->decode_sources, 8).total(), 4);
        }
        const auto bound = environment.device_pool->decodeOrigin(environment.block_id, "default");
        if (known_source) {
            ASSERT_TRUE(bound);
            EXPECT_TRUE(*bound == *source);
        } else {
            EXPECT_FALSE(bound);
        }
    }
}

template<typename Code>
std::vector<std::pair<Code, bool>> failedCompletionCases(bool write = false) {
    std::vector<std::pair<Code, bool>> cases{{write ? Code::ER_SDKWRITE_ERROR : Code::ER_SDKREAD_ERROR, false},
                                             {Code::ER_INVALID_PARAMS, false}};
    if constexpr (kvcm::detail::HasExplicitBufferSafeCodes<Code>::value) {
        cases.emplace_back(Code::ER_SDK_TIMEOUT_BUFFER_SAFE, true);
        cases.emplace_back(Code::ER_SDK_ERROR_BUFFER_SAFE, true);
    }
    return cases;
}

TEST(KVCMLocalTest, EnabledReadQuarantinesUnprovenSdkErrorsButNotExplicitSafeResults) {
    autil::EnvGuard async("RTP_LLM_STREAM_ASYNC", "0");
    using EC = kv_cache_manager::ClientErrorCode;
    for (const bool enabled : {false, true}) {
        for (const auto& [ec, safe] : failedCompletionCases<EC>()) {
            SCOPED_TRACE(::testing::Message() << "enabled=" << enabled << " ec=" << static_cast<int>(ec));
            autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", enabled ? "1" : "0");
            auto            environment = makeBackendEnvironment("kvcm_completion_proof");
            auto            client      = std::make_shared<MockClientWrapper>();
            EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
            EXPECT_CALL(*client, shutdown()).Times(1);
            auto backend = makeCompletionProofBackend(environment, client);
            ASSERT_TRUE(initSingleRank(*backend.backend, environment));
            kv_cache_manager::Locations locations{{{"tp0_Fdefault", "pace://read_uri"}}};
            EXPECT_CALL(*client, match(_, _, _, _, _, _)).WillOnce(Return(std::make_pair(true, locations)));
            auto request     = makeStorageRequest(environment);
            auto observation = match(*backend.backend, request);
            ASSERT_TRUE(observation.success);
            ASSERT_NE(observation.match_meta, nullptr);
            client->load_status = ec;
            const auto error    = readResult(*backend.backend, request, std::move(observation.match_meta));
            EXPECT_FALSE(error.ok());
            ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
            const bool unknown = enabled && !safe;
            EXPECT_EQ(static_cast<bool>(backend.backend->protected_quarantine_), unknown);
            EXPECT_FALSE(environment.device_pool->isFrozen(environment.block_id));
            std::lock_guard<std::mutex> lock(backend.backend->quarantine_mutex_);
            EXPECT_EQ(
                backend.backend->quarantined_breakdown_[0][static_cast<size_t>(StorageQuarantineReason::INVALID_RESPONSE)]
                    .task_count,
                unknown ? 1u : 0u);
        }
    }
}

TEST(KVCMLocalTest, CallerBufferSafetyDoesNotProveRemoteAllocationQuiescence) {
    autil::EnvGuard async("RTP_LLM_STREAM_ASYNC", "0");
    using EC = kv_cache_manager::ClientErrorCode;
    for (const bool enabled : {false, true}) {
        for (const auto& [ec, safe] : failedCompletionCases<EC>(/*write=*/true)) {
            autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", enabled ? "1" : "0");
            auto            environment = makeBackendEnvironment("kvcm_write_completion_proof");
            auto            client      = std::make_shared<MockClientWrapper>();
            EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
            EXPECT_CALL(*client, shutdown()).Times(1);
            auto backend = makeCompletionProofBackend(environment, client);
            ASSERT_TRUE(initSingleRank(*backend.backend, environment));
            kv_cache_manager::WriteLocation allocation;
            allocation.write_session_id = "pending-session";
            allocation.block_mask       = kv_cache_manager::BlockMaskOffset{0};
            allocation.locations        = {{{"tp0_Fdefault", "pace://write_uri"}}};
            EXPECT_CALL(*client, getWriteLocation(_, _, _, _, _, _, 0))
                .WillOnce(Return(std::make_pair(true, allocation)));
            const bool unknown = enabled && !safe;
            if (enabled) {
                EXPECT_CALL(*client, finishWrite(_, _, _, _, _)).Times(0);
            } else {
                EXPECT_CALL(*client, finishWrite(_, _, "pending-session", _, _)).WillOnce(Return(true));
            }
            client->save_status = ec;
            auto task           = backend.backend->prepareWrite(makeStorageRequest(environment));
            ASSERT_TRUE(static_cast<bool>(task));
            ASSERT_TRUE(backend.backend->write(std::move(task)));
            ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
            EXPECT_EQ(static_cast<bool>(backend.backend->protected_quarantine_), unknown);
            EXPECT_FALSE(environment.device_pool->isFrozen(environment.block_id));
        }
    }
}

TEST(KVCMLocalTest, CommitAckFailureIsUnknownPublicationWithoutUnknownDma) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1");
    autil::EnvGuard async("RTP_LLM_STREAM_ASYNC", "0");
    for (const bool throws : {false, true}) {
        auto environment = makeBackendEnvironment("kvcm_commit_ack_failure");
        auto client      = std::make_shared<MockClientWrapper>();
        EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
        EXPECT_CALL(*client, shutdown()).Times(1);
        auto backend = makeCompletionProofBackend(environment, client);
        ASSERT_TRUE(initSingleRank(*backend.backend, environment));
        kv_cache_manager::WriteLocation allocation;
        allocation.write_session_id = "pending-session";
        allocation.block_mask       = kv_cache_manager::BlockMaskOffset{0};
        allocation.locations        = {{{"tp0_Fdefault", "pace://write_uri"}}};
        EXPECT_CALL(*client, getWriteLocation(_, _, _, _, _, _, 0)).WillOnce(Return(std::make_pair(true, allocation)));
        // Exactly one attempted commit; no second mask-zero abort.
        EXPECT_CALL(*client, finishWrite(_, _, "pending-session", _, _))
            .WillOnce(Invoke([throws](const auto&, const auto&, const auto&, const auto&, const auto&) -> bool {
                if (throws)
                    throw std::runtime_error("lost commit acknowledgement");
                return false;
            }));
        client->save_status = kv_cache_manager::ClientErrorCode::ER_OK;
        std::mutex               events_mutex;
        std::vector<std::string> outcomes;
        auto                     accounting = std::make_shared<kvcm::DecodePublicationAccounting>(
            1024,
            1,
            kvcm::DecodePublicationAccounting::SnapshotObserver{},
            [&](const char* outcome, std::optional<int64_t> latency) {
                std::lock_guard<std::mutex> lock(events_mutex);
                outcomes.emplace_back(outcome);
                EXPECT_FALSE(latency.has_value());
            });
        auto request                 = makeStorageRequest(environment);
        request.completion_admission = accounting->tryAcquire(128, kvcm::DecodePublicationAccounting::Clock::now());
        ASSERT_TRUE(request.completion_admission);
        const auto refs_before = environment.device_pool->refCount(environment.block_id);
        auto       task        = backend.backend->prepareWrite(std::move(request));
        ASSERT_TRUE(static_cast<bool>(task));
        EXPECT_EQ(environment.device_pool->refCount(environment.block_id), refs_before + 1);
        ASSERT_TRUE(backend.backend->write(std::move(task)));
        ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
        EXPECT_FALSE(environment.device_pool->isFrozen(environment.block_id));
        EXPECT_EQ(accounting->snapshot().quarantined, 0u);
        EXPECT_EQ(environment.device_pool->refCount(environment.block_id), refs_before);
        // Join the worker before inspecting ownership of its retired closure.
        backend.backend->shutdown();
        EXPECT_EQ(accounting->snapshot().bytes, 0u);
        EXPECT_EQ(accounting->snapshot().jobs, 0u);
        std::lock_guard<std::mutex> lock(events_mutex);
        EXPECT_EQ(outcomes, std::vector<std::string>{"completion_unknown"});
    }
}

TEST(KVCMLocalTest, FailureBeforeTransferStillAbortsItsUnusedAllocation) {
    autil::EnvGuard async("RTP_LLM_STREAM_ASYNC", "0");
    for (const bool enabled : {false, true}) {
        autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", enabled ? "1" : "0");
        auto            environment = makeBackendEnvironment("kvcm_pretransfer_failure");
        auto            client      = std::make_shared<MockClientWrapper>();
        EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
        EXPECT_CALL(*client, shutdown()).Times(1);
        auto backend = makeCompletionProofBackend(environment, client);
        ASSERT_TRUE(initSingleRank(*backend.backend, environment));
        kv_cache_manager::WriteLocation allocation;
        allocation.write_session_id = "unused-session";
        allocation.block_mask       = kv_cache_manager::BlockMaskOffset{0};
        allocation.locations        = {{{"unknown-spec-before-io", "write_uri"}}};
        EXPECT_CALL(*client, getWriteLocation(_, _, _, _, _, _, 0)).WillOnce(Return(std::make_pair(true, allocation)));
        EXPECT_CALL(*client,
                    finishWrite(_,
                                _,
                                "unused-session",
                                ::testing::Eq(kv_cache_manager::BlockMask{kv_cache_manager::BlockMaskOffset{0}}),
                                _))
            .WillOnce(Return(true));
        EXPECT_CALL(*client, saveKvCachesForTag(_, _, _, _)).Times(0);
        auto task = backend.backend->prepareWrite(makeStorageRequest(environment));
        ASSERT_TRUE(static_cast<bool>(task));
        ASSERT_TRUE(backend.backend->write(std::move(task)));
        ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
        EXPECT_FALSE(environment.device_pool->isFrozen(environment.block_id));
    }
}

TEST(KVCMLocalTest, RuntimeDisableAfterAllocationAbortsBeforePayloadAndCanRecover) {
    autil::EnvGuard async("RTP_LLM_STREAM_ASYNC", "0");
    for (const bool protected_io : {false, true}) {
        autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", protected_io ? "1" : "0");
        auto environment = makeBackendEnvironment("kvcm_control_after_allocation");
        auto client = std::make_shared<MockClientWrapper>();
        EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
        EXPECT_CALL(*client, shutdown()).Times(1);
        auto backend = makeCompletionProofBackend(environment, client);
        ASSERT_TRUE(initSingleRank(*backend.backend, environment));
        const auto refs = environment.device_pool->refCount(environment.block_id);
        kv_cache_manager::WriteLocation allocation;
        allocation.write_session_id = "disabled-before-save";
        allocation.block_mask = kv_cache_manager::BlockMaskOffset{0};
        allocation.locations = {{{"tp0_Fdefault", "pace://write_uri"}}};
        EXPECT_CALL(*client, getWriteLocation(_, _, _, _, _, _, 0))
            .WillOnce(Invoke([&](const auto&, const auto&, const auto&, const auto&, const auto&, auto, auto) {
                backend.backend->setRemoteCacheEnabled(false);
                return std::make_pair(true, allocation);
            }));
        EXPECT_CALL(*client, saveKvCachesForTag(_, _, _, _)).Times(0);
        EXPECT_CALL(*client,
                    finishWrite(_, _, "disabled-before-save",
                                ::testing::Eq(kv_cache_manager::BlockMask{kv_cache_manager::BlockMaskOffset{0}}),
                                ::testing::IsEmpty()))
            .WillOnce(Return(true));
        EXPECT_TRUE(backend.backend->write(backend.backend->prepareWrite(makeStorageRequest(environment))));
        ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
        EXPECT_EQ(environment.device_pool->refCount(environment.block_id), refs);
        EXPECT_FALSE(environment.device_pool->isFrozen(environment.block_id));
        EXPECT_FALSE(backend.backend->protected_quarantine_);
        EXPECT_FALSE(backend.backend->prepareWrite(makeStorageRequest(environment)));

        backend.backend->setRemoteCacheEnabled(true);
        kv_cache_manager::WriteLocation empty;
        // All keys already exist: the returned mask must cover the request.
        empty.block_mask = kv_cache_manager::BlockMaskOffset{1};
        EXPECT_CALL(*client, getWriteLocation(_, _, _, _, _, _, 0)).WillOnce(Return(std::make_pair(true, empty)));
        EXPECT_TRUE(backend.backend->write(backend.backend->prepareWrite(makeStorageRequest(environment))));
        ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
        EXPECT_EQ(environment.device_pool->refCount(environment.block_id), refs);
    }
}

TEST(KVCMLocalTest, RuntimeDisableAfterMatchSkipsLoadWithoutQuarantine) {
    autil::EnvGuard async("RTP_LLM_STREAM_ASYNC", "0");
    for (const bool protected_io : {false, true}) {
        autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", protected_io ? "1" : "0");
        auto environment = makeBackendEnvironment("kvcm_control_after_match");
        auto client = std::make_shared<MockClientWrapper>();
        EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
        EXPECT_CALL(*client, shutdown()).Times(1);
        auto backend = makeCompletionProofBackend(environment, client);
        ASSERT_TRUE(initSingleRank(*backend.backend, environment));
        kv_cache_manager::Locations locations{{{"tp0_Fdefault", "pace://read_uri"}}};
        EXPECT_CALL(*client, match(_, _, _, _, _, _)).WillOnce(Return(std::make_pair(true, locations)));
        auto observation = match(*backend.backend, makeStorageRequest(environment));
        ASSERT_TRUE(observation.success);
        ASSERT_NE(observation.match_meta, nullptr);
        EXPECT_CALL(*client, loadKvCachesForTag(_, _, _, _)).Times(0);
        const auto refs = environment.device_pool->refCount(environment.block_id);
        backend.backend->setRemoteCacheEnabled(false);
        const auto result = readResult(*backend.backend, makeStorageRequest(environment), observation.match_meta);
        EXPECT_FALSE(result.ok());
        EXPECT_NE(result.code(), ErrorCode::DEADLINE_EXCEEDED);
        EXPECT_EQ(environment.device_pool->refCount(environment.block_id), refs);
        EXPECT_FALSE(backend.backend->protected_quarantine_);
        EXPECT_FALSE(environment.device_pool->isFrozen(environment.block_id));
        auto local_request = makeStorageRequest(environment, {101, 102}, 1);
        const auto local = match(*backend.backend, std::move(local_request));
        EXPECT_TRUE(local.success);
        EXPECT_EQ(local.matched_blocks_num, 1u);
        EXPECT_EQ(local.match_meta, nullptr);
    }
}

TEST(KVCMLocalTest, RuntimeDisableDuringSaveStillFinishesAdmittedWrite) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "0");
    auto environment = makeBackendEnvironment("kvcm_control_during_save");
    auto client = std::make_shared<MockClientWrapper>();
    EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), client);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));
    kv_cache_manager::WriteLocation allocation;
    allocation.write_session_id = "already-submitted";
    allocation.block_mask = kv_cache_manager::BlockMaskOffset{0};
    allocation.locations = {{{"tp0_Fdefault", "pace://write_uri"}}};
    EXPECT_CALL(*client, getWriteLocation(_, _, _, _, _, _, 0)).WillOnce(Return(std::make_pair(true, allocation)));
    EXPECT_CALL(*client, saveKvCachesForTag(_, _, _, _))
        .WillOnce(Invoke([&](const auto&, const auto& uris, const auto&, const auto&) {
            backend.backend->setRemoteCacheEnabled(false);
            return std::make_pair(true, uris);
        }));
    EXPECT_CALL(*client,
                finishWrite(_, _, "already-submitted",
                            ::testing::Eq(kv_cache_manager::BlockMask{kv_cache_manager::BlockMaskOffset{1}}), _))
        .WillOnce(Return(true));
    const auto refs = environment.device_pool->refCount(environment.block_id);
    EXPECT_TRUE(backend.backend->write(backend.backend->prepareWrite(makeStorageRequest(environment))));
    ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
    EXPECT_EQ(environment.device_pool->refCount(environment.block_id), refs);
    EXPECT_FALSE(backend.backend->prepareWrite(makeStorageRequest(environment)));
}

TEST(KVCMLocalTest, RuntimeDisableDuringTp2FanoutStillCompletesBothRanksAndFinishesWrite) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "0");
    auto environment = makeBackendEnvironment("kvcm_control_tp2");
    auto client = std::make_shared<MockClientWrapper>();
    EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client, shutdown()).Times(1);
    std::atomic<StorageBackend*> control_backend{nullptr};
    auto first_state = std::make_shared<KVCMBroadcastState>();
    auto second_state = std::make_shared<KVCMBroadcastState>();
    for (const auto& state : {first_state, second_state}) {
        state->on_request = [&] {
            if (auto* backend = control_backend.load()) {
                backend->setRemoteCacheEnabled(false);
            }
        };
    }
    KVCMBroadcastRpcServer first(0, first_state);
    KVCMBroadcastRpcServer second(1, second_state);
    ASSERT_TRUE(first.start());
    ASSERT_TRUE(second.start());
    auto broadcast = std::make_shared<BroadcastManager>(std::vector<std::string>{first.address(), second.address()});
    ASSERT_TRUE(broadcast->init());
    auto parallel = singleRankConfig();
    parallel.tp_size = 2;
    auto backend = makeBackend(environment, parallel, client, broadcast);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));
    control_backend.store(backend.backend.get());
    kv_cache_manager::WriteLocation allocation;
    allocation.write_session_id = "tp2-admitted-before-disable";
    allocation.block_mask = kv_cache_manager::BlockMaskOffset{0};
    allocation.locations = {{{"tp0_Fdefault", "pace://rank0"}, {"tp1_Fdefault", "pace://rank1"}}};
    EXPECT_CALL(*client, getWriteLocation(_, _, _, _, _, _, 0)).WillOnce(Return(std::make_pair(true, allocation)));
    EXPECT_CALL(*client, saveKvCachesForTag(_, _, _, _)).Times(0);
    EXPECT_CALL(*client,
                finishWrite(_, _, "tp2-admitted-before-disable",
                            ::testing::Eq(kv_cache_manager::BlockMask{kv_cache_manager::BlockMaskOffset{1}}), _))
        .WillOnce(Return(true));
    const auto refs = environment.device_pool->refCount(environment.block_id);
    EXPECT_TRUE(backend.backend->write(backend.backend->prepareWrite(makeStorageRequest(environment))));
    ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
    EXPECT_FALSE(backend.backend->remoteCacheEnabled());
    for (const auto& state : {first_state, second_state}) {
        const auto requests = snapshotRequests(state);
        ASSERT_EQ(requests.size(), 1u);
        EXPECT_EQ(requests.front().op(), REMOTE_OPERATION_WRITE);
        EXPECT_EQ(requests.front().uris_size(), 1);
    }
    EXPECT_EQ(environment.device_pool->refCount(environment.block_id), refs);
    EXPECT_FALSE(backend.backend->prepareWrite(makeStorageRequest(environment)));
    control_backend.store(nullptr);
}

TEST(KVCMLocalTest, RemoteCacheControlGaugeReportsDisabledAndEnabled) {
    auto reporter = std::make_shared<kmonitor::MetricsReporter>("", "", kmonitor::MetricsTags{});
    RtpLLMRemoteCacheControlMetricsCollector collector;
    for (const bool enabled : {false, true}) {
        collector.enabled = enabled;
        reporter->report<RtpLLMRemoteCacheControlMetrics, RtpLLMRemoteCacheControlMetricsCollector>(nullptr,
                                                                                                &collector);
        auto* metrics = reporter->getMetricsGroup<RtpLLMRemoteCacheControlMetrics>();
        ASSERT_NE(metrics, nullptr);
        EXPECT_DOUBLE_EQ(snapshotGauge(metrics->enabled_metric), enabled ? 1 : 0);
    }
}

TEST(KVCMLocalTest, EnabledReadAndWriteResponsesCarryExplicitCompletionProof) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1");
    autil::EnvGuard async("RTP_LLM_STREAM_ASYNC", "0");
    using EC = kv_cache_manager::ClientErrorCode;
    for (const auto& [ec, safe] : failedCompletionCases<EC>()) {
        for (const auto op : {REMOTE_OPERATION_READ, REMOTE_OPERATION_WRITE}) {
            // Unknown completion closes an owner. Each disposition therefore
            // needs a fresh mock owner, not another call on the closed one.
            auto environment = makeBackendEnvironment("kvcm_response_completion_proof");
            auto client = std::make_shared<MockClientWrapper>();
            EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
            EXPECT_CALL(*client, shutdown()).Times(1);
            auto backend = makeCompletionProofBackend(environment, client);
            ASSERT_TRUE(initSingleRank(*backend.backend, environment));
            client->load_status = ec;
            client->save_status = ec;
            RemoteOperationRequestPB request;
            request.set_op(op);
            request.add_group_tags("default");
            request.add_block_ids(environment.block_id);
            request.add_uris("pace://uri");
            RemoteOperationResponsePB response;
            EXPECT_FALSE(backend->execute(request, response));
            EXPECT_EQ(response.transfer_status(), ec == EC::ER_SDK_TIMEOUT ? REMOTE_TRANSFER_STATUS_TIMEOUT :
                                                                            REMOTE_TRANSFER_STATUS_FAILED);
            EXPECT_EQ(response.payload_buffer_safe(), safe);
            if (!safe) {
                RemoteOperationResponsePB rejected;
                EXPECT_FALSE(backend->execute(request, rejected));
                EXPECT_EQ(rejected.transfer_status(), REMOTE_TRANSFER_STATUS_FAILED);
                EXPECT_TRUE(rejected.payload_buffer_safe());
            }
        }
    }
    auto environment = makeBackendEnvironment("kvcm_preflight_completion_proof");
    auto client = std::make_shared<MockClientWrapper>();
    EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client, shutdown()).Times(1);
    EXPECT_CALL(*client, loadKvCachesForTag(_, _, _, _)).Times(0);
    auto backend = makeCompletionProofBackend(environment, client);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));
    RemoteOperationRequestPB local_reject;
    local_reject.set_op(REMOTE_OPERATION_READ);
    local_reject.add_group_tags("foreign");
    local_reject.add_block_ids(environment.block_id);
    local_reject.add_uris("pace://uri");
    RemoteOperationResponsePB response;
    EXPECT_FALSE(backend->execute(local_reject, response));
    EXPECT_EQ(response.transfer_status(), REMOTE_TRANSFER_STATUS_FAILED);
    EXPECT_TRUE(response.payload_buffer_safe());
}

TEST(KVCMLocalTest, UnknownPaceBatchRetainsEveryDestinationGroupUntilQuiescence) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1");
    autil::EnvGuard async("RTP_LLM_STREAM_ASYNC", "0");
    auto environment = makeHybridBackendEnvironment("kvcm_group_completion_proof");
    auto client = std::make_shared<MockClientWrapper>();
    std::string first_tag;
    EXPECT_CALL(*client, initForPools(_, _, _, _))
        .WillOnce(Invoke([&](const auto&, auto, const auto&, const auto& tags) {
            first_tag = tags.front();
            return true;
        }));
    EXPECT_CALL(*client, shutdown()).Times(1);
    auto backend = makeCompletionProofBackend(environment, client);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));
    kv_cache_manager::Locations locations{
        {{"tp0_Ffull1", "pace://full_uri"}, {"tp0_Llinear", "pace://linear_uri"}}};
    EXPECT_CALL(*client, match(_, _, _, _, _, _)).WillOnce(Return(std::make_pair(true, locations)));
    auto request = makeGroupedStorageRequest(environment, {101}, 0, {environment.block_id});
    auto observation = match(*backend.backend, request);
    ASSERT_TRUE(observation.success);
    ASSERT_NE(observation.match_meta, nullptr);
    std::map<std::string, size_t> references;
    for (const auto& [tag, pool] : environment.pools_by_tag)
        references[tag] = pool->refCount(environment.block_id);
    // All PACE groups go through the registered first client in one batch.
    EXPECT_CALL(*client, loadKvCachesForTag(first_tag, _, _, _))
        .WillOnce(Invoke([](const auto&, const auto& uris, auto& buffers, const auto&) {
            EXPECT_EQ(uris.size(), 2u);
            EXPECT_EQ(buffers.size(), 2u);
            return false;
        }));
    const auto error = readResult(*backend.backend, request, std::move(observation.match_meta));
    EXPECT_FALSE(error.ok());
    ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
    EXPECT_TRUE(backend.backend->protected_quarantine_);
    for (const auto& [tag, pool] : environment.pools_by_tag) {
        SCOPED_TRACE(tag);
        EXPECT_FALSE(pool->isFrozen(environment.block_id));
        EXPECT_EQ(pool->refCount(environment.block_id), references.at(tag) + 1);
    }
    auto* mock = dynamic_cast<MockQuiescentKVCMBackend*>(backend.backend.get());
    ASSERT_NE(mock, nullptr);
    mock->proveMockQuiescence();
    backend.backend->shutdown();
    for (const auto& [tag, pool] : environment.pools_by_tag)
        EXPECT_EQ(pool->refCount(environment.block_id), references.at(tag));
}

TEST(KVCMLocalTest, MatchAndReadUseReturnedLocation) {
    auto environment    = makeBackendEnvironment("kvcm_storage_backend_match_read");
    auto client_wrapper = std::make_shared<MockClientWrapper>();

    EXPECT_CALL(*client_wrapper, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), client_wrapper);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));

    kv_cache_manager::Locations locations{
        kv_cache_manager::Location{kv_cache_manager::LocationSpecUnit{"tp0_Fdefault", "read_uri"}}};
    EXPECT_CALL(*client_wrapper,
                match(_, _, kv_cache_manager::QueryType::QT_PREFIX_MATCH, std::vector<int64_t>{101}, _, _))
        .WillOnce(Return(std::make_pair(true, locations)));
    auto observation = match(*backend.backend, makeStorageRequest(environment));
    ASSERT_TRUE(observation.success);
    ASSERT_EQ(observation.matched_blocks_num, 1u);
    ASSERT_NE(observation.match_meta, nullptr);

    EXPECT_CALL(*client_wrapper, loadKvCachesForTag("default", kv_cache_manager::UriStrVec{"read_uri"}, _, _))
        .WillOnce(Invoke([](const std::string&,
                            const kv_cache_manager::UriStrVec&,
                            kv_cache_manager::BlockBuffers& buffers,
                            const std::shared_ptr<kv_cache_manager::TransferTraceInfo>&) {
            EXPECT_EQ(buffers.size(), 1u);
            return true;
        }));
    EXPECT_TRUE(read(*backend.backend, makeStorageRequest(environment), std::move(observation.match_meta)));
}

TEST(KVCMLocalTest, SuccessfulMatchPreservesLocalPrefixAndReadsOnlyRemoteSuffix) {
    auto environment    = makeBackendEnvironment("kvcm_storage_backend_partial_local_prefix");
    auto client_wrapper = std::make_shared<MockClientWrapper>();

    EXPECT_CALL(*client_wrapper, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), client_wrapper);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));
    ScopedReferencedBlocks source_blocks(environment.device_pool, 3);
    const auto&            block_ids         = source_blocks.get();
    const auto             second_block_info = environment.device_pool->convertIndexToBuffer(0, block_ids[1]);
    const auto             third_block_info  = environment.device_pool->convertIndexToBuffer(0, block_ids[2]);
    ASSERT_EQ(second_block_info.size(), 1u);
    ASSERT_EQ(third_block_info.size(), 1u);

    kv_cache_manager::Locations locations{
        kv_cache_manager::Location{kv_cache_manager::LocationSpecUnit{"tp0_Fdefault", "read_uri_102"}},
        kv_cache_manager::Location{kv_cache_manager::LocationSpecUnit{"tp0_Fdefault", "read_uri_103"}},
    };
    EXPECT_CALL(*client_wrapper,
                match(_, _, kv_cache_manager::QueryType::QT_PREFIX_MATCH, std::vector<int64_t>({101, 102, 103}), _, _))
        .WillOnce(Invoke([locations](const std::string&,
                                     const std::string&,
                                     kv_cache_manager::QueryType,
                                     const std::vector<int64_t>&,
                                     const kv_cache_manager::BlockMask& block_mask,
                                     const kv_cache_manager::ForwardContext&) {
            const auto* offset = std::get_if<kv_cache_manager::BlockMaskOffset>(&block_mask);
            EXPECT_NE(offset, nullptr);
            if (offset != nullptr) {
                EXPECT_EQ(*offset, 1u);
            }
            return std::make_pair(true, locations);
        }));
    auto request =
        makeStorageRequest(environment, /*keys=*/{101, 102, 103}, /*local_matched_blocks=*/1, /*block_ids=*/block_ids);
    auto observation = match(*backend.backend, request);
    ASSERT_TRUE(observation.success);
    ASSERT_EQ(observation.matched_blocks_num, 3u);
    ASSERT_NE(observation.match_meta, nullptr);

    EXPECT_CALL(*client_wrapper,
                loadKvCachesForTag("default", kv_cache_manager::UriStrVec({"read_uri_102", "read_uri_103"}), _, _))
        .WillOnce(Invoke([second_base = second_block_info.front().addr, third_base = third_block_info.front().addr](
                             const std::string&,
                             const kv_cache_manager::UriStrVec&,
                             kv_cache_manager::BlockBuffers& buffers,
                             const std::shared_ptr<kv_cache_manager::TransferTraceInfo>&) {
            if (buffers.size() != 2u || buffers[0].iovs.size() != 1u || buffers[1].iovs.size() != 1u) {
                ADD_FAILURE() << "KVCM read received an invalid block-buffer shape";
                return false;
            }
            EXPECT_EQ(buffers[0].iovs[0].base, second_base);
            EXPECT_EQ(buffers[1].iovs[0].base, third_base);
            return true;
        }));
    EXPECT_TRUE(read(*backend.backend, std::move(request), std::move(observation.match_meta)));
}

TEST(KVCMLocalTest, MatchFailurePreservesLocalPrefix) {
    auto environment    = makeBackendEnvironment("kvcm_storage_backend_match_fallback");
    auto client_wrapper = std::make_shared<MockClientWrapper>();

    EXPECT_CALL(*client_wrapper, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), client_wrapper);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));

    EXPECT_CALL(*client_wrapper,
                match(_, _, kv_cache_manager::QueryType::QT_PREFIX_MATCH, std::vector<int64_t>({101, 102}), _, _))
        .WillOnce(Return(std::make_pair(false, kv_cache_manager::Locations{})));
    auto observation =
        match(*backend.backend, makeStorageRequest(environment, /*keys=*/{101, 102}, /*local_matched_blocks=*/1));
    EXPECT_TRUE(observation.success);
    EXPECT_EQ(observation.matched_blocks_num, 1u);
    EXPECT_EQ(observation.match_meta, nullptr);
}

TEST(KVCMLocalTest, InvalidMatchLocationReportsNoHitAndDoesNotDispatchPayload) {
    auto environment    = makeBackendEnvironment("kvcm_storage_backend_invalid_location");
    auto client_wrapper = std::make_shared<MockClientWrapper>();

    EXPECT_CALL(*client_wrapper, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), client_wrapper);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));

    kv_cache_manager::Locations locations{
        kv_cache_manager::Location{kv_cache_manager::LocationSpecUnit{"unknown_spec", "read_uri"}}};
    EXPECT_CALL(*client_wrapper, match(_, _, _, _, _, _)).WillOnce(Return(std::make_pair(true, locations)));
    auto observation = match(*backend.backend, makeStorageRequest(environment));
    EXPECT_CALL(*client_wrapper, loadKvCachesForTag("default", _, _, _)).Times(0);
    EXPECT_FALSE(observation.success);
    EXPECT_EQ(observation.matched_blocks_num, 0u);
    EXPECT_EQ(observation.match_meta, nullptr);
}

TEST(KVCMLocalTest, TP2DuplicateRankMatchReportsNoHitAndDoesNotDispatchPayload) {
    auto environment    = makeBackendEnvironment("kvcm_storage_backend_tp2_duplicate_rank");
    auto client_wrapper = std::make_shared<MockClientWrapper>();

    ParallelismConfig parallelism_config;
    parallelism_config.tp_size    = 2;
    parallelism_config.tp_rank    = 0;
    parallelism_config.local_rank = 0;

    EXPECT_CALL(*client_wrapper, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    EXPECT_CALL(*client_wrapper, loadKvCachesForTag("default", _, _, _)).Times(0);
    auto broadcast_manager =
        std::make_shared<BroadcastManager>(std::vector<std::string>{"unused-rank-0", "unused-rank-1"});
    auto backend = makeBackend(environment, parallelism_config, client_wrapper, std::move(broadcast_manager));
    ASSERT_TRUE(backend->init(environment.cache_config.topologyPtr(),
                              environment.pools_by_tag,
                              [&](int layer_id, const std::string&, int block_id) {
                                  return environment.device_pool->convertIndexToBuffer(layer_id, block_id);
                              }));

    kv_cache_manager::Locations locations{kv_cache_manager::Location{
        kv_cache_manager::LocationSpecUnit{"tp0_Fdefault", "rank0_uri"},
        kv_cache_manager::LocationSpecUnit{"tp0_Fdefault", "duplicate_rank0_uri"},
    }};
    EXPECT_CALL(*client_wrapper, match(_, _, _, _, _, _)).WillOnce(Return(std::make_pair(true, locations)));
    const auto observation = match(*backend.backend, makeStorageRequest(environment));
    EXPECT_FALSE(observation.success);
    EXPECT_EQ(observation.matched_blocks_num, 0u);
    EXPECT_EQ(observation.match_meta, nullptr);
}

TEST(KVCMLocalTest, PayloadReadFailurePropagatesToCompletion) {
    auto environment    = makeBackendEnvironment("kvcm_storage_backend_read_failure");
    auto client_wrapper = std::make_shared<MockClientWrapper>();

    EXPECT_CALL(*client_wrapper, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), client_wrapper);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));

    kv_cache_manager::Locations locations{
        kv_cache_manager::Location{kv_cache_manager::LocationSpecUnit{"tp0_Fdefault", "read_uri"}}};
    EXPECT_CALL(*client_wrapper, match(_, _, _, _, _, _)).WillOnce(Return(std::make_pair(true, locations)));
    auto observation = match(*backend.backend, makeStorageRequest(environment));
    ASSERT_TRUE(observation.success);
    ASSERT_NE(observation.match_meta, nullptr);

    const auto ref_count_before = environment.device_pool->refCount(environment.block_id);
    EXPECT_CALL(*client_wrapper, loadKvCachesForTag("default", kv_cache_manager::UriStrVec{"read_uri"}, _, _))
        .WillOnce(Return(false));
    EXPECT_FALSE(read(*backend.backend, makeStorageRequest(environment), std::move(observation.match_meta)));
    EXPECT_EQ(environment.device_pool->refCount(environment.block_id), ref_count_before);
    EXPECT_FALSE(environment.device_pool->isFrozen(environment.block_id));
    EXPECT_EQ(environment.device_pool->treeRefCount(environment.block_id), 0u);
}

TEST(KVCMLocalTest, PayloadReadTimeoutFreezesOnlyDestinationBlock) {
    autil::EnvGuard quarantine_abort_ratio("RTP_LLM_REMOTE_CACHE_QUARANTINE_ABORT_RATIO", "0.125");
    auto environment    = makeBackendEnvironment("kvcm_storage_backend_read_timeout");
    auto client_wrapper = std::make_shared<MockClientWrapper>();
    auto metrics_reporter =
        std::make_shared<kmonitor::MetricsReporter>("", "", kmonitor::MetricsTags{});

    EXPECT_CALL(*client_wrapper, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    auto backend = makeBackend(environment,
                               singleRankConfig(),
                               client_wrapper,
                               nullptr,
                               metrics_reporter,
                               /*gdr_enabled=*/false);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));
    auto* quarantine_metrics = metrics_reporter->getMetricsGroup<RtpLLMRemoteCacheQuarantineMetrics>();
    ASSERT_NE(quarantine_metrics, nullptr);
    EXPECT_DOUBLE_EQ(snapshotGauge(quarantine_metrics->quarantined_lease_count_metric), 0);
    EXPECT_DOUBLE_EQ(snapshotGauge(quarantine_metrics->quarantined_block_count_metric), 0);
    EXPECT_DOUBLE_EQ(snapshotGauge(quarantine_metrics->quarantined_block_ratio_metric), 0);

    kv_cache_manager::Locations locations{
        kv_cache_manager::Location{kv_cache_manager::LocationSpecUnit{"tp0_Fdefault", "read_uri"}}};
    EXPECT_CALL(*client_wrapper, match(_, _, _, _, _, _)).WillOnce(Return(std::make_pair(true, locations)));
    auto observation = match(*backend.backend, makeStorageRequest(environment));
    ASSERT_TRUE(observation.success);
    ASSERT_NE(observation.match_meta, nullptr);

    const auto ref_count_before = environment.device_pool->refCount(environment.block_id);
    client_wrapper->load_status = kv_cache_manager::ClientErrorCode::ER_SDK_TIMEOUT;
    EXPECT_CALL(*client_wrapper, loadKvCachesForTag(_, _, _, _)).Times(0);
    const auto error =
        readResult(*backend.backend, makeStorageRequest(environment), std::move(observation.match_meta));
    EXPECT_EQ(error.code(), ErrorCode::DEADLINE_EXCEEDED);
    EXPECT_EQ(environment.device_pool->refCount(environment.block_id), ref_count_before + 1);
    EXPECT_EQ(environment.device_pool->treeRefCount(environment.block_id), 1u);
    EXPECT_TRUE(environment.device_pool->isFrozen(environment.block_id));
    EXPECT_EQ(environment.device_pool->referencedBlocksNum(BlockTreeRefType::FROZEN), 1u);

    uint64_t quarantine_generation;
    {
        std::lock_guard<std::mutex> lock(backend.backend->quarantine_mutex_);
        quarantine_generation = backend.backend->quarantine_generation_;
    }

    ScopedReferencedBlocks healthy_blocks(environment.device_pool, 1);
    auto                   healthy_request =
        makeStorageRequest(environment, /*keys=*/{202}, /*local_matched_blocks=*/0, /*block_ids=*/healthy_blocks.get());

    kv_cache_manager::Locations healthy_locations{
        kv_cache_manager::Location{kv_cache_manager::LocationSpecUnit{"tp0_Fdefault", "healthy_read_uri"}}};
    EXPECT_CALL(*client_wrapper, match(_, _, _, _, _, _)).WillOnce(Return(std::make_pair(true, healthy_locations)));
    auto healthy_observation = match(*backend.backend, healthy_request);
    ASSERT_TRUE(healthy_observation.success);
    ASSERT_NE(healthy_observation.match_meta, nullptr);

    client_wrapper->load_status = kv_cache_manager::ClientErrorCode::ER_OK;
    EXPECT_TRUE(read(*backend.backend, healthy_request, std::move(healthy_observation.match_meta)));
    EXPECT_FALSE(environment.device_pool->isFrozen(healthy_blocks.get().front()));

    kv_cache_manager::WriteLocation empty_write_location;
    EXPECT_CALL(*client_wrapper, getWriteLocation(_, _, _, _, _, _, 0))
        .WillOnce(Return(std::make_pair(true, empty_write_location)));
    auto write_task = backend.backend->prepareWrite(healthy_request);
    ASSERT_TRUE(write_task);
    EXPECT_TRUE(backend.backend->write(std::move(write_task)));
    ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));

    EXPECT_EQ(environment.device_pool->refCount(environment.block_id), ref_count_before + 1);
    // Snapshot consumes gauge samples; inspect the unchanged quarantine state only once.
    EXPECT_DOUBLE_EQ(snapshotGauge(quarantine_metrics->quarantined_lease_count_metric), 1);
    EXPECT_DOUBLE_EQ(snapshotGauge(quarantine_metrics->quarantined_block_count_metric), 1);
    EXPECT_DOUBLE_EQ(snapshotGauge(quarantine_metrics->quarantined_block_ratio_metric), 0.125);
    // Healthy I/O must not change the quarantine generation or state.
    {
        std::lock_guard<std::mutex> lock(backend.backend->quarantine_mutex_);
        EXPECT_EQ(backend.backend->quarantine_generation_, quarantine_generation);
        EXPECT_EQ(backend.backend->quarantined_task_count_, 1u);
        EXPECT_EQ(backend.backend->quarantined_block_count_, 1u);
    }

    backend.backend->shutdown();
    EXPECT_EQ(environment.device_pool->refCount(environment.block_id), ref_count_before + 1);
    EXPECT_TRUE(environment.device_pool->isFrozen(environment.block_id));
    EXPECT_EQ(environment.device_pool->referencedBlocksNum(BlockTreeRefType::FROZEN), 1u);
    EXPECT_DOUBLE_EQ(snapshotGauge(quarantine_metrics->quarantined_lease_count_metric), 0);
    EXPECT_DOUBLE_EQ(snapshotGauge(quarantine_metrics->quarantined_block_count_metric), 0);
    EXPECT_DOUBLE_EQ(snapshotGauge(quarantine_metrics->quarantined_block_ratio_metric), 0);
}

TEST(KVCMLocalTest, RankZeroAbortsWhenQuarantinedBlocksExceedConfiguredRatio) {
    // Earlier cases initialize CUDA. Forking that runtime cannot initialize a
    // new GPU pool; exec a fresh test process so the assertion reaches the
    // quarantine capacity guard rather than failing in cudaGetDevice.
    struct RestoreDeathTestStyle {
        std::string prior = ::testing::GTEST_FLAG(death_test_style);
        ~RestoreDeathTestStyle() {
            ::testing::GTEST_FLAG(death_test_style) = prior;
        }
    } restore_death_test_style;
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    ASSERT_EXIT(
        ([]() {
            (void)dup2(STDERR_FILENO, STDOUT_FILENO);
            autil::EnvGuard feature_off("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "0");
            autil::EnvGuard console_log("FT_SERVER_TEST", "1");
            if (!initLogger()) {
                _exit(5);
            }
            autil::EnvGuard quarantine_abort_ratio("RTP_LLM_REMOTE_CACHE_QUARANTINE_ABORT_RATIO", "0.1");
            auto            environment    = makeBackendEnvironment("kvcm_storage_backend_quarantine_abort");
            auto            client_wrapper = std::make_shared<MockClientWrapper>();

            EXPECT_CALL(*client_wrapper, initForPools(_, _, _, _)).WillOnce(Return(true));
            auto backend = makeBackend(environment,
                                       singleRankConfig(),
                                       client_wrapper,
                                       nullptr,
                                       nullptr,
                                       /*gdr_enabled=*/false);
            if (!initSingleRank(*backend.backend, environment)) {
                _exit(2);
            }

            kv_cache_manager::Locations locations{
                kv_cache_manager::Location{kv_cache_manager::LocationSpecUnit{"tp0_Fdefault", "read_uri"}}};
            EXPECT_CALL(*client_wrapper, match(_, _, _, _, _, _)).WillOnce(Return(std::make_pair(true, locations)));
            auto observation = match(*backend.backend, makeStorageRequest(environment));
            if (!observation.success || observation.match_meta == nullptr) {
                _exit(3);
            }

            client_wrapper->load_status = kv_cache_manager::ClientErrorCode::ER_SDK_TIMEOUT;
            (void)readResult(*backend.backend, makeStorageRequest(environment), std::move(observation.match_meta));
            _exit(4);
        }()),
        ::testing::KilledBySignal(SIGABRT),
        "KVCM quarantined block capacity exceeded on rank 0");
}

}  // namespace
}  // namespace rtp_llm
