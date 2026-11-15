#include "rtp_llm/cpp/utils/DecodePublicationTest.h"
#include "rtp_llm/cpp/utils/RemoteCacheConfig.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test/KVCMMockTestBase.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/StorageBackendExecutor.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <csignal>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
#include <unistd.h>

#include "kmonitor/client/MetricsReporter.h"
#include "kmonitor/client/KMonitorFactory.h"
#include "kmonitor/client/core/MetricsData.h"
#include "rtp_llm/cpp/cache/DeviceBlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/GroupPolicy.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/events/KVCacheEventPublisher.h"
#include "rtp_llm/cpp/cache/events/KVCacheEventReporter.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/utils/AsyncProfilingScope.h"
#include "rtp_llm/cpp/utils/AsyncProfileTrace.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/SdkBufferSafety.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/DecodePublicationAccounting.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BoundedThreadTestUtils.h"

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
            "extra_info":"{\"rtp_remote_cache_enabled\":)")
                   + (enabled.load() ? "true" : "false") + R"(}"})";
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
    std::mutex                                       mutex_;
    std::condition_variable                          cv_;
    std::vector<std::pair<std::string, std::string>> requests_;
};

TEST(KVCMLocalTest, ManagerKeepsRemoteControlIndependentOfHbmPublicationSettings) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "0");
    ::rtp_llm::test::withDecodePublicationStartup([&] {
    for (int mode = 0; mode < 10; ++mode) {
        SCOPED_TRACE(mode);
        auto        environment = makeBackendEnvironment("kvcm_control_only_" + std::to_string(mode));
        auto        client      = std::make_shared<MockClientWrapper>();
        std::string expected_registration;
        EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Invoke([&](const auto& configs, auto, auto&, auto&) {
            expected_registration = configs.at("")->registerInstanceRequest();
            return true;
        }));
        EXPECT_CALL(*client, shutdown()).Times(1);
        KVCacheConfig config;
        config.enable_remote_cache                 = true;
        config.kvcm_server_address                 = "outer-rpc";
        config.kv_cache_event_host_ip_port         = "127.0.0.1:19999";
        config.kv_cache_event_snapshot_interval_ms = mode == 4 ? 0 : 600000;
        config.kv_cache_event_publisher_type = (mode == 0 || mode == 9) ? "none" :
            (mode == 8 ? "" : (mode == 6 ? "unknown" : "kvcm"));
        auto parallel = singleRankConfig();
        parallel.role_type = mode == 9 ? RoleType::DECODE : RoleType::PREFILL;
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
            auto custom = std::make_shared<kvcm::KVCMConfig>(
                false,
                "",
                8,
                "custom-group",
                "custom-remote",
                std::vector<std::string>{"rpc-a", "rpc-b"},
                infos,
                std::make_shared<kvcm::MetaChannelConfig>(),
                std::make_shared<kvcm::SdkWrapperConfig>(),
                groups,
                kvcm::ModelDeployment(runtime.model_name, "fp16", false, 1, 1, 1, "extra", "user-data"));
            config.kvcm_client_config = autil::legacy::ToJsonString(kvcm::ClientWrapper::ConfigMap{{"alias", custom}});
        }
        auto backend = std::make_shared<KVCMStorageBackend>(
            environment.cache_config, config, runtime, parallel,
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
            auto groups                               = manager_config.topology().groups();
            groups.front().policy.enable_prefix_reuse = false;
            auto layers                               = manager_config.topology().layers();
            manager_config.setTopology(std::move(groups), std::move(layers));
        }
        KVCacheManager manager(manager_config, false, nullptr, config, parallel, runtime);
        auto full = std::make_shared<FullGroupSet>(
            std::vector<DeviceBlockPoolPtr>{environment.device_pool}, nullptr, nullptr);
        full->initialize(0, manager_config.topologyPtr(), {"default"});
        manager.block_tree_cache_                   = block_tree_cache_test::makeBlockTreeCacheForTest({full});
        manager.block_tree_cache_->storage_backend_ = backend;
        auto reporter                               = std::make_shared<ControlOnlyReporter>();
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
    });
}

TEST(KVCMLocalTest, Dsv41PublisherUsesEffectiveBackendFullSpecMerge) {
    for (const std::string model_name : {"DeepSeekV41", "DeepSeekV41DSpark"}) {
        for (const bool compact_indexer : {false, true}) {
            for (const bool tair_backend : {false, true}) {
                SCOPED_TRACE(model_name);
                SCOPED_TRACE(compact_indexer);
                SCOPED_TRACE(tair_backend);
                std::vector<std::string> tags{"hca_kv", "csa_kv", "indexer_kv"};
                if (compact_indexer) {
                    tags.push_back("indexer_kv_c2");
                }
                BackendEnvironment environment;
                environment.cache_config.dtype              = DataType::TYPE_FP16;
                environment.cache_config.layer_num          = tags.size();
                environment.cache_config.seq_size_per_block = 8;
                std::vector<KVCacheSpecPtr>   specs;
                std::vector<std::vector<int>> layers;
                for (size_t index = 0; index < tags.size(); ++index) {
                    specs.push_back(test::makeMhaSpec(
                        tags[index], 8, DataType::TYPE_FP16, 1, tags[index] == "indexer_kv_c2" ? 1 : 2));
                    layers.push_back({static_cast<int>(index)});
                }
                environment.cache_config.fromGroupedSpecs(
                    specs, layers, std::vector<CacheGroupType>(tags.size(), CacheGroupType::FULL), tags);
                environment.cache_config.finalizeBlockNums(8, RuntimeConfig{});
                initializeEnvironmentPools(environment, "kvcm_compact_indexer_control", 8);

                auto                                  client = std::make_shared<MockClientWrapper>();
                kvcm::KVCMConfig::LocationSpecInfoMap actual_payload_specs;
                EXPECT_CALL(*client, initForPools(_, _, _, _))
                    .WillOnce(Invoke([&](const auto& configs, auto, auto&, auto&) {
                        const auto fields =
                            autil::legacy::AnyCast<autil::legacy::json::JsonMap>(autil::legacy::ToJson(configs.at("")));
                        autil::legacy::FromJson(actual_payload_specs, fields.at("location_spec_infos"));
                        return true;
                    }));
                EXPECT_CALL(*client, shutdown()).Times(1);
                KVCacheConfig config;
                config.enable_remote_cache           = true;
                config.kvcm_server_address           = "outer-rpc";
                config.kvcm_model_sdk_config         = tair_backend ? R"([{"type":"pace"}])" : R"([{"type":"file"}])";
                config.reuse_cache                   = true;
                config.enable_device_cache           = true;
                config.kv_cache_event_publisher_type = "kvcm";
                config.kv_cache_event_instance_id    = "hbm-event-instance";
                config.kv_cache_event_host_ip_port   = "127.0.0.1:19999";
                config.kv_cache_event_snapshot_interval_ms = 600000;
                RuntimeConfig runtime;
                runtime.model_name = model_name;
                auto backend       = std::make_shared<KVCMStorageBackend>(environment.cache_config,
                                                                    config,
                                                                    runtime,
                                                                    singleRankConfig(),
                                                                    SpeculativeExecutionConfig{},
                                                                    nullptr,
                                                                    false,
                                                                    nullptr,
                                                                    client);
                ASSERT_TRUE(initSingleRank(*backend, environment));
                ASSERT_TRUE(backend->controlConfig().has_value());
                EXPECT_EQ(backend->controlConfig()->merge_full_specs, tair_backend);
                std::set<std::string> payload_names;
                for (const auto& [name, bytes] : actual_payload_specs) {
                    EXPECT_GT(bytes, 0);
                    payload_names.insert(name);
                }
                std::set<std::string> expected_payload{"F0_TP0", "F1_TP0", "F2_TP0"};
                if (compact_indexer) {
                    expected_payload.insert("F2C2_TP0");
                }
                EXPECT_EQ(payload_names, tair_backend ? std::set<std::string>{"F_TP0"} : expected_payload);

                KVCacheManager manager(environment.cache_config, false, nullptr, config, singleRankConfig(), runtime);
                std::vector<DeviceBlockPoolPtr> pools;
                for (const auto& tag : tags) {
                    pools.push_back(environment.pools_by_tag.at(tag));
                }
                auto full = std::make_shared<FullGroupSet>(std::move(pools), nullptr, nullptr);
                full->initialize(0, environment.cache_config.topologyPtr(), tags);
                manager.block_tree_cache_                   = block_tree_cache_test::makeBlockTreeCacheForTest({full});
                manager.block_tree_cache_->storage_backend_ = backend;
                auto reporter                               = std::make_shared<ControlOnlyReporter>();
                reporter->enabled.store(true);
                manager.initCacheEventPublisher(reporter);
                ASSERT_NE(manager.cache_event_publisher_, nullptr);
                ASSERT_TRUE(reporter->waitFor("EVENT_NODE_REGISTER"));
                EXPECT_EQ(manager.block_tree_cache_->tree()->publication_groups_.size(),
                          tair_backend ? 1 : tags.size());
                manager.stopCacheEventPublisher();
                std::set<std::string> event_names;
                for (const auto& [route, body] : reporter->requests()) {
                    if (route != "/api/registerInstance") {
                        continue;
                    }
                    autil::legacy::json::JsonMap registration;
                    autil::legacy::FromJsonString(registration, body);
                    const auto infos =
                        autil::legacy::AnyCast<autil::legacy::json::JsonArray>(registration.at("location_spec_infos"));
                    for (const auto& info : infos) {
                        const auto fields = autil::legacy::AnyCast<autil::legacy::json::JsonMap>(info);
                        event_names.insert(autil::legacy::AnyCast<std::string>(fields.at("name")));
                    }
                }
                std::set<std::string> expected_events{"F0", "F1", "F2"};
                if (compact_indexer) {
                    expected_events.insert("F2C2");
                }
                EXPECT_EQ(event_names, tair_backend ? std::set<std::string>{"F"} : expected_events);
            }
        }
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
    const auto [failed_save_ok, failed_actual_uris] = wrapper.saveKvCachesForTag("default", failed_uris, buffers);
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

TEST(KVCMLocalTest, PrefillAndPdfusionIgnoreDecodeOnlyRestrictionsWithEitherSwitch) {
    for (const bool enabled : {false, true}) {
    for (const auto role : {RoleType::PREFILL, RoleType::PDFUSION}) {
    SCOPED_TRACE(::testing::PrintToString(std::make_pair(enabled, role)));
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", enabled ? "1" : "0");
    ::rtp_llm::test::withDecodePublicationStartup([&] {
    autil::EnvGuard budget("RTP_LLM_REMOTE_CACHE_DECODE_PENDING_MIB", "invalid");
    autil::EnvGuard jobs("RTP_LLM_REMOTE_CACHE_DECODE_PENDING_JOBS", "invalid");
    auto            environment = makeBackendEnvironment("kvcm_disabled_decode_preserves_prefill");
    auto            client      = std::make_shared<MockClientWrapper>();
    EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client, shutdown()).Times(1);
    KVCacheConfig config;
    config.kvcm_server_address   = "unused-test-address";
    config.kvcm_model_sdk_config = R"([{"type":"pace_ssd"}])";
    // Deliberately outside D SSD/GDR pairing restrictions: P delegates
    // ordinary configuration to the original SDK path with either switch.
    config.kvcm_read_backend_type = 0;
    RuntimeConfig runtime;
    runtime.model_name = "kvcm_disabled_decode";
    auto parallel = singleRankConfig();
    parallel.role_type = role;
    BackendHandle backend(std::make_unique<KVCMStorageBackend>(environment.cache_config,
                                                               config,
                                                               runtime,
                                                               parallel,
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
    });
    }
    }
}

TEST(KVCMLocalTest, InvalidDecodeSwitchPreservesPrefillTrafficButStillRejectsDecode) {
    for (const char* value : {"", "true", " 1", "2"}) {
        autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", value);
        ASSERT_EQ(setenv("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", value, 1), 0); // Empty is present, not unset.
        test::withDecodePublicationStartup([&] {
            EXPECT_FALSE(remoteCacheDecodePublicationEnabled());
            EXPECT_FALSE(remoteCacheDecodePublicationConfigValid());
            // P then D share the same cached invalid setting; P must not
            // rewrite it to valid/off or make a later D init succeed.
            for (auto role : {RoleType::PREFILL, RoleType::PDFUSION, RoleType::DECODE}) {
                SCOPED_TRACE(value);
                SCOPED_TRACE(role);
                auto environment = makeBackendEnvironment("role_scoped_invalid_decode_toggle");
                auto client = std::make_shared<MockClientWrapper>();
                const bool decode = role == RoleType::DECODE;
                EXPECT_CALL(*client, initForPools(_, _, _, _)).Times(decode ? 0 : 1).WillRepeatedly(Return(true));
                EXPECT_CALL(*client, shutdown()).Times(1);
                auto parallel = singleRankConfig();
                parallel.role_type = role;
                parallel.prefill_cp_config.prefill_cp_size = 1;
                auto backend = makeBackend(environment, parallel, client);
                ASSERT_EQ(initSingleRank(*backend.backend, environment), !decode);
                if (!decode) {
                    kv_cache_manager::Locations locations{
                        kv_cache_manager::Location{{"tp0_Fdefault", "legacy_read_uri"}}};
                    EXPECT_CALL(*client, match(_, _, _, std::vector<int64_t>{101}, _, _))
                        .WillOnce(Return(std::make_pair(true, locations)));
                    auto matched = match(*backend.backend, makeStorageRequest(environment));
                    ASSERT_TRUE(matched.success);
                    ASSERT_EQ(matched.matched_blocks_num, 1u);
                    EXPECT_CALL(*client, loadKvCachesForTag("default", kv_cache_manager::UriStrVec{"legacy_read_uri"}, _, _))
                        .WillOnce(Return(true));
                    EXPECT_TRUE(read(*backend.backend, makeStorageRequest(environment), std::move(matched.match_meta)));
                }
                EXPECT_FALSE(remoteCacheDecodePublicationEnabled());
                EXPECT_FALSE(remoteCacheDecodePublicationConfigValid());
            }
        });
    }
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
    auto environment    = makeBackendEnvironment("kvcm_storage_backend_gdr_span");
    auto client_wrapper = std::make_shared<MockClientWrapper>();

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

TEST(KVCMLocalTest, PrefillKeepsHostFallbackWhileEnabledDecodeRequiresNativeGpuPools) {
    autil::EnvGuard async("RTP_LLM_STREAM_ASYNC", "0");
    for (const bool decode : {false, true}) {
    for (const bool enabled : {false, true}) {
        SCOPED_TRACE(::testing::PrintToString(std::make_pair(decode, enabled)));
        const bool reject_host = decode && enabled;
        autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", enabled ? "1" : "0");
        ::rtp_llm::test::withDecodePublicationStartup([&] {
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
        if (reject_host) {
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
        // P always keeps the original fallback. Only enabled D requires
        // native GPU pools for its source projection.
        KVCacheConfig config;
        config.kvcm_server_address                 = "unused-test-address";
        config.kvcm_instance_id_salt               = "host-pool-fallback-contract";
        config.kvcm_model_sdk_config               = R"([{"type":"pace"}])";
        config.kvcm_read_backend_type              = 3;
        auto parallel                              = singleRankConfig();
        parallel.role_type                         = decode ? RoleType::DECODE : RoleType::PREFILL;
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
        EXPECT_EQ(initSingleRank(*backend.backend, environment), !reject_host);
        });
    }
    }
}

TEST(KVCMLocalTest, DecodePublicationParsesSdkOptionsAndKeepsBackendContracts) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1");
    ::rtp_llm::test::withDecodePublicationStartup([&] {
    struct Case { std::string json; int read_backend; bool gdr; bool valid; };
    const std::vector<Case> cases{
        {R"([{"type":"pace","sdk_log_file_path":"logs/pace client.log","sdk_log_level":"INFO"}])", 0, true, true},
        {R"( [ { "sdk_log_level" : "INFO", "type" : "pace", "sdk_log_file_path" : "logs/pace client.log" } ] )", 3, true, true},
        {R"([{"sdk_log_level":"INFO","type":"pace_ssd","sdk_log_file_path":"logs/pace client.log"}])", 9, true, true},
        {R"([{"type":"pace_ssd"}])", 0, true, false},
        {R"([{"type":"pace"}])", 9, true, false},
        {R"([{"type":"pace_ssd"}])", 9, false, false},
        {R"([{"type":"pace"}])", 0, false, false},
        {R"([{"type":"pace"}])", 0, true, true},
        {R"(null)", 0, true, false},
        {R"([])", 0, true, false},
        {R"({"type":"pace"})", 0, true, false},
        {R"([1])", 0, true, false},
        {R"([{}])", 0, true, false},
        {R"([{"type":1}])", 0, true, false},
        {R"([{"type":"pace_typo"}])", 0, true, false},
        {R"([{"type":"pace"},{"type":"pace_ssd"}])", 0, true, false},
        {R"([{"type":"pace"},{"type":"local"}])", 0, true, false},
        {R"([{"type":"pace"},{"type":"pace"}])", 0, true, false},
        {R"([{"type":"pace"})", 0, true, false},
    };
    for (const auto& value : cases) {
        SCOPED_TRACE(value.json);
        auto environment = makeBackendEnvironment("publication_sdk_options");
        auto client = std::make_shared<MockClientWrapper>();
        EXPECT_CALL(*client, shutdown()).Times(1);
        EXPECT_CALL(*client, initForPools(_, _, _, _)).Times(value.valid ? 1 : 0)
            .WillRepeatedly(Invoke([&](const auto& configs, auto, const auto&, const auto&) {
                // Observe the effective SDK config passed to real client creation,
                // not just a standalone JSON parse or the admission helper.
                const auto fields = autil::legacy::AnyCast<autil::legacy::json::JsonMap>(
                    autil::legacy::ToJson(configs.at("")));
                const auto sdk_fields = autil::legacy::AnyCast<autil::legacy::json::JsonMap>(
                    fields.at("sdk_config"));
                kvcm::SdkWrapperConfig expected;
                expected.parseBackendConfigs(value.json);
                const auto actual = autil::legacy::ToJsonString(sdk_fields.at("sdk_backend_configs"));
                EXPECT_EQ(actual, autil::legacy::ToJsonString(expected.sdk_backend_configs()));
                if (value.json.find("sdk_log_file_path") != std::string::npos) {
                    const auto backends = autil::legacy::AnyCast<autil::legacy::json::JsonArray>(
                        sdk_fields.at("sdk_backend_configs"));
                    EXPECT_EQ(backends.size(), 1u);
                    if (backends.size() != 1u) return false;
                    const auto fields = autil::legacy::AnyCast<autil::legacy::json::JsonMap>(backends.front());
                    EXPECT_EQ(autil::legacy::AnyCast<std::string>(fields.at("sdk_log_file_path")),
                              "logs/pace client.log");
                    EXPECT_EQ(autil::legacy::AnyCast<std::string>(fields.at("sdk_log_level")), "INFO");
                }
                return true;
            }));
        KVCacheConfig config;
        config.kvcm_model_sdk_config = value.json;
        config.kvcm_read_backend_type = value.read_backend;
        config.kvcm_instance_id_salt = "sdk-options-contract";
        auto parallel = singleRankConfig();
        parallel.role_type = RoleType::DECODE;
        parallel.prefill_cp_config.prefill_cp_size = 1;
        auto backend = makeBackend(environment, parallel, client, nullptr, nullptr, value.gdr, config);
        bool initialized = false;
        EXPECT_NO_THROW(initialized = initSingleRank(*backend.backend, environment));
        EXPECT_EQ(initialized, value.valid);
    }
    });
}

TEST(KVCMLocalTest, OptionalSaltPreservesPrefillDecodeIdentityAndLayoutIsolation) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1");
    ::rtp_llm::test::withDecodePublicationStartup([&] {
    const auto instance = [](RoleType role, int cp, const std::string& salt) {
        auto environment = makeBackendEnvironment("publication_optional_salt");
        auto client = std::make_shared<MockClientWrapper>();
        std::string id;
        EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Invoke(
            [&](const auto& configs, auto, const auto&, const auto&) {
                id = configs.at("")->instance_id();
                return true;
            }));
        EXPECT_CALL(*client, shutdown()).Times(1);
        KVCacheConfig config;
        config.kvcm_model_sdk_config = R"([{"type":"pace","sdk_log_level":"INFO"}])";
        config.kvcm_instance_id_salt = salt;
        auto parallel = singleRankConfig();
        parallel.role_type = role;
        parallel.prefill_cp_config.prefill_cp_size = cp;
        parallel.prefill_cp_config.kv_cache_sharded = true;
        auto backend = makeBackend(environment, parallel, client, nullptr, nullptr, true, config);
        EXPECT_TRUE(initSingleRank(*backend.backend, environment));
        EXPECT_FALSE(id.empty());
        backend.backend->shutdown();
        return id;
    };
    const auto p = instance(RoleType::PREFILL, 1, "");
    const auto d = instance(RoleType::DECODE, 1, "");
    EXPECT_EQ(p, d);
    EXPECT_EQ(instance(RoleType::PREFILL, 1, "explicit-salt"), "explicit-salt_" + p);
    EXPECT_EQ(instance(RoleType::DECODE, 1, "explicit-salt"), "explicit-salt_" + d);
    // CP2 is an identity-generation check, not multi-rank transfer acceptance.
    EXPECT_NE(instance(RoleType::DECODE, 2, ""), d);
    EXPECT_NE(instance(RoleType::DECODE, 2, "explicit-salt"), "explicit-salt_" + d);
    });
}

TEST(KVCMLocalTest, PrefillIdentityAndSpecsUseOwnGeometryWhileOnlyDecodeProjects) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1");
    ::rtp_llm::test::withDecodePublicationStartup([&] {
        // Config-generation coverage only: these mocks perform no collective,
        // SDK transfer or model execution, and do not certify runtime topologies.
        const auto instance = [](RoleType role, int tp, int dp, int target_cp) {
            auto environment = makeBackendEnvironment("role_scoped_geometry");
            auto client = std::make_shared<MockClientWrapper>();
            const int wire_tp = role == RoleType::DECODE ? target_cp : tp;
            std::string identity;
            EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Invoke(
                [&](const auto& configs, auto, const auto& registrations, const auto&) {
                    identity = configs.at("")->instance_id();
                    const auto fields = autil::legacy::AnyCast<autil::legacy::json::JsonMap>(
                        autil::legacy::ToJson(configs.at("")));
                    kvcm::KVCMConfig::LocationSpecInfoMap infos;
                    autil::legacy::FromJson(infos, fields.at("location_spec_infos"));
                    EXPECT_EQ(infos.size(), static_cast<size_t>(wire_tp));
                    for (int rank = 0; rank < wire_tp; ++rank)
                        EXPECT_EQ(infos.count("tp" + std::to_string(rank) + "_Fdefault"), 1u);
                    EXPECT_EQ(registrations.size(), role == RoleType::DECODE ? static_cast<size_t>(target_cp) : 1u);
                    return true;
                }));
            EXPECT_CALL(*client, shutdown()).Times(1);
            KVCacheConfig config;
            config.kvcm_model_sdk_config = R"([{"type":"pace"}])";
            auto parallel = singleRankConfig();
            parallel.role_type = role;
            parallel.tp_size = tp;
            parallel.dp_size = dp;
            parallel.prefill_cp_config.prefill_cp_size = target_cp;
            parallel.prefill_cp_config.kv_cache_sharded = false;
            std::shared_ptr<BroadcastManager> broadcast;
            if (tp > 1)
                broadcast = std::make_shared<BroadcastManager>(std::vector<std::string>(tp, "unused-geometry-peer"));
            auto backend = makeBackend(environment, parallel, client, std::move(broadcast), nullptr, true, config);
            EXPECT_TRUE(initSingleRank(*backend.backend, environment));
            EXPECT_FALSE(identity.empty());
            backend.backend->shutdown();
            return identity;
        };
        const auto p11 = instance(RoleType::PREFILL, 1, 1, 1);
        EXPECT_EQ(instance(RoleType::PREFILL, 1, 1, 2), p11);
        const auto p21 = instance(RoleType::PREFILL, 2, 1, 2);
        EXPECT_EQ(instance(RoleType::PREFILL, 2, 1, 1), p21);
        EXPECT_NE(p21, p11);
        const auto p12 = instance(RoleType::PREFILL, 1, 2, 1);
        EXPECT_NE(p12, p11);
        EXPECT_EQ(instance(RoleType::PREFILL, 1, 2, 2), p12);
        EXPECT_EQ(instance(RoleType::DECODE, 1, 1, 2), p21);
        EXPECT_EQ(instance(RoleType::DECODE, 1, 2, 2), p21);
    });
}

TEST(KVCMLocalTest, MirroredNativeV41DescriptorsGiveMatchingPrefillAndDecodeWireConfigs) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1");
    ::rtp_llm::test::withDecodePublicationStartup([&] {
    createDevice();
    // Test-only mirror of build_v41_kv_cache_spec_descs, paired with the Python
    // native-generator contract test. This does not execute a Python->C++ bridge,
    // collective, model kernel or SDK transfer, or certify runtime topologies.
    const auto model = [](bool bounded, bool draft) {
        ModelConfig config;
        config.model_type = "deepseek_v41";
        config.num_layers = draft ? 3 : 5;
        config.data_type = DataType::TYPE_BF16;
        config.attn_config.head_num = 1;
        config.attn_config.kv_head_num = 1;
        config.attn_config.size_per_head = 512;
        config.attn_config.use_mla = true;
        config.attn_config.tokens_per_block = 256;
        config.attn_config.kernel_tokens_per_block = 128;
        config.attn_config.sliding_window = 128;
        config.cache_min_replay_tokens = bounded ? 128 : 1;
        config.cache_key_hash_seed = bounded ? 0x4453563431494231LL : 0x4453563431494631LL;
        config.hybrid_attention_config.hybrid_attention_types.assign(config.num_layers, HybridAttentionType::NONE);
        auto csa = test::makeDsv4Desc("csa_kv", "compressed_kv", 288, DataType::TYPE_UINT8, 2);
        auto hca = test::makeDsv4Desc("hca_kv", "compressed_kv", 288, DataType::TYPE_UINT8, 1);
        csa.block_stride_bytes_alignment = hca.block_stride_bytes_alignment = 4608;
        auto indexer = test::makeDsv4Desc("indexer_kv", "compressed_kv", 68, DataType::TYPE_UINT8, 1);
        auto state = test::makeDsv4Desc("csa_state", "fixed_state", 1024, DataType::TYPE_FP32);
        state.compression_ratio = 2;
        state.state_ring_overlap = 0;
        auto swa = test::makeDsv4Desc("swa_kv", "sliding_window_kv", 528, DataType::TYPE_UINT8);
        swa.block_stride_bytes_alignment = 16896;
        auto decoder = swa; // Rename after inheriting the native SWA CP policies.
        decoder.tag = "decoder_swa_kv";
        decoder.reuse->enable_prefix_reuse = false;
        for (int layer = 0; layer < config.num_layers; ++layer) {
            std::vector<KVCacheSpecDesc> descs{bounded && (draft || layer >= 4) ? decoder : swa};
            if (!draft && (layer == 1 || layer == 3)) {
                descs.push_back(layer == 1 ? csa : hca);
                descs.push_back(indexer);
                if (layer == 1) descs.push_back(state);
            }
            config.kv_cache_spec_descs.push_back(std::move(descs));
        }
        return config;
    };
    struct Observation {
        bool initialized{false};
        std::string identity, config_json, backend_type;
        kvcm::KVCMConfig::LocationSpecInfoMap infos;
        kvcm::KVCMConfig::LocationSpecGroups groups;
    };
    const auto instance = [&](RoleType role, int cp, bool sharded, bool bounded, bool dspark, int gamma = 5,
                              bool ssd = false, const std::string& custom = {},
                              bool expected_init = true, int decode_dp = 1) {
        auto parallel = singleRankConfig();
        parallel.role_type = role;
        if (role == RoleType::DECODE) parallel.dp_size = decode_dp;
        parallel.tp_size = role == RoleType::PREFILL ? cp : 1;
        parallel.prefill_cp_config.prefill_cp_size = cp;
        parallel.prefill_cp_config.kv_cache_sharded = sharded;
        parallel.prefill_cp_config.method = cp == 1 ? CPRotateMethod::DISABLED :
            (role == RoleType::PREFILL ? CPRotateMethod::ALL_GATHER : CPRotateMethod::PREFILL_CP);
        KVCacheConfig settings;
        settings.kvcm_server_address = "unused-test-address";
        settings.kvcm_client_config = custom;
        settings.kvcm_read_backend_type = ssd ? 9 : 0;
        // A selected custom config must not depend on the unused native JSON.
        settings.kvcm_model_sdk_config = !custom.empty() ? "{invalid-unselected-native-sdk" :
            (ssd ? R"([{"type":"pace_ssd"}])" : R"([{"type":"pace"}])");
        SpeculativeExecutionConfig speculation;
        speculation.type = dspark ? SP_TYPE_DSPARK : SP_TYPE_NONE;
        speculation.gen_num_per_cycle = dspark ? gamma : 0; // Draft layers remain three, independent of gamma.
        const auto target = model(bounded, false);
        const auto draft = model(bounded, true);
        BackendEnvironment environment;
        auto& cache = environment.cache_config;
        cache = CacheConfigCreator::createConfig(target, parallel, settings, speculation, dspark ? &draft : nullptr);
        cache.finalizeBlockNums(4, RuntimeConfig{});
        EXPECT_EQ(cache.cache_key_hash_seed, bounded ? 0x4453563431494231LL : 0x4453563431494631LL);
        EXPECT_EQ(cache.seq_size_per_block, 256u);
        EXPECT_EQ(cache.layer_all_num(), dspark ? 8u : 5u);
        EXPECT_EQ(cache.mtp_sub_configs.size(), dspark ? 1u : 0u);
        if (dspark) {
            EXPECT_EQ(cache.mtp_sub_configs.front()->layer_num, 3u);
        }
        const size_t shards = sharded ? cp : 1;
        // Independent literal wire sizes: two kernel pages per logical FULL
        // block; STATE has 2 (+3 rounded to 6, or +5 rounded to 8) FP32
        // entries, CP-aligned; SWA has 128 (+3/+5 rounded/aligned) rows
        // with 16896-byte stride alignment. The gamma variants below are 3/5.
        const size_t state_full_bytes = dspark ? (gamma == 3 && shards < 4 ? 24576 : 32768) :
                                                (shards == 4 ? 16384 : 8192);
        const size_t state_bytes = state_full_bytes / shards;
        const size_t swa_bytes = (dspark ? 84480 : 67584) / shards;
        const std::map<std::string, size_t> expected_bytes{
            {"csa_kv", 36864}, {"hca_kv", 73728}, {"indexer_kv", 34816},
            {"csa_state", state_bytes}, {"swa_kv", swa_bytes * (bounded ? 4 : (dspark ? 8 : 5))}};
        kvcm::KVCMConfig::LocationSpecInfoMap expected_infos;
        const bool merged = !ssd && (custom.empty() || custom.find("F_TP0") != std::string::npos);
        for (const auto& [tag, bytes] : expected_bytes) {
            const bool full = tag != "swa_kv" && tag != "csa_state";
            const auto& group = cache.topology().group(tag);
            EXPECT_EQ(group.seqSizePerBlock(), full ? 256u : 256u * shards);
            EXPECT_EQ(cache.blockSizeBytesForGroup(tag),
                      bytes * (role == RoleType::DECODE && !full ? shards : 1));
            for (int rank = 0; rank < cp; ++rank)
                expected_infos[kvcm::genLocationSpecName(rank, (full ? "F" : "L") + tag, "DeepSeekV41", merged)] += bytes;
        }
        if (bounded) {
            EXPECT_FALSE(cache.topology().group("decoder_swa_kv").policy.enable_prefix_reuse);
        }
        initializeEnvironmentPools(environment, "v41_native_contract", 4);
        Observation observed;
        auto client = std::make_shared<MockClientWrapper>();
        EXPECT_CALL(*client, initForPools(_, _, _, _)).Times(expected_init ? 1 : 0).WillRepeatedly(Invoke(
            [&](const auto& configs, auto, const auto&, const auto&) {
                observed.identity = configs.at("")->instance_id();
                observed.config_json = autil::legacy::ToJsonString(configs);
                const auto fields = autil::legacy::AnyCast<autil::legacy::json::JsonMap>(
                    autil::legacy::ToJson(configs.at("")));
                autil::legacy::FromJson(observed.infos, fields.at("location_spec_infos"));
                autil::legacy::FromJson(observed.groups, fields.at("location_spec_groups"));
                const auto sdk = autil::legacy::AnyCast<autil::legacy::json::JsonMap>(fields.at("sdk_config"));
                const auto backends = autil::legacy::AnyCast<autil::legacy::json::JsonArray>(sdk.at("sdk_backend_configs"));
                EXPECT_EQ(backends.size(), 1u);
                if (!backends.empty()) {
                    const auto backend = autil::legacy::AnyCast<autil::legacy::json::JsonMap>(backends.front());
                    autil::legacy::FromJson(observed.backend_type, backend.at("type"));
                }
                return true;
            }));
        EXPECT_CALL(*client, shutdown()).Times(1);
        RuntimeConfig runtime;
        runtime.model_name = "DeepSeekV41";
        std::shared_ptr<BroadcastManager> broadcast;
        if (role == RoleType::PREFILL && cp > 1)
            broadcast = std::make_shared<BroadcastManager>(std::vector<std::string>(cp, "unused-contract-peer"));
        BackendHandle backend(std::make_unique<KVCMStorageBackend>(cache, settings, runtime, parallel,
            speculation, std::move(broadcast), true, nullptr, client));
        observed.initialized = initSingleRank(*backend.backend, environment);
        EXPECT_EQ(observed.initialized, expected_init);
        if (!observed.initialized) {
            return observed;
        }
        EXPECT_EQ(observed.backend_type, ssd ? "pace_ssd" : "pace");
        EXPECT_FALSE(observed.identity.empty());
        EXPECT_EQ(observed.infos, expected_infos);
        EXPECT_EQ(observed.groups.count("F"), 1u);
        EXPECT_EQ(observed.groups.count("FL"), 1u);
        EXPECT_EQ(observed.groups.count("F0"), merged ? 0u : 1u);
        EXPECT_EQ(observed.groups.count("F1"), merged ? 0u : 1u);
        EXPECT_EQ(observed.groups.count("F2"), merged ? 0u : 1u);
        EXPECT_EQ(observed.groups.at("L3").size(), static_cast<size_t>(cp));
        EXPECT_EQ(observed.groups.at("L4").size(), static_cast<size_t>(cp));
        if (role == RoleType::PREFILL && cp == 1 && bounded && dspark && merged && custom.empty()) {
            RemoteOperationRequestPB grouped;
            grouped.set_op(REMOTE_OPERATION_READ);
            identifyReadRequest(grouped);
            grouped.set_timeout_ms(1000);
            grouped.add_uris("pace://merged-full");
            grouped.add_uri_block_counts(3);
            for (const auto* tag : {"csa_kv", "hca_kv", "indexer_kv"}) {
                grouped.add_group_tags(tag);
                grouped.add_block_ids(environment.block_id);
            }
            EXPECT_CALL(*client, loadKvCachesForTag("csa_kv", kv_cache_manager::UriStrVec{"pace://merged-full"}, _, _))
                .WillOnce(Invoke([&](const auto&, const auto&, auto& buffers, const auto&) {
                    EXPECT_EQ(buffers.size(), 1u);
                    if (buffers.size() != 1) return false;
                    size_t bytes = 0, previous = 0;
                    for (const auto& iov : buffers.front().iovs) {
                        EXPECT_NE(iov.base, nullptr);
                        EXPECT_GE(iov.size, previous);
                        previous = iov.size;
                        bytes += iov.size;
                    }
                    EXPECT_EQ(bytes, expected_bytes.at("csa_kv") + expected_bytes.at("hca_kv")
                                     + expected_bytes.at("indexer_kv"));
                    return true;
                }));
            RemoteOperationResponsePB response;
            EXPECT_TRUE(backend->execute(grouped, response));
            EXPECT_TRUE(response.payload_buffer_safe());
            for (const uint32_t invalid : {0u, 2u, 4u}) {
                auto malformed = grouped;
                identifyReadRequest(malformed);
                malformed.set_uri_block_counts(0, invalid);
                RemoteOperationResponsePB rejected;
                EXPECT_FALSE(backend->execute(malformed, rejected));
                EXPECT_TRUE(rejected.payload_buffer_safe());
            }
            auto reversed = grouped;
            identifyReadRequest(reversed);
            reversed.set_group_tags(0, "hca_kv");
            reversed.set_group_tags(1, "csa_kv");
            RemoteOperationResponsePB rejected;
            EXPECT_FALSE(backend->execute(reversed, rejected));
            EXPECT_TRUE(rejected.payload_buffer_safe());
        }
        backend.backend->shutdown();
        return observed;
    };
    std::map<std::tuple<bool, int, bool>, std::string> target_only_identities;
    for (bool bounded : {false, true}) for (bool dspark : {false, true}) for (int cp : {1, 2, 4}) {
        std::string replicated_identity;
        for (bool sharded : {false, true}) {
            SCOPED_TRACE(::testing::PrintToString(std::make_tuple(bounded, dspark, cp, sharded)));
            const auto p = instance(RoleType::PREFILL, cp, sharded, bounded, dspark);
            const auto d = instance(RoleType::DECODE, cp, sharded, bounded, dspark);
            EXPECT_EQ(p.identity, d.identity);
            EXPECT_EQ(p.infos, d.infos);
            EXPECT_EQ(p.groups, d.groups);
            const auto key = std::make_tuple(bounded, cp, sharded);
            if (!dspark) target_only_identities.emplace(key, p.identity);
            else EXPECT_NE(p.identity, target_only_identities.at(key));
            if (dspark && cp == 1 && !sharded) {
                const auto p3 = instance(RoleType::PREFILL, cp, sharded, bounded, true, 3);
                const auto d3 = instance(RoleType::DECODE, cp, sharded, bounded, true, 3);
                EXPECT_EQ(p3.identity, d3.identity);
                EXPECT_EQ(p3.infos, d3.infos);
                EXPECT_EQ(p3.groups, d3.groups);
                EXPECT_NE(p3.identity, p.identity); // Original speculation identity retains gamma.
            }
            if (bounded && dspark && cp == 4) {
                // Feed the actual target P config to D TP1/DP4. No model or SDK
                // runs here; this checks the selected config/projection contract.
                const auto d_custom = instance(RoleType::DECODE, cp, sharded, true, true, 5,
                                               false, p.config_json, true, 4);
                EXPECT_EQ(d_custom.identity, p.identity);
                EXPECT_EQ(d_custom.infos, p.infos);
                EXPECT_EQ(d_custom.groups, p.groups);
                const auto p_ssd = instance(RoleType::PREFILL, cp, sharded, true, true, 5, true);
                const auto d_ssd = instance(RoleType::DECODE, cp, sharded, true, true, 5,
                                            true, p_ssd.config_json, true, 4);
                EXPECT_EQ(d_ssd.identity, p_ssd.identity);
                EXPECT_EQ(d_ssd.infos, p_ssd.infos);
                EXPECT_EQ(d_ssd.groups, p_ssd.groups);
                if (sharded) {
                    using Map = autil::legacy::json::JsonMap;
                    const auto config_fields = [&](const std::string& json) {
                        Map selected;
                        autil::legacy::FromJsonString(selected, json);
                        return autil::legacy::AnyCast<Map>(selected.at(""));
                    };
                    const auto custom_json = [](const Map& fields) {
                        return autil::legacy::ToJsonString(Map{{"target-p-alias", autil::legacy::ToJson(fields)}});
                    };
                    const auto required = p_ssd.infos.begin()->first;
                    for (int invalid = 0; invalid < 5; ++invalid) {
                        SCOPED_TRACE(invalid);
                        auto fields = config_fields(p_ssd.config_json);
                        kvcm::KVCMConfig::LocationSpecInfoMap infos = p_ssd.infos;
                        if (invalid == 0) { // Must fail BEFORE disabled-decoder pruning rewrites the map.
                            ++infos.at(required);
                            fields["location_spec_infos"] = autil::legacy::ToJson(infos);
                        } else if (invalid == 1) {
                            fields["block_size"] = autil::legacy::ToJson(int32_t{128});
                        } else if (invalid == 2) {
                            infos.erase(required);
                            fields["location_spec_infos"] = autil::legacy::ToJson(infos);
                        } else if (invalid == 3) {
                            fields.erase("block_size"); // Required by the original KVCM parser.
                        } else {
                            auto sdk = autil::legacy::AnyCast<Map>(fields.at("sdk_config"));
                            sdk["sdk_backend_configs"] = autil::legacy::ToJson(
                                autil::legacy::json::JsonArray{autil::legacy::ToJson(Map{{"type", std::string("mooncake")}})});
                            fields["sdk_config"] = autil::legacy::ToJson(sdk);
                        }
                        EXPECT_FALSE(instance(RoleType::DECODE, cp, sharded, true, true, 5,
                                              true, custom_json(fields), false, 4).initialized);
                    }
                    // Original instance/group aliases and disabled-group pruning
                    // also work for D; required geometry is still checked first.
                    auto fields = config_fields(p_ssd.config_json);
                    auto infos = p_ssd.infos;
                    auto groups = p_ssd.groups;
                    infos["unused_decoder"] = 16;
                    groups["disabled_alias"] = {"unused_decoder"};
                    groups["safe_alias"] = {required};
                    fields["location_spec_infos"] = autil::legacy::ToJson(infos);
                    fields["location_spec_groups"] = autil::legacy::ToJson(groups);
                    const auto aliased = instance(RoleType::DECODE, cp, sharded, true, true, 5,
                                                   true, custom_json(fields), true, 4);
                    EXPECT_EQ(aliased.identity, p_ssd.identity);
                    EXPECT_EQ(aliased.infos, p_ssd.infos);
                    EXPECT_EQ(aliased.groups.at("safe_alias"), (std::vector<std::string>{required}));
                    EXPECT_EQ(aliased.groups.count("disabled_alias"), 0u);
                    const auto p_custom = instance(RoleType::PREFILL, cp, sharded, true, true, 5,
                                                    true, p_ssd.config_json);
                    EXPECT_EQ(p_custom.identity, p_ssd.identity);
                }
            }
            if (!sharded) {
                replicated_identity = p.identity;
            } else if (cp > 1) {
                EXPECT_NE(p.identity, replicated_identity);
            } else {
                EXPECT_EQ(p.identity, replicated_identity);
            }
        }
    }
    });
}

TEST(KVCMLocalTest, DsparkDecodeAdmitsNativeCacheTagsAndRejectsIncompleteDraftConfiguration) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1");
    ::rtp_llm::test::withDecodePublicationStartup([&] {
    createDevice();
    for (const auto role : {RoleType::DECODE}) {
        // Native bounded-replay layout: target encoder STATE/SWA, plus an
        // independent non-reusable decoder SWA segment for all three draft layers.
        for (int invalid = 0; invalid < 10; ++invalid) {
            SCOPED_TRACE(::testing::PrintToString(std::make_tuple(static_cast<int>(role), invalid)));
            BackendEnvironment environment;
            auto& target = environment.cache_config;
            target.dtype = DataType::TYPE_FP16;
            target.layer_num = 2;
            target.seq_size_per_block = 8;
            const std::string state_tag = invalid == 1 ? "Lcsa_state" : "csa_state";
            const std::string swa_tag = invalid == 2 ? "Lswa_kv" : "swa_kv";
            auto full = test::makeMhaSpec("csa_kv", 8, DataType::TYPE_FP16, 1, 2);
            auto state = test::makeResolvedOpaqueSpec(true, state_tag, DataType::TYPE_FP32, 32, 8);
            auto swa = test::makeResolvedOpaqueSpec(true, swa_tag, DataType::TYPE_FP16, 32, 8);
            auto decoder = test::makeResolvedOpaqueSpec(true, "decoder_swa_kv", DataType::TYPE_FP16, 32, 8);
            auto full_policy = defaultCacheGroupPolicy(CacheGroupType::FULL);
            auto swa_policy = defaultCacheGroupPolicy(CacheGroupType::SWA);
            auto decoder_policy = swa_policy;
            decoder_policy.enable_prefix_reuse = false;
            target.fromGroupedSpecs({full, state, swa, decoder}, {{0}, {0}, {0}, {1}},
                {CacheGroupType::FULL, CacheGroupType::SWA, CacheGroupType::SWA, CacheGroupType::SWA},
                {"csa_kv", state_tag, swa_tag, "decoder_swa_kv"},
                {full_policy, swa_policy, swa_policy, decoder_policy});
            CacheConfig draft;
            draft.dtype = target.dtype;
            draft.layer_num = 3;
            draft.seq_size_per_block = target.seq_size_per_block;
            draft.fromGroupedSpecs({decoder}, {{0, 1, 2}}, {CacheGroupType::SWA},
                {"decoder_swa_kv"}, {decoder_policy});
            target.mtp_sub_configs.push_back(target.mergeMTPModule(draft, 0, target.layer_num));
            target.finalizeBlockNums(8, RuntimeConfig{});
            ASSERT_EQ(target.mtp_sub_configs.size(), 1u);
            EXPECT_EQ(target.layer_all_num(), 5u);
            EXPECT_EQ(target.mtp_sub_configs.front()->layerIdsForGroup("decoder_swa_kv"),
                      (std::vector<int>{0, 1, 2}));
            EXPECT_EQ(target.layerIdsForGroup("decoder_swa_kv"), (std::vector<int>{1, 2, 3, 4}));
            for (int layer = 2; layer < 5; ++layer)
                EXPECT_EQ(&target.physicalGroupForLayer(layer, "decoder_swa_kv"),
                          &target.mtp_sub_configs.front()->topology().groupForLayer(layer - 2, "decoder_swa_kv"));
            initializeEnvironmentPools(environment, "dspark_native_tags", 8);
            if (invalid == 3) target.mtp_sub_configs.clear();
            if (invalid == 4) target.mtp_sub_configs.push_back(target.mtp_sub_configs.front());
            if (invalid == 5) target.mtp_sub_configs.front().reset();
            if (invalid == 6) target.mtp_sub_configs.front()->layer_num = 0;
            SpeculativeExecutionConfig speculation;
            speculation.type = invalid == 7 ? SP_TYPE_EAGLE : (invalid == 8 ? SP_TYPE_MTP : SP_TYPE_DSPARK);
            speculation.gen_num_per_cycle = 5; // Proposal width is not the draft layer/module count.
            auto client = std::make_shared<MockClientWrapper>();
            EXPECT_CALL(*client, initForPools(_, _, _, _)).Times(invalid == 0 ? 1 : 0).WillRepeatedly(Return(true));
            EXPECT_CALL(*client, shutdown()).Times(1);
            KVCacheConfig config;
            config.kvcm_server_address = "unused-test-address";
            config.kvcm_instance_id_salt = "dspark-native-tag-admission";
            config.kvcm_model_sdk_config = R"([{"type":"pace_ssd"}])";
            config.kvcm_read_backend_type = 9;
            auto parallel = singleRankConfig();
            parallel.role_type = role;
            parallel.prefill_cp_config.prefill_cp_size = 1;
            RuntimeConfig runtime;
            runtime.model_name = invalid == 9 ? "DeepSeekV4" : "DeepSeekV41";
            BackendHandle backend(std::make_unique<KVCMStorageBackend>(target, config, runtime, parallel,
                speculation, nullptr, true, nullptr, client));
            EXPECT_EQ(initSingleRank(*backend.backend, environment), invalid == 0);
            if (invalid == 0 && role == RoleType::DECODE) {
                BatchKVCacheResource resource;
                resource.resetBatchSize(1);
                resource.initGroups(target.topologyPtr());
                EXPECT_FALSE(resource.decode_endpoint);
                EXPECT_CALL(*client, getWriteLocation(_, _, _, _, _, _, _)).Times(0);
                EXPECT_CALL(*client, saveKvCachesForTag(_, _, _, _)).Times(0);
                for (size_t attempt = 1; attempt <= 3; ++attempt) {
                    EXPECT_TRUE(backend.backend->captureCompletedDecode(resource, 8));
                    EXPECT_EQ(resource.decodeCompletionLifecycle().admissionAttempts(), std::min(attempt, size_t{2}));
                    EXPECT_FALSE(resource.decodeCompletionSubmitted());
                }
            }
        }
    }
    });
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
            identifyReadRequest(*remote);
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

TEST(KVCMLocalTest, PrefillNonrootBackendDoesNotInstallDecodePayloadObserver) {
    autil::EnvGuard async("RTP_LLM_STREAM_ASYNC", "0");
    autil::EnvGuard manual("kmonitorManuallyMode", "true");
    for (const bool enabled : {false, true}) {
        autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", enabled ? "1" : "0");
        ::rtp_llm::test::withDecodePublicationStartup([&] {
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
                    EXPECT_FALSE(client->transfer_observer_);
                    backend.backend->shutdown();
                    EXPECT_FALSE(client->transfer_observer_);
                }
            }
        }
        });
    }
}

TEST(KVCMLocalTest, PrefillForwardsDecoratedOrPlainUriAndExactDestinationDescriptors) {
    for (const bool known_source : {true, false}) {
        SCOPED_TRACE(known_source);
        autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1");
        ::rtp_llm::test::withDecodePublicationStartup([&] {
        autil::EnvGuard async("RTP_LLM_STREAM_ASYNC", "0");
        auto            environment = makeBackendEnvironment("kvcm_physical_source_binding");
        auto            client      = std::make_shared<MockClientWrapper>();
        EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
        EXPECT_CALL(*client, shutdown()).Times(1);
        auto backend = makeCompletionProofBackend(environment, client);
        ASSERT_TRUE(initSingleRank(*backend.backend, environment));
        const std::string           canonical = "pace://allocated/1?media_type=2&node_id=1&size=32";
        const auto uri = known_source ? canonical + "&dtop_origin=v1." + std::string(32, 'a') + ".4.0.8.8.9" : canonical;
        kv_cache_manager::Locations locations{{{"tp0_Fdefault", uri}}};
        EXPECT_CALL(*client, match(_, _, _, _, _, _)).WillOnce(Return(std::make_pair(true, locations)));
        auto observation = match(*backend.backend, makeStorageRequest(environment));
        ASSERT_TRUE(observation.success);
        ASSERT_NE(observation.match_meta, nullptr);
        EXPECT_CALL(*client, loadKvCachesForTag("default", kv_cache_manager::UriStrVec{uri}, _, _))
            .WillOnce(Invoke([&](const auto&, const auto&, auto& buffers, const auto&) {
                // The original read still reaches the actual destination buffer.
                const auto expected = environmentBuffers(environment, 0, "default", environment.block_id);
                EXPECT_EQ(buffers.size(), 1u);
                if (buffers.size() == 1) {
                    EXPECT_EQ(buffers.front().iovs.size(), expected.size());
                    for (size_t i = 0; i < std::min(buffers.front().iovs.size(), expected.size()); ++i) {
                        EXPECT_EQ(buffers.front().iovs[i].base, expected[i].addr);
                        EXPECT_EQ(buffers.front().iovs[i].size, expected[i].size_bytes);
                    }
                }
                return true;
            }));
        EXPECT_TRUE(read(*backend.backend, makeStorageRequest(environment), std::move(observation.match_meta)));
        });
    }
}

template<typename Code>
std::vector<std::pair<Code, bool>> failedCompletionCases(bool write = false) {
    std::vector<std::pair<Code, bool>> cases{{write ? Code::ER_SDKWRITE_ERROR : Code::ER_SDKREAD_ERROR, false},
                                             {Code::ER_INVALID_PARAMS, false},
                                             {Code::ER_SDK_TIMEOUT, false}};
    if constexpr (kvcm::detail::HasExplicitBufferSafeCodes<Code>::value) {
        cases.emplace_back(Code::ER_SDK_TIMEOUT_BUFFER_SAFE, true);
        cases.emplace_back(Code::ER_SDK_ERROR_BUFFER_SAFE, true);
    }
    return cases;
}

TEST(KVCMLocalTest, PrefillUsesExplicitBufferProofForFailureAndTimeoutWithEitherSwitch) {
    autil::EnvGuard async("RTP_LLM_STREAM_ASYNC", "0");
    using EC = kv_cache_manager::ClientErrorCode;
    for (const bool enabled : {false, true}) {
        for (const auto& [ec, safe] : failedCompletionCases<EC>()) {
            SCOPED_TRACE(::testing::Message() << "enabled=" << enabled << " ec=" << static_cast<int>(ec));
            autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", enabled ? "1" : "0");
            const auto code = ec;
            ::rtp_llm::test::withDecodePublicationStartup([&] {
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
            const auto ref_count_before = environment.device_pool->refCount(environment.block_id);
            client->load_status = code;
            const auto error    = readResult(*backend.backend, request, std::move(observation.match_meta));
            EXPECT_FALSE(error.ok());
            ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
            const bool unknown = !safe;
            EXPECT_FALSE(backend.backend->protected_quarantine_);
            EXPECT_EQ(environment.device_pool->isFrozen(environment.block_id), unknown);
            EXPECT_EQ(backend->frozenTasks().size(), unknown ? 1u : 0u);
            EXPECT_EQ(environment.device_pool->refCount(environment.block_id), ref_count_before + (unknown ? 1u : 0u));
            std::lock_guard<std::mutex> lock(backend.backend->quarantine_mutex_);
            EXPECT_EQ(
                backend.backend->quarantined_breakdown_[0][static_cast<size_t>(StorageQuarantineReason::SDK_TIMEOUT)]
                    .task_count,
                unknown && code == EC::ER_SDK_TIMEOUT ? 1u : 0u);
            });
        }
    }
}

TEST(KVCMLocalTest, PrefillRpcRequiresBufferProofIndependentlyOfTransferStatus) {
    for (const bool enabled : {false, true}) for (const bool safe : {false, true}) {
        for (const auto status : {REMOTE_TRANSFER_STATUS_FAILED, REMOTE_TRANSFER_STATUS_TIMEOUT}) {
            autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", enabled ? "1" : "0");
            ::rtp_llm::test::withDecodePublicationStartup([&] {
                auto environment = makeBackendEnvironment("prefill_rpc_buffer_proof");
                auto first = std::make_shared<KVCMBroadcastState>();
                auto second = std::make_shared<KVCMBroadcastState>();
                second->execute = [=](const auto&, auto& response) {
                    response.set_transfer_status(status);
                    response.set_payload_buffer_safe(safe);
                };
                KVCMBroadcastRpcServer rank0(0, first), rank1(1, second);
                ASSERT_TRUE(rank0.start()); ASSERT_TRUE(rank1.start());
                auto broadcast = std::make_shared<BroadcastManager>(std::vector<std::string>{rank0.address(), rank1.address()});
                ASSERT_TRUE(broadcast->init());
                auto client = std::make_shared<MockClientWrapper>();
                EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
                EXPECT_CALL(*client, shutdown()).Times(1);
                auto parallel = singleRankConfig();
                parallel.role_type = RoleType::PREFILL;
                parallel.tp_size = 2;
                auto backend = makeBackend(environment, parallel, client, broadcast);
                ASSERT_TRUE(initSingleRank(*backend.backend, environment));
                const kv_cache_manager::Locations locations{{{"tp0_Fdefault", "pace://rank0"}, {"tp1_Fdefault", "pace://rank1"}}};
                EXPECT_CALL(*client, match(_, _, _, _, _, _)).WillOnce(Return(std::make_pair(true, locations)));
                auto request = makeStorageRequest(environment);
                auto matched = match(*backend.backend, request);
                ASSERT_TRUE(matched.success);
                const auto refs = environment.device_pool->refCount(environment.block_id);
                EXPECT_FALSE(readResult(*backend.backend, request, matched.match_meta).ok());
                ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
                EXPECT_EQ(environment.device_pool->isFrozen(environment.block_id), !safe);
                EXPECT_EQ(environment.device_pool->refCount(environment.block_id), refs + (safe ? 0u : 1u));
                EXPECT_FALSE(backend.backend->protected_quarantine_);
                backend.backend->shutdown();
            });
        }
    }
}

TEST(KVCMLocalTest, PrefillRpcPreservesExceptionalUnknownFromRawSdkAndStillAcceptsNextRead) {
    using EC = kv_cache_manager::ClientErrorCode;
    for (const bool enabled : {false, true}) {
        for (const bool throws : {false, true}) {
          for (const bool write : {false, true}) {
           for (const bool peer_rpc_failure : {false, true}) {
            SCOPED_TRACE(::testing::Message() << "enabled=" << enabled << " throws=" << throws << " write=" << write
                                                << " peer_rpc_failure=" << peer_rpc_failure);
            autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", enabled ? "1" : "0");
            ::rtp_llm::test::withDecodePublicationStartup([&] {
                auto destination = makeBackendEnvironment("prefill_rpc_destination");
                auto worker_environment = makeBackendEnvironment("prefill_rpc_worker");
                auto factory = std::make_unique<kvcm::MockClientFactory>();
                auto subscriber = std::make_unique<kvcm::MockSubscriber>();
                EXPECT_CALL(*subscriber, init(_)).WillOnce(Return(true));
                EXPECT_CALL(*subscriber, getAddresses(_)).Times(0);
                EXPECT_CALL(*factory, createSubscriber(false)).WillOnce(Invoke([&](bool) { return std::move(subscriber); }));
                auto meta = std::make_unique<kv_cache_manager::MockMetaClient>();
                static const std::string storage_config = R"({"sdk_backend_configs":[]})";
                EXPECT_CALL(*meta, GetStorageConfig()).WillOnce(::testing::ReturnRef(storage_config));
                EXPECT_CALL(*factory, createMetaClient(_, _)).WillOnce(Invoke([&](const auto&, const auto&) { return std::move(meta); }));
                auto sdk = std::make_unique<kv_cache_manager::MockTransferClient>(std::make_shared<int>(0));
                if (write) {
                    EXPECT_CALL(*sdk, SaveKvCaches(_, _, _))
                        .WillOnce(Invoke([throws](const auto&, const auto&, const auto&) -> std::pair<EC, kv_cache_manager::UriStrVec> {
                            if (throws) throw std::runtime_error("SDK exceptional write exit after entry");
                            return {EC::ER_SDKWRITE_ERROR, {}};
                        }));
                    EXPECT_CALL(*sdk, LoadKvCaches(_, _, _)).WillOnce(Return(EC::ER_OK));
                } else {
                    EXPECT_CALL(*sdk, LoadKvCaches(_, _, _))
                        .WillOnce(Invoke([throws](const auto&, const auto&, const auto&) -> EC {
                            if (throws) throw std::runtime_error("SDK exceptional read exit after entry");
                            return EC::ER_SDKREAD_ERROR;
                        }))
                        .WillOnce(Return(EC::ER_OK));
                }
                EXPECT_CALL(*factory, createTransferClient(_, _))
                    .WillOnce(Invoke([&](const auto&, const auto&) { return std::move(sdk); }));
                auto wrapper = std::make_shared<kvcm::ClientWrapper>(std::move(factory), std::chrono::seconds(10), false);
                auto worker_parallel = singleRankConfig();
                worker_parallel.tp_size = 2;
                worker_parallel.tp_rank = 1;
                worker_parallel.role_type = RoleType::PREFILL;
                auto worker = makeBackend(worker_environment, worker_parallel, wrapper);
                ASSERT_TRUE(initSingleRank(*worker.backend, worker_environment));
                auto rank0 = std::make_shared<KVCMBroadcastState>();
                auto rank1 = std::make_shared<KVCMBroadcastState>();
                rank0->fail = peer_rpc_failure;
                std::atomic<int> exceptional{0}, ordinary{0};
                rank1->execute = [&](const auto& request, auto& response) {
                    (void)worker.backend->execute(request, response);
                    if (response.transfer_status() == REMOTE_TRANSFER_STATUS_EXCEPTION_UNKNOWN) ++exceptional;
                    if (response.transfer_status() == REMOTE_TRANSFER_STATUS_FAILED) ++ordinary;
                };
                KVCMBroadcastRpcServer server0(0, rank0), server1(1, rank1);
                ASSERT_TRUE(server0.start());
                ASSERT_TRUE(server1.start());
                auto broadcast = std::make_shared<BroadcastManager>(std::vector<std::string>{server0.address(), server1.address()});
                ASSERT_TRUE(broadcast->init());
                auto controller_client = std::make_shared<MockClientWrapper>();
                EXPECT_CALL(*controller_client, initForPools(_, _, _, _)).WillOnce(Return(true));
                EXPECT_CALL(*controller_client, shutdown()).Times(1);
                auto parallel = singleRankConfig();
                parallel.tp_size = 2;
                parallel.role_type = RoleType::PREFILL;
                auto controller = makeBackend(destination, parallel, controller_client, broadcast);
                ASSERT_TRUE(initSingleRank(*controller.backend, destination));
                const kv_cache_manager::Locations locations{{{"tp0_Fdefault", "pace://rank0"}, {"tp1_Fdefault", "pace://rank1"}}};
                EXPECT_CALL(*controller_client, match(_, _, _, _, _, _))
                    .WillRepeatedly(Return(std::make_pair(true, locations)));
                auto request = makeStorageRequest(destination);
                const auto refs = destination.device_pool->refCount(destination.block_id);
                auto matched = match(*controller.backend, request);
                ASSERT_TRUE(matched.success);
                if (write) {
                    kv_cache_manager::WriteLocation allocation;
                    allocation.write_session_id = "exception-boundary-session";
                    allocation.block_mask = kv_cache_manager::BlockMaskOffset{0};
                    allocation.locations = locations;
                    EXPECT_CALL(*controller_client, getWriteLocation(_, _, _, _, _, 40, 0))
                        .WillOnce(Return(std::make_pair(true, allocation)));
                    if (throws) {
                        EXPECT_CALL(*controller_client, finishWrite(_, _, _, _, _)).Times(0);
                    } else {
                        EXPECT_CALL(*controller_client, finishWrite(_, _, "exception-boundary-session",
                            ::testing::Eq(kv_cache_manager::BlockMask{kv_cache_manager::BlockMaskOffset{0}}),
                            ::testing::IsEmpty())).WillOnce(Return(true));
                    }
                    EXPECT_TRUE(controller.backend->write(controller.backend->prepareWrite(request)));
                } else {
                    auto result = readResult(*controller.backend, request, matched.match_meta);
                    EXPECT_FALSE(result.ok());
                }
                ASSERT_TRUE(waitForBackendOperationsForTest(*controller.backend));
                EXPECT_EQ(exceptional.load(), throws ? 1 : 0);
                EXPECT_EQ(ordinary.load(), throws ? 0 : 1);
                EXPECT_EQ(destination.device_pool->isFrozen(destination.block_id), !write);
                EXPECT_EQ(destination.device_pool->refCount(destination.block_id), refs + (write ? 0u : 1u));
                EXPECT_FALSE(controller.backend->protected_quarantine_);
                rank0->fail = false; // The independent next request must still make progress.
                ScopedReferencedBlocks next_destination(destination, 1), next_worker(worker_environment, 1);
                ASSERT_EQ(next_destination.get(), next_worker.get());
                auto next = makeStorageRequest(destination, {102}, 0, next_destination.get());
                auto next_match = match(*controller.backend, next);
                ASSERT_TRUE(next_match.success);
                EXPECT_TRUE(read(*controller.backend, next, next_match.match_meta));
                EXPECT_FALSE(destination.device_pool->isFrozen(next_destination.get().front()));
                EXPECT_EQ(destination.device_pool->refCount(next_destination.get().front()), 1u);
                controller.backend->shutdown();
                worker.backend->shutdown();
            });
           }
          }
        }
    }
}

TEST(KVCMLocalTest, PrefillFailedPayloadClosesOnceWithoutFreezingLocalSources) {
    autil::EnvGuard async("RTP_LLM_STREAM_ASYNC", "0");
    using EC = kv_cache_manager::ClientErrorCode;
    for (const bool enabled : {false, true}) {
        for (const auto& [ec, safe] : failedCompletionCases<EC>(/*write=*/true)) {
            autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", enabled ? "1" : "0");
            const auto code = ec;
            ::rtp_llm::test::withDecodePublicationStartup([&] {
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
            // Original P policy: best-effort zero-mask close, even on timeout.
            // The mock performs no DMA; this does not establish SDK quiescence.
            EXPECT_CALL(*client, finishWrite(_, _, "pending-session",
                ::testing::Eq(kv_cache_manager::BlockMask{kv_cache_manager::BlockMaskOffset{0}}),
                ::testing::IsEmpty())).WillOnce(Return(true));
            const auto ref_count_before = environment.device_pool->refCount(environment.block_id);
            client->save_status = code;
            auto task           = backend.backend->prepareWrite(makeStorageRequest(environment));
            ASSERT_TRUE(static_cast<bool>(task));
            ASSERT_TRUE(backend.backend->write(std::move(task)));
            ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
            EXPECT_FALSE(backend.backend->protected_quarantine_);
            EXPECT_FALSE(environment.device_pool->isFrozen(environment.block_id));
            EXPECT_TRUE(backend->frozenTasks().empty());
            EXPECT_EQ(environment.device_pool->refCount(environment.block_id), ref_count_before);
            });
        }
    }
}

TEST(KVCMLocalTest, PrefillCommitAckFailureRemainsSingleFinishWithoutDmaQuarantine) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1");
    ::rtp_llm::test::withDecodePublicationStartup([&] {
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
        // Production P requests carry no D-only admission lease.
        auto request = makeStorageRequest(environment);
        const auto refs_before = environment.device_pool->refCount(environment.block_id);
        auto       task        = backend.backend->prepareWrite(std::move(request));
        ASSERT_TRUE(static_cast<bool>(task));
        EXPECT_EQ(environment.device_pool->refCount(environment.block_id), refs_before + 1);
        ASSERT_TRUE(backend.backend->write(std::move(task)));
        ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
        EXPECT_FALSE(environment.device_pool->isFrozen(environment.block_id));
        EXPECT_FALSE(backend.backend->protected_quarantine_);
        EXPECT_EQ(environment.device_pool->refCount(environment.block_id), refs_before);
        // Join the worker before inspecting ownership of its retired closure.
        backend.backend->shutdown();
    }
    });
}

TEST(KVCMLocalTest, SynchronousDecodeAdapterRejectsBeforeMetadataOrPayload) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1");
    ::rtp_llm::test::withDecodePublicationStartup([&] {
        auto environment = makeBackendEnvironment("sync_decode_adapter");
        auto client = std::make_shared<MockClientWrapper>();
        EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
        EXPECT_CALL(*client, shutdown()).Times(1);
        KVCacheConfig config;
        config.kvcm_model_sdk_config = R"([{"type":"pace"}])";
        auto parallel = singleRankConfig();
        parallel.role_type = RoleType::DECODE;
        parallel.prefill_cp_config.prefill_cp_size = 1;
        auto backend = makeBackend(environment, parallel, client, nullptr, nullptr, true, config);
        ASSERT_TRUE(initSingleRank(*backend.backend, environment));
        auto request = makeStorageRequest(environment);
        request.decode_completion = true;
        const auto refs = environment.device_pool->refCount(environment.block_id);
        EXPECT_CALL(*client, getWriteLocation(_, _, _, _, _, _, _)).Times(0);
        EXPECT_CALL(*client, saveKvCachesForTag(_, _, _, _)).Times(0);
        EXPECT_CALL(*client, finishWrite(_, _, _, _, _)).Times(0);
        EXPECT_THROW(backend.backend->writeImpl(request), StorageOperationBufferSafeFailure);
        EXPECT_EQ(environment.device_pool->refCount(environment.block_id), refs);
        EXPECT_FALSE(environment.device_pool->isFrozen(environment.block_id));
        EXPECT_FALSE(backend.backend->hasUnresolvedExternalIo());
        backend.backend->shutdown();
    });
}

TEST(KVCMLocalTest, FailedOrThrowingStartNeverDispatchesOrClosesPayload) {
    for (const bool enabled : {false, true}) {
        for (const bool throws : {false, true}) {
            autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", enabled ? "1" : "0");
            ::rtp_llm::test::withDecodePublicationStartup([&] {
                auto environment = makeBackendEnvironment("start_failure_before_payload");
                auto client = std::make_shared<MockClientWrapper>();
                EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
                EXPECT_CALL(*client, shutdown()).Times(1);
                auto backend = makeCompletionProofBackend(environment, client);
                ASSERT_TRUE(initSingleRank(*backend.backend, environment));
                const auto refs = environment.device_pool->refCount(environment.block_id);
                EXPECT_CALL(*client, getWriteLocation(_, _, _, _, _, _, 0))
                    .WillOnce(Invoke([&](const auto&, const auto&, const auto&, const auto&, const auto&, auto, auto)
                                         -> std::pair<bool, kv_cache_manager::WriteLocation> {
                        if (throws) throw std::runtime_error("lost StartWrite acknowledgement");
                        return {false, {}};
                    }));
                EXPECT_CALL(*client, saveKvCachesForTag(_, _, _, _)).Times(0);
                EXPECT_CALL(*client, finishWrite(_, _, _, _, _)).Times(0);
                ASSERT_TRUE(backend.backend->write(backend.backend->prepareWrite(makeStorageRequest(environment))));
                ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
                backend.backend->shutdown();
                EXPECT_EQ(environment.device_pool->refCount(environment.block_id), refs);
                EXPECT_FALSE(environment.device_pool->isFrozen(environment.block_id));
                EXPECT_FALSE(backend.backend->protected_quarantine_);
            });
        }
    }
}

// Exercise the non-reservation (queue_size=0 or custom executor) failed-post
// fallback, without changing the default executor or its completion reservations.
class RejectCompletionExecutor final: public StorageBackendExecutor {
public:
    explicit RejectCompletionExecutor(std::shared_ptr<StorageBackendExecutor> delegate): delegate_(std::move(delegate)) {}
    bool start() override { return delegate_->start(); }
    bool submit(Task task, BlockTreeTaskClass task_class) override {
        return delegate_->submit(std::move(task), task_class);
    }
    bool submitCompletion(Task) override { ++rejected; return false; }
    void stopAdmission() noexcept override { delegate_->stopAdmission(); }
    void shutdown() noexcept override { delegate_->shutdown(); }
    std::atomic<int> rejected{0};
private:
    std::shared_ptr<StorageBackendExecutor> delegate_;
};

TEST(KVCMLocalTest, RejectedPrefillCompletionPostDoesNoMetadataIoAndReleasesWriteSources) {
    for (const bool enabled : {false, true}) {
        for (const bool unknown : {false, true}) {
            autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", enabled ? "1" : "0");
            ::rtp_llm::test::withDecodePublicationStartup([&] {
                auto environment = makeBackendEnvironment("off_worker_settlement");
                auto client = std::make_shared<MockClientWrapper>();
                EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
                EXPECT_CALL(*client, shutdown()).Times(1);
                auto backend = makeCompletionProofBackend(environment, client);
                auto executor = std::make_shared<RejectCompletionExecutor>(backend.backend->executor_);
                backend.backend->executor_ = executor; // Test-only injection before init/start.
                ASSERT_TRUE(initSingleRank(*backend.backend, environment));
                const auto refs = environment.device_pool->refCount(environment.block_id);
                kv_cache_manager::WriteLocation location;
                location.write_session_id = "fallback-open-session";
                location.block_mask = kv_cache_manager::BlockMaskOffset{0};
                location.locations = {{{"tp0_Fdefault", "pace://allocated"}}};
                EXPECT_CALL(*client, getWriteLocation(_, _, _, _, _, _, 0))
                    .WillOnce(Return(std::make_pair(true, location)));
                EXPECT_CALL(*client, finishWrite(_, _, _, _, _)).Times(0);
                client->save_status = unknown ? kv_cache_manager::ClientErrorCode::ER_SDK_TIMEOUT :
                                               kv_cache_manager::ClientErrorCode::ER_OK;
                ASSERT_TRUE(backend.backend->write(backend.backend->prepareWrite(makeStorageRequest(environment))));
                ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
                EXPECT_EQ(executor->rejected.load(), 1);
                // Ordinary P writes only read local sources; UNKNOWN keeps no local FROZEN target.
                EXPECT_EQ(environment.device_pool->refCount(environment.block_id), refs);
                EXPECT_FALSE(environment.device_pool->isFrozen(environment.block_id));
                EXPECT_FALSE(backend.backend->protected_quarantine_);
                EXPECT_FALSE(backend.backend->hasUnresolvedExternalIo());
                backend.backend->shutdown();
                EXPECT_EQ(environment.device_pool->refCount(environment.block_id), refs);
                EXPECT_FALSE(environment.device_pool->isFrozen(environment.block_id));
                EXPECT_FALSE(backend.backend->protected_quarantine_);
            });
        }
    }
}

TEST(KVCMLocalTest, LegacySessionIdsAndRetiredQuotaSettingsDoNotGateWrites) {
    for (const bool enabled : {false, true}) {
        autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", enabled ? "1" : "0");
        ::rtp_llm::test::withDecodePublicationStartup([&] {
            autil::EnvGuard slots("RTP_LLM_REMOTE_CACHE_SESSION_SLOTS", "0");
            autil::EnvGuard bytes("RTP_LLM_REMOTE_CACHE_SESSION_BYTES", "invalid");
            auto environment = makeBackendEnvironment("legacy_session_identity");
            auto client = std::make_shared<MockClientWrapper>();
            EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
            EXPECT_CALL(*client, shutdown()).Times(1);
            auto backend = makeCompletionProofBackend(environment, client);
            ASSERT_TRUE(initSingleRank(*backend.backend, environment));
            const auto refs = environment.device_pool->refCount(environment.block_id);
            for (const std::string id : {std::string{}, std::string(1025, 'x')}) {
                kv_cache_manager::WriteLocation location;
                location.write_session_id = id;
                location.block_mask = kv_cache_manager::BlockMaskOffset{0};
                location.locations = {{{"tp0_Fdefault", "pace://allocated"}}};
                EXPECT_CALL(*client, getWriteLocation(_, _, _, _, _, _, 0))
                    .WillOnce(Return(std::make_pair(true, location)));
                EXPECT_CALL(*client, saveKvCachesForTag(_, _, _, _))
                    .WillOnce(Return(std::make_pair(true, kv_cache_manager::UriStrVec{"pace://actual"})));
                EXPECT_CALL(*client, finishWrite(_, _, id, _, _)).WillOnce(Return(true));
                ASSERT_TRUE(backend.backend->write(backend.backend->prepareWrite(makeStorageRequest(environment))));
                ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
                EXPECT_EQ(environment.device_pool->refCount(environment.block_id), refs);
                EXPECT_FALSE(environment.device_pool->isFrozen(environment.block_id));
            }
            backend.backend->shutdown();
        });
    }
}

TEST(KVCMLocalTest, FinishIsSingleAttemptAndCannotAbortOrBlockLaterWrites) {
    for (const bool enabled : {false, true}) {
        for (const bool unused : {false, true}) {
            for (const bool throws : {false, true}) {
                autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", enabled ? "1" : "0");
                ::rtp_llm::test::withDecodePublicationStartup([&] {
                auto environment = makeBackendEnvironment("single_finish_attempt");
                auto client = std::make_shared<MockClientWrapper>();
                EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
                EXPECT_CALL(*client, shutdown()).Times(1);
                auto backend = makeCompletionProofBackend(environment, client);
                ASSERT_TRUE(initSingleRank(*backend.backend, environment));
                const auto refs = environment.device_pool->refCount(environment.block_id);
                std::atomic<int> starts{0}, finishes{0};
                EXPECT_CALL(*client, getWriteLocation(_, _, _, _, _, _, 0))
                    .WillRepeatedly(Invoke([&](const auto&, const auto&, const auto&, const auto&,
                                               const auto&, auto, auto) {
                        kv_cache_manager::WriteLocation location;
                        location.write_session_id = "single-finish-" + std::to_string(++starts);
                        location.block_mask = kv_cache_manager::BlockMaskOffset{0};
                        location.locations = {{{"tp0_Fdefault", "pace://allocated"}}};
                        if (unused && starts == 1) backend.backend->setRemoteCacheEnabled(false);
                        return std::make_pair(true, location);
                    }));
                EXPECT_CALL(*client, saveKvCachesForTag(_, _, _, _)).Times(unused ? 1 : 2)
                    .WillRepeatedly(Return(std::make_pair(true, kv_cache_manager::UriStrVec{"pace://actual"})));
                EXPECT_CALL(*client, finishWrite(_, _, _, _, _))
                    .WillOnce(Invoke([&](const auto&, const auto&, const auto& session, const auto& mask, const auto& locations) {
                        ++finishes;
                        EXPECT_EQ(session, "single-finish-" + std::to_string(finishes.load()));
                        EXPECT_EQ(std::get<kv_cache_manager::BlockMaskOffset>(mask), unused ? 0u : 1u);
                        if (unused) {
                            EXPECT_TRUE(locations.empty());
                        } else {
                            EXPECT_EQ(locations.size(), 1u);
                            if (locations.size() == 1) {
                                EXPECT_EQ(locations.front().size(), 1u);
                                if (locations.front().size() == 1) {
                                    EXPECT_EQ(locations.front().front().spec_name, "tp0_Fdefault");
                                    EXPECT_EQ(locations.front().front().uri, "pace://actual");
                                }
                            }
                        }
                        if (throws) throw std::runtime_error("lost FinishWrite acknowledgement");
                        return false;
                    }))
                    .WillOnce(Invoke([&](const auto&, const auto&, const auto& session, const auto& mask, const auto& locations) {
                        ++finishes;
                        EXPECT_EQ(session, "single-finish-" + std::to_string(finishes.load()));
                        EXPECT_EQ(std::get<kv_cache_manager::BlockMaskOffset>(mask), 1u);
                        EXPECT_EQ(locations.size(), 1u);
                        if (locations.size() == 1) {
                            EXPECT_EQ(locations.front().size(), 1u);
                            if (locations.front().size() == 1) {
                                EXPECT_EQ(locations.front().front().spec_name, "tp0_Fdefault");
                                EXPECT_EQ(locations.front().front().uri, "pace://actual");
                            }
                        }
                        return true;
                    }));
                ASSERT_TRUE(backend.backend->write(backend.backend->prepareWrite(makeStorageRequest(environment))));
                ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
                EXPECT_EQ(finishes.load(), 1); // No retry and no mask-zero replay after a commit attempt.
                EXPECT_EQ(environment.device_pool->refCount(environment.block_id), refs);
                EXPECT_FALSE(environment.device_pool->isFrozen(environment.block_id));
                backend.backend->setRemoteCacheEnabled(true);
                ASSERT_TRUE(backend.backend->write(backend.backend->prepareWrite(makeStorageRequest(environment))));
                ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
                backend.backend->shutdown();
                EXPECT_EQ(starts.load(), 2);
                EXPECT_EQ(finishes.load(), 2); // A metadata ACK failure does not exhaust a global pool.
                EXPECT_EQ(environment.device_pool->refCount(environment.block_id), refs);
                });
            }
        }
    }
}

TEST(KVCMLocalTest, PrefillQueueRejectAndExpiryNeverStartSessionOrRetainUnknown) {
    for (const bool enabled : {false, true}) {
        for (const bool expire_queued : {false, true}) {
            SCOPED_TRACE(::testing::Message() << "enabled=" << enabled << " expire=" << expire_queued);
            autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", enabled ? "1" : "0");
            ::rtp_llm::test::withDecodePublicationStartup([&] {
                auto environment = makeBackendEnvironment("prefill_original_queue");
                auto client = std::make_shared<MockClientWrapper>();
                EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
                EXPECT_CALL(*client, shutdown()).Times(1);
                KVCacheConfig config;
                config.kvcm_model_sdk_config = R"([{"type":"pace"}])";
                config.kvcm_asyncwrapper_thread_num = 1;
                config.kvcm_asyncwrapper_queue_size = 1;
                config.kvcm_task_timeout_ms = 30000;
                auto parallel = singleRankConfig();
                parallel.role_type = RoleType::PREFILL;
                auto backend = makeBackend(environment, parallel, client, nullptr, nullptr, true, config);
                ASSERT_TRUE(initSingleRank(*backend.backend, environment));
                const auto refs = environment.device_pool->refCount(environment.block_id);
                std::promise<void> release, entered, release_worker, worker_entered;
                auto released = release.get_future().share();
                auto started = entered.get_future();
                auto worker_released = release_worker.get_future().share();
                auto worker_started = worker_entered.get_future();
                block_tree_cache_detail::ScopeRollback unblock([&] { release.set_value(); });
                block_tree_cache_detail::ScopeRollback unblock_worker([&] { release_worker.set_value(); });
                block_tree_cache_detail::ScopeRollback cleanup([&] {
                    unblock.run(); unblock_worker.run(); backend.backend->shutdown();
                });
                std::atomic<int> starts{0}, saves{0}, commits{0}, closes{0};
                EXPECT_CALL(*client, getWriteLocation(_, _, _, _, _, 40, 0))
                    .WillRepeatedly(Invoke([&](const auto&, const auto&, const auto&, const auto&, const auto&, auto, auto) {
                        kv_cache_manager::WriteLocation value;
                        value.write_session_id = "prefill-" + std::to_string(++starts);
                        value.block_mask = kv_cache_manager::BlockMaskOffset{0};
                        value.locations = {{{"tp0_Fdefault", "pace://write"}}};
                        return std::make_pair(true, value);
                    }));
                EXPECT_CALL(*client, saveKvCachesForTag(_, _, _, _))
                    .WillRepeatedly(Invoke([&](const auto&, const auto& uris, const auto&, const auto&) {
                        if (++saves == 1) {
                            entered.set_value();
                            EXPECT_EQ(released.wait_for(std::chrono::seconds(10)), std::future_status::ready);
                        }
                        return std::make_pair(true, uris);
                    }));
                EXPECT_CALL(*client, finishWrite(_, _, _, _, _))
                    .WillRepeatedly(Invoke([&](const auto&, const auto&, const auto&, const auto& mask, const auto& locations) {
                        const auto offset = std::get<kv_cache_manager::BlockMaskOffset>(mask);
                        if (offset == 0) ++closes; else { ++commits; EXPECT_EQ(offset, 1u); }
                        EXPECT_TRUE(locations.empty());
                        return true;
                    }));
                ASSERT_TRUE(backend.backend->write(backend.backend->prepareWrite(makeStorageRequest(environment))));
                ASSERT_EQ(started.wait_for(std::chrono::seconds(5)), std::future_status::ready);
                // SDK work no longer occupies the business worker. Hold that
                // worker explicitly before filling its one BACKGROUND slot.
                ASSERT_TRUE(backend.backend->executor_->submit([&] {
                    worker_entered.set_value();
                    EXPECT_EQ(worker_released.wait_for(std::chrono::seconds(10)), std::future_status::ready);
                }, BlockTreeTaskClass::LOAD));
                ASSERT_EQ(worker_started.wait_for(std::chrono::seconds(5)), std::future_status::ready);
                auto queued = makeStorageRequest(environment);
                queued.transfer_task.emplace(std::vector<TransferDescriptor>{},
                    expire_queued ? std::chrono::milliseconds(300) : std::chrono::seconds(30));
                ASSERT_TRUE(backend.backend->write(backend.backend->prepareWrite(queued)));
                EXPECT_EQ(backend.backend->queueSizes().background, 1u);
                const auto held_refs = environment.device_pool->refCount(environment.block_id);
                for (int i = 0; i < 3; ++i) {
                    EXPECT_FALSE(backend.backend->write(backend.backend->prepareWrite(makeStorageRequest(environment))));
                    EXPECT_EQ(environment.device_pool->refCount(environment.block_id), held_refs);
                    EXPECT_EQ(starts.load(), 1);
                    EXPECT_EQ(closes.load(), 0); // No session was opened for any rejected task.
                    EXPECT_FALSE(environment.device_pool->isFrozen(environment.block_id));
                }
                if (expire_queued) {
                    std::promise<void> never;
                    auto until_deadline = never.get_future();
                    EXPECT_EQ(until_deadline.wait_until(queued.transfer_task->deadline()), std::future_status::timeout);
                }
                unblock.run();
                unblock_worker.run();
                ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
                EXPECT_EQ(starts.load(), expire_queued ? 1 : 2);
                EXPECT_EQ(saves.load(), expire_queued ? 1 : 2);
                EXPECT_EQ(commits.load(), expire_queued ? 1 : 2);
                EXPECT_EQ(closes.load(), 0);
                EXPECT_EQ(environment.device_pool->refCount(environment.block_id), refs);
                EXPECT_FALSE(environment.device_pool->isFrozen(environment.block_id));
                ASSERT_TRUE(backend.backend->write(backend.backend->prepareWrite(makeStorageRequest(environment))));
                ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
                EXPECT_EQ(commits.load(), expire_queued ? 2 : 3);
                EXPECT_EQ(environment.device_pool->refCount(environment.block_id), refs);
            });
        }
    }
}

TEST(KVCMLocalTest, BroadcastWithoutAnyRankSubmissionClosesUnusedSession) {
    for (const bool enabled : {false, true}) {
        autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", enabled ? "1" : "0");
        ::rtp_llm::test::withDecodePublicationStartup([&] {
        auto environment = makeBackendEnvironment("broadcast_not_submitted");
        auto client = std::make_shared<MockClientWrapper>();
        EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
        EXPECT_CALL(*client, shutdown()).Times(1);
        auto parallel = singleRankConfig();
        parallel.tp_size = 2;
        parallel.role_type = RoleType::PREFILL;
        parallel.prefill_cp_config.prefill_cp_size = 2;
        KVCacheConfig config;
        config.kvcm_model_sdk_config = R"([{"type":"pace"}])";
        // No init(), hence no RPC pool/CQ and no rank can be submitted.
        auto broadcast = std::make_shared<BroadcastManager>(std::vector<std::string>{"unused-0", "unused-1"});
        config.kvcm_server_address = "unused-test-address";
        config.kvcm_instance_id_salt = "broadcast-not-submitted-" + std::to_string(enabled);
        RuntimeConfig runtime;
        runtime.model_name = "kvcm_test_model";
        BackendHandle backend(std::make_unique<MockQuiescentKVCMBackend>(environment.cache_config,
            config, runtime, parallel, SpeculativeExecutionConfig{}, broadcast, true, nullptr, client));
        ASSERT_TRUE(initSingleRank(*backend.backend, environment));
        const auto refs = environment.device_pool->refCount(environment.block_id);
        kv_cache_manager::WriteLocation allocation;
        allocation.write_session_id = "no-rank-started";
        allocation.block_mask = kv_cache_manager::BlockMaskOffset{0};
        allocation.locations = {{{"tp0_Fdefault", "pace://rank0"}, {"tp1_Fdefault", "pace://rank1"}}};
        EXPECT_CALL(*client, getWriteLocation(_, _, _, _, _, _, 0)).WillOnce(Return(std::make_pair(true, allocation)));
        EXPECT_CALL(*client, saveKvCachesForTag(_, _, _, _)).Times(0);
        EXPECT_CALL(*client, finishWrite(_, _, "no-rank-started",
                    ::testing::Eq(kv_cache_manager::BlockMask{kv_cache_manager::BlockMaskOffset{0}}),
                    ::testing::IsEmpty())).WillOnce(Return(true));
        ASSERT_TRUE(backend.backend->write(backend.backend->prepareWrite(makeStorageRequest(environment))));
        ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
        ASSERT_FALSE(backend.backend->protected_quarantine_); // On failure, the existing mock destructor can drain safely.
        backend.backend->shutdown();
        EXPECT_EQ(environment.device_pool->refCount(environment.block_id), refs);
        EXPECT_FALSE(environment.device_pool->isFrozen(environment.block_id));
        });
    }
}

TEST(KVCMLocalTest, PrefillKeepsOriginalTagRoutingForNonPaceUris) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1");
    ::rtp_llm::test::withDecodePublicationStartup([&] {
    auto environment = makeBackendEnvironment("local_preflight_not_started");
    auto client = std::make_shared<MockClientWrapper>();
    EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client, shutdown()).Times(1);
    auto backend = makeCompletionProofBackend(environment, client);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));
    const auto refs = environment.device_pool->refCount(environment.block_id);
    kv_cache_manager::WriteLocation allocation;
    allocation.write_session_id = "local-preflight";
    allocation.block_mask = kv_cache_manager::BlockMaskOffset{0};
    allocation.locations = {{{"tp0_Fdefault", "not-pace://rejected-before-sdk"}}};
    EXPECT_CALL(*client, getWriteLocation(_, _, _, _, _, _, 0)).WillOnce(Return(std::make_pair(true, allocation)));
    EXPECT_CALL(*client, saveKvCachesForTag("default", kv_cache_manager::UriStrVec{"not-pace://rejected-before-sdk"}, _, _))
        .WillOnce(Return(std::make_pair(true, kv_cache_manager::UriStrVec{"not-pace://actual"})));
    EXPECT_CALL(*client, finishWrite(_, _, "local-preflight",
                ::testing::Eq(kv_cache_manager::BlockMask{kv_cache_manager::BlockMaskOffset{1}}),
                ::testing::Not(::testing::IsEmpty()))).WillOnce(Return(true));
    ASSERT_TRUE(backend.backend->write(backend.backend->prepareWrite(makeStorageRequest(environment))));
    ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
    backend.backend->shutdown();
    EXPECT_EQ(environment.device_pool->refCount(environment.block_id), refs);
    EXPECT_FALSE(environment.device_pool->isFrozen(environment.block_id));
    });
}

TEST(KVCMLocalTest, FailureBeforeTransferStillAbortsItsUnusedAllocation) {
    autil::EnvGuard async("RTP_LLM_STREAM_ASYNC", "0");
    for (const bool enabled : {false, true}) {
        autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", enabled ? "1" : "0");
        ::rtp_llm::test::withDecodePublicationStartup([&] {
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
        });
    }
}

TEST(KVCMLocalTest, RuntimeDisableAfterAllocationAbortsBeforePayloadAndCanRecover) {
    autil::EnvGuard async("RTP_LLM_STREAM_ASYNC", "0");
    for (const bool protected_io : {false, true}) {
        autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", protected_io ? "1" : "0");
        ::rtp_llm::test::withDecodePublicationStartup([&] {
        auto environment = makeBackendEnvironment("kvcm_control_after_allocation");
        auto client = std::make_shared<MockClientWrapper>();
        EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
        EXPECT_CALL(*client, shutdown()).Times(1);
        auto backend = makeCompletionProofBackend(environment, client);
        ASSERT_TRUE(initSingleRank(*backend.backend, environment));
        const auto                      refs = environment.device_pool->refCount(environment.block_id);
        kv_cache_manager::WriteLocation allocation;
        allocation.write_session_id = "disabled-before-save";
        allocation.block_mask       = kv_cache_manager::BlockMaskOffset{0};
        allocation.locations        = {{{"tp0_Fdefault", "pace://write_uri"}}};
        EXPECT_CALL(*client, getWriteLocation(_, _, _, _, _, _, 0))
            .WillOnce(Invoke([&](const auto&, const auto&, const auto&, const auto&, const auto&, auto, auto) {
                backend.backend->setRemoteCacheEnabled(false);
                return std::make_pair(true, allocation);
            }));
        EXPECT_CALL(*client, saveKvCachesForTag(_, _, _, _)).Times(0);
        EXPECT_CALL(*client,
                    finishWrite(_,
                                _,
                                "disabled-before-save",
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
        });
    }
}

TEST(KVCMLocalTest, RuntimeDisableAfterMatchSkipsLoadWithoutQuarantine) {
    autil::EnvGuard async("RTP_LLM_STREAM_ASYNC", "0");
    for (const bool protected_io : {false, true}) {
        autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", protected_io ? "1" : "0");
        ::rtp_llm::test::withDecodePublicationStartup([&] {
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
        auto       local_request = makeStorageRequest(environment, {101, 102}, 1);
        const auto local         = match(*backend.backend, std::move(local_request));
        EXPECT_TRUE(local.success);
        EXPECT_EQ(local.matched_blocks_num, 1u);
        EXPECT_EQ(local.match_meta, nullptr);
        });
    }
}

TEST(KVCMLocalTest, RuntimeDisableDuringSaveStillFinishesAdmittedWrite) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "0");
    ::rtp_llm::test::withDecodePublicationStartup([&] {
    auto environment = makeBackendEnvironment("kvcm_control_during_save");
    auto client = std::make_shared<MockClientWrapper>();
    EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client, shutdown()).Times(1);
    auto backend = makeBackend(environment, singleRankConfig(), client);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));
    kv_cache_manager::WriteLocation allocation;
    allocation.write_session_id = "already-submitted";
    allocation.block_mask       = kv_cache_manager::BlockMaskOffset{0};
    allocation.locations        = {{{"tp0_Fdefault", "pace://write_uri"}}};
    EXPECT_CALL(*client, getWriteLocation(_, _, _, _, _, _, 0)).WillOnce(Return(std::make_pair(true, allocation)));
    EXPECT_CALL(*client, saveKvCachesForTag(_, _, _, _))
        .WillOnce(Invoke([&](const auto&, const auto& uris, const auto&, const auto&) {
            backend.backend->setRemoteCacheEnabled(false);
            return std::make_pair(true, uris);
        }));
    EXPECT_CALL(*client,
                finishWrite(_,
                            _,
                            "already-submitted",
                            ::testing::Eq(kv_cache_manager::BlockMask{kv_cache_manager::BlockMaskOffset{1}}),
                            _))
        .WillOnce(Return(true));
    const auto refs = environment.device_pool->refCount(environment.block_id);
    EXPECT_TRUE(backend.backend->write(backend.backend->prepareWrite(makeStorageRequest(environment))));
    ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
    EXPECT_EQ(environment.device_pool->refCount(environment.block_id), refs);
    EXPECT_FALSE(backend.backend->prepareWrite(makeStorageRequest(environment)));
    });
}

TEST(KVCMLocalTest, RuntimeDisableDuringTp2FanoutStillCompletesBothRanksAndFinishesWrite) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "0");
    ::rtp_llm::test::withDecodePublicationStartup([&] {
    auto environment = makeBackendEnvironment("kvcm_control_tp2");
    auto client = std::make_shared<MockClientWrapper>();
    EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client, shutdown()).Times(1);
    std::atomic<StorageBackend*> control_backend{nullptr};
    auto                         first_state  = std::make_shared<KVCMBroadcastState>();
    auto                         second_state = std::make_shared<KVCMBroadcastState>();
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
    auto parallel    = singleRankConfig();
    parallel.tp_size = 2;
    auto backend     = makeBackend(environment, parallel, client, broadcast);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));
    control_backend.store(backend.backend.get());
    kv_cache_manager::WriteLocation allocation;
    allocation.write_session_id = "tp2-admitted-before-disable";
    allocation.block_mask       = kv_cache_manager::BlockMaskOffset{0};
    allocation.locations        = {{{"tp0_Fdefault", "pace://rank0"}, {"tp1_Fdefault", "pace://rank1"}}};
    EXPECT_CALL(*client, getWriteLocation(_, _, _, _, _, _, 0)).WillOnce(Return(std::make_pair(true, allocation)));
    EXPECT_CALL(*client, saveKvCachesForTag(_, _, _, _)).Times(0);
    EXPECT_CALL(*client,
                finishWrite(_,
                            _,
                            "tp2-admitted-before-disable",
                            ::testing::Eq(kv_cache_manager::BlockMask{kv_cache_manager::BlockMaskOffset{1}}),
                            _))
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
    });
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

TEST(KVCMLocalTest, PrefillWorkerReportsSdkDispositionWithoutBackendWideStop) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1");
    ::rtp_llm::test::withDecodePublicationStartup([&] {
    autil::EnvGuard async("RTP_LLM_STREAM_ASYNC", "0");
    using EC = kv_cache_manager::ClientErrorCode;
    for (const auto& [ec, safe] : failedCompletionCases<EC>()) {
        for (const auto op : {REMOTE_OPERATION_READ, REMOTE_OPERATION_WRITE}) {
            // P worker keeps serving after one SDK error; controller status
            // handling decides ordinary failure versus timeout freezing.
            auto environment = makeBackendEnvironment("kvcm_response_completion_proof");
            auto client      = std::make_shared<MockClientWrapper>();
            EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
            EXPECT_CALL(*client, shutdown()).Times(1);
            auto backend = makeCompletionProofBackend(environment, client);
            ASSERT_TRUE(initSingleRank(*backend.backend, environment));
            client->load_status = ec;
            client->save_status = ec;
            RemoteOperationRequestPB request;
            request.set_op(op);
            identifyReadRequest(request);
            request.add_group_tags("default");
            request.add_block_ids(environment.block_id);
            request.add_uris("pace://uri");
            RemoteOperationResponsePB response;
            EXPECT_FALSE(backend->execute(request, response));
            EXPECT_EQ(response.transfer_status(),
                      ec == EC::ER_SDK_TIMEOUT ? REMOTE_TRANSFER_STATUS_TIMEOUT : REMOTE_TRANSFER_STATUS_FAILED);
            EXPECT_EQ(response.payload_buffer_safe(), safe);
            {
                // This checks a new operation, not replay of the original
                // execution identity.
                identifyReadRequest(request);
                RemoteOperationResponsePB rejected;
                EXPECT_FALSE(backend->execute(request, rejected));
                EXPECT_EQ(rejected.transfer_status(), response.transfer_status());
                EXPECT_EQ(rejected.payload_buffer_safe(), safe);
            }
        }
    }
    auto environment = makeBackendEnvironment("kvcm_preflight_completion_proof");
    auto client      = std::make_shared<MockClientWrapper>();
    EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client, shutdown()).Times(1);
    EXPECT_CALL(*client, loadKvCachesForTag(_, _, _, _)).Times(0);
    auto backend = makeCompletionProofBackend(environment, client);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));
    RemoteOperationRequestPB local_reject;
    local_reject.set_op(REMOTE_OPERATION_READ);
    identifyReadRequest(local_reject);
    local_reject.add_group_tags("foreign");
    local_reject.add_block_ids(environment.block_id);
    local_reject.add_uris("pace://uri");
    RemoteOperationResponsePB response;
    EXPECT_FALSE(backend->execute(local_reject, response));
    EXPECT_EQ(response.transfer_status(), REMOTE_TRANSFER_STATUS_FAILED);
    EXPECT_TRUE(response.payload_buffer_safe());
    });
}

TEST(KVCMLocalTest, PrefillUnsafePaceBatchFreezesEveryDestinationGroupWithoutGlobalStop) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1");
    ::rtp_llm::test::withDecodePublicationStartup([&] {
    autil::EnvGuard async("RTP_LLM_STREAM_ASYNC", "0");
    auto            environment = makeHybridBackendEnvironment("kvcm_group_completion_proof");
    auto            client      = std::make_shared<MockClientWrapper>();
    std::string     first_tag;
    EXPECT_CALL(*client, initForPools(_, _, _, _))
        .WillOnce(Invoke([&](const auto&, auto, const auto&, const auto& tags) {
            first_tag = tags.front();
            return true;
        }));
    EXPECT_CALL(*client, shutdown()).Times(1);
    auto backend = makeCompletionProofBackend(environment, client);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));
    kv_cache_manager::Locations locations{{{"tp0_Ffull1", "pace://full_uri"}, {"tp0_Llinear", "pace://linear_uri"}}};
    EXPECT_CALL(*client, match(_, _, _, _, _, _)).WillOnce(Return(std::make_pair(true, locations)));
    auto request     = makeGroupedStorageRequest(environment, {101}, 0, {environment.block_id});
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
    EXPECT_FALSE(backend.backend->protected_quarantine_);
    for (const auto& [tag, pool] : environment.pools_by_tag) {
        SCOPED_TRACE(tag);
        EXPECT_TRUE(pool->isFrozen(environment.block_id));
        EXPECT_EQ(pool->refCount(environment.block_id), references.at(tag) + 1u);
    }
    EXPECT_FALSE(backend.backend->hasUnresolvedExternalIo());
    backend.backend->shutdown();
    for (const auto& [tag, pool] : environment.pools_by_tag)
        EXPECT_EQ(pool->refCount(environment.block_id), references.at(tag) + 1u);
    });
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

TEST(KVCMLocalTest, TP2PayloadWaitReleasesCoordinatorAndShutdownDrainsSettlement) {
    autil::EnvGuard decode_publish("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "0");
    ::rtp_llm::test::withDecodePublicationStartup([&] {
    for (const auto& [write, late] : {std::make_pair(false, false), std::make_pair(false, true),
                                      std::make_pair(true, false), std::make_pair(true, true)}) {
        SCOPED_TRACE(::testing::Message() << "write=" << write << ", late=" << late);
        auto environment = std::make_shared<BackendEnvironment>(
            makeBackendEnvironment(write ? "kvcm_tp2_deferred_write" : "kvcm_tp2_deferred_read"));
        auto client = std::make_shared<MockClientWrapper>();
        EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
        EXPECT_CALL(*client, shutdown()).Times(1);
        auto first_state         = std::make_shared<KVCMBroadcastState>();
        auto second_state        = std::make_shared<KVCMBroadcastState>();
        auto entered             = std::make_shared<std::promise<void>>();
        auto release             = std::make_shared<std::promise<void>>();
        auto rank_entered        = entered->get_future();
        auto released            = release->get_future().share();
        second_state->on_request = [entered, released, first_call = std::make_shared<std::atomic<bool>>(true)] {
            if (first_call->exchange(false)) {
                entered->set_value();
                released.wait();
            }
        };
        KVCMBroadcastRpcServer first(0, first_state), second(1, second_state);
        ASSERT_TRUE(first.start());
        ASSERT_TRUE(second.start());
        auto broadcast =
            std::make_shared<BroadcastManager>(std::vector<std::string>{first.address(), second.address()});
        ASSERT_TRUE(broadcast->init());
        auto parallel    = singleRankConfig();
        parallel.tp_size = 2;
        KVCacheConfig config;
        config.kvcm_asyncwrapper_thread_num = 1;
        config.kvcm_asyncwrapper_queue_size = 0;
        auto backend                        = std::make_shared<BackendHandle>(
            makeBackend(*environment, parallel, client, broadcast, nullptr, false, config));
        block_tree_cache_detail::ScopeRollback unblock([release] { release->set_value(); });
        ASSERT_TRUE(initSingleRank(*backend->backend, *environment));
        auto profile = AsyncProfileSession::start(128);
        ASSERT_TRUE(profile);
        block_tree_cache_detail::ScopeRollback stop_profile([&] { profile->finish(); });
        const kv_cache_manager::Locations      locations{
                 {{"tp0_Fdefault", "pace://read0"}, {"tp1_Fdefault", "pace://read1"}}};
        auto request       = makeStorageRequest(*environment);
        request.request_id = 42;
        if (late)
            request.transfer_task.emplace(std::vector<TransferDescriptor>{}, std::chrono::seconds(2));
        const auto refs       = environment->device_pool->refCount(environment->block_id);
        auto       completed  = std::make_shared<std::promise<void>>();
        auto       result     = completed->get_future();
        auto       progressed = std::make_shared<std::promise<bool>>();
        auto       next       = progressed->get_future();
        if (write) {
            kv_cache_manager::WriteLocation allocation;
            allocation.write_session_id      = "deferred-write";
            allocation.block_mask            = kv_cache_manager::BlockMaskOffset{0};
            allocation.locations             = locations;
            auto next_allocation             = allocation;
            next_allocation.write_session_id = "next-write";
            EXPECT_CALL(*client, getWriteLocation(_, _, _, _, _, _, 0))
                .WillOnce(Return(std::make_pair(true, allocation)))
                .WillOnce(Return(std::make_pair(true, next_allocation)));
            const auto all_written = ::testing::Eq(kv_cache_manager::BlockMask{kv_cache_manager::BlockMaskOffset{1}});
            EXPECT_CALL(*client, finishWrite(_, _, "next-write", all_written, _))
                .WillOnce(::testing::InvokeWithoutArgs([progressed] {
                    progressed->set_value(true);
                    return true;
                }));
            auto acknowledge = ::testing::InvokeWithoutArgs([completed] {
                completed->set_value();
                return true;
            });
            if (late) {
                EXPECT_CALL(*client, finishWrite(_, _, "deferred-write", all_written, _)).Times(0);
                EXPECT_CALL(*client, finishWrite(_, _, "deferred-write",
                            ::testing::Eq(kv_cache_manager::BlockMask{kv_cache_manager::BlockMaskOffset{0}}),
                            ::testing::IsEmpty())).WillOnce(acknowledge);
            } else {
                EXPECT_CALL(*client, finishWrite(_, _, "deferred-write", all_written, _))
                    .WillOnce(acknowledge);
            }
            ASSERT_TRUE(backend->backend->write(backend->backend->prepareWrite(request)));
        } else {
            EXPECT_CALL(*client, match(_, _, _, _, _, _)).WillRepeatedly(Return(std::make_pair(true, locations)));
            auto matched = match(*backend->backend, request);
            ASSERT_TRUE(matched.success);
            backend->backend->read(request, matched.match_meta, [completed, expect_late = late](ErrorInfo e) {
                if (expect_late) {
                    EXPECT_EQ(e.code(), ErrorCode::DEADLINE_EXCEEDED);
                } else {
                    EXPECT_TRUE(e.ok());
                }
                completed->set_value();
            });
        }
        ASSERT_EQ(rank_entered.wait_for(std::chrono::seconds(5)), std::future_status::ready);
        EXPECT_EQ(result.wait_for(std::chrono::milliseconds(0)), std::future_status::timeout);
        EXPECT_EQ(environment->device_pool->refCount(environment->block_id), refs + 1);
        if (write) {
            auto next_request       = makeStorageRequest(*environment, {102});
            next_request.request_id = 43;
            ASSERT_TRUE(backend->backend->write(backend->backend->prepareWrite(std::move(next_request))));
        } else {
            backend->backend->match(request, [progressed](size_t, auto, bool ok) { progressed->set_value(ok); });
        }
        ASSERT_EQ(next.wait_for(std::chrono::seconds(5)), std::future_status::ready);
        EXPECT_TRUE(next.get());
        if (late) {
            auto                                   worker_entered = std::make_shared<std::promise<void>>();
            auto                                   worker         = worker_entered->get_future();
            std::promise<void>                     release_worker;
            auto                                   worker_released = release_worker.get_future().share();
            block_tree_cache_detail::ScopeRollback unblock_worker([&] { release_worker.set_value(); });
            ASSERT_TRUE(backend->backend->executor_->submit(
                [worker_entered, worker_released] {
                    worker_entered->set_value();
                    worker_released.wait();
                },
                BlockTreeTaskClass::LOAD));
            ASSERT_EQ(worker.wait_for(std::chrono::seconds(5)), std::future_status::ready);
            unblock.run();
            const auto deadline = request.transfer_task->deadline();
            // Observe successful RPC completion queued before the deadline, then
            // delay only its settlement. A transport timeout is a different case.
            while (backend->backend->executor_->queueSizes().completion == 0
                   && std::chrono::steady_clock::now() < deadline)
                ASSERT_EQ(result.wait_for(std::chrono::milliseconds(1)), std::future_status::timeout);
            ASSERT_EQ(backend->backend->executor_->queueSizes().completion, 1u);
            ASSERT_EQ(result.wait_until(deadline), std::future_status::timeout);
        }
        block_tree_cache_test::BoundedThread<void> stopping([backend, environment] { backend->backend->shutdown(); });
        if (!late) {
            EXPECT_EQ(stopping.waitFor(std::chrono::milliseconds(50)), std::future_status::timeout);
            EXPECT_EQ(result.wait_for(std::chrono::milliseconds(0)), std::future_status::timeout);
            unblock.run();
        }
        ASSERT_EQ(stopping.waitFor(std::chrono::seconds(5)), std::future_status::ready);
        stopping.get();
        // Writes have no public completion callback. Shutdown must drain the
        // actual business settlement, not manufacture an abort ACK to signal it.
        ASSERT_TRUE(waitForBackendOperationsForTest(*backend->backend));
        if (!write || !late) {
            ASSERT_EQ(result.wait_for(std::chrono::seconds(5)), std::future_status::ready);
            result.get();
        }
        EXPECT_FALSE(environment->device_pool->isFrozen(environment->block_id));
        EXPECT_EQ(environment->device_pool->refCount(environment->block_id), refs);
        const auto events = profile->finish();
        stop_profile.dismiss();
        const std::string io_stage = write ? "write_io" : "read_io";
        const auto        count    = std::count_if(events.begin(), events.end(), [&](const auto& event) {
            return event.record.stage == io_stage && event.record.kind == AsyncProfileKind::Lifecycle;
        });
        EXPECT_EQ(count, write ? 2 : 1);
        const auto trace = mergeAsyncProfileTrace(R"({"traceEvents":[]})", events);
        EXPECT_EQ(trace.find("cache.transfer.storage_backend." + io_stage + "("), std::string::npos);
        for (const auto* stage : {write ? "write_submit" : "read_submit", write ? "write_complete" : "read_complete"}) {
            size_t executions = 0;
            for (const auto& event : events) {
                if (event.record.stage != stage)
                    continue;
                ++executions;
                EXPECT_EQ(event.record.kind, AsyncProfileKind::Execution);
                EXPECT_GT(event.record.start_tid, 0);
                EXPECT_EQ(event.record.start_tid, event.end_tid);
                EXPECT_GE(event.record.request_id, 0);
                EXPECT_NE(trace.find(event.name), std::string::npos);
            }
            EXPECT_EQ(executions, write ? 2u : 1u) << stage;
        }
    }
    });
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
    const auto tree_refs_before = environment.device_pool->treeRefCount(environment.block_id);
    EXPECT_CALL(*client_wrapper, loadKvCachesForTag("default", kv_cache_manager::UriStrVec{"read_uri"}, _, _))
        .WillOnce(Return(false));
    const auto error = readResult(*backend.backend, makeStorageRequest(environment), std::move(observation.match_meta));
    EXPECT_EQ(error.code(), ErrorCode::EXECUTION_EXCEPTION);
    ASSERT_TRUE(waitForBackendOperationsForTest(*backend.backend));
    // A returned error without explicit buffer safety follows the upstream
    // task-local freeze policy; it must not be downgraded to ordinary failure.
    EXPECT_EQ(environment.device_pool->refCount(environment.block_id), ref_count_before + 1);
    EXPECT_TRUE(environment.device_pool->isFrozen(environment.block_id));
    EXPECT_EQ(environment.device_pool->treeRefCount(environment.block_id), tree_refs_before + 1);
}

TEST(KVCMLocalTest, PayloadReadTimeoutFreezesOnlyDestinationBlock) {
    autil::EnvGuard quarantine_abort_ratio("RTP_LLM_REMOTE_CACHE_QUARANTINE_ABORT_RATIO", "0.125");
    auto            environment      = makeBackendEnvironment("kvcm_storage_backend_read_timeout");
    auto            client_wrapper   = std::make_shared<MockClientWrapper>();
    auto            metrics_reporter = std::make_shared<kmonitor::MetricsReporter>("", "", kmonitor::MetricsTags{});

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
    const auto error = readResult(*backend.backend, makeStorageRequest(environment), std::move(observation.match_meta));
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
    // Healthy I/O must leave the reported quarantine stocks unchanged.
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
