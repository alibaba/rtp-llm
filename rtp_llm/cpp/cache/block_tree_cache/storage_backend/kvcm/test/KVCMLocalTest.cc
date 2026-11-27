#include "rtp_llm/cpp/utils/test/DecodePublicationTestStartup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test/KVCMMockTestBase.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <unistd.h>

#include "rtp_llm/cpp/cache/DeviceBlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/events/KVCacheEventPublisher.h"
#include "rtp_llm/cpp/cache/events/KVCacheEventReporter.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"
#include "rtp_llm/cpp/utils/AsyncProfilingScope.h"
#include "rtp_llm/cpp/utils/AsyncProfileTrace.h"
#include "rtp_llm/cpp/utils/Logger.h"
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
    for (int mode = 0; mode < 8; ++mode) {
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
        config.kv_cache_event_publisher_type       = mode == 0 ? "none" : (mode == 6 ? "unknown" : "kvcm");
        config.reuse_cache                         = mode != 1;
        config.enable_device_cache                 = mode != 2;
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
        auto backend = std::make_shared<KVCMStorageBackend>(environment.cache_config,
                                                            config,
                                                            runtime,
                                                            singleRankConfig(),
                                                            SpeculativeExecutionConfig{},
                                                            nullptr,
                                                            false,
                                                            nullptr,
                                                            client);
        auto manager_config = environment.cache_config;
        if (mode == 5) {
            auto groups                               = manager_config.topology().groups();
            groups.front().policy.enable_prefix_reuse = false;
            auto layers                               = manager_config.topology().layers();
            manager_config.setTopology(std::move(groups), std::move(layers));
        }
        KVCacheManager manager(manager_config, false, nullptr, config, singleRankConfig(), runtime);
        auto           full =
            std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{environment.device_pool}, nullptr, nullptr);
        full->initialize(0, manager_config.topologyPtr(), {"default"});
        manager.block_tree_cache_                   = block_tree_cache_test::makeBlockTreeCacheForTest({full});
        manager.block_tree_cache_->storage_backend_ = backend;
        backend->setReclaimer(manager.block_tree_cache_->reclaimer_);
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

TEST(KVCMLocalTest, Dsv41PublisherUsesEffectiveBackendFullSpecMerge) {
    for (const std::string model_name : {"DeepSeekV41", "DeepSeekV41DSpark"}) {
        for (const bool tair_backend : {false, true}) {
            SCOPED_TRACE(model_name);
            SCOPED_TRACE(tair_backend);
            std::vector<std::string> tags{"hca_kv", "csa_kv", "indexer_kv"};
            BackendEnvironment       environment;
            environment.cache_config.dtype              = DataType::TYPE_FP16;
            environment.cache_config.layer_num          = tags.size();
            environment.cache_config.seq_size_per_block = 8;
            std::vector<KVCacheSpecPtr>   specs;
            std::vector<std::vector<int>> layers;
            for (size_t index = 0; index < tags.size(); ++index) {
                specs.push_back(test::makeMhaSpec(tags[index], 8, DataType::TYPE_FP16, 1, 2));
                layers.push_back({static_cast<int>(index)});
            }
            environment.cache_config.fromGroupedSpecs(
                specs, layers, std::vector<CacheGroupType>(tags.size(), CacheGroupType::FULL), tags);
            environment.cache_config.finalizeBlockNums(8, RuntimeConfig{});
            initializeEnvironmentPools(environment, "kvcm_three_full_control", 8);

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
            config.enable_remote_cache                 = true;
            config.kvcm_server_address                 = "outer-rpc";
            config.kvcm_model_sdk_config               = tair_backend ? R"([{"type":"pace"}])" : R"([{"type":"file"}])";
            config.reuse_cache                         = true;
            config.enable_device_cache                 = true;
            config.kv_cache_event_publisher_type       = "kvcm";
            config.kv_cache_event_instance_id          = "hbm-event-instance";
            config.kv_cache_event_host_ip_port         = "127.0.0.1:19999";
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
            KVCacheManager manager(environment.cache_config, false, nullptr, config, singleRankConfig(), runtime);
            std::vector<DeviceBlockPoolPtr> pools;
            for (const auto& tag : tags) {
                pools.push_back(environment.pools_by_tag.at(tag));
            }
            auto full = std::make_shared<FullGroupSet>(std::move(pools), nullptr, nullptr);
            full->initialize(0, environment.cache_config.topologyPtr(), tags);
            manager.block_tree_cache_                   = block_tree_cache_test::makeBlockTreeCacheForTest({full});
            manager.block_tree_cache_->storage_backend_ = backend;
            backend->setReclaimer(manager.block_tree_cache_->reclaimer_);
            ASSERT_TRUE(initSingleRank(*backend, environment));
            ASSERT_TRUE(backend->controlConfig().has_value());
            EXPECT_EQ(backend->controlConfig()->merge_full_specs, tair_backend);
            std::set<std::string> payload_names;
            for (const auto& [name, bytes] : actual_payload_specs) {
                EXPECT_GT(bytes, 0);
                payload_names.insert(name);
            }
            std::set<std::string> expected_payload{"F0_TP0", "F1_TP0", "F2_TP0"};
            EXPECT_EQ(payload_names, tair_backend ? std::set<std::string>{"F_TP0"} : expected_payload);
            auto reporter = std::make_shared<ControlOnlyReporter>();
            reporter->enabled.store(true);
            manager.initCacheEventPublisher(reporter);
            ASSERT_NE(manager.cache_event_publisher_, nullptr);
            ASSERT_TRUE(reporter->waitFor("EVENT_NODE_REGISTER"));
            EXPECT_EQ(manager.block_tree_cache_->tree()->publication_groups_.size(), tair_backend ? 1 : tags.size());
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
            EXPECT_EQ(event_names, tair_backend ? std::set<std::string>{"F"} : expected_events);
        }
    }
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
    kv_cache_manager::BlockBuffers buffers;
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

TEST(KVCMLocalTest, GdrFallsBackToStandardRegistrationForHostPinnedPool) {
    auto environment = makeHybridBackendEnvironment("kvcm_storage_backend_mixed_gdr_span");
    ASSERT_EQ(environment.cache_config.groupTags().size(), 2u);

    const auto& host_tag = environment.cache_config.groupTags().back();
    ASSERT_NE(host_tag, environment.cache_config.groupTags().front());
    auto old_pool = environment.pools_by_tag.at(host_tag);
    old_pool->decRef(environment.block_id);

    auto host_config       = std::make_shared<DeviceBlockPoolConfig>(DeviceBlockPoolConfigHelper::createConfigForGroup(
        environment.cache_config, environment.cache_config.topology().group(host_tag)));
    host_config->pool_name = "mixed_gdr_host_pinned";
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
    EXPECT_CALL(*client_wrapper, initForPools(_, _, _, _))
        .WillOnce(Invoke([&](const kvcm::ClientWrapper::ConfigMap&,
                             kv_cache_manager::RoleType,
                             const std::vector<kvcm::ClientWrapper::PoolRegistration>& registrations,
                             const std::vector<std::string>&                           tags) {
            EXPECT_EQ(registrations.size(), environment.cache_config.groupTags().size());
            EXPECT_EQ(tags.size(), registrations.size());
            if (registrations.size() != tags.size()) {
                return false;
            }
            for (size_t index = 0; index < tags.size(); ++index) {
                const auto& pool         = environment.pools_by_tag.at(tags[index]);
                const auto& registration = registrations[index];
                EXPECT_EQ(registration.span.base, pool->getBaseAddress());
                if (tags[index] == host_tag) {
                    EXPECT_EQ(pool->where(), MemoryType::MEMORY_CPU_PINNED);
                    EXPECT_EQ(registration.span.size, pool->getTotalSizeBytes());
                    EXPECT_FALSE(registration.memory_registrations.has_value());
                } else {
                    EXPECT_TRUE(pool->usesDedicatedDeviceAllocation());
                    EXPECT_EQ(registration.span.size, pool->getAllocationSizeBytes());
                    if (!registration.memory_registrations.has_value()) {
                        ADD_FAILURE() << "expected GDR memory registration for device pool";
                        return false;
                    }
                    if (registration.memory_registrations->gpu.size() != 1u) {
                        ADD_FAILURE() << "expected one GPU registration for device pool";
                        return false;
                    }
                    EXPECT_EQ(registration.memory_registrations->gpu.front().base, pool->getBaseAddress());
                    EXPECT_EQ(registration.memory_registrations->gpu.front().size, pool->getAllocationSizeBytes());
                }
            }
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
    for (const auto& [write, late] : {std::make_pair(false, false),
                                      std::make_pair(false, true),
                                      std::make_pair(true, false),
                                      std::make_pair(true, true)}) {
        SCOPED_TRACE(::testing::Message() << "write=" << write << ", late=" << late);
        auto environment = std::make_shared<BackendEnvironment>(
            makeBackendEnvironment(write ? "kvcm_tp2_deferred_write" : "kvcm_tp2_deferred_read"));
        auto client = std::make_shared<MockClientWrapper>();
        EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
        EXPECT_CALL(*client, shutdown()).Times(1);
        auto first_state            = std::make_shared<KVCMBroadcastState>();
        auto second_state           = std::make_shared<KVCMBroadcastState>();
        auto entered                = std::make_shared<std::promise<void>>();
        auto release                = std::make_shared<std::promise<void>>();
        auto rank_entered           = entered->get_future();
        auto released               = release->get_future().share();
        second_state->worker_action = [entered, released, first_call = std::make_shared<std::atomic<bool>>(true)] {
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
                EXPECT_CALL(*client, finishWrite(_, _, "deferred-write", _, _)).Times(0);
                EXPECT_CALL(*client, abortWrite(_, _, "deferred-write")).WillOnce(acknowledge);
            } else {
                EXPECT_CALL(*client, abortWrite(_, _, _)).Times(0);
                EXPECT_CALL(*client, finishWrite(_, _, "deferred-write", all_written, _)).WillOnce(acknowledge);
            }
            ASSERT_TRUE(backend->backend->write(backend->backend->prepareWrite(request)));
        } else {
            EXPECT_CALL(*client, match(_, _, _, _, _, _)).WillRepeatedly(Return(std::make_pair(true, locations)));
            auto matched = match(*backend->backend, request);
            ASSERT_TRUE(matched.success);
            backend->backend->read(request, matched.match_meta, [completed, expect_late = late](ErrorInfo e) {
                if (expect_late)
                    EXPECT_EQ(e.code(), ErrorCode::DEADLINE_EXCEEDED);
                else
                    EXPECT_TRUE(e.ok());
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
        ASSERT_EQ(result.wait_for(std::chrono::seconds(5)), std::future_status::ready);
        result.get();
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
    auto environment    = makeBackendEnvironment("kvcm_storage_backend_read_timeout");
    auto client_wrapper = std::make_shared<MockClientWrapper>();

    EXPECT_CALL(*client_wrapper, initForPools(_, _, _, _)).WillOnce(Return(true));
    EXPECT_CALL(*client_wrapper, shutdown()).Times(1);
    auto backend =
        makeBackend(environment, singleRankConfig(), client_wrapper, nullptr, nullptr, /*gdr_enabled=*/false);
    ASSERT_TRUE(initSingleRank(*backend.backend, environment));

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
    EXPECT_EQ(environment.device_pool->referencedBlocksNum(BlockTreeRefType::FROZEN), 1u);

    backend.backend->shutdown();
    EXPECT_EQ(environment.device_pool->refCount(environment.block_id), ref_count_before + 1);
    EXPECT_TRUE(environment.device_pool->isFrozen(environment.block_id));
    EXPECT_EQ(environment.device_pool->referencedBlocksNum(BlockTreeRefType::FROZEN), 1u);
}

TEST(KVCMLocalTest, AutomaticIdentityUsesResolvedCheckpointWithoutChangingEnvironment) {
    const auto instance = [](const std::string& source, const std::string& resolved) {
        autil::EnvGuard checkpoint("CHECKPOINT_PATH", source);
        auto environment = makeBackendEnvironment("checkpoint_identity");
        auto client = std::make_shared<MockClientWrapper>();
        std::string id;
        EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Invoke(
            [&](const auto& configs, auto, const auto&, const auto&) {
                id = configs.at("")->instance_id();
                return true;
            }));
        EXPECT_CALL(*client, shutdown()).Times(1);
        KVCacheConfig config;
        config.kvcm_model_checkpoint_path = resolved;
        auto backend = makeBackend(environment, singleRankConfig(), client, nullptr, nullptr, false, config);
        EXPECT_TRUE(initSingleRank(*backend.backend, environment));
        EXPECT_FALSE(id.empty());
        EXPECT_EQ(autil::EnvUtil::getEnv("CHECKPOINT_PATH", std::string("")), source);
        backend.backend->shutdown();
        return id;
    };
    // Equivalent to the old env-sync hash, while leaving the raw URI visible to other readers.
    const auto local = instance("/mnt/fuse/a", "");
    EXPECT_EQ(instance("oss://models/target", "/mnt/fuse/a"), local);
    EXPECT_EQ(instance("oss://models/stale", "/mnt/fuse/a"), local);
    EXPECT_NE(instance("oss://models/target", ""), local);
    EXPECT_NE(instance("oss://models/target", "/mnt/fuse/b"), local);
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
        {R"([{"type":"pace"}])", 0, false, true},
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

std::pair<std::string, std::string> observeRoleGeometry(RoleType role, int tp, int dp, int target_cp,
                                                        const std::string& salt, bool check_read = false) {
    auto environment = makeBackendEnvironment("read_publish_geometry");
    auto client = std::make_shared<MockClientWrapper>();
    auto metadata = std::make_shared<MockClientWrapper>();
    const bool projected = role == RoleType::DECODE && (target_cp != tp || dp != 1);
    std::string native_id;
    std::string publication_id;
    const auto check_specs = [](const kvcm::KVCMConfigPtr& config, int ranks) {
        const auto fields = autil::legacy::AnyCast<autil::legacy::json::JsonMap>(autil::legacy::ToJson(config));
        kvcm::KVCMConfig::LocationSpecInfoMap infos;
        autil::legacy::FromJson(infos, fields.at("location_spec_infos"));
        EXPECT_EQ(infos.size(), static_cast<size_t>(ranks));
        for (int rank = 0; rank < ranks; ++rank) {
            EXPECT_EQ(infos.count("tp" + std::to_string(rank) + "_Fdefault"), 1u);
        }
    };
    EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Invoke(
        [&](const auto& configs, auto, const auto& registrations, const auto& tags) {
            native_id = configs.at("")->instance_id();
            check_specs(configs.at(""), tp);
            EXPECT_EQ(tags, std::vector<std::string>{"default"});
            EXPECT_EQ(registrations.size(), 1u);
            return true;
        }));
    EXPECT_CALL(*client, shutdown()).Times(1);
    EXPECT_CALL(*metadata, shutdown()).Times(projected ? 1 : 0);
    EXPECT_CALL(*client, createMetadataOnlyClient(_)).Times(projected ? 1 : 0).WillRepeatedly(Invoke(
        [&](const auto& config) {
            publication_id = config->instance_id();
            check_specs(config, target_cp);
            return metadata;
        }));
    KVCacheConfig config;
    config.kvcm_model_sdk_config = R"([{"type":"pace","sdk_log_level":"INFO"}])";
    config.kvcm_instance_id_salt = salt;
    auto parallel = singleRankConfig();
    parallel.role_type = role;
    parallel.tp_size = tp;
    parallel.dp_size = dp;
    parallel.prefill_cp_config.prefill_cp_size = target_cp;
    std::shared_ptr<BroadcastManager> broadcast;
    if (tp > 1)
        broadcast = std::make_shared<BroadcastManager>(std::vector<std::string>(tp, "unused-geometry-peer"));
    auto backend = makeBackend(environment, parallel, client, std::move(broadcast), nullptr, true, config);
    EXPECT_TRUE(initSingleRank(*backend.backend, environment));
    EXPECT_FALSE(native_id.empty());
    if (check_read) {
        const kv_cache_manager::Locations expected{{{"tp0_Fdefault", "pace://native"}}};
        EXPECT_CALL(*metadata, match(_, _, _, _, _, _)).Times(0);
        EXPECT_CALL(*client, match("", _, _, _, _, _)).WillOnce(Return(std::make_pair(true, expected)));
        EXPECT_CALL(*client, loadKvCachesForTag("default", kv_cache_manager::UriStrVec{"pace://native"}, _, _))
            .WillOnce(Return(true));
        auto observation = match(*backend.backend, makeStorageRequest(environment));
        EXPECT_TRUE(observation.success);
        EXPECT_NE(observation.match_meta, nullptr);
        if (observation.match_meta) {
            EXPECT_TRUE(read(*backend.backend, makeStorageRequest(environment), std::move(observation.match_meta)));
        }
    }
    backend.backend->shutdown();
    return std::make_pair(native_id, projected ? publication_id : native_id);
}

TEST(KVCMLocalTest, OptionalSaltPreservesPrefillDecodeIdentityAndLayoutIsolation) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1");
    ::rtp_llm::test::withDecodePublicationStartup([&] {
        const auto p = observeRoleGeometry(RoleType::PREFILL, 1, 1, 1, "").first;
        const auto d = observeRoleGeometry(RoleType::DECODE, 1, 1, 1, "").second;
        EXPECT_EQ(p, d);
        EXPECT_EQ(observeRoleGeometry(RoleType::PREFILL, 1, 1, 1, "explicit-salt").first, "explicit-salt_" + p);
        EXPECT_EQ(observeRoleGeometry(RoleType::DECODE, 1, 1, 1, "explicit-salt").second, "explicit-salt_" + d);
        EXPECT_NE(observeRoleGeometry(RoleType::DECODE, 1, 1, 2, "").second, d);
        EXPECT_NE(observeRoleGeometry(RoleType::DECODE, 1, 1, 2, "explicit-salt").second, "explicit-salt_" + d);
    });
}

TEST(KVCMLocalTest, PrefillIdentityAndSpecsUseOwnGeometryWhileOnlyDecodeProjects) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1");
    ::rtp_llm::test::withDecodePublicationStartup([&] {
        const auto p11 = observeRoleGeometry(RoleType::PREFILL, 1, 1, 1, "").first;
        EXPECT_EQ(observeRoleGeometry(RoleType::PREFILL, 1, 1, 2, "").first, p11);
        const auto p12 = observeRoleGeometry(RoleType::PREFILL, 1, 2, 1, "").first;
        EXPECT_NE(p12, p11);
        EXPECT_EQ(observeRoleGeometry(RoleType::PREFILL, 1, 2, 2, "").first, p12);
        EXPECT_EQ(observeRoleGeometry(RoleType::DECODE, 1, 1, 1, "").first, p11);
        EXPECT_EQ(observeRoleGeometry(RoleType::DECODE, 1, 1, 1, "explicit-salt").second, "explicit-salt_" + p11);
        for (const auto cp : {2, 4}) {
            const auto p = observeRoleGeometry(RoleType::PREFILL, cp, 1, cp, "").first;
            EXPECT_EQ(observeRoleGeometry(RoleType::PREFILL, cp, 1, 1, "").first, p);
            EXPECT_NE(p, p11);
            EXPECT_EQ(observeRoleGeometry(RoleType::DECODE, 1, 1, cp, ""), std::make_pair(p11, p));
            EXPECT_EQ(observeRoleGeometry(RoleType::DECODE, 1, 2, cp, ""), std::make_pair(p12, p));
        }
    });
}

TEST(KVCMLocalTest, ExplicitDecodePublicationPreservesMergedAndUnmergedLayouts) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1");
    ::rtp_llm::test::withDecodePublicationStartup([&] {
        BackendEnvironment env;
        auto& cache = env.cache_config;
        cache.dtype = DataType::TYPE_FP16;
        cache.layer_num = 3;
        cache.seq_size_per_block = 8;
        const std::vector<std::string> tags{"hca_kv", "csa_kv", "indexer_kv"};
        std::vector<KVCacheSpecPtr> specs;
        for (const auto& tag : tags) {
            specs.push_back(test::makeMhaSpec(tag, 8, DataType::TYPE_FP16, 1, 2));
        }
        cache.fromGroupedSpecs(specs, {{0}, {1}, {2}}, std::vector<CacheGroupType>(3, CacheGroupType::FULL), tags);
        cache.finalizeBlockNums(8, RuntimeConfig{});
        initializeEnvironmentPools(env, "kvcm_explicit_projection", 8);
        const auto config_for = [&](int tp, bool merged, int block, const std::string& id) {
            auto sdk = std::make_shared<kvcm::SdkWrapperConfig>();
            sdk->parseBackendConfigs(R"([{"type":"pace"}])");
            auto infos = std::make_shared<kvcm::KVCMConfig::LocationSpecInfoMap>();
            auto groups = std::make_shared<kvcm::KVCMConfig::LocationSpecGroups>();
            for (size_t group = 0; group < tags.size(); ++group) {
                const auto name = "F" + std::to_string(group);
                for (int rank = 0; rank < tp; ++rank) {
                    const auto spec = (merged ? std::string("F") : name) + "_TP" + std::to_string(rank);
                    (*infos)[spec] += cache.blockSizeBytesForGroup(tags[group]);
                    if (!merged) {
                        (*groups)[name].push_back(spec);
                    }
                    if (merged && group == 0) {
                        (*groups)["F"].push_back(spec);
                    }
                }
            }
            if (!merged) {
                // KVCM's aggregate uses canonical tag order (csa, hca, indexer).
                for (const auto* name : {"F1", "F0", "F2"}) {
                    const auto& source = groups->at(name);
                    auto& full = (*groups)["F"];
                    full.insert(full.end(), source.begin(), source.end());
                }
            }
            return std::make_shared<kvcm::KVCMConfig>(
                false, "", block, "explicit", id, std::vector<std::string>{"unused"}, infos,
                std::make_shared<kvcm::MetaChannelConfig>(), sdk, groups,
                kvcm::ModelDeployment("DeepSeekV41", "fp16", false, tp, 1, 1, "", ""));
        };
        auto parallel = singleRankConfig();
        parallel.role_type = RoleType::DECODE;
        RuntimeConfig runtime;
        runtime.model_name = "DeepSeekV41";
        std::string auto_native_id;
        {
            auto client = std::make_shared<MockClientWrapper>();
            EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Invoke(
                [&](const auto& map, auto, const auto&, const auto&) {
                    EXPECT_TRUE(map.at("")->hasLocationSpec("F_TP0"));
                    auto_native_id = map.at("")->instance_id();
                    return true;
                }));
            EXPECT_CALL(*client, shutdown()).Times(1);
            KVCacheConfig settings;
            settings.kvcm_server_address = "unused";
            settings.kvcm_instance_group = "explicit";
            settings.kvcm_model_sdk_config = R"([{"type":"pace"}])";
            BackendHandle baseline(std::make_unique<KVCMStorageBackend>(
                cache, settings, runtime, parallel, SpeculativeExecutionConfig{}, nullptr, false, nullptr, client));
            ASSERT_TRUE(initSingleRank(*baseline.backend, env));
            ASSERT_FALSE(auto_native_id.empty());
        }
        for (const int cp : {2, 4}) {
            for (const bool merged : {false, true}) {
                for (const bool separate : {false, true}) {
                    for (const bool correct_block : {false, true}) {
                        SCOPED_TRACE(::testing::Message() << cp << '/' << merged << '/' << separate << '/' << correct_block);
                        auto metadata = std::make_shared<MockClientWrapper>();
                        auto native = std::make_shared<MockClientWrapper>();
                        auto projected = config_for(cp, merged, correct_block ? 8 : 16, "prefill-id");
                        kvcm::ClientWrapper::ConfigMap configs;
                        if (separate) {
                            configs.emplace("", config_for(1, false, 8, "decode-id"));
                            configs.emplace("decode_publication", projected);
                        } else {
                            configs.emplace("", projected);
                        }
                        EXPECT_CALL(*native, initForPools(_, _, _, _)).Times(correct_block ? 1 : 0)
                            .WillRepeatedly(Invoke([&](const auto& map, auto, const auto&, const auto&) {
                                EXPECT_EQ(map.at("")->hasLocationSpec("F_TP0"), !separate);
                                EXPECT_EQ(map.at("")->hasLocationSpec("F0_TP0"), separate);
                                EXPECT_FALSE(map.at("")->hasLocationSpec("F0_TP1"));
                                EXPECT_EQ(map.at("")->instance_id(), separate ? "decode-id" : auto_native_id);
                                return true;
                            }));
                        EXPECT_CALL(*native, createMetadataOnlyClient(_)).Times(correct_block ? 1 : 0)
                            .WillRepeatedly(Invoke([&](const auto& effective) {
                                EXPECT_EQ(effective->instance_id(), "prefill-id");
                                EXPECT_EQ(effective->blockSize(), 8);
                                EXPECT_EQ(effective->hasLocationSpec("F_TP1"), merged);
                                EXPECT_EQ(effective->hasLocationSpec("F0_TP1"), !merged);
                                return metadata;
                            }));
                        EXPECT_CALL(*native, shutdown()).Times(1);
                        EXPECT_CALL(*metadata, shutdown()).Times(correct_block ? 1 : 0);
                        KVCacheConfig settings;
                        settings.kvcm_client_config = autil::legacy::ToJsonString(configs);
                        parallel.prefill_cp_config.prefill_cp_size = cp;
                        BackendHandle backend(std::make_unique<KVCMStorageBackend>(
                            cache, settings, runtime, parallel, SpeculativeExecutionConfig{}, nullptr, false, nullptr, native));
                        EXPECT_EQ(initSingleRank(*backend.backend, env), correct_block);
                    }
                }
            }
        }
        for (const bool bad_block : {false, true}) {
            SCOPED_TRACE(bad_block);
            auto native_config = config_for(1, false, bad_block ? 16 : 8, "decode-id");
            if (!bad_block) {
                ++native_config->locationSpecSizes()->at("F0_TP0");
            }
            KVCacheConfig settings;
            settings.kvcm_client_config = autil::legacy::ToJsonString(kvcm::ClientWrapper::ConfigMap{
                {"", native_config}, {"decode_publication", config_for(2, true, 8, "prefill-id")}});
            parallel.prefill_cp_config.prefill_cp_size = 2;
            auto client = std::make_shared<MockClientWrapper>();
            EXPECT_CALL(*client, initForPools(_, _, _, _)).Times(0);
            EXPECT_CALL(*client, createMetadataOnlyClient(_)).Times(0);
            EXPECT_CALL(*client, shutdown()).Times(1);
            BackendHandle backend(std::make_unique<KVCMStorageBackend>(
                cache, settings, runtime, parallel, SpeculativeExecutionConfig{}, nullptr, false, nullptr, client));
            EXPECT_FALSE(initSingleRank(*backend.backend, env));
        }
    });
}

TEST(KVCMLocalTest, NativeDecodeReadsAndProjectedPublicationUseTheirOwnNamespaces) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1");
    ::rtp_llm::test::withDecodePublicationStartup([&] {
        for (const int cp : {2, 4}) {
            const auto [read_id, publication_id] = observeRoleGeometry(RoleType::DECODE, 1, 2, cp, "", true);
            EXPECT_NE(read_id, publication_id);
        }
    });
}

TEST(KVCMLocalTest, DecodeOffSkipsRemoteWritesBeforePinningButKeepsLocalTreeAndRemoteReads) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "0");
    for (const auto role : {RoleType::PREFILL, RoleType::PDFUSION, RoleType::DECODE}) {
        SCOPED_TRACE(static_cast<int>(role));
        ::rtp_llm::test::withDecodePublicationStartup([&] {
            const bool decode = role == RoleType::DECODE;
            auto environment = makeBackendEnvironment("publication_write_admission");
            const auto original_refs = environment.device_pool->refCount(environment.block_id);
            auto client = std::make_shared<MockClientWrapper>();
            EXPECT_CALL(*client, initForPools(_, _, _, _)).WillOnce(Return(true));
            EXPECT_CALL(*client, shutdown()).Times(1);
            kv_cache_manager::WriteLocation allocation;
            allocation.write_session_id = "prefill-write";
            allocation.block_mask = kv_cache_manager::BlockMaskOffset{0};
            allocation.locations = {{{"tp0_Fdefault", "write_uri"}}};
            EXPECT_CALL(*client, getWriteLocation(_, _, _, _, _, _, _))
                .Times(decode ? 0 : 1).WillRepeatedly(Return(std::make_pair(true, allocation)));
            EXPECT_CALL(*client, saveKvCachesForTag(_, _, _, _))
                .Times(decode ? 0 : 1)
                .WillRepeatedly(Return(std::make_pair(true, kv_cache_manager::UriStrVec{"write_uri"})));
            EXPECT_CALL(*client, finishWrite(_, _, _, _, _)).Times(decode ? 0 : 1)
                .WillRepeatedly(Invoke([](const auto&, const auto&, const auto&, const auto& mask, const auto&) {
                    const auto* offset = std::get_if<kv_cache_manager::BlockMaskOffset>(&mask);
                    EXPECT_NE(offset, nullptr);
                    if (offset) {
                        EXPECT_EQ(*offset, 1u);
                    }
                    return true;
                }));
            auto parallel = singleRankConfig();
            parallel.role_type = role;
            auto handle = makeBackend(environment, parallel, client);
            auto backend = std::shared_ptr<KVCMStorageBackend>(std::move(handle.backend));
            auto group = std::make_shared<FullGroupSet>(
                std::vector<DeviceBlockPoolPtr>{environment.device_pool}, nullptr, nullptr);
            group->initialize(0, environment.cache_config.topologyPtr(), environment.cache_config.groupTags());
            BlockTreeCacheConfig config;
            config.enable_remote_cache = true;
            auto cache = block_tree_cache_test::makeBlockTreeCacheForTest({group}, config, backend);
            ASSERT_NE(cache, nullptr);
            ASSERT_TRUE(backend->remoteCacheEnabled());
            // Exercise virtual dispatch through the same base type the tree uses.
            StorageBackend& remote = *backend;
            auto projected_request = makeStorageRequest(environment);
            projected_request.decode_write = std::make_shared<DecodeWriteContext>();
            EXPECT_FALSE(remote.prepareWrite(std::move(projected_request)));
            if (decode)
                EXPECT_FALSE(remote.prepareWrite(makeStorageRequest(environment)));
            EXPECT_EQ(environment.device_pool->refCount(environment.block_id), original_refs);
            std::vector<std::vector<GroupSetResource>> resources(1, std::vector<GroupSetResource>(1));
            resources[0][0].device_blocks = {environment.block_id};
            cache->insert({101}, resources, Tier::DEVICE);
            ASSERT_TRUE(waitForBackendOperationsForTest(remote));
            EXPECT_EQ(cache->getKeySnapshot().keys, CacheKeysType({101}));
            auto local = cache->match({101}, {}, /*enable_storage_backend=*/false);
            EXPECT_EQ(local.matched_device_blocks, 1u);
            block_tree_cache_test::releaseRequestRefsForTest(*cache, local.matched_device_resources);
            local.matched_device_resources.clear();
            EXPECT_EQ(environment.device_pool->treeRefCount(environment.block_id), 1u);
            EXPECT_EQ(environment.device_pool->refCount(environment.block_id), original_refs + 1);
            EXPECT_FALSE(environment.device_pool->isFrozen(environment.block_id));
            if (decode) {
                const kv_cache_manager::Locations locations{{{"tp0_Fdefault", "read_uri"}}};
                EXPECT_CALL(*client, match(_, _, _, _, _, _))
                    .WillOnce(Return(std::make_pair(true, locations)));
                EXPECT_CALL(*client, loadKvCachesForTag("default", kv_cache_manager::UriStrVec{"read_uri"}, _, _))
                    .WillOnce(Return(true));
                auto observation = match(*backend, makeStorageRequest(environment));
                ASSERT_TRUE(observation.success);
                ASSERT_NE(observation.match_meta, nullptr);
                EXPECT_TRUE(read(*backend, makeStorageRequest(environment), std::move(observation.match_meta)));
                ASSERT_TRUE(waitForBackendOperationsForTest(remote));
            }
            cache.reset();
            EXPECT_EQ(environment.device_pool->refCount(environment.block_id), original_refs);
        });
    }
}

TEST(KVCMLocalTest, DsparkDecodeAdmitsNativeCacheTagsAndRejectsIncompleteDraftConfiguration) {
    autil::EnvGuard feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1");
    ::rtp_llm::test::withDecodePublicationStartup([&] {
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
                    EXPECT_EQ(resource.decodePublicationAttempts(), std::min(attempt, size_t{2}));
                    EXPECT_FALSE(resource.decodeCompletionSubmitted());
                }
            }
        }
    }
    });
}

}  // namespace
}  // namespace rtp_llm
