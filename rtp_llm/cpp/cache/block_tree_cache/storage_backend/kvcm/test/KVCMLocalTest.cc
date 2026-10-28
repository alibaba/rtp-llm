#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test/KVCMMockTestBase.h"

#include <array>
#include <atomic>
#include <csignal>
#include <condition_variable>
#include <cstdint>
#include <unistd.h>

#include "kmonitor/client/MetricsReporter.h"
#include "kmonitor/client/core/MetricsData.h"
#include "rtp_llm/cpp/cache/DeviceBlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/events/KVCacheEventPublisher.h"
#include "rtp_llm/cpp/cache/events/KVCacheEventReporter.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/utils/Logger.h"

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

TEST(KVCMLocalTest, InitRejectsInvalidQuarantineAbortRatio) {
    autil::EnvGuard quarantine_abort_ratio("RTP_LLM_REMOTE_CACHE_QUARANTINE_ABORT_RATIO", "invalid");
    auto            environment    = makeBackendEnvironment("kvcm_storage_backend_invalid_quarantine_ratio");
    auto            client_wrapper = std::make_shared<MockClientWrapper>();
    EXPECT_CALL(*client_wrapper, initForPools(_, _, _, _)).Times(0);
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

TEST(KVCMLocalTest, GdrFallsBackToStandardRegistrationForHostPinnedPool) {
    auto environment = makeHybridBackendEnvironment("kvcm_storage_backend_mixed_gdr_span");
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

    int operation_sequence = 0;
    for (const auto ec :
         {kv_cache_manager::ClientErrorCode::ER_CLIENT_NOT_EXISTS, kv_cache_manager::ClientErrorCode::ER_SDK_TIMEOUT}) {
        client_wrapper->load_status = ec;
        client_wrapper->save_status = ec;
        for (const auto operation : {REMOTE_OPERATION_READ, REMOTE_OPERATION_WRITE}) {
            FunctionRequestPB request;
            auto*             remote = request.mutable_remote_request();
            remote->set_op(operation);
            const auto operation_id = "transfer_" + std::to_string(operation_sequence++);
            remote->set_operation_id(operation_id);
            remote->add_group_tags("default");
            remote->add_block_ids(environment.block_id);
            remote->add_uris("transfer_uri");
            FunctionResponsePB response;
            EXPECT_TRUE(manager.executeFunction(request, response));
            ASSERT_TRUE(response.has_remote_response());
            const bool timed_out = ec == kv_cache_manager::ClientErrorCode::ER_SDK_TIMEOUT;
            EXPECT_EQ(response.remote_response().transfer_status(),
                      timed_out ? REMOTE_TRANSFER_STATUS_TIMEOUT : REMOTE_TRANSFER_STATUS_FAILED);

            QueryKVCMOperationsRequestPB  query_request;
            QueryKVCMOperationsResponsePB query_response;
            query_request.add_operation_ids(operation_id);
            ASSERT_TRUE(manager.queryKVCMOperations(query_request, query_response));
            ASSERT_EQ(query_response.operations_size(), 1);
            EXPECT_EQ(query_response.operations(0).state(),
                      QueryKVCMOperationsResponsePB::OperationState::RUNNING);
            EXPECT_FALSE(manager.forgetKVCMOperation(operation_id));
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

    backend.backend->shutdown();
    EXPECT_EQ(environment.device_pool->refCount(environment.block_id), ref_count_before + 1);
    EXPECT_TRUE(environment.device_pool->isFrozen(environment.block_id));
    EXPECT_EQ(environment.device_pool->referencedBlocksNum(BlockTreeRefType::FROZEN), 1u);
    EXPECT_DOUBLE_EQ(snapshotGauge(quarantine_metrics->quarantined_lease_count_metric), 0);
    EXPECT_DOUBLE_EQ(snapshotGauge(quarantine_metrics->quarantined_block_count_metric), 0);
    EXPECT_DOUBLE_EQ(snapshotGauge(quarantine_metrics->quarantined_block_ratio_metric), 0);
}

TEST(KVCMLocalTest, RankZeroAbortsWhenQuarantinedBlocksExceedConfiguredRatio) {
    // CUDA needs a fresh process rather than the default fork-only death-test child.
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    ASSERT_EXIT(
        ([]() {
            (void)dup2(STDERR_FILENO, STDOUT_FILENO);
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
