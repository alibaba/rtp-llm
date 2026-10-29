#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "autil/EnvUtil.h"

#include <array>
#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <string>
#include <stdexcept>
#include <thread>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/ClientWrapper.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/SdkBufferSafety.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test/MockKVCMClient.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BoundedThreadTestUtils.h"
#include "kvcm_client/HostCacheStateCompat.h"

namespace rtp_llm::kvcm {
namespace {

using ::testing::_;
using ::testing::DoAll;
using ::testing::Invoke;
using ::testing::Return;
using ::testing::ReturnRef;
using ::testing::SetArgReferee;

TEST(ClientFactoryGdrTest, SsdAutomaticallyRequiresGpuRegistrations) {
    autil::EnvGuard              enabled("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1");
    ClientFactory                factory;
    kv_cache_manager::InitParams params{};
    const std::string            ssd = R"({"sdk_config":{"sdk_backend_configs":[{"type":"pace_ssd"}]}})";
    EXPECT_THROW(factory.createTransferClient(ssd, params), std::invalid_argument);
    const std::string mixed = R"({"sdk_config":{"sdk_backend_configs":[{"type":"pace"},{"type":"pace_ssd"}]}})";
    kv_cache_manager::ClientMemoryRegistrations registrations;
    EXPECT_THROW(factory.createTransferClientWithMemory(mixed, params, registrations), std::invalid_argument);
    EXPECT_THROW(factory.createTransferClientWithMemory(ssd, params, registrations), std::invalid_argument);
}

struct InitObservation {
    const kv_cache_manager::RegistSpan* registration_descriptor = nullptr;
    void*                               registration_base       = nullptr;
    size_t                              registration_size       = 0;
};

KVCMConfigPtr makeConfig(bool enable_vipserver, const std::string& endpoint, uint32_t retry_time = 1) {
    auto                           location_infos  = std::make_shared<KVCMConfig::LocationSpecInfoMap>();
    auto                           location_groups = std::make_shared<KVCMConfig::LocationSpecGroups>();
    auto                           channel         = std::make_shared<MetaChannelConfig>(retry_time,
                                                       /*connection_timeout=*/1000,
                                                       /*call_timeout=*/100);
    auto                           sdk             = std::make_shared<SdkWrapperConfig>();
    const std::vector<std::string> addresses =
        enable_vipserver || endpoint.empty() ? std::vector<std::string>{} : std::vector<std::string>{endpoint};
    return std::make_shared<KVCMConfig>(enable_vipserver,
                                        enable_vipserver ? endpoint : "",
                                        /*block_size=*/8,
                                        "instance_group",
                                        "instance_id",
                                        addresses,
                                        location_infos,
                                        channel,
                                        sdk,
                                        location_groups,
                                        ModelDeployment());
}

std::shared_ptr<ClientWrapper>
initializeClient(bool                                                  enable_vipserver,
                 const std::string&                                    endpoint,
                 kv_cache_manager::RegistSpan&                         registration_span,
                 InitObservation&                                      observation,
                 std::shared_ptr<int>                                  transfer_destruction_count,
                 kv_cache_manager::MockMetaClient**                    initial_meta_client      = nullptr,
                 std::shared_ptr<std::atomic<int>>                     reinit_attempts          = nullptr,
                 kv_cache_manager::MockTransferClient**                transfer_client          = nullptr,
                 std::unique_ptr<kv_cache_manager::MockTransferClient> supplied_transfer_client = nullptr) {
    auto  factory        = std::make_unique<MockClientFactory>();
    auto* factory_ptr    = factory.get();
    auto  subscriber     = std::make_unique<MockSubscriber>();
    auto* subscriber_ptr = subscriber.get();

    EXPECT_CALL(*factory_ptr, createSubscriber(enable_vipserver)).WillOnce(Invoke([&subscriber](bool) {
        return std::move(subscriber);
    }));
    EXPECT_CALL(*subscriber_ptr, init(std::vector<std::string>{endpoint})).WillOnce(Return(true));
    if (enable_vipserver) {
        EXPECT_CALL(*subscriber_ptr, getAddresses(_))
            .WillOnce(DoAll(SetArgReferee<0>(std::vector<std::string>{endpoint + "_resolved"}), Return(true)));
    } else if (reinit_attempts) {
        EXPECT_CALL(*subscriber_ptr, getAddresses(_))
            .WillRepeatedly(DoAll(SetArgReferee<0>(std::vector<std::string>{endpoint}), Return(true)));
    } else {
        EXPECT_CALL(*subscriber_ptr, getAddresses(_)).Times(0);
    }

    static const std::string storage_config = R"({"sdk_backend_configs":[]})";
    auto                     meta_client    = std::make_unique<kv_cache_manager::MockMetaClient>();
    if (initial_meta_client != nullptr) {
        *initial_meta_client = meta_client.get();
    }
    EXPECT_CALL(*meta_client, GetStorageConfig()).WillOnce(ReturnRef(storage_config));
    if (reinit_attempts) {
        EXPECT_CALL(*factory_ptr, createMetaClient(_, _))
            .WillOnce(Invoke([&meta_client](const std::string&, const kv_cache_manager::InitParams&) {
                return std::move(meta_client);
            }))
            .WillRepeatedly(
                Invoke([reinit_attempts, &observation](const std::string&, const kv_cache_manager::InitParams& params) {
                    EXPECT_EQ(params.regist_span, observation.registration_descriptor);
                    EXPECT_NE(params.regist_span, nullptr);
                    if (params.regist_span) {
                        EXPECT_EQ(params.regist_span->base, observation.registration_base);
                        EXPECT_EQ(params.regist_span->size, observation.registration_size);
                    }
                    reinit_attempts->fetch_add(1, std::memory_order_release);
                    return std::unique_ptr<kv_cache_manager::MetaClient>{};
                }));
    } else {
        EXPECT_CALL(*factory_ptr, createMetaClient(_, _))
            .WillOnce(Invoke([&meta_client](const std::string&, const kv_cache_manager::InitParams&) {
                return std::move(meta_client);
            }));
    }
    EXPECT_CALL(*factory_ptr, createTransferClient(_, _))
        .WillOnce(Invoke([&](const std::string&, const kv_cache_manager::InitParams& params) {
            if (params.regist_span != nullptr) {
                observation.registration_descriptor = params.regist_span;
                observation.registration_base       = params.regist_span->base;
                observation.registration_size       = params.regist_span->size;
            }
            auto client = supplied_transfer_client ?
                              std::move(supplied_transfer_client) :
                              std::make_unique<kv_cache_manager::MockTransferClient>(transfer_destruction_count);
            if (transfer_client != nullptr) {
                *transfer_client = client.get();
            }
            return client;
        }));

    auto                         wrapper = std::make_shared<ClientWrapper>(std::move(factory));
    ClientWrapper::ConfigMap     config_map{{"", makeConfig(enable_vipserver, endpoint)}};
    kv_cache_manager::InitParams params{kv_cache_manager::RoleType::HYBRID, &registration_span, "tp0_Fgroup"};
    if (!wrapper->init(config_map, params, "default")) {
        return nullptr;
    }
    return wrapper;
}

template<typename Match>
Match hostStateFixture(const std::string& host, int64_t local, int64_t remote) {
    Match result{};
    result.host_ip_port = host;
    result.local        = local;
    if constexpr (HasGlobalHostCacheMatch<Match>::value) {
        result.global = remote;
    } else {
        result.p2p_1_fetch       = 1;
        result.p2p_1_total_match = remote;
    }
    return result;
}

template<typename Match>
void expectHostState(const Match& value, const Match& expected) {
    EXPECT_EQ(value.host_ip_port, expected.host_ip_port);
    EXPECT_EQ(value.local, expected.local);
    if constexpr (HasGlobalHostCacheMatch<Match>::value) {
        EXPECT_EQ(value.global, expected.global);
    } else {
        EXPECT_EQ(value.p2p_1_fetch, expected.p2p_1_fetch);
        EXPECT_EQ(value.p2p_1_total_match, expected.p2p_1_total_match);
    }
}

// Exercise the actual ClientWrapper -> compile-time SDK adapter boundary.
// The backend-aware SDK must request P2P's L2 backend explicitly: {} would
// silently disable the estimate even though the metadata call succeeds.
TEST(ClientWrapperTest, HostStatePreservesP2pSelectionAndSdkFailureWithPublicationOnAndOff) {
    using namespace kv_cache_manager;
    const std::vector<int64_t>     keys{101, 202};
    const std::vector<std::string> medium{"hbm"};
    const HostCacheState           expected{hostStateFixture<HostCacheMatch>("host-a:1", 2, 3),
                                  hostStateFixture<HostCacheMatch>("host-b:2", 5, 7)};
    for (const std::string enabled : {"0", "1"}) {
        SCOPED_TRACE(enabled);
        autil::EnvGuard      feature("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", enabled);
        std::array<char, 32> data{};
        RegistSpan           span{data.data(), data.size()};
        InitObservation      observation;
        MockMetaClient*      meta      = nullptr;
        auto                 destroyed = std::make_shared<int>(0);
        auto                 wrapper   = initializeClient(false, "direct", span, observation, destroyed, &meta);
        ASSERT_NE(wrapper, nullptr);
        if constexpr (std::is_invocable_v<decltype(&MetaClient::GetHostCacheState),
                                          MetaClient&,
                                          const std::string&,
                                          QueryType,
                                          const std::vector<int64_t>&,
                                          const std::vector<std::string>&,
                                          int32_t,
                                          const std::vector<StorageType>&>) {
            EXPECT_CALL(*meta, GetHostCacheState(_, _, _, _, _)).Times(0);
            EXPECT_CALL(*meta,
                        GetHostCacheState("host-state",
                                          QueryType::QT_PREFIX_MATCH,
                                          keys,
                                          medium,
                                          2,
                                          std::vector<StorageType>{StorageType::ST_EVENT_REPORT_L2}))
                .WillOnce(Return(std::make_pair(ER_OK, expected)))
                .WillOnce(Return(std::make_pair(ER_INVALID_PARAMS, expected)));
        } else {
            EXPECT_CALL(*meta, GetHostCacheState(_, _, _, _, _, _)).Times(0);
            EXPECT_CALL(*meta, GetHostCacheState("host-state", QueryType::QT_PREFIX_MATCH, keys, medium, 2))
                .WillOnce(Return(std::make_pair(ER_OK, expected)))
                .WillOnce(Return(std::make_pair(ER_INVALID_PARAMS, expected)));
        }
        auto [ok, hosts] = wrapper->getHostCacheState("", "host-state", QueryType::QT_PREFIX_MATCH, keys, medium, 2);
        ASSERT_TRUE(ok);
        ASSERT_EQ(hosts.size(), expected.size());
        for (size_t i = 0; i < hosts.size(); ++i) {
            expectHostState(hosts[i], expected[i]);
        }
        auto [failed, ignored] =
            wrapper->getHostCacheState("", "host-state", QueryType::QT_PREFIX_MATCH, keys, medium, 2);
        EXPECT_FALSE(failed);
        EXPECT_TRUE(ignored.empty());
        wrapper->shutdown();
        EXPECT_EQ(*destroyed, 1);
    }
}

// Reproduce the legacy Manager response observed on the real PACE fixture.
// Every result remains indexed by the original key, including masks and misses.
TEST(ClientWrapperTest, BackendMasksPreserveOriginalKeyPositions) {
    autil::EnvGuard enabled("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1");
    using namespace kv_cache_manager;
    const auto             type = StorageType::ST_TAIRMEMPOOL;
    const BackendLocations all{
        {{type, 16, {{"full", "uri101"}}}}, {{type, 16, {{"full", "uri102"}}}}, {{type, 16, {{"full", "uri103"}}}}};
    struct Case {
        BlockMask         mask;
        BackendLocations  response;
        std::vector<bool> expected_hits;
    };
    BackendLocations miss = all;
    miss[1].clear();
    BackendLocations already_masked = all;
    already_masked[0].clear();
    const std::vector<Case> cases{
        {BlockMaskOffset{1}, all, {false, true, true}},
        {BlockMaskVector{true, false, true}, all, {false, true, false}},
        {BlockMaskOffset{1}, miss, {false, false, true}},
        {BlockMaskOffset{1}, already_masked, {false, true, true}},
        {BlockMaskVector{}, all, {true, true, true}},
        {BlockMaskOffset{3}, all, {false, false, false}},
    };
    for (size_t c = 0; c < cases.size(); ++c) {
        SCOPED_TRACE(c);
        std::array<char, 32> data{};
        RegistSpan           span{data.data(), data.size()};
        InitObservation      observation;
        MockMetaClient*      meta      = nullptr;
        auto                 destroyed = std::make_shared<int>(0);
        auto                 wrapper   = initializeClient(false, "direct", span, observation, destroyed, &meta);
        ASSERT_NE(wrapper, nullptr);
        const auto& test = cases[c];
        EXPECT_CALL(*meta,
                    GetCacheLocationsByBackend("backend",
                                               std::vector<int64_t>({101, 102, 103}),
                                               std::vector<int64_t>{},
                                               test.mask,
                                               std::vector<std::string>{"full"},
                                               type,
                                               BackendSelectStrategy::LSS_WEIGHTED_RANDOM))
            .WillOnce(Return(std::make_pair(ClientErrorCode::ER_OK, test.response)));
        auto [ok, result] =
            wrapper->getCacheLocationsByBackend("", "backend", {101, 102, 103}, {}, test.mask, {"full"}, type);
        ASSERT_TRUE(ok);
        ASSERT_EQ(result.size(), 3u);
        for (size_t i = 0; i < result.size(); ++i) {
            EXPECT_EQ(!result[i].empty(), test.expected_hits[i]);
            if (!result[i].empty()) {
                ASSERT_EQ(result[i].size(), 1u);
                ASSERT_EQ(result[i][0].location_specs.size(), 1u);
                EXPECT_EQ(result[i][0].location_specs[0].spec_name, "full");
                EXPECT_EQ(result[i][0].location_specs[0].uri, all[i][0].location_specs[0].uri);
            }
        }
        wrapper->shutdown();
        EXPECT_EQ(*destroyed, 1);
    }
}

TEST(ClientWrapperTest, BackendRejectsMalformedShapeAndPropagatesSdkFailure) {
    autil::EnvGuard enabled("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1");
    using namespace kv_cache_manager;
    const auto                                                type = StorageType::ST_TAIRMEMPOOL;
    const BackendLocations                                    all(3, {{type, 16, {{"full", "uri"}}}});
    const std::vector<std::pair<BlockMask, BackendLocations>> malformed{
        {BlockMaskOffset{0}, BackendLocations(2)},
        {BlockMaskOffset{4}, all},
        {BlockMaskVector{false, true}, all},
    };
    std::array<char, 32> data{};
    RegistSpan           span{data.data(), data.size()};
    InitObservation      observation;
    MockMetaClient*      meta      = nullptr;
    auto                 destroyed = std::make_shared<int>(0);
    auto                 wrapper   = initializeClient(false, "direct", span, observation, destroyed, &meta);
    ASSERT_NE(wrapper, nullptr);
    for (const auto& [mask, response] : malformed) {
        EXPECT_CALL(*meta, GetCacheLocationsByBackend("bad_shape", _, _, mask, _, type, _))
            .WillOnce(Return(std::make_pair(ClientErrorCode::ER_OK, response)));
        auto [ok, result] = wrapper->getCacheLocationsByBackend("", "bad_shape", {101, 102, 103}, {}, mask, {}, type);
        EXPECT_FALSE(ok);
        EXPECT_TRUE(result.empty());
    }
    EXPECT_CALL(*meta, GetCacheLocationsByBackend("sdk_error", _, _, _, _, type, _))
        .WillOnce(Return(std::make_pair(ClientErrorCode::ER_INVALID_PARAMS, all)));
    auto [ok, result] =
        wrapper->getCacheLocationsByBackend("", "sdk_error", {101, 102, 103}, {}, BlockMaskOffset{0}, {}, type);
    EXPECT_FALSE(ok);
    EXPECT_TRUE(result.empty());
    wrapper->shutdown();
}

TEST(ClientWrapperTest, FeatureOffPreservesBackendResponseAndSdkError) {
    autil::EnvGuard disabled("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "0");
    using namespace kv_cache_manager;
    const auto type = StorageType::ST_TAIRMEMPOOL;
    const BackendLocations all{
        {{type, 16, {{"full", "uri101"}}}}, {{type, 16, {{"full", "uri102"}}}}, {{type, 16, {{"full", "uri103"}}}}};
    const std::vector<std::pair<BlockMask, BackendLocations>> cases{
        {BlockMaskOffset{1}, all},
        {BlockMaskVector{true, false, true}, all},
        {BlockMaskOffset{4}, all},
        {BlockMaskOffset{0}, BackendLocations{all[0], all[1]}},
    };
    std::array<char, 32> data{};
    RegistSpan span{data.data(), data.size()};
    InitObservation observation;
    MockMetaClient* meta = nullptr;
    auto destroyed = std::make_shared<int>(0);
    auto wrapper = initializeClient(false, "direct", span, observation, destroyed, &meta);
    ASSERT_NE(wrapper, nullptr);
    for (const auto& [mask, response] : cases) {
        EXPECT_CALL(*meta, GetCacheLocationsByBackend("legacy", _, _, mask, _, type, _))
            .WillOnce(Return(std::make_pair(ClientErrorCode::ER_OK, response)));
        auto [ok, result] = wrapper->getCacheLocationsByBackend("", "legacy", {101, 102, 103}, {}, mask, {}, type);
        ASSERT_TRUE(ok);
        ASSERT_EQ(result.size(), response.size());
        for (size_t i = 0; i < result.size(); ++i) {
            ASSERT_EQ(result[i].size(), 1u);
            ASSERT_EQ(result[i][0].location_specs.size(), 1u);
            EXPECT_EQ(result[i][0].location_specs[0].spec_name, response[i][0].location_specs[0].spec_name);
            EXPECT_EQ(result[i][0].location_specs[0].uri, response[i][0].location_specs[0].uri);
        }
    }
    EXPECT_CALL(*meta, GetCacheLocationsByBackend("legacy_error", _, _, _, _, type, _))
        .WillOnce(Return(std::make_pair(ClientErrorCode::ER_INVALID_PARAMS, all)));
    auto [ok, result] = wrapper->getCacheLocationsByBackend(
        "", "legacy_error", {101, 102, 103}, {}, BlockMaskOffset{0}, {}, type);
    EXPECT_FALSE(ok);
    EXPECT_TRUE(result.empty());
    wrapper->shutdown();
    EXPECT_EQ(*destroyed, 1);
}

TEST(ClientWrapperTest, RejectsInvalidTagsBeforeCreatingClients) {
    auto factory = std::make_unique<MockClientFactory>();
    EXPECT_CALL(*factory, createSubscriber(_)).Times(0);
    EXPECT_CALL(*factory, createTransferClient(_, _)).Times(0);
    ClientWrapper                                      wrapper(std::move(factory));
    std::array<char, 32>                               data{};
    const std::vector<ClientWrapper::PoolRegistration> registrations{{{data.data(), 16}, "tp0_Fzeta"},
                                                                     {{data.data() + 16, 16}, "tp0_Falpha"}};
    const ClientWrapper::ConfigMap                     configs{{"", makeConfig(false, "direct")}};
    for (const auto& tags : std::vector<std::vector<std::string>>{{}, {"zeta"}, {"zeta", ""}, {"zeta", "zeta"}}) {
        EXPECT_FALSE(wrapper.initForPools(configs, kv_cache_manager::RoleType::HYBRID, registrations, tags));
    }
    kv_cache_manager::BlockBuffers buffers;
    EXPECT_FALSE(wrapper.loadKvCachesForTag("zeta", {}, buffers));
    EXPECT_FALSE(wrapper.saveKvCachesForTag("alpha", {}, buffers).first);
}

TEST(ClientWrapperTest, TagRoutingUsesRegistrationOrderAndFailedInitPublishesNoRoutes) {
    for (bool fail_second : {false, true}) {
        auto                     factory        = std::make_unique<MockClientFactory>();
        auto                     subscriber     = std::make_unique<MockSubscriber>();
        auto                     meta           = std::make_unique<kv_cache_manager::MockMetaClient>();
        static const std::string storage_config = "{}";
        EXPECT_CALL(*meta, GetStorageConfig()).WillOnce(ReturnRef(storage_config));
        EXPECT_CALL(*subscriber, init(std::vector<std::string>{"direct"})).WillOnce(Return(true));
        EXPECT_CALL(*factory, createSubscriber(false)).WillOnce(Invoke([&](bool) { return std::move(subscriber); }));
        EXPECT_CALL(*factory, createMetaClient(_, _)).WillOnce(Invoke([&](const auto&, const auto&) {
            return std::move(meta);
        }));
        auto                                               destroyed = std::make_shared<int>(0);
        std::vector<const kv_cache_manager::RegistSpan*>   retained;
        std::vector<kv_cache_manager::MockTransferClient*> clients;
        EXPECT_CALL(*factory, createTransferClient(_, _))
            .Times(2)
            .WillRepeatedly(Invoke([&](const auto&, const kv_cache_manager::InitParams& params)
                                       -> std::unique_ptr<kv_cache_manager::TransferClient> {
                EXPECT_EQ(params.self_location_spec_name, retained.empty() ? "tp0_Fzeta" : "tp0_Falpha");
                EXPECT_EQ(params.role_type,
                          retained.empty() ? kv_cache_manager::RoleType::HYBRID : kv_cache_manager::RoleType::WORKER);
                retained.push_back(params.regist_span);
                if (fail_second && retained.size() == 2) {
                    return nullptr;
                }
                auto client = std::make_unique<kv_cache_manager::MockTransferClient>(destroyed);
                clients.push_back(client.get());
                return client;
            }));
        ClientWrapper                                wrapper(std::move(factory));
        std::array<char, 32>                         data{};
        std::vector<ClientWrapper::PoolRegistration> registrations{{{data.data(), 16}, "tp0_Fzeta"},
                                                                   {{data.data() + 16, 16}, "tp0_Falpha"}};
        EXPECT_EQ(wrapper.initForPools({{"", makeConfig(false, "direct")}},
                                       kv_cache_manager::RoleType::HYBRID,
                                       registrations,
                                       {"zeta", "alpha"}),
                  !fail_second);
        kv_cache_manager::BlockBuffers buffers;
        if (fail_second) {
            EXPECT_EQ(*destroyed, 1);
            EXPECT_FALSE(wrapper.loadKvCachesForTag("zeta", {}, buffers));
            EXPECT_FALSE(wrapper.saveKvCachesForTag("alpha", {}, buffers).first);
        } else {
            registrations.clear();
            ASSERT_EQ(retained.size(), 2u);
            EXPECT_EQ(retained[0]->base, data.data());
            EXPECT_EQ(retained[1]->base, data.data() + 16);
            EXPECT_EQ(retained[0]->size, 16u);
            EXPECT_CALL(*clients[1], LoadKvCaches(kv_cache_manager::UriStrVec{"alpha_uri"}, _, _))
                .WillOnce(Return(kv_cache_manager::ClientErrorCode::ER_OK));
            EXPECT_CALL(*clients[0], SaveKvCaches(kv_cache_manager::UriStrVec{"zeta_uri"}, _, _))
                .WillOnce(Return(
                    std::make_pair(kv_cache_manager::ClientErrorCode::ER_OK, kv_cache_manager::UriStrVec{"actual"})));
            // Exercise routing with a valid one-object transfer now that
            // bounded batching always checks URI/buffer geometry first.
            buffers.emplace_back();
            buffers.back().iovs.push_back({kv_cache_manager::MemoryType::CPU, data.data() + 16, 16, false});
            EXPECT_TRUE(wrapper.loadKvCachesForTag("alpha", {"alpha_uri"}, buffers));
            buffers.back().iovs.front().base = data.data();
            EXPECT_EQ(wrapper.saveKvCachesForTag("zeta", {"zeta_uri"}, buffers).second,
                      (kv_cache_manager::UriStrVec{"actual"}));
            EXPECT_FALSE(wrapper.loadKvCachesForTag("missing", {}, buffers));
        }
        wrapper.shutdown();
        EXPECT_EQ(*destroyed, fail_second ? 1 : 2);
        EXPECT_FALSE(wrapper.loadKvCachesForTag("zeta", {}, buffers));
    }
}

TEST(ClientWrapperTest, RejectsInvalidConfigBeforeCreatingSubscriberOrClients) {
    auto                         valid      = makeConfig(false, "endpoint");
    auto                         serialized = autil::legacy::ToJsonString(valid);
    autil::legacy::json::JsonMap fields;
    autil::legacy::FromJsonString(fields, serialized);
    fields.erase("meta_channel_config");
    KVCMConfigPtr missing;
    autil::legacy::FromJsonString(missing, autil::legacy::ToJsonString(fields));
    fields["meta_channel_config"] = autil::legacy::Any();
    KVCMConfigPtr null_channel;
    autil::legacy::FromJsonString(null_channel, autil::legacy::ToJsonString(fields));
    const std::vector<ClientWrapper::ConfigMap> invalid = {{},
                                                           {{"null", nullptr}},
                                                           {{"empty", std::make_shared<KVCMConfig>()}},
                                                           {{"missing", missing}},
                                                           {{"null_channel", null_channel}},
                                                           {{"zero_retry", makeConfig(false, "endpoint", 0)}}};
    for (const auto& configs : invalid) {
        auto factory = std::make_unique<MockClientFactory>();
        EXPECT_CALL(*factory, createSubscriber(_)).Times(0);
        EXPECT_CALL(*factory, createMetaClient(_, _)).Times(0);
        EXPECT_CALL(*factory, createTransferClient(_, _)).Times(0);
        ClientWrapper wrapper(std::move(factory));
        EXPECT_FALSE(wrapper.init(configs, {kv_cache_manager::RoleType::HYBRID, nullptr, "tp0_Ffull"}, "default"));
        wrapper.shutdown();
    }
}

TEST(ClientWrapperTest, RejectsMultipleDirectEndpointsAndVipDomainsBeforeSideEffects) {
    for (bool vip : {false, true}) {
        for (auto role : {kv_cache_manager::RoleType::HYBRID, kv_cache_manager::RoleType::WORKER}) {
            auto factory = std::make_unique<MockClientFactory>();
            EXPECT_CALL(*factory, createSubscriber(_)).Times(0);
            EXPECT_CALL(*factory, createMetaClient(_, _)).Times(0);
            EXPECT_CALL(*factory, createTransferClient(_, _)).Times(0);
            ClientWrapper wrapper(std::move(factory));
            EXPECT_FALSE(wrapper.init({{"a", makeConfig(vip, "first")}, {"b", makeConfig(vip, "second")}},
                                      {role, nullptr, "tp0_Ffull"},
                                      "default"));
            wrapper.shutdown();
        }
    }
}

TEST(ClientWrapperTest, RecreatesTransferClientForDifferentRegistrationSpan) {
    std::array<char, 64>         first_pool{};
    std::array<char, 96>         second_pool{};
    kv_cache_manager::RegistSpan first_span{first_pool.data(), first_pool.size()};
    kv_cache_manager::RegistSpan second_span{second_pool.data(), second_pool.size()};
    InitObservation              first_observation;
    InitObservation              second_observation;
    auto                         first_destruction_count  = std::make_shared<int>(0);
    auto                         second_destruction_count = std::make_shared<int>(0);

    auto first = initializeClient(
        /*enable_vipserver=*/false, "first", first_span, first_observation, first_destruction_count);
    ASSERT_NE(first, nullptr);
    EXPECT_NE(first_observation.registration_descriptor, &first_span);
    EXPECT_EQ(first_observation.registration_base, first_pool.data());
    EXPECT_EQ(first_observation.registration_size, first_pool.size());
    first.reset();
    EXPECT_EQ(*first_destruction_count, 1);

    auto second = initializeClient(
        /*enable_vipserver=*/false, "second", second_span, second_observation, second_destruction_count);
    ASSERT_NE(second, nullptr);
    EXPECT_NE(second_observation.registration_descriptor, &second_span);
    EXPECT_EQ(second_observation.registration_base, second_pool.data());
    EXPECT_EQ(second_observation.registration_size, second_pool.size());
    EXPECT_NE(second_observation.registration_base, first_observation.registration_base);
    second.reset();
    EXPECT_EQ(*second_destruction_count, 1);
}

TEST(ClientWrapperTest, ForwardsGpuMemoryRegistrationToTransferClient) {
    auto  factory        = std::make_unique<MockClientFactory>();
    auto* factory_ptr    = factory.get();
    auto  subscriber     = std::make_unique<MockSubscriber>();
    auto* subscriber_ptr = subscriber.get();

    EXPECT_CALL(*factory_ptr, createSubscriber(false)).WillOnce(Invoke([&subscriber](bool) {
        return std::move(subscriber);
    }));
    EXPECT_CALL(*subscriber_ptr, init(std::vector<std::string>{"direct"})).WillOnce(Return(true));
    EXPECT_CALL(*subscriber_ptr, getAddresses(_)).Times(0);

    static const std::string storage_config = R"({"sdk_backend_configs":[]})";
    auto                     meta_client    = std::make_unique<kv_cache_manager::MockMetaClient>();
    EXPECT_CALL(*meta_client, GetStorageConfig()).WillOnce(ReturnRef(storage_config));
    EXPECT_CALL(*factory_ptr, createMetaClient(_, _))
        .WillOnce(Invoke([&meta_client](const std::string&, const kv_cache_manager::InitParams&) {
            return std::move(meta_client);
        }));
    EXPECT_CALL(*factory_ptr, createTransferClient(_, _)).Times(0);

    std::array<char, 64> pool{};
    auto*                gpu_base = reinterpret_cast<void*>(0x100000);
    EXPECT_CALL(*factory_ptr, createTransferClientWithMemory(_, _, _))
        .WillOnce(Invoke([&](const std::string&,
                             const kv_cache_manager::InitParams& params,
                             const kv_cache_manager::ClientMemoryRegistrations& registrations) {
            EXPECT_NE(params.regist_span, nullptr);
            EXPECT_EQ(registrations.gpu.size(), 1u);
            if (!registrations.gpu.empty()) {
                EXPECT_EQ(registrations.gpu.front().base, gpu_base);
                EXPECT_EQ(registrations.gpu.front().size, 4096u);
                EXPECT_EQ(registrations.gpu.front().device_id, 3);
            }
            return std::make_unique<kv_cache_manager::MockTransferClient>(std::make_shared<int>(0));
        }));

    kv_cache_manager::ClientMemoryRegistrations memory_registrations;
    memory_registrations.gpu.push_back({gpu_base, 4096, 3});
    const std::vector<ClientWrapper::PoolRegistration> registrations{
        {{pool.data(), pool.size()}, "tp0_Ffull", std::move(memory_registrations)}};
    ClientWrapper wrapper(std::move(factory));
    ASSERT_TRUE(wrapper.initForPools({{"", makeConfig(false, "direct")}},
                                    kv_cache_manager::RoleType::HYBRID,
                                    registrations,
                                    {"full"}));
    wrapper.shutdown();
}

TEST(ClientWrapperTest, RecreatesSubscriberWhenSwitchingFromDirectToVipServer) {
    std::array<char, 64>         direct_pool{};
    std::array<char, 64>         vip_pool{};
    kv_cache_manager::RegistSpan direct_span{direct_pool.data(), direct_pool.size()};
    kv_cache_manager::RegistSpan vip_span{vip_pool.data(), vip_pool.size()};
    InitObservation              direct_observation;
    InitObservation              vip_observation;
    auto                         direct_destruction_count = std::make_shared<int>(0);
    auto                         vip_destruction_count    = std::make_shared<int>(0);

    auto direct = initializeClient(
        /*enable_vipserver=*/false, "direct_address", direct_span, direct_observation, direct_destruction_count);
    ASSERT_NE(direct, nullptr);
    direct.reset();
    EXPECT_EQ(*direct_destruction_count, 1);

    auto vip = initializeClient(
        /*enable_vipserver=*/true, "vip_domain", vip_span, vip_observation, vip_destruction_count);
    ASSERT_NE(vip, nullptr);
    vip.reset();
    EXPECT_EQ(*vip_destruction_count, 1);
}

TEST(ClientWrapperTest, ShutdownInterruptsFailedReRegistrationBeforeCreatingNewSpanClient) {
    std::array<char, 64>                  old_pool{};
    kv_cache_manager::RegistSpan          old_span{old_pool.data(), old_pool.size()};
    InitObservation                       old_observation;
    auto                                  old_destruction_count   = std::make_shared<int>(0);
    auto                                  reinit_attempts         = std::make_shared<std::atomic<int>>(0);
    kv_cache_manager::MockMetaClient*     initial_meta_client     = nullptr;
    kv_cache_manager::MockTransferClient* initial_transfer_client = nullptr;

    auto old_client = initializeClient(/*enable_vipserver=*/false,
                                       "old_address",
                                       old_span,
                                       old_observation,
                                       old_destruction_count,
                                       &initial_meta_client,
                                       reinit_attempts,
                                       &initial_transfer_client);
    ASSERT_NE(old_client, nullptr);
    ASSERT_NE(initial_meta_client, nullptr);
    ASSERT_NE(initial_transfer_client, nullptr);
    EXPECT_CALL(*initial_meta_client, FinishWrite(_, _, _, _))
        .WillOnce(Return(kv_cache_manager::ClientErrorCode::ER_SERVICE_INSTANCE_NOT_EXIST));
    EXPECT_FALSE(old_client->finishWrite("", "trace", "session", {}, {}));

    const auto attempt_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (reinit_attempts->load(std::memory_order_acquire) == 0
           && std::chrono::steady_clock::now() < attempt_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_GT(reinit_attempts->load(std::memory_order_acquire), 0);

    EXPECT_CALL(*initial_transfer_client, LoadKvCaches(_, _, _))
        .WillOnce(Return(kv_cache_manager::ClientErrorCode::ER_OK));
    kv_cache_manager::UriStrVec    uris{"old_uri"};
    kv_cache_manager::BlockBuffers buffers(1);
    buffers.front().iovs.push_back({kv_cache_manager::MemoryType::CPU, old_pool.data(), old_pool.size(), false});
    std::promise<bool>             transfer_result_promise;
    auto                           transfer_result = transfer_result_promise.get_future();
    std::thread                    transfer_thread(
        [&] { transfer_result_promise.set_value(old_client->loadKvCachesForTag("default", uris, buffers)); });
    const auto transfer_status = transfer_result.wait_for(std::chrono::seconds(1));
    if (transfer_status != std::future_status::ready) {
        // Keep a regressed implementation from hanging the test indefinitely.
        old_client->shutdown();
    }
    EXPECT_EQ(transfer_status, std::future_status::ready);
    transfer_thread.join();
    EXPECT_TRUE(transfer_result.get());

    const auto shutdown_start = std::chrono::steady_clock::now();
    old_client->shutdown();
    const auto shutdown_elapsed = std::chrono::steady_clock::now() - shutdown_start;
    EXPECT_LT(shutdown_elapsed, std::chrono::seconds(1));
    EXPECT_EQ(*old_destruction_count, 1);
    old_client.reset();
    EXPECT_EQ(*old_destruction_count, 1);

    std::array<char, 96>         new_pool{};
    kv_cache_manager::RegistSpan new_span{new_pool.data(), new_pool.size()};
    InitObservation              new_observation;
    auto                         new_destruction_count = std::make_shared<int>(0);
    auto                         new_client            = initializeClient(
        /*enable_vipserver=*/false, "new_address", new_span, new_observation, new_destruction_count);
    ASSERT_NE(new_client, nullptr);
    EXPECT_EQ(new_observation.registration_base, new_pool.data());
    new_client.reset();
    EXPECT_EQ(*new_destruction_count, 1);
}

TEST(ClientWrapperTest, RejectsEmptyDirectAddressBeforeCreatingClients) {
    auto  factory        = std::make_unique<MockClientFactory>();
    auto* factory_ptr    = factory.get();
    auto  subscriber     = std::make_unique<MockSubscriber>();
    auto* subscriber_ptr = subscriber.get();
    EXPECT_CALL(*factory_ptr, createSubscriber(false)).WillOnce(Invoke([&subscriber](bool) {
        return std::move(subscriber);
    }));
    EXPECT_CALL(*subscriber_ptr, init(std::vector<std::string>{})).WillOnce(Return(true));
    EXPECT_CALL(*factory_ptr, createMetaClient(_, _)).Times(0);
    EXPECT_CALL(*factory_ptr, createTransferClient(_, _)).Times(0);

    ClientWrapper                wrapper(std::move(factory));
    std::array<char, 16>         registration{};
    kv_cache_manager::RegistSpan span{registration.data(), registration.size()};
    EXPECT_FALSE(wrapper.init(
        {{"", makeConfig(false, "")}}, {kv_cache_manager::RoleType::HYBRID, &span, "tp0_Ffull"}, "default"));
}

TEST(ClientWrapperTest, RetriesMetadataClientCreationAccordingToKVCMConfig) {
    auto  factory           = std::make_unique<MockClientFactory>();
    auto* factory_ptr       = factory.get();
    auto  subscriber        = std::make_unique<MockSubscriber>();
    auto* subscriber_ptr    = subscriber.get();
    auto  meta_client       = std::make_unique<kv_cache_manager::MockMetaClient>();
    auto* meta_ptr          = meta_client.get();
    auto  destruction_count = std::make_shared<int>(0);
    auto  transfer_client   = std::make_unique<kv_cache_manager::MockTransferClient>(destruction_count);

    EXPECT_CALL(*factory_ptr, createSubscriber(false)).WillOnce(Invoke([&subscriber](bool) {
        return std::move(subscriber);
    }));
    EXPECT_CALL(*subscriber_ptr, init(std::vector<std::string>{"direct"})).WillOnce(Return(true));
    EXPECT_CALL(*factory_ptr, createMetaClient(_, _))
        .WillOnce(Invoke([](const auto&, const auto&) { return std::unique_ptr<kv_cache_manager::MetaClient>{}; }))
        .WillOnce(Invoke([&meta_client](const auto&, const auto&) { return std::move(meta_client); }));
    static const std::string storage_config = R"({"sdk_backend_configs":[]})";
    EXPECT_CALL(*meta_ptr, GetStorageConfig()).WillOnce(ReturnRef(storage_config));
    EXPECT_CALL(*factory_ptr, createTransferClient(_, _)).WillOnce(Invoke([&transfer_client](const auto&, const auto&) {
        return std::move(transfer_client);
    }));

    ClientWrapper                wrapper(std::move(factory));
    std::array<char, 16>         registration{};
    kv_cache_manager::RegistSpan span{registration.data(), registration.size()};
    EXPECT_TRUE(wrapper.init({{"", makeConfig(false, "direct", /*retry_time=*/2)}},
                             {kv_cache_manager::RoleType::HYBRID, &span, "tp0_Ffull"},
                             "default"));
    wrapper.shutdown();
    EXPECT_EQ(*destruction_count, 1);
}

TEST(ClientWrapperTest, ReusesMetadataClientWhenVipAddressSnapshotIsUnchanged) {
    auto  factory           = std::make_unique<MockClientFactory>();
    auto* factory_ptr       = factory.get();
    auto  subscriber        = std::make_unique<MockSubscriber>();
    auto* subscriber_ptr    = subscriber.get();
    auto  meta_client       = std::make_unique<kv_cache_manager::MockMetaClient>();
    auto* meta_ptr          = meta_client.get();
    auto  destruction_count = std::make_shared<int>(0);
    auto  transfer_client   = std::make_unique<kv_cache_manager::MockTransferClient>(destruction_count);

    EXPECT_CALL(*factory_ptr, createSubscriber(true)).WillOnce(Invoke([&subscriber](bool) {
        return std::move(subscriber);
    }));
    EXPECT_CALL(*subscriber_ptr, init(std::vector<std::string>{"vip"})).WillOnce(Return(true));
    EXPECT_CALL(*subscriber_ptr, getAddresses(_))
        .Times(2)
        .WillRepeatedly(DoAll(SetArgReferee<0>(std::vector<std::string>{"address"}), Return(true)));
    EXPECT_CALL(*factory_ptr, createMetaClient(_, _))
        .Times(1)
        .WillOnce(Invoke([&meta_client](const auto&, const auto&) { return std::move(meta_client); }));
    static const std::string storage_config = R"({"sdk_backend_configs":[]})";
    EXPECT_CALL(*meta_ptr, GetStorageConfig()).WillOnce(ReturnRef(storage_config));
    EXPECT_CALL(*meta_ptr, MatchLocation("match", _, _, _, _, _, _))
        .WillOnce(Return(std::make_pair(kv_cache_manager::ClientErrorCode::ER_OK, kv_cache_manager::Locations{})));
    EXPECT_CALL(*factory_ptr, createTransferClient(_, _)).WillOnce(Invoke([&transfer_client](const auto&, const auto&) {
        return std::move(transfer_client);
    }));

    ClientWrapper                wrapper(std::move(factory));
    std::array<char, 32>         registration{};
    kv_cache_manager::RegistSpan span{registration.data(), registration.size()};
    ASSERT_TRUE(wrapper.init(
        {{"", makeConfig(true, "vip")}}, {kv_cache_manager::RoleType::HYBRID, &span, "tp0_Ffull"}, "default"));
    EXPECT_TRUE(wrapper
                    .match("",
                           "match",
                           kv_cache_manager::QueryType::QT_PREFIX_MATCH,
                           {1},
                           kv_cache_manager::BlockMaskOffset{0},
                           {})
                    .first);
    wrapper.shutdown();
    EXPECT_EQ(*destruction_count, 1);
}

TEST(ClientWrapperTest, RecreatesMetadataClientWhenVipAddressChanges) {
    auto  factory           = std::make_unique<MockClientFactory>();
    auto* factory_ptr       = factory.get();
    auto  subscriber        = std::make_unique<MockSubscriber>();
    auto* subscriber_ptr    = subscriber.get();
    auto  old_meta_client   = std::make_unique<kv_cache_manager::MockMetaClient>();
    auto* old_meta_ptr      = old_meta_client.get();
    auto  new_meta_client   = std::make_unique<kv_cache_manager::MockMetaClient>();
    auto* new_meta_ptr      = new_meta_client.get();
    auto  destruction_count = std::make_shared<int>(0);
    auto  transfer_client   = std::make_unique<kv_cache_manager::MockTransferClient>(destruction_count);

    EXPECT_CALL(*factory_ptr, createSubscriber(true)).WillOnce(Invoke([&subscriber](bool) {
        return std::move(subscriber);
    }));
    EXPECT_CALL(*subscriber_ptr, init(std::vector<std::string>{"vip"})).WillOnce(Return(true));
    EXPECT_CALL(*subscriber_ptr, getAddresses(_))
        .WillOnce(DoAll(SetArgReferee<0>(std::vector<std::string>{"old_address"}), Return(true)))
        .WillOnce(DoAll(SetArgReferee<0>(std::vector<std::string>{"new_address"}), Return(true)));
    EXPECT_CALL(*factory_ptr, createMetaClient(_, _))
        .WillOnce(Invoke([&old_meta_client](const auto&, const auto&) { return std::move(old_meta_client); }))
        .WillOnce(Invoke([&new_meta_client](const auto&, const auto&) { return std::move(new_meta_client); }));
    static const std::string storage_config = R"({"sdk_backend_configs":[]})";
    EXPECT_CALL(*old_meta_ptr, GetStorageConfig()).WillOnce(ReturnRef(storage_config));
    EXPECT_CALL(*new_meta_ptr, MatchLocation("match", _, _, _, _, _, _))
        .WillOnce(Return(std::make_pair(kv_cache_manager::ClientErrorCode::ER_OK, kv_cache_manager::Locations{})));
    EXPECT_CALL(*factory_ptr, createTransferClient(_, _)).WillOnce(Invoke([&transfer_client](const auto&, const auto&) {
        return std::move(transfer_client);
    }));

    ClientWrapper                wrapper(std::move(factory));
    std::array<char, 32>         registration{};
    kv_cache_manager::RegistSpan span{registration.data(), registration.size()};
    ASSERT_TRUE(wrapper.init(
        {{"", makeConfig(true, "vip")}}, {kv_cache_manager::RoleType::HYBRID, &span, "tp0_Ffull"}, "default"));
    EXPECT_TRUE(wrapper
                    .match("",
                           "match",
                           kv_cache_manager::QueryType::QT_PREFIX_MATCH,
                           {1},
                           kv_cache_manager::BlockMaskOffset{0},
                           {})
                    .first);
    wrapper.shutdown();
    EXPECT_EQ(*destruction_count, 1);
}

class MetadataClientLifetimeTest: public ::testing::TestWithParam<std::string> {};

TEST_P(MetadataClientLifetimeTest, VipRefreshRetainsInFlightClient) {
    using block_tree_cache_test::BoundedThread;
    class TrackedMetaClient: public kv_cache_manager::MockMetaClient {
    public:
        explicit TrackedMetaClient(std::shared_ptr<std::atomic<int>> destroyed): destroyed_(std::move(destroyed)) {}
        ~TrackedMetaClient() override {
            destroyed_->fetch_add(1);
        }

    private:
        std::shared_ptr<std::atomic<int>> destroyed_;
    };

    auto               destroyed  = std::make_shared<std::atomic<int>>(0);
    auto               old_client = std::make_unique<TrackedMetaClient>(destroyed);
    auto               new_client = std::make_unique<kv_cache_manager::MockMetaClient>();
    auto               factory    = std::make_unique<MockClientFactory>();
    auto               subscriber = std::make_unique<MockSubscriber>();
    auto               refresh    = std::make_shared<std::atomic<bool>>(false);
    auto               entered    = std::make_shared<std::promise<void>>();
    auto               ready      = entered->get_future();
    std::promise<void> release;
    auto               released = release.get_future().share();
    auto               hold     = [entered, released] {
        entered->set_value();
        released.wait();
    };
    static const std::string storage_config = R"({"sdk_backend_configs":[]})";
    EXPECT_CALL(*old_client, GetStorageConfig()).WillOnce(ReturnRef(storage_config));
    if (GetParam() == "match") {
        EXPECT_CALL(*old_client, MatchLocation(_, _, _, _, _, _, _)).WillOnce(Invoke([hold](const auto&...) {
            hold();
            return std::make_pair(kv_cache_manager::ClientErrorCode::ER_OK, kv_cache_manager::Locations{});
        }));
    } else if (GetParam() == "start_write") {
        EXPECT_CALL(*old_client, StartWrite(_, _, _, _, _, 0)).WillOnce(Invoke([hold](const auto&...) {
            hold();
            return std::make_pair(kv_cache_manager::ClientErrorCode::ER_OK, kv_cache_manager::WriteLocation{});
        }));
    } else {
        EXPECT_CALL(*old_client, FinishWrite(_, _, _, _)).WillOnce(Invoke([hold](const auto&...) {
            hold();
            return kv_cache_manager::ClientErrorCode::ER_OK;
        }));
    }
    EXPECT_CALL(*new_client, FinishWrite(_, _, _, _)).WillOnce(Return(kv_cache_manager::ClientErrorCode::ER_OK));
    EXPECT_CALL(*subscriber, init(std::vector<std::string>{"vip"})).WillOnce(Return(true));
    EXPECT_CALL(*subscriber, getAddresses(_)).WillRepeatedly(Invoke([refresh](auto& addresses) {
        addresses = {refresh->load() ? "new_address" : "old_address"};
        return true;
    }));
    EXPECT_CALL(*factory, createSubscriber(true)).WillOnce(Invoke([&subscriber](bool) {
        return std::move(subscriber);
    }));
    EXPECT_CALL(*factory, createMetaClient(_, _))
        .WillOnce(Invoke([&old_client](const auto&, const auto&) { return std::move(old_client); }))
        .WillOnce(Invoke([&new_client](const auto&, const auto&) { return std::move(new_client); }));
    EXPECT_CALL(*factory, createTransferClient(_, _)).WillOnce(Invoke([](const auto&, const auto&) {
        return std::make_unique<kv_cache_manager::MockTransferClient>(std::make_shared<int>(0));
    }));
    auto wrapper = std::make_shared<ClientWrapper>(std::move(factory));
    ASSERT_TRUE(wrapper->init(
        {{"", makeConfig(true, "vip")}}, {kv_cache_manager::RoleType::HYBRID, nullptr, "tp0_Ffull"}, "default"));
    BoundedThread<bool> call([wrapper, operation = GetParam()] {
        if (operation == "match") {
            return wrapper
                ->match("",
                        "match",
                        kv_cache_manager::QueryType::QT_PREFIX_MATCH,
                        {1},
                        kv_cache_manager::BlockMaskOffset{0},
                        {})
                .first;
        }
        if (operation == "start_write") {
            return wrapper->getWriteLocation("", "start", {1}, {}, {}, 10).first;
        }
        return wrapper->finishWrite("", "finish", "session", {}, {});
    });
    const auto          status = ready.wait_for(std::chrono::seconds(5));
    if (status != std::future_status::ready) {
        release.set_value();
        FAIL() << "metadata RPC did not enter the old client";
    }
    refresh->store(true);
    BoundedThread<bool> refreshed([wrapper] { return wrapper->finishWrite("", "refresh", "session", {}, {}); });
    const auto          refreshed_status = refreshed.waitFor(std::chrono::seconds(5));
    EXPECT_EQ(refreshed_status, std::future_status::ready);
    EXPECT_EQ(destroyed->load(), 0) << "old client destroyed while the RPC was still executing";
    release.set_value();
    ASSERT_EQ(refreshed.waitFor(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_TRUE(refreshed.get());
    ASSERT_EQ(call.waitFor(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_TRUE(call.get());
    EXPECT_EQ(destroyed->load(), 1);
    wrapper->shutdown();
    EXPECT_FALSE(wrapper->finishWrite("", "stopped", "session", {}, {}));
}

INSTANTIATE_TEST_SUITE_P(MetadataCalls,
                         MetadataClientLifetimeTest,
                         ::testing::Values("match", "start_write", "finish_write"),
                         [](const ::testing::TestParamInfo<std::string>& info) { return info.param; });

TEST(ClientWrapperTest, VipRefreshFailureDoesNotCallStaleMetadataClient) {
    auto  factory           = std::make_unique<MockClientFactory>();
    auto* factory_ptr       = factory.get();
    auto  subscriber        = std::make_unique<MockSubscriber>();
    auto* subscriber_ptr    = subscriber.get();
    auto  meta_client       = std::make_unique<kv_cache_manager::MockMetaClient>();
    auto* meta_ptr          = meta_client.get();
    auto  destruction_count = std::make_shared<int>(0);
    auto  transfer_client   = std::make_unique<kv_cache_manager::MockTransferClient>(destruction_count);

    EXPECT_CALL(*factory_ptr, createSubscriber(true)).WillOnce(Invoke([&subscriber](bool) {
        return std::move(subscriber);
    }));
    EXPECT_CALL(*subscriber_ptr, init(std::vector<std::string>{"vip"})).WillOnce(Return(true));
    EXPECT_CALL(*subscriber_ptr, getAddresses(_))
        .WillOnce(DoAll(SetArgReferee<0>(std::vector<std::string>{"old_address"}), Return(true)))
        .WillOnce(DoAll(SetArgReferee<0>(std::vector<std::string>{}), Return(false)));
    EXPECT_CALL(*factory_ptr, createMetaClient(_, _)).WillOnce(Invoke([&meta_client](const auto&, const auto&) {
        return std::move(meta_client);
    }));
    static const std::string storage_config = R"({"sdk_backend_configs":[]})";
    EXPECT_CALL(*meta_ptr, GetStorageConfig()).WillOnce(ReturnRef(storage_config));
    EXPECT_CALL(*meta_ptr, FinishWrite(_, _, _, _)).Times(0);
    EXPECT_CALL(*factory_ptr, createTransferClient(_, _)).WillOnce(Invoke([&transfer_client](const auto&, const auto&) {
        return std::move(transfer_client);
    }));

    ClientWrapper                wrapper(std::move(factory));
    std::array<char, 32>         registration{};
    kv_cache_manager::RegistSpan span{registration.data(), registration.size()};
    ASSERT_TRUE(wrapper.init(
        {{"", makeConfig(true, "vip")}}, {kv_cache_manager::RoleType::HYBRID, &span, "tp0_Ffull"}, "default"));
    EXPECT_FALSE(wrapper.finishWrite("", "finish", "session", kv_cache_manager::BlockMaskOffset{0}, {}));
    wrapper.shutdown();
    EXPECT_EQ(*destruction_count, 1);
}

TEST(ClientWrapperTest, RetriesVipAddressChangeAfterClientCreationFailure) {
    auto  factory           = std::make_unique<MockClientFactory>();
    auto* factory_ptr       = factory.get();
    auto  subscriber        = std::make_unique<MockSubscriber>();
    auto* subscriber_ptr    = subscriber.get();
    auto  old_meta_client   = std::make_unique<kv_cache_manager::MockMetaClient>();
    auto* old_meta_ptr      = old_meta_client.get();
    auto  new_meta_client   = std::make_unique<kv_cache_manager::MockMetaClient>();
    auto* new_meta_ptr      = new_meta_client.get();
    auto  destruction_count = std::make_shared<int>(0);
    auto  transfer_client   = std::make_unique<kv_cache_manager::MockTransferClient>(destruction_count);

    EXPECT_CALL(*factory_ptr, createSubscriber(true)).WillOnce(Invoke([&subscriber](bool) {
        return std::move(subscriber);
    }));
    EXPECT_CALL(*subscriber_ptr, init(std::vector<std::string>{"vip"})).WillOnce(Return(true));
    EXPECT_CALL(*subscriber_ptr, getAddresses(_))
        .WillOnce(DoAll(SetArgReferee<0>(std::vector<std::string>{"old_address"}), Return(true)))
        .WillOnce(DoAll(SetArgReferee<0>(std::vector<std::string>{"new_address"}), Return(true)))
        .WillOnce(DoAll(SetArgReferee<0>(std::vector<std::string>{"new_address"}), Return(true)));
    EXPECT_CALL(*factory_ptr, createMetaClient(_, _))
        .WillOnce(Invoke([&old_meta_client](const auto&, const auto&) { return std::move(old_meta_client); }))
        .WillOnce(Invoke([](const auto&, const auto&) { return std::unique_ptr<kv_cache_manager::MetaClient>{}; }))
        .WillOnce(Invoke([&new_meta_client](const auto&, const auto&) { return std::move(new_meta_client); }));
    static const std::string storage_config = R"({"sdk_backend_configs":[]})";
    EXPECT_CALL(*old_meta_ptr, GetStorageConfig()).WillOnce(ReturnRef(storage_config));
    EXPECT_CALL(*old_meta_ptr, FinishWrite(_, _, _, _)).Times(0);
    EXPECT_CALL(*new_meta_ptr, FinishWrite("finish_2", "session", _, _))
        .WillOnce(Return(kv_cache_manager::ClientErrorCode::ER_OK));
    EXPECT_CALL(*factory_ptr, createTransferClient(_, _)).WillOnce(Invoke([&transfer_client](const auto&, const auto&) {
        return std::move(transfer_client);
    }));

    ClientWrapper                wrapper(std::move(factory));
    std::array<char, 32>         registration{};
    kv_cache_manager::RegistSpan span{registration.data(), registration.size()};
    ASSERT_TRUE(wrapper.init(
        {{"", makeConfig(true, "vip")}}, {kv_cache_manager::RoleType::HYBRID, &span, "tp0_Ffull"}, "default"));
    EXPECT_FALSE(wrapper.finishWrite("", "finish_1", "session", kv_cache_manager::BlockMaskOffset{0}, {}));
    EXPECT_TRUE(wrapper.finishWrite("", "finish_2", "session", kv_cache_manager::BlockMaskOffset{0}, {}));
    wrapper.shutdown();
    EXPECT_EQ(*destruction_count, 1);
}

TEST(ClientWrapperTest, ServiceInstanceFailureReRegistersAndRoutesTheNextRequest) {
    auto  factory           = std::make_unique<MockClientFactory>();
    auto* factory_ptr       = factory.get();
    auto  subscriber        = std::make_unique<MockSubscriber>();
    auto* subscriber_ptr    = subscriber.get();
    auto  old_meta_client   = std::make_unique<kv_cache_manager::MockMetaClient>();
    auto* old_meta_ptr      = old_meta_client.get();
    auto  new_meta_client   = std::make_unique<kv_cache_manager::MockMetaClient>();
    auto* new_meta_ptr      = new_meta_client.get();
    auto  destruction_count = std::make_shared<int>(0);
    auto  transfer_client   = std::make_unique<kv_cache_manager::MockTransferClient>(destruction_count);
    auto  reinit_started    = std::make_shared<std::promise<void>>();
    auto  reinit_ready      = reinit_started->get_future();

    EXPECT_CALL(*factory_ptr, createSubscriber(false)).WillOnce(Invoke([&subscriber](bool) {
        return std::move(subscriber);
    }));
    EXPECT_CALL(*subscriber_ptr, init(std::vector<std::string>{"direct"})).WillOnce(Return(true));
    EXPECT_CALL(*subscriber_ptr, getAddresses(_))
        .WillOnce(DoAll(SetArgReferee<0>(std::vector<std::string>{"direct"}), Return(true)));
    EXPECT_CALL(*factory_ptr, createMetaClient(_, _))
        .WillOnce(Invoke([&old_meta_client](const auto&, const auto&) { return std::move(old_meta_client); }))
        .WillOnce(Invoke([&new_meta_client, reinit_started](const auto&, const auto&) {
            auto result = std::move(new_meta_client);
            reinit_started->set_value();
            return result;
        }));
    static const std::string storage_config = R"({"sdk_backend_configs":[]})";
    EXPECT_CALL(*old_meta_ptr, GetStorageConfig()).WillOnce(ReturnRef(storage_config));
    EXPECT_CALL(*old_meta_ptr, FinishWrite("finish_1", "session", _, _))
        .WillOnce(Return(kv_cache_manager::ClientErrorCode::ER_SERVICE_INSTANCE_NOT_EXIST));
    EXPECT_CALL(*new_meta_ptr, FinishWrite("finish_2", "session", _, _))
        .WillOnce(Return(kv_cache_manager::ClientErrorCode::ER_OK));
    EXPECT_CALL(*factory_ptr, createTransferClient(_, _)).WillOnce(Invoke([&transfer_client](const auto&, const auto&) {
        return std::move(transfer_client);
    }));

    ClientWrapper                wrapper(std::move(factory));
    std::array<char, 32>         registration{};
    kv_cache_manager::RegistSpan span{registration.data(), registration.size()};
    ASSERT_TRUE(wrapper.init(
        {{"", makeConfig(false, "direct")}}, {kv_cache_manager::RoleType::HYBRID, &span, "tp0_Ffull"}, "default"));
    EXPECT_FALSE(wrapper.finishWrite("", "finish_1", "session", kv_cache_manager::BlockMaskOffset{0}, {}));
    ASSERT_EQ(reinit_ready.wait_for(std::chrono::seconds(3)), std::future_status::ready);

    bool       routed   = false;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (!routed && std::chrono::steady_clock::now() < deadline) {
        routed = wrapper.finishWrite("", "finish_2", "session", kv_cache_manager::BlockMaskOffset{0}, {});
        if (!routed) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    EXPECT_TRUE(routed);
    wrapper.shutdown();
    EXPECT_EQ(*destruction_count, 1);
}

TEST(ClientWrapperTest, DisabledPublicationPreservesSdkCallAndTraceIdentity) {
    autil::EnvGuard disabled("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "0");
    using namespace kv_cache_manager;
    unsigned char       byte = 7;
    RegistSpan          span{&byte, 1};
    InitObservation     observation;
    MockTransferClient* mock   = nullptr;
    auto                client = initializeClient(
        false, "upstream_passthrough", span, observation, std::make_shared<int>(0), nullptr, nullptr, &mock);
    ASSERT_NE(client, nullptr);
    // Above both the new block/IOV chunk limits: ordinary SDK clients retain
    // their original call granularity and receive the caller's descriptors.
    UriStrVec    uris(300, "allocated_uri"), actual(300, "actual_uri");
    BlockBuffers buffers(300);
    for (auto& block : buffers)
        for (size_t j = 0; j < 4; ++j)
            block.iovs.push_back({MemoryType::CPU, &byte, 1, false});
    auto trace          = std::make_shared<TransferTraceInfo>();
    trace->block_ids    = uris;
    size_t observations = 0;
    client->setTransferObserver([&](bool, size_t, bool) { ++observations; });
    EXPECT_CALL(*mock, SaveKvCaches(_, _, _)).WillOnce(Invoke([&](const auto& u, const auto& b, const auto& t) {
        EXPECT_EQ(&u, &uris);
        EXPECT_EQ(&b, &buffers);
        EXPECT_EQ(t, trace);
        return std::make_pair(ClientErrorCode::ER_OK, actual);
    }));
    const auto saved = client->saveKvCachesForTagWithStatus("default", uris, buffers, trace);
    EXPECT_EQ(saved.first, ClientErrorCode::ER_OK);
    EXPECT_EQ(saved.second, actual);
    EXPECT_CALL(*mock, LoadKvCaches(_, _, _)).WillOnce(Invoke([&](const auto& u, const auto& b, const auto& t) {
        EXPECT_EQ(&u, &actual);
        EXPECT_EQ(&b, &buffers);
        EXPECT_EQ(t, trace);
        *static_cast<unsigned char*>(b[0].iovs[0].base) = 9;
        return ClientErrorCode::ER_SDK_TIMEOUT;
    }));
    EXPECT_EQ(client->loadKvCachesForTagWithStatus("default", actual, buffers, trace), ClientErrorCode::ER_SDK_TIMEOUT);
    EXPECT_EQ(byte, 9u);
    EXPECT_EQ(buffers[0].iovs[0].size, 1u);
    EXPECT_CALL(*mock, SaveKvCaches(_, _, _))
        .WillOnce(Return(std::make_pair(ClientErrorCode::ER_SDK_TIMEOUT, UriStrVec{"partial_uri"})));
    const auto failed = client->saveKvCachesForTag("default", uris, buffers, trace);
    EXPECT_FALSE(failed.first);
    EXPECT_TRUE(failed.second.empty());
    EXPECT_EQ(observations, 0u);
    client->shutdown();
}

TEST(ClientWrapperTest, BoundedSdkChunksPreserveUrisAndDestinationBytes) {
    autil::EnvGuard enabled("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1");
    using namespace kv_cache_manager;
    std::vector<unsigned char> data(250 * 19), destination(data.size(), 0);
    for (size_t i = 0; i < data.size(); ++i)
        data[i] = static_cast<unsigned char>(i * 31 + 7);
    RegistSpan          span{data.data(), data.size()};
    InitObservation     observation;
    MockTransferClient* mock = nullptr;
    auto                client =
        initializeClient(false, "bounded", span, observation, std::make_shared<int>(0), nullptr, nullptr, &mock);
    ASSERT_NE(client, nullptr);
    ASSERT_NE(mock, nullptr);
    size_t read_bytes = 0, written_bytes = 0, unknown_calls = 0;
    client->setTransferObserver([&](bool read, size_t bytes, bool unknown) {
        (read ? read_bytes : written_bytes) += bytes;
        unknown_calls += unknown;
    });
    UriStrVec    uris;
    BlockBuffers source, target;
    for (size_t i = 0; i < 250; ++i) {
        uris.push_back("uri" + std::to_string(i));
        source.emplace_back();
        target.emplace_back();
        for (size_t j = 0; j < 19; ++j) {
            source.back().iovs.push_back({MemoryType::CPU, data.data() + i * 19 + j, 1, false});
            target.back().iovs.push_back({MemoryType::CPU, destination.data() + i * 19 + j, 1, false});
        }
    }
    auto trace        = std::make_shared<TransferTraceInfo>();
    trace->need_print = true;
    trace->block_ids  = uris;
    size_t saved = 0, loaded = 0;
    EXPECT_CALL(*mock, SaveKvCaches(_, _, _))
        .Times(5)
        .WillRepeatedly(Invoke([&](const auto& u, const auto& b, const auto& part_trace) {
            EXPECT_NE(part_trace, nullptr);
            EXPECT_EQ(part_trace->block_ids, u);
            EXPECT_LE(u.size() * 19, 1024u);
            UriStrVec actual;
            for (size_t i = 0; i < u.size(); ++i) {
                EXPECT_EQ(u[i], uris[saved + i]);
                EXPECT_EQ(b[i].iovs.size(), 19u);
                for (size_t j = 0; j < 19; ++j)
                    EXPECT_EQ(b[i].iovs[j].base, source[saved + i].iovs[j].base);
                actual.push_back("actual_" + u[i]);
            }
            saved += u.size();
            return std::make_pair(ClientErrorCode::ER_OK, actual);
        }));
    auto written = client->saveKvCachesForTagWithStatus("default", uris, source, trace);
    ASSERT_EQ(written.first, ClientErrorCode::ER_OK);
    ASSERT_EQ(saved, 250u);
    ASSERT_EQ(written.second.size(), 250u);
    EXPECT_CALL(*mock, LoadKvCaches(_, _, _))
        .Times(5)
        .WillRepeatedly(Invoke([&](const auto& u, auto& b, const auto& part_trace) {
            EXPECT_NE(part_trace, nullptr);
            EXPECT_EQ(part_trace->block_ids.size(), u.size());
            for (size_t i = 0; i < u.size(); ++i)
                EXPECT_EQ(part_trace->block_ids[i], uris[loaded + i]);
            EXPECT_LE(u.size() * 19, 1024u);
            for (size_t i = 0; i < u.size(); ++i) {
                EXPECT_EQ(u[i], "actual_" + uris[loaded + i]);
                for (size_t j = 0; j < 19; ++j)
                    *static_cast<unsigned char*>(b[i].iovs[j].base) = data[(loaded + i) * 19 + j];
            }
            loaded += u.size();
            return ClientErrorCode::ER_OK;
        }));
    ASSERT_EQ(client->loadKvCachesForTagWithStatus("default", written.second, target, trace), ClientErrorCode::ER_OK);
    EXPECT_EQ(loaded, 250u);
    EXPECT_EQ(data, destination);
    EXPECT_EQ(read_bytes, data.size());
    EXPECT_EQ(written_bytes, data.size());
    EXPECT_EQ(unknown_calls, 0u);
    client->shutdown();
}
TEST(ClientWrapperTest, BoundedSdkStopsOnFailureAndRejectsUnsafeGeometry) {
    autil::EnvGuard enabled("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1");
    using namespace kv_cache_manager;
    unsigned char       byte = 7;
    RegistSpan          span{&byte, 1};
    InitObservation     observation;
    MockTransferClient* mock   = nullptr;
    auto                client = initializeClient(
        false, "bounded_failure", span, observation, std::make_shared<int>(0), nullptr, nullptr, &mock);
    ASSERT_NE(client, nullptr);
    size_t confirmed = 0, expected_confirmed = 0, unknown_calls = 0;
    client->setTransferObserver([&](bool read, size_t bytes, bool unknown) {
        EXPECT_FALSE(read);
        confirmed += bytes;
        unknown_calls += unknown;
    });
    UriStrVec    uris(120, "uri");
    BlockBuffers buffers(120);
    for (auto& b : buffers)
        for (int j = 0; j < 19; ++j)
            b.iovs.push_back({MemoryType::CPU, &byte, 1, false});
    {
        ::testing::InSequence ordered;
        EXPECT_CALL(*mock, SaveKvCaches(_, _, _)).WillOnce(Invoke([&](const auto&, const auto& batch, const auto&) {
            for (const auto& block : batch)
                for (const auto& iov : block.iovs)
                    expected_confirmed += iov.size;
            return std::make_pair(ClientErrorCode::ER_OK, UriStrVec{});
        }));
        EXPECT_CALL(*mock, SaveKvCaches(_, _, _))
            .WillOnce(Return(std::make_pair(ClientErrorCode::ER_INVALID_PARAMS, UriStrVec{})));
    }
    auto completion = ClientWrapper::TransferCompletion::NOT_STARTED;
    auto result     = client->saveKvCachesForTagWithStatus("default", uris, buffers, nullptr, &completion);
    EXPECT_EQ(result.first, ClientErrorCode::ER_INVALID_PARAMS);
    EXPECT_TRUE(result.second.empty());
    EXPECT_GT(confirmed, 0u);
    EXPECT_EQ(confirmed, expected_confirmed);
    EXPECT_EQ(unknown_calls, 1u);
    EXPECT_EQ(completion, ClientWrapper::TransferCompletion::UNKNOWN);
    ::testing::Mock::VerifyAndClearExpectations(mock);
    EXPECT_CALL(*mock, SaveKvCaches(_, _, _)).Times(0);
    EXPECT_CALL(*mock, LoadKvCaches(_, _, _)).Times(0);
    EXPECT_EQ(client->saveKvCachesForTagWithStatus("default", UriStrVec{"one"}, buffers, nullptr, &completion).first,
              ClientErrorCode::ER_INVALID_PARAMS);
    EXPECT_EQ(completion, ClientWrapper::TransferCompletion::NOT_STARTED);
    BlockBuffers unsafe(1);
    for (int j = 0; j < 1025; ++j)
        unsafe[0].iovs.push_back({MemoryType::CPU, &byte, 1, false});
    EXPECT_EQ(client->loadKvCachesForTagWithStatus("default", UriStrVec{"one"}, unsafe, nullptr, &completion),
              ClientErrorCode::ER_INVALID_PARAMS);
    EXPECT_EQ(completion, ClientWrapper::TransferCompletion::NOT_STARTED);
    EXPECT_EQ(unknown_calls, 1u);  // Rejected geometry never attempted SDK I/O.
    client->shutdown();
}

class ObservedTransferClient: public kv_cache_manager::MockTransferClient, public SdkTransferObservable {
public:
    using MockTransferClient::MockTransferClient;
    void setSdkTransferObserver(SdkTransferObserver next) override {
        ++setter_calls;
        observer = std::move(next);
    }
    SdkTransferObserver observer;
    size_t              setter_calls{0};
};

TEST(ClientWrapperTest, InnerSdkObserverOwnsPartialBytesAndSuppressesOuterFailureOrException) {
    autil::EnvGuard enabled("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1");
    using namespace kv_cache_manager;
    unsigned char   byte = 7;
    RegistSpan      span{&byte, 1};
    InitObservation observation;
    auto            destruction_count = std::make_shared<int>(0);
    auto            inner             = std::make_unique<ObservedTransferClient>(destruction_count);
    auto*           observed          = inner.get();
    auto            client            = initializeClient(
        false, "inner_sdk", span, observation, destruction_count, nullptr, nullptr, nullptr, std::move(inner));
    ASSERT_NE(client, nullptr);
    size_t completed = 0, unknown = 0, events = 0;
    client->setTransferObserver([&](bool read, size_t bytes, bool uncertain) {
        EXPECT_TRUE(read);
        completed += bytes;
        unknown += uncertain;
        ++events;
    });
    ASSERT_TRUE(static_cast<bool>(observed->observer));
    BlockBuffers buffers(1);
    buffers[0].iovs.push_back({MemoryType::CPU, &byte, 1, false});
    EXPECT_CALL(*observed, LoadKvCaches(_, _, _)).WillOnce(Invoke([&](const auto&, const auto&, const auto&) {
        observed->observer(true, 6, false);
        observed->observer(true, 0, true);
        return ClientErrorCode::ER_SDKREAD_ERROR;
    }));
    EXPECT_EQ(client->loadKvCachesForTagWithStatus("default", {"uri"}, buffers), ClientErrorCode::ER_SDKREAD_ERROR);
    EXPECT_EQ(completed, 6u);
    EXPECT_EQ(unknown, 1u);
    EXPECT_EQ(events, 2u);
    EXPECT_CALL(*observed, LoadKvCaches(_, _, _))
        .WillOnce(Invoke([&](const auto&, const auto&, const auto&) -> ClientErrorCode {
            observed->observer(true, 0, true);
            throw std::runtime_error("inner SDK unknown");
        }));
    EXPECT_THROW(client->loadKvCachesForTagWithStatus("default", {"uri"}, buffers), std::runtime_error);
    EXPECT_EQ(unknown, 2u);
    EXPECT_EQ(events, 3u);  // No outer exception duplicates the inner event.
    client->shutdown();
    EXPECT_EQ(*destruction_count, 1);
}

TEST(ClientWrapperTest, InnerSdkPreIoFailureProducesNoEventAndClearingOrShutdownReleasesCallback) {
    autil::EnvGuard enabled("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1");
    using namespace kv_cache_manager;
    unsigned char   byte = 7;
    RegistSpan      span{&byte, 1};
    InitObservation observation;
    auto            destruction_count = std::make_shared<int>(0);
    auto            inner             = std::make_unique<ObservedTransferClient>(destruction_count);
    auto*           observed          = inner.get();
    auto            client            = initializeClient(
        false, "inner_pre_io", span, observation, destruction_count, nullptr, nullptr, nullptr, std::move(inner));
    ASSERT_NE(client, nullptr);
    auto               owner    = std::make_shared<int>(0);
    std::weak_ptr<int> lifetime = owner;
    size_t             events   = 0;
    client->setTransferObserver([owner, &events](bool, size_t, bool) { ++events; });
    owner.reset();
    BlockBuffers buffers(1);
    buffers[0].iovs.push_back({MemoryType::CPU, &byte, 1, false});
    EXPECT_CALL(*observed, SaveKvCaches(_, _, _))
        .WillOnce(Return(std::make_pair(ClientErrorCode::ER_INVALID_PARAMS, UriStrVec{})));
    EXPECT_EQ(client->saveKvCachesForTagWithStatus("default", {"uri"}, buffers).first,
              ClientErrorCode::ER_INVALID_PARAMS);
    EXPECT_EQ(events, 0u);  // Packer rejected before any inner SDK call.
    EXPECT_FALSE(lifetime.expired());
    client->setTransferObserver({});
    EXPECT_FALSE(static_cast<bool>(observed->observer));
    EXPECT_TRUE(lifetime.expired());
    client->shutdown();
}

TEST(ClientWrapperTest, FeatureOffDoesNotInstallAnInnerSdkObserver) {
    autil::EnvGuard disabled("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "0");
    using namespace kv_cache_manager;
    unsigned char   byte = 7;
    RegistSpan      span{&byte, 1};
    InitObservation observation;
    auto            destruction_count = std::make_shared<int>(0);
    auto            inner             = std::make_unique<ObservedTransferClient>(destruction_count);
    auto*           observed          = inner.get();
    auto            client            = initializeClient(
        false, "off_inner_sdk", span, observation, destruction_count, nullptr, nullptr, nullptr, std::move(inner));
    ASSERT_NE(client, nullptr);
    client->setTransferObserver([](bool, size_t, bool) {});
    EXPECT_EQ(observed->setter_calls, 0u);
    EXPECT_FALSE(static_cast<bool>(observed->observer));
    client->shutdown();
}

template<typename Code>
std::vector<std::pair<Code, bool>> completionProofCases() {
    std::vector<std::pair<Code, bool>> cases{{Code::ER_OK, true},
                                             {Code::ER_INVALID_PARAMS, false},
                                             {Code::ER_SDKREAD_ERROR, false},
                                             {Code::ER_SDK_TIMEOUT, false}};
    if constexpr (detail::HasExplicitBufferSafeCodes<Code>::value) {
        cases.emplace_back(Code::ER_SDK_TIMEOUT_BUFFER_SAFE, true);
        cases.emplace_back(Code::ER_SDK_ERROR_BUFFER_SAFE, true);
    }
    return cases;
}

TEST(ClientWrapperTest, CompletionProofDistinguishesLocalRejectionFromActualSdkError) {
    autil::EnvGuard enabled("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1");
    using namespace kv_cache_manager;
    using Completion         = ClientWrapper::TransferCompletion;
    unsigned char       byte = 7;
    RegistSpan          span{&byte, 1};
    InitObservation     observation;
    MockTransferClient* mock   = nullptr;
    auto                client = initializeClient(
        false, "completion_proof", span, observation, std::make_shared<int>(0), nullptr, nullptr, &mock);
    ASSERT_NE(client, nullptr);
    BlockBuffers buffers(1);
    buffers[0].iovs.push_back({MemoryType::CPU, &byte, 1, false});
    auto completion = Completion::UNKNOWN;
    EXPECT_EQ(client->loadKvCachesForTagWithStatus("missing", {"uri"}, buffers, nullptr, &completion),
              ClientErrorCode::ER_CLIENT_NOT_EXISTS);
    EXPECT_EQ(completion, Completion::NOT_STARTED);
    EXPECT_EQ(client->loadKvCachesForTagWithStatus("default", {"uri", "extra"}, buffers, nullptr, &completion),
              ClientErrorCode::ER_INVALID_PARAMS);
    EXPECT_EQ(completion, Completion::NOT_STARTED);
    for (const auto& [ec, safe] : completionProofCases<ClientErrorCode>()) {
        SCOPED_TRACE(static_cast<int>(ec));
        EXPECT_CALL(*mock, LoadKvCaches(_, _, _)).WillOnce(Return(ec));
        EXPECT_EQ(client->loadKvCachesForTagWithStatus("default", {"uri"}, buffers, nullptr, &completion), ec);
        EXPECT_EQ(completion, safe ? Completion::BUFFER_SAFE : Completion::UNKNOWN);
        EXPECT_CALL(*mock, SaveKvCaches(_, _, _)).WillOnce(Return(std::make_pair(ec, UriStrVec{})));
        EXPECT_EQ(client->saveKvCachesForTagWithStatus("default", {"uri"}, buffers, nullptr, &completion).first, ec);
        EXPECT_EQ(completion, safe ? Completion::BUFFER_SAFE : Completion::UNKNOWN);
    }
    EXPECT_CALL(*mock, LoadKvCaches(_, _, _))
        .WillOnce(Invoke([](const auto&, const auto&, const auto&) -> ClientErrorCode {
            throw std::runtime_error("SDK may retain caller buffers");
        }));
    EXPECT_THROW(client->loadKvCachesForTagWithStatus("default", {"uri"}, buffers, nullptr, &completion),
                 std::runtime_error);
    EXPECT_EQ(completion, Completion::UNKNOWN);
    client->shutdown();
}

TEST(ClientWrapperTest, ThrowingSdkMarksUnknownAndShutdownReleasesObserver) {
    autil::EnvGuard enabled("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "1");
    using namespace kv_cache_manager;
    unsigned char       byte = 7;
    RegistSpan          span{&byte, 1};
    InitObservation     observation;
    MockTransferClient* mock = nullptr;
    auto                client =
        initializeClient(false, "throwing_sdk", span, observation, std::make_shared<int>(0), nullptr, nullptr, &mock);
    ASSERT_NE(client, nullptr);
    auto               owner   = std::make_shared<int>(0);
    std::weak_ptr<int> alive   = owner;
    size_t             unknown = 0;
    client->setTransferObserver([owner, &unknown](bool read, size_t bytes, bool uncertain) {
        (void)owner;
        EXPECT_TRUE(read);
        EXPECT_EQ(bytes, 0u);
        EXPECT_TRUE(uncertain);
        ++unknown;
        throw std::runtime_error("reporting failed");
    });
    owner.reset();
    BlockBuffers buffers(1);
    buffers[0].iovs.push_back({MemoryType::CPU, &byte, 1, false});
    EXPECT_CALL(*mock, LoadKvCaches(_, _, _)).WillOnce(Invoke([](const auto&, auto&, const auto&) -> ClientErrorCode {
        throw std::runtime_error("SDK completion unknown");
    }));
    EXPECT_THROW(client->loadKvCachesForTagWithStatus("default", UriStrVec{"uri"}, buffers), std::runtime_error);
    EXPECT_EQ(unknown, 1u);
    EXPECT_FALSE(alive.expired());
    client->shutdown();
    EXPECT_TRUE(alive.expired());
}

}  // namespace
}  // namespace rtp_llm::kvcm
