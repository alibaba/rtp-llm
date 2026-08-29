#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "rtp_llm/cpp/cache/connector/remote_connector/ClientWrapper.h"
#include "rtp_llm/cpp/cache/connector/remote_connector/test/MockKVCMClient.h"

namespace rtp_llm::remote_connector {
namespace {

using ::testing::_;
using ::testing::DoAll;
using ::testing::Invoke;
using ::testing::Return;
using ::testing::ReturnRef;
using ::testing::SetArgReferee;

struct InitObservation {
    const kv_cache_manager::RegistSpan* registration_descriptor = nullptr;
    void*                               registration_base       = nullptr;
    size_t                              registration_size       = 0;
};

RemoteConnectorConfigPtr makeConfig(bool enable_vipserver, const std::string& endpoint) {
    auto location_infos  = std::make_shared<RemoteConnectorConfig::LocationSpecInfoMap>();
    auto location_groups = std::make_shared<RemoteConnectorConfig::LocationSpecGroups>();
    auto channel         = std::make_shared<MetaChannelConfig>(/*retry_time=*/1,
                                                       /*connection_timeout=*/1000,
                                                       /*call_timeout=*/100);
    auto sdk             = std::make_shared<SdkWrapperConfig>();
    return std::make_shared<RemoteConnectorConfig>(enable_vipserver,
                                                   enable_vipserver ? endpoint : "",
                                                   /*block_size=*/8,
                                                   "instance_group",
                                                   "instance_id",
                                                   enable_vipserver ? std::vector<std::string>{} :
                                                                      std::vector<std::string>{endpoint},
                                                   location_infos,
                                                   channel,
                                                   sdk,
                                                   location_groups,
                                                   ModelDeployment());
}

std::shared_ptr<ClientWrapper> initializeClient(bool                                   enable_vipserver,
                                                const std::string&                     endpoint,
                                                kv_cache_manager::RegistSpan&          registration_span,
                                                InitObservation&                       observation,
                                                std::shared_ptr<int>                   transfer_destruction_count,
                                                kv_cache_manager::MockMetaClient**     initial_meta_client = nullptr,
                                                std::shared_ptr<std::atomic<int>>      reinit_attempts     = nullptr,
                                                kv_cache_manager::MockTransferClient** transfer_client     = nullptr) {
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
            .WillRepeatedly(Invoke([reinit_attempts](const std::string&, const kv_cache_manager::InitParams&) {
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
            auto client = std::make_unique<kv_cache_manager::MockTransferClient>(transfer_destruction_count);
            if (transfer_client != nullptr) {
                *transfer_client = client.get();
            }
            return client;
        }));

    auto                         wrapper = std::make_shared<ClientWrapper>(std::move(factory));
    ClientWrapper::ConfigMap     config_map{{"", makeConfig(enable_vipserver, endpoint)}};
    kv_cache_manager::InitParams params{kv_cache_manager::RoleType::HYBRID, &registration_span, "tp0_Fgroup"};
    if (!wrapper->init(config_map, params)) {
        return nullptr;
    }
    return wrapper;
}

TEST(ClientWrapperLifecycleTest, RecreatesTransferClientForDifferentRegistrationSpan) {
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

TEST(ClientWrapperLifecycleTest, RecreatesSubscriberWhenSwitchingFromDirectToVipServer) {
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

TEST(ClientWrapperLifecycleTest, ShutdownInterruptsFailedReRegistrationBeforeCreatingNewSpanClient) {
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
    kv_cache_manager::UriStrVec    uris;
    kv_cache_manager::BlockBuffers buffers;
    std::promise<bool>             transfer_result_promise;
    auto                           transfer_result = transfer_result_promise.get_future();
    std::thread transfer_thread([&] { transfer_result_promise.set_value(old_client->loadKvCaches(uris, buffers)); });
    const auto  transfer_status = transfer_result.wait_for(std::chrono::seconds(1));
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

}  // namespace
}  // namespace rtp_llm::remote_connector
