#include <thread>
#include <chrono>
#include "MockKVCMClient.h"
#include "rtp_llm/cpp/cache/connector/remote_connector/ClientWrapper.h"
#include "rtp_llm/cpp/utils/Logger.h"

using namespace ::testing;

namespace rtp_llm {
namespace remote_connector {

class ClientWrapperTest: public ::testing::Test {
public:
    void SetUp() override {
        rtp_llm::initLogger();
        ClientWrapper::client_factory_     = std::make_unique<MockClientFactory>();
        mock_client_factory_               = dynamic_cast<MockClientFactory*>(ClientWrapper::client_factory_.get());
        ClientWrapper::subscriber_         = std::make_unique<MockSubscriber>();
        mock_subscriber_                   = dynamic_cast<MockSubscriber*>(ClientWrapper::subscriber_.get());
        client_wrapper_                    = std::make_shared<ClientWrapper>();
        client_wrapper_->address_snapshot_ = init_addresses_;

        auto& config_map                  = client_wrapper_->config_map_;
        config_map[""]                    = std::make_shared<RemoteConnectorConfig>();
        config_map[""]->enable_vipserver_ = true;
        config_map[""]->set_addresses(init_addresses_);
        config_map[""]->instance_id_ = "default_instance";

        default_meta_client_ = std::make_shared<kv_cache_manager::MockMetaClient>();

        client_wrapper_->meta_client_map_[""] = default_meta_client_;
    }

    void TearDown() override {
        ClientWrapper::subscriber_.reset();
        ClientWrapper::client_factory_.reset();
    }

private:
    ClientWrapper::ConfigMap makeConfigMap() {
        auto config =
            std::make_shared<RemoteConnectorConfig>(false,
                                                    "",
                                                    8,
                                                    "group",
                                                    "instance",
                                                    std::vector<std::string>{"address"},
                                                    std::make_shared<RemoteConnectorConfig::LocationSpecInfoMap>(),
                                                    std::make_shared<MetaChannelConfig>(),
                                                    std::make_shared<SdkWrapperConfig>(),
                                                    std::make_shared<RemoteConnectorConfig::LocationSpecGroups>(),
                                                    ModelDeployment());
        return {{"", std::move(config)}};
    }

    MockClientFactory*                                mock_client_factory_ = nullptr;
    MockSubscriber*                                   mock_subscriber_     = nullptr;
    std::shared_ptr<kv_cache_manager::MockMetaClient> default_meta_client_;
    std::shared_ptr<ClientWrapper>                    client_wrapper_;
    inline static const std::vector<std::string>      init_addresses_ = {"init_address"};
};

TEST_F(ClientWrapperTest, test_no_need_reinit) {
    EXPECT_CALL(*mock_client_factory_, CreateMetaClient(_, _)).Times(0);
    EXPECT_CALL(*mock_subscriber_, getAddresses(_)).WillOnce(DoAll(SetArgReferee<0>(init_addresses_), Return(true)));
    EXPECT_CALL(*default_meta_client_, FinishWrite(Eq("default_trace"), _, _, _))
        .WillOnce(Return(kv_cache_manager::ClientErrorCode::ER_OK));
    ASSERT_TRUE(client_wrapper_->finishWrite("", "default_trace", "", {}, {}));
}

TEST_F(ClientWrapperTest, test_no_invalid_addresses) {
    EXPECT_CALL(*mock_client_factory_, CreateMetaClient(_, _)).Times(0);
    const std::vector<std::string> empty_addresses = {};
    EXPECT_CALL(*mock_subscriber_, getAddresses(_)).WillOnce(DoAll(SetArgReferee<0>(empty_addresses), Return(false)));
    EXPECT_CALL(*default_meta_client_, FinishWrite(_, _, _, _)).Times(0);
    ASSERT_FALSE(client_wrapper_->finishWrite("", "", "", {}, {}));
    ASSERT_EQ(init_addresses_, client_wrapper_->address_snapshot_);
}

TEST_F(ClientWrapperTest, test_reinit_with_new_addresses) {
    auto new_default_meta_client     = std::make_unique<kv_cache_manager::MockMetaClient>();
    auto raw_new_default_meta_client = new_default_meta_client.get();
    EXPECT_CALL(*mock_client_factory_, CreateMetaClient(_, _))
        .WillOnce(Invoke([&](const std::string&, const kv_cache_manager::InitParams&) {
            return std::move(new_default_meta_client);
        }));
    const std::vector<std::string> new_addresses = {"new_address"};
    EXPECT_CALL(*mock_subscriber_, getAddresses(_))
        .Times(2)
        .WillOnce(DoAll(SetArgReferee<0>(init_addresses_), Return(true)))
        .WillOnce(DoAll(SetArgReferee<0>(new_addresses), Return(true)));
    EXPECT_CALL(*default_meta_client_, FinishWrite(Eq("trace_1"), _, _, _))
        .WillOnce(Return(kv_cache_manager::ClientErrorCode::ER_OK));
    ASSERT_TRUE(client_wrapper_->finishWrite("", "trace_1", "", {}, {}));
    ASSERT_EQ(init_addresses_, client_wrapper_->address_snapshot_);
    ASSERT_EQ(default_meta_client_.get(), client_wrapper_->meta_client_map_.at("").get());

    // reinit default instance
    EXPECT_CALL(*raw_new_default_meta_client, FinishWrite(Eq("trace_2"), _, _, _))
        .WillOnce(Return(kv_cache_manager::ClientErrorCode::ER_OK));
    ASSERT_TRUE(client_wrapper_->finishWrite("", "trace_2", "", {}, {}));
    ASSERT_EQ(new_addresses, client_wrapper_->address_snapshot_);
    ASSERT_EQ(new_addresses, client_wrapper_->config_map_.at("")->addresses_);
    ASSERT_EQ(raw_new_default_meta_client, client_wrapper_->meta_client_map_.at("").get());
}

TEST_F(ClientWrapperTest, test_new_address_create_client_first_fail_second_success) {
    auto new_default_meta_client     = std::make_unique<kv_cache_manager::MockMetaClient>();
    auto raw_new_default_meta_client = new_default_meta_client.get();
    EXPECT_CALL(*mock_client_factory_, CreateMetaClient(_, _))
        .WillOnce(Invoke([&](const std::string&, const kv_cache_manager::InitParams&) { return nullptr; }))
        .WillOnce(Invoke([&](const std::string&, const kv_cache_manager::InitParams&) {
            return std::move(new_default_meta_client);
        }));
    const std::vector<std::string> new_addresses = {"new_address"};
    EXPECT_CALL(*mock_subscriber_, getAddresses(_))
        .Times(3)
        .WillOnce(DoAll(SetArgReferee<0>(init_addresses_), Return(true)))
        .WillOnce(DoAll(SetArgReferee<0>(new_addresses), Return(true)))
        .WillOnce(DoAll(SetArgReferee<0>(new_addresses), Return(true)));
    // init address
    EXPECT_CALL(*default_meta_client_, FinishWrite(Eq("trace_1"), _, _, _))
        .WillOnce(Return(kv_cache_manager::ClientErrorCode::ER_OK));
    ASSERT_TRUE(client_wrapper_->finishWrite("", "trace_1", "", {}, {}));
    ASSERT_EQ(init_addresses_, client_wrapper_->address_snapshot_);
    ASSERT_EQ(default_meta_client_.get(), client_wrapper_->meta_client_map_.at("").get());
    // first : new address, but failed to create new meta client
    ASSERT_FALSE(client_wrapper_->finishWrite("", "trace_2", "", {}, {}));
    const std::vector<std::string> empty_addresses = {};
    ASSERT_EQ(empty_addresses, client_wrapper_->address_snapshot_);
    ASSERT_EQ(empty_addresses, client_wrapper_->config_map_.at("")->addresses_);
    // second : new address, succeed to create to new meta client
    ASSERT_TRUE(client_wrapper_->finishWrite("", "trace_3", "", {}, {}));
    ASSERT_EQ(new_addresses, client_wrapper_->address_snapshot_);
    ASSERT_EQ(new_addresses, client_wrapper_->config_map_.at("")->addresses_);
    ASSERT_EQ(raw_new_default_meta_client, client_wrapper_->meta_client_map_.at("").get());
}

TEST_F(ClientWrapperTest, test_registration) {
    auto new_default_meta_client     = std::make_unique<kv_cache_manager::MockMetaClient>();
    auto raw_new_default_meta_client = new_default_meta_client.get();
    EXPECT_CALL(*mock_client_factory_, CreateMetaClient(_, _))
        .WillOnce(Invoke([&](const std::string&, const kv_cache_manager::InitParams&) {
            return std::move(new_default_meta_client);
        }));
    EXPECT_CALL(*mock_subscriber_, getAddresses(_))
        .Times(3)
        .WillRepeatedly(DoAll(SetArgReferee<0>(init_addresses_), Return(true)));
    EXPECT_CALL(*default_meta_client_, FinishWrite(Eq("trace_1"), _, _, _))
        .WillOnce(Return(kv_cache_manager::ClientErrorCode::ER_SERVICE_INSTANCE_NOT_EXIST));
    ASSERT_FALSE(client_wrapper_->finishWrite("", "trace_1", "", {}, {}));
    while (true) {
        // busy wait for reinitAllMetaClients thread working
        if (client_wrapper_->rr_other_working_.load(std::memory_order_acquire)) {
            break;
        }
    }
    int i = 0;
    for (i = 0; i < 100; i++) {
        // wait for reinitAllMetaClients thread finish
        if (!client_wrapper_->rr_other_working_.load(std::memory_order_acquire)) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    ASSERT_LT(i, 100);
    EXPECT_CALL(*raw_new_default_meta_client, FinishWrite(Eq("trace_2"), _, _, _))
        .WillOnce(Return(kv_cache_manager::ClientErrorCode::ER_OK));
    ASSERT_TRUE(client_wrapper_->finishWrite("", "trace_2", "", {}, {}));
    ASSERT_EQ(init_addresses_, client_wrapper_->address_snapshot_);
    ASSERT_EQ(raw_new_default_meta_client, client_wrapper_->meta_client_map_.at("").get());
}

TEST_F(ClientWrapperTest, transfer_clients_route_by_tag_and_restore_uri_order) {
    auto  full_client                            = std::make_unique<kv_cache_manager::MockTransferClient>();
    auto  linear_client                          = std::make_unique<kv_cache_manager::MockTransferClient>();
    auto* full_mock                              = full_client.get();
    auto* linear_mock                            = linear_client.get();
    client_wrapper_->transfer_clients_["full"]   = std::move(full_client);
    client_wrapper_->transfer_clients_["linear"] = std::move(linear_client);

    const std::vector<std::string>    tags{"full", "linear", "full"};
    const kv_cache_manager::UriStrVec uris{"full_0", "linear_0", "full_1"};
    kv_cache_manager::BlockBuffers    buffers(uris.size());

    EXPECT_CALL(*full_mock, LoadKvCaches(Eq(kv_cache_manager::UriStrVec{"full_0", "full_1"}), _, IsNull()))
        .WillOnce(Return(kv_cache_manager::ClientErrorCode::ER_OK));
    EXPECT_CALL(*linear_mock, LoadKvCaches(Eq(kv_cache_manager::UriStrVec{"linear_0"}), _, IsNull()))
        .WillOnce(Return(kv_cache_manager::ClientErrorCode::ER_OK));
    EXPECT_TRUE(client_wrapper_->loadKvCaches(tags, uris, buffers));

    EXPECT_CALL(*full_mock, SaveKvCaches(Eq(kv_cache_manager::UriStrVec{"full_0", "full_1"}), _, IsNull()))
        .WillOnce(Return(std::make_pair(kv_cache_manager::ClientErrorCode::ER_OK,
                                        kv_cache_manager::UriStrVec{"actual_full_0", "actual_full_1"})));
    EXPECT_CALL(*linear_mock, SaveKvCaches(Eq(kv_cache_manager::UriStrVec{"linear_0"}), _, IsNull()))
        .WillOnce(Return(
            std::make_pair(kv_cache_manager::ClientErrorCode::ER_OK, kv_cache_manager::UriStrVec{"actual_linear_0"})));
    const auto result = client_wrapper_->saveKvCaches(tags, uris, buffers);
    ASSERT_TRUE(result.first);
    EXPECT_EQ(result.second, (kv_cache_manager::UriStrVec{"actual_full_0", "actual_linear_0", "actual_full_1"}));

    kv_cache_manager::BlockBuffers missing_buffers(1);
    EXPECT_CALL(*full_mock, LoadKvCaches(_, _, _)).Times(0);
    EXPECT_CALL(*linear_mock, LoadKvCaches(_, _, _)).Times(0);
    EXPECT_FALSE(client_wrapper_->loadKvCaches({"missing"}, {"missing_uri"}, missing_buffers));
}

TEST_F(ClientWrapperTest, init_creates_one_instance_local_transfer_client_per_tag) {
    auto                     wrapper        = std::make_shared<ClientWrapper>();
    auto                     meta_client    = std::make_unique<kv_cache_manager::MockMetaClient>();
    auto*                    meta_mock      = meta_client.get();
    static const std::string storage_config = "storage";

    EXPECT_CALL(*mock_subscriber_, init(Eq(std::vector<std::string>{"address"}))).WillOnce(Return(true));
    EXPECT_CALL(*mock_client_factory_, CreateMetaClient(_, _))
        .WillOnce(
            Invoke([&](const std::string&, const kv_cache_manager::InitParams&) { return std::move(meta_client); }));
    EXPECT_CALL(*meta_mock, GetStorageConfig()).WillOnce(ReturnRef(storage_config));

    std::map<std::string, std::pair<void*, size_t>> created_registrations;
    EXPECT_CALL(*mock_client_factory_, CreateTransferClient(_, _))
        .Times(2)
        .WillRepeatedly(Invoke([&](const std::string&, const kv_cache_manager::InitParams& init_params) {
            auto client = std::make_unique<kv_cache_manager::MockTransferClient>();
            if (init_params.regist_span == nullptr) {
                ADD_FAILURE() << "missing per-tag registration span";
                return client;
            }
            created_registrations.emplace(init_params.self_location_spec_name,
                                          std::make_pair(init_params.regist_span->base, init_params.regist_span->size));
            return client;
        }));

    ClientWrapper::TransferRegistrationMap registrations;
    registrations.emplace("linear", ClientWrapper::TransferRegistration{reinterpret_cast<void*>(0x2000), 32, "tp0_L"});
    registrations.emplace("full", ClientWrapper::TransferRegistration{reinterpret_cast<void*>(0x1000), 64, "tp0_F"});
    ASSERT_TRUE(wrapper->init(makeConfigMap(), kv_cache_manager::RoleType::HYBRID, registrations));
    EXPECT_EQ(wrapper->transfer_clients_.size(), 2u);
    EXPECT_EQ(wrapper->transfer_clients_.count("full"), 1u);
    EXPECT_EQ(wrapper->transfer_clients_.count("linear"), 1u);
    EXPECT_EQ(created_registrations,
              (std::map<std::string, std::pair<void*, size_t>>{{"tp0_F", {reinterpret_cast<void*>(0x1000), 64}},
                                                               {"tp0_L", {reinterpret_cast<void*>(0x2000), 32}}}));
}

TEST_F(ClientWrapperTest, init_failure_clears_all_transfer_clients) {
    auto                     wrapper        = std::make_shared<ClientWrapper>();
    auto                     meta_client    = std::make_unique<kv_cache_manager::MockMetaClient>();
    auto*                    meta_mock      = meta_client.get();
    static const std::string storage_config = "storage";

    EXPECT_CALL(*mock_subscriber_, init(Eq(std::vector<std::string>{"address"}))).WillOnce(Return(true));
    EXPECT_CALL(*mock_client_factory_, CreateMetaClient(_, _))
        .WillOnce(
            Invoke([&](const std::string&, const kv_cache_manager::InitParams&) { return std::move(meta_client); }));
    EXPECT_CALL(*meta_mock, GetStorageConfig()).WillOnce(ReturnRef(storage_config));

    size_t creation_count = 0;
    EXPECT_CALL(*mock_client_factory_, CreateTransferClient(_, _))
        .Times(2)
        .WillRepeatedly(Invoke([&](const std::string&, const kv_cache_manager::InitParams&) {
            if (++creation_count == 2) {
                return std::unique_ptr<kv_cache_manager::TransferClient>();
            }
            return std::unique_ptr<kv_cache_manager::TransferClient>(
                std::make_unique<kv_cache_manager::MockTransferClient>());
        }));

    ClientWrapper::TransferRegistrationMap registrations{{"full", {reinterpret_cast<void*>(0x1000), 64, "tp0_F"}},
                                                         {"linear", {reinterpret_cast<void*>(0x2000), 32, "tp0_L"}}};
    EXPECT_FALSE(wrapper->init(makeConfigMap(), kv_cache_manager::RoleType::HYBRID, registrations));
    EXPECT_TRUE(wrapper->transfer_clients_.empty());
}

}  // namespace remote_connector
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
