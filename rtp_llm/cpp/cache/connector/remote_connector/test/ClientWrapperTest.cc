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

        config_map["lora"]                    = std::make_shared<RemoteConnectorConfig>();
        config_map["lora"]->enable_vipserver_ = true;
        config_map["lora"]->set_addresses(init_addresses_);
        config_map["lora"]->instance_id_ = "lora_instance";

        default_meta_client_ = std::make_shared<kv_cache_manager::MockMetaClient>();
        lora_meta_client_    = std::make_shared<kv_cache_manager::MockMetaClient>();

        client_wrapper_->meta_client_map_[""]     = default_meta_client_;
        client_wrapper_->meta_client_map_["lora"] = lora_meta_client_;
    }

    void TearDown() override {}

private:
    MockClientFactory*                                mock_client_factory_ = nullptr;
    MockSubscriber*                                   mock_subscriber_     = nullptr;
    std::shared_ptr<kv_cache_manager::MockMetaClient> default_meta_client_;
    std::shared_ptr<kv_cache_manager::MockMetaClient> lora_meta_client_;
    std::shared_ptr<ClientWrapper>                    client_wrapper_;
    inline static const std::vector<std::string>      init_addresses_ = {"init_address"};
};

TEST_F(ClientWrapperTest, test_no_need_reinit) {
    EXPECT_CALL(*mock_client_factory_, CreateMetaClient(_, _)).Times(0);
    EXPECT_CALL(*mock_subscriber_, getAddresses(_))
        .Times(2)
        .WillRepeatedly(DoAll(SetArgReferee<0>(init_addresses_), Return(true)));
    EXPECT_CALL(*default_meta_client_, FinishWrite(Eq("default_trace"), _, _, _))
        .WillOnce(Return(kv_cache_manager::ClientErrorCode::ER_OK));
    EXPECT_CALL(*lora_meta_client_, FinishWrite(Eq("lora_trace"), _, _, _))
        .WillOnce(Return(kv_cache_manager::ClientErrorCode::ER_OK));
    ASSERT_TRUE(client_wrapper_->finishWrite("", "default_trace", "", {}, {}));
    ASSERT_TRUE(client_wrapper_->finishWrite("lora", "lora_trace", "", {}, {}));
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
    auto new_lora_meta_client        = std::make_unique<kv_cache_manager::MockMetaClient>();
    auto raw_new_lora_meta_client    = new_lora_meta_client.get();
    EXPECT_CALL(*mock_client_factory_, CreateMetaClient(_, _))
        .WillOnce(Return(std::move(new_default_meta_client)))
        .WillOnce(Return(std::move(new_lora_meta_client)));
    const std::vector<std::string> new_addresses = {"new_address"};
    EXPECT_CALL(*mock_subscriber_, getAddresses(_))
        .Times(3)
        .WillOnce(DoAll(SetArgReferee<0>(init_addresses_), Return(true)))
        .WillOnce(DoAll(SetArgReferee<0>(new_addresses), Return(true)))
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

    // lora instance not reinit now
    ASSERT_EQ(init_addresses_, client_wrapper_->config_map_.at("lora")->addresses_);
    ASSERT_EQ(lora_meta_client_.get(), client_wrapper_->meta_client_map_.at("lora").get());
    // reinit lora instace
    EXPECT_CALL(*raw_new_lora_meta_client, FinishWrite(Eq("trace_3"), _, _, _))
        .WillOnce(Return(kv_cache_manager::ClientErrorCode::ER_OK));
    ASSERT_TRUE(client_wrapper_->finishWrite("lora", "trace_3", "", {}, {}));
    ASSERT_EQ(new_addresses, client_wrapper_->config_map_.at("lora")->addresses_);
    ASSERT_EQ(raw_new_lora_meta_client, client_wrapper_->meta_client_map_.at("lora").get());
}

TEST_F(ClientWrapperTest, test_new_address_create_client_first_fail_second_success) {
    auto new_default_meta_client     = std::make_unique<kv_cache_manager::MockMetaClient>();
    auto raw_new_default_meta_client = new_default_meta_client.get();
    EXPECT_CALL(*mock_client_factory_, CreateMetaClient(_, _))
        .WillOnce(Return(nullptr))
        .WillOnce(Return(std::move(new_default_meta_client)));
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
    auto new_lora_meta_client        = std::make_unique<kv_cache_manager::MockMetaClient>();
    auto raw_new_lora_meta_client    = new_lora_meta_client.get();
    EXPECT_CALL(*mock_client_factory_, CreateMetaClient(_, _))
        .Times(2)
        .WillOnce(Return(std::move(new_default_meta_client)))
        .WillOnce(Return(std::move(new_lora_meta_client)));
    EXPECT_CALL(*mock_subscriber_, getAddresses(_))
        .Times(4)
        .WillRepeatedly(DoAll(SetArgReferee<0>(init_addresses_), Return(true)));
    EXPECT_CALL(*default_meta_client_, FinishWrite(Eq("trace_1"), _, _, _))
        .WillOnce(Return(kv_cache_manager::ClientErrorCode::ER_SERVICE_INSTANCE_NOT_EXIST));
    ASSERT_FALSE(client_wrapper_->finishWrite("", "trace_1", "", {}, {}));
    while (true) {
        // busy wait for reRegistration thread working
        if (client_wrapper_->rr_other_working_.load(std::memory_order_acquire)) {
            break;
        }
    }
    int i = 0;
    for (i = 0; i < 100; i++) {
        // wait for reRegistration thread finish
        if (!client_wrapper_->rr_other_working_.load(std::memory_order_acquire)) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    ASSERT_LT(i, 100);
    EXPECT_CALL(*raw_new_default_meta_client, FinishWrite(Eq("trace_2"), _, _, _))
        .WillOnce(Return(kv_cache_manager::ClientErrorCode::ER_OK));
    EXPECT_CALL(*raw_new_lora_meta_client, FinishWrite(Eq("trace_3"), _, _, _))
        .WillOnce(Return(kv_cache_manager::ClientErrorCode::ER_OK));
    ASSERT_TRUE(client_wrapper_->finishWrite("", "trace_2", "", {}, {}));
    ASSERT_TRUE(client_wrapper_->finishWrite("lora", "trace_3", "", {}, {}));
    ASSERT_EQ(init_addresses_, client_wrapper_->address_snapshot_);
    ASSERT_EQ(raw_new_default_meta_client, client_wrapper_->meta_client_map_.at("").get());
    ASSERT_EQ(raw_new_lora_meta_client, client_wrapper_->meta_client_map_.at("lora").get());
}

}  // namespace remote_connector
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
