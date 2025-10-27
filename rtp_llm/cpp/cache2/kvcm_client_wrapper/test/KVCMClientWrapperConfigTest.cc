#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache2/kv_cache_connector/KVCMClientWrapperConfig.h"

namespace rtp_llm {
namespace test {

class KVCMClientWrapperConfigTest: public ::testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(KVCMClientWrapperConfigTest, SdkWrapperConfigTest) {
    {
        SdkWrapperConfig wrapper_config;
        std::string      json = R"(
{
    "thread_num":4,
    "queue_size":2000,
    "sdk_backend_configs":[{"type":"local","sdk_log_level":"DEBUG"}],
    "timeout_config": {
        "put_timeout_ms":2000,
        "get_timeout_ms":2000
    }
}
)";
        autil::legacy::FromJsonString(wrapper_config, json);
        ASSERT_EQ(1, wrapper_config.sdk_backend_configs_.size());
        std::shared_ptr<SdkBackendConfig> config =
            std::dynamic_pointer_cast<SdkBackendConfig>(wrapper_config.sdk_backend_configs_[0]);
        ASSERT_EQ(DataStorageType::DATA_STORAGE_TYPE_LOCAL, config->type_);
        ASSERT_EQ("DEBUG", config->sdk_log_level_);
    }
    {
        SdkWrapperConfig wrapper_config;
        std::string      json = R"(
{
    "thread_num":2,
    "queue_size":2000,
    "sdk_backend_configs":[{"type":"local","sdk_log_level":"DEBUG"}],
    "timeout_config": {
        "put_timeout_ms":2000,
        "get_timeout_ms":2000
    }
}
)";
        autil::legacy::FromJsonString(wrapper_config, json);
        ASSERT_EQ(2, wrapper_config.thread_num_);
        ASSERT_EQ(1, wrapper_config.sdk_backend_configs_.size());
        std::shared_ptr<SdkBackendConfig> config =
            std::dynamic_pointer_cast<SdkBackendConfig>(wrapper_config.sdk_backend_configs_[0]);
        ASSERT_TRUE(config);
        ASSERT_EQ(DataStorageType::DATA_STORAGE_TYPE_LOCAL, config->type_);
        ASSERT_EQ("DEBUG", config->sdk_log_level_);
    }
    {
        SdkWrapperConfig wrapper_config;
        std::string      json = R"(
{
    "sdk_backend_configs":[
    {
        "type":"local",
        "sdk_log_level":"DEBUG"
    },
    {
        "type":"3fs",
        "sdk_log_level":"INFO",
        "enable_async_write":false,
        "write_thread_num":3,
        "write_queue_size":100,
        "read_iov_block_size":999,
        "read_iov_size":1999
    }]
}
)";
        autil::legacy::FromJsonString(wrapper_config, json);
        ASSERT_EQ(4, wrapper_config.thread_num_);
        ASSERT_EQ(2, wrapper_config.sdk_backend_configs_.size());
        {
            std::shared_ptr<SdkBackendConfig> config =
                std::dynamic_pointer_cast<SdkBackendConfig>(wrapper_config.sdk_backend_configs_[0]);
            ASSERT_TRUE(config);
            ASSERT_EQ(DataStorageType::DATA_STORAGE_TYPE_LOCAL, config->type_);
            ASSERT_EQ("DEBUG", config->sdk_log_level_);
        }
        {
            std::shared_ptr<Hf3fsSdkConfig> config =
                std::dynamic_pointer_cast<Hf3fsSdkConfig>(wrapper_config.sdk_backend_configs_[1]);
            ASSERT_TRUE(config);
            ASSERT_EQ(DataStorageType::DATA_STORAGE_TYPE_3FS, config->type_);
            ASSERT_EQ("INFO", config->sdk_log_level_);
            ASSERT_FALSE(config->enable_async_write_);
            ASSERT_EQ(3, config->write_thread_num_);
            ASSERT_EQ(100, config->write_queue_size_);
            ASSERT_EQ(999, config->read_iov_block_size_);
            ASSERT_EQ(1999, config->read_iov_size_);
            ASSERT_EQ((1ULL << 20), config->write_iov_block_size_);
            ASSERT_EQ((1ULL << 32), config->write_iov_size_);
        }
    }
}

}  // namespace test
}  // namespace rtp_llm

// int main(int argc, char** argv) {
//     ::testing::InitGoogleTest(&argc, argv);
//     return RUN_ALL_TESTS();
// }
