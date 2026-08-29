#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "autil/legacy/jsonizable.h"
#include "rtp_llm/cpp/cache/connector/remote_connector/RemoteConnectorConfig.h"

namespace rtp_llm {
namespace {

void expectInvalidSdkConfig(const std::string& value) {
    SdkWrapperConfig config;
    EXPECT_THROW(autil::legacy::FromJsonString(config, value), autil::legacy::ExceptionBase);
}

TEST(RemoteConnectorConfigTest, RejectsNonObjectSdkBackend) {
    expectInvalidSdkConfig(R"({"sdk_backend_configs":[1]})");
}

TEST(RemoteConnectorConfigTest, RejectsSdkBackendWithoutType) {
    expectInvalidSdkConfig(R"({"sdk_backend_configs":[{}]})");
}

TEST(RemoteConnectorConfigTest, RejectsSdkBackendWithNonStringType) {
    expectInvalidSdkConfig(R"({"sdk_backend_configs":[{"type":1}]})");
}

TEST(RemoteConnectorConfigTest, AcceptsValidSdkBackend) {
    SdkWrapperConfig config;
    EXPECT_NO_THROW(autil::legacy::FromJsonString(config, R"({"sdk_backend_configs":[{"type":"local"}]})"));
}

TEST(RemoteConnectorConfigTest, FailedSdkBackendReplacementPreservesPreviousValidState) {
    SdkWrapperConfig config;
    ASSERT_NO_THROW(autil::legacy::FromJsonString(config, R"({"sdk_backend_configs":[{"type":"file"}]})"));
    ASSERT_EQ(config.sdk_backend_configs().size(), 1u);
    ASSERT_NE(std::dynamic_pointer_cast<NfsSdkConfig>(config.sdk_backend_configs().front()), nullptr);
    const std::string previous_config = autil::legacy::ToJsonString(config);

    EXPECT_THROW(autil::legacy::FromJsonString(
                     config, R"({"thread_num":999,"sdk_backend_configs":[{"type":"local"},{"type":1}]})"),
                 autil::legacy::ExceptionBase);
    ASSERT_EQ(config.sdk_backend_configs().size(), 1u);
    EXPECT_NE(std::dynamic_pointer_cast<NfsSdkConfig>(config.sdk_backend_configs().front()), nullptr);
    EXPECT_EQ(autil::legacy::ToJsonString(config), previous_config);
}

}  // namespace
}  // namespace rtp_llm
