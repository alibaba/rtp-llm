#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "autil/legacy/jsonizable.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/KVCMConfig.h"

namespace rtp_llm {
namespace kvcm {
namespace {

void expectInvalidSdkConfig(const std::string& value) {
    SdkWrapperConfig config;
    EXPECT_THROW(autil::legacy::FromJsonString(config, value), autil::legacy::ExceptionBase);
}

TEST(KVCMConfigTest, RejectsNonObjectSdkBackend) {
    expectInvalidSdkConfig(R"({"sdk_backend_configs":[1]})");
}

TEST(KVCMConfigTest, RejectsSdkBackendWithoutType) {
    expectInvalidSdkConfig(R"({"sdk_backend_configs":[{}]})");
}

TEST(KVCMConfigTest, RejectsSdkBackendWithNonStringType) {
    expectInvalidSdkConfig(R"({"sdk_backend_configs":[{"type":1}]})");
}

TEST(KVCMConfigTest, AcceptsValidSdkBackend) {
    SdkWrapperConfig config;
    EXPECT_NO_THROW(autil::legacy::FromJsonString(config, R"({"sdk_backend_configs":[{"type":"local"}]})"));
}

TEST(KVCMConfigTest, FailedSdkBackendReplacementPreservesPreviousValidState) {
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

TEST(KVCMConfigTest, PreservesPaceBackendConfiguration) {
    SdkWrapperConfig config;
    ASSERT_NO_THROW(config.parseBackendConfigs(R"([{"type":"pace","sdk_log_level":"INFO"}])"));
    ASSERT_EQ(config.sdk_backend_configs().size(), 1u);
    EXPECT_NE(std::dynamic_pointer_cast<TairMempoolSdkConfig>(config.sdk_backend_configs().front()), nullptr);
    const auto serialized = autil::legacy::ToJsonString(config);
    EXPECT_NE(serialized.find("pace"), std::string::npos);
    EXPECT_NE(serialized.find("INFO"), std::string::npos);
    SdkWrapperConfig restored;
    ASSERT_NO_THROW(autil::legacy::FromJsonString(restored, serialized));
    EXPECT_EQ(autil::legacy::ToJsonString(restored), serialized);
}

TEST(KVCMConfigTest, PreservesExplicitPaceSsdWireBackend) {
    SdkWrapperConfig config;
    ASSERT_NO_THROW(config.parseBackendConfigs(R"([{"type":"pace_ssd","sdk_log_level":"INFO"}])"));
    ASSERT_EQ(config.sdk_backend_configs().size(), 1u);
    const auto serialized = autil::legacy::ToJsonString(config);
    EXPECT_NE(serialized.find("pace_ssd"), std::string::npos);
    EXPECT_NE(serialized.find("INFO"), std::string::npos);
    SdkWrapperConfig restored;
    ASSERT_NO_THROW(autil::legacy::FromJsonString(restored, serialized));
    EXPECT_EQ(autil::legacy::ToJsonString(restored), serialized);
}

TEST(KVCMConfigTest, RoundTripsSsdThroughCompleteModelClientConfig) {
    auto sdk = std::make_shared<SdkWrapperConfig>();
    ASSERT_NO_THROW(sdk->parseBackendConfigs(R"([{"type":"pace_ssd"}])"));
    auto infos = std::make_shared<KVCMConfig::LocationSpecInfoMap>();
    infos->emplace("tp0_Ffull", 4096);
    auto groups = std::make_shared<KVCMConfig::LocationSpecGroups>();
    groups->emplace("Ffull", std::vector<std::string>{"tp0_Ffull"});
    KVCMConfig config(false,
                      "",
                      128,
                      "decode_remote_test",
                      "fixture",
                      {"127.0.0.1:1"},
                      infos,
                      std::make_shared<MetaChannelConfig>(),
                      sdk,
                      groups,
                      ModelDeployment("fixture", "fp16", false, 2, 1, 1, "", ""));
    const auto serialized = autil::legacy::ToJsonString(config);
    KVCMConfig restored;
    ASSERT_NO_THROW(autil::legacy::FromJsonString(restored, serialized));
    EXPECT_EQ(autil::legacy::ToJsonString(restored), serialized);
    EXPECT_NE(serialized.find("pace_ssd"), std::string::npos);
}

TEST(KVCMConfigTest, ControlRegistrationPreservesEffectiveMetadataWithoutSdkSettings) {
    auto infos = std::make_shared<KVCMConfig::LocationSpecInfoMap>();
    infos->emplace("F0_TP0", 4096);
    infos->emplace("F0_TP1", 4096);
    auto groups = std::make_shared<KVCMConfig::LocationSpecGroups>();
    groups->emplace("custom_group", std::vector<std::string>{"F0_TP0", "F0_TP1"});
    KVCMConfig config(false, "", 128, "remote_group", "stable_remote", {"rpc-a", "rpc-b"}, infos,
                      std::make_shared<MetaChannelConfig>(), std::make_shared<SdkWrapperConfig>(), groups,
                      ModelDeployment("fixture", "fp16", false, 2, 1, 1, "extra", "user-data"));
    config.set_default_query_type(2);
    autil::legacy::json::JsonMap request;
    ASSERT_NO_THROW(autil::legacy::FromJsonString(request, config.registerInstanceRequest()));
    autil::legacy::json::JsonMap client;
    ASSERT_NO_THROW(autil::legacy::FromJsonString(client, autil::legacy::ToJsonString(config)));
    for (const auto* field :
         {"instance_group", "instance_id", "block_size", "default_query_type", "model_deployment"}) {
        EXPECT_EQ(autil::legacy::ToJsonString(request.at(field)), autil::legacy::ToJsonString(client.at(field)));
    }
    EXPECT_EQ(request.count("address"), 0u);
    EXPECT_EQ(request.count("sdk_config"), 0u);
    EXPECT_EQ(request.count("meta_channel_config"), 0u);
    std::vector<autil::legacy::json::JsonMap> wire_infos;
    autil::legacy::FromJson(wire_infos, request.at("location_spec_infos"));
    ASSERT_EQ(wire_infos.size(), 2u);
    EXPECT_EQ(autil::legacy::AnyCast<std::string>(wire_infos[0].at("name")), "F0_TP0");
    int64_t size = 0;
    autil::legacy::FromJson(size, wire_infos[0].at("size"));
    EXPECT_EQ(size, 4096);
    std::vector<autil::legacy::json::JsonMap> wire_groups;
    autil::legacy::FromJson(wire_groups, request.at("location_spec_groups"));
    ASSERT_EQ(wire_groups.size(), 1u);
    EXPECT_EQ(autil::legacy::AnyCast<std::string>(wire_groups[0].at("name")), "custom_group");
    std::vector<std::string> names;
    autil::legacy::FromJson(names, wire_groups[0].at("spec_names"));
    EXPECT_EQ(names, (std::vector<std::string>{"F0_TP0", "F0_TP1"}));
}

TEST(KVCMConfigTest, RejectsUnknownSsdNameWithoutReplacingValidState) {
    SdkWrapperConfig config;
    ASSERT_NO_THROW(config.parseBackendConfigs(R"([{"type":"pace_ssd"}])"));
    const auto original = autil::legacy::ToJsonString(config);
    EXPECT_THROW(config.parseBackendConfigs(R"([{"type":"pace_ssd_typo"}])"), autil::legacy::ExceptionBase);
    EXPECT_EQ(autil::legacy::ToJsonString(config), original);
}

}  // namespace
}  // namespace kvcm
}  // namespace rtp_llm
