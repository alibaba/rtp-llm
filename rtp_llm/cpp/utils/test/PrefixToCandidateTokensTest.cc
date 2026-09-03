
#include "gtest/gtest.h"

#include "rtp_llm/cpp/models/logits_processor/PrefixToCandidateTokens.h"

#include <chrono>
#include <memory>
#include <thread>

using namespace std;

namespace rtp_llm {

class DFAUtilTest: public ::testing::Test {
protected:
};

TEST_F(DFAUtilTest, testConfigJsonize) {
    string schemaStr =
        "{\"start_token_id\": 224, \"end_token_id\": 1, \"prefix_dict\": {\"1_2\": [3,4,5], \"1_2_3\": [3,4,5]}}";
    TreeDecodeConfig config;
    autil::legacy::FromJsonString(config, schemaStr);
    ASSERT_EQ(224, config.start_token_id);
    ASSERT_EQ(1, config.end_token_id);
    ASSERT_EQ("_", config.sep);
    ASSERT_EQ((size_t)2, config.prefix_dict.size());
    std::vector<int32_t> vec = {3, 4, 5};
    ASSERT_EQ(vec, config.prefix_dict["1_2"]);
    ASSERT_EQ(vec, config.prefix_dict["1_2_3"]);
}

TEST_F(DFAUtilTest, testReloadFile) {
    std::string file_path = "./rtp_llm/cpp/utils/test/gir_prefix_dict.json";
    const auto  result    = PrefixToCandidateTokens::instance()->reloadPrefixDict(file_path);
    ASSERT_TRUE(result.ok()) << result.message;
    const auto snapshot = PrefixToCandidateTokens::instance()->snapshot();
    ASSERT_NE(nullptr, snapshot);
    ASSERT_EQ(225, snapshot->startTokenId());
    ASSERT_EQ(2, snapshot->endTokenId());
    std::vector<int32_t> vec = {3, 4, 5};
    ASSERT_EQ(vec, snapshot->getCandidateTokens("1_2"));
    ASSERT_EQ(vec, snapshot->getCandidateTokens("1_2_3"));
}

TEST_F(DFAUtilTest, testVersionedHotSwapPinsSnapshot) {
    auto manager = PrefixToCandidateTokens::instance();

    TreeDecodeConfig version_one;
    version_one.prefix_dict      = {{"225", {10}}, {"225_10", {2}}};
    const uint64_t first_version = manager->currentVersion() + 1;
    const auto     first_result  = manager->updatePrefixDict(first_version, version_one);
    ASSERT_EQ(PrefixTreeUpdateCode::UPDATED, first_result.code);
    const auto first_snapshot = manager->snapshot();
    ASSERT_EQ(first_version, first_snapshot->version());

    TreeDecodeConfig version_two;
    version_two.prefix_dict       = {{"225", {11}}, {"225_11", {2}}};
    const uint64_t second_version = first_version + 1;
    const auto     second_result  = manager->updatePrefixDict(second_version, version_two);
    ASSERT_EQ(PrefixTreeUpdateCode::UPDATED, second_result.code);

    const auto active_snapshot = manager->snapshot();
    EXPECT_EQ(second_version, active_snapshot->version());
    EXPECT_EQ((std::vector<int32_t>{11}), active_snapshot->getCandidateTokens("225"));

    // A request that started before the update still sees its original tree.
    EXPECT_EQ(first_version, first_snapshot->version());
    EXPECT_EQ((std::vector<int32_t>{10}), first_snapshot->getCandidateTokens("225"));
}

TEST_F(DFAUtilTest, testRejectsStaleAndInvalidUpdatesWithoutChangingActiveTree) {
    auto manager = PrefixToCandidateTokens::instance();

    TreeDecodeConfig valid_config;
    valid_config.prefix_dict      = {{"225", {20}}, {"225_20", {2}}};
    const uint64_t active_version = manager->currentVersion() + 1;
    ASSERT_EQ(PrefixTreeUpdateCode::UPDATED, manager->updatePrefixDict(active_version, valid_config).code);

    TreeDecodeConfig stale_config;
    stale_config.prefix_dict = {{"225", {21}}, {"225_21", {2}}};
    const auto stale_result  = manager->updatePrefixDict(active_version - 1, stale_config);
    EXPECT_EQ(PrefixTreeUpdateCode::STALE_VERSION, stale_result.code);

    TreeDecodeConfig invalid_config;
    const auto       invalid_result = manager->updatePrefixDict(active_version + 1, invalid_config);
    EXPECT_EQ(PrefixTreeUpdateCode::INVALID_CONFIG, invalid_result.code);

    TreeDecodeConfig same_boundary_config;
    same_boundary_config.start_token_id = 2;
    same_boundary_config.end_token_id   = 2;
    same_boundary_config.prefix_dict    = {{"2", {3}}};
    const auto same_boundary_result     = manager->updatePrefixDict(active_version + 1, same_boundary_config);
    EXPECT_EQ(PrefixTreeUpdateCode::INVALID_CONFIG, same_boundary_result.code);

    const auto active_snapshot = manager->snapshot();
    ASSERT_NE(nullptr, active_snapshot);
    EXPECT_EQ(active_version, active_snapshot->version());
    EXPECT_EQ((std::vector<int32_t>{20}), active_snapshot->getCandidateTokens("225"));
}

TEST_F(DFAUtilTest, testAcceptsMasterArtifactJson) {
    auto              manager  = PrefixToCandidateTokens::instance();
    const uint64_t    version  = manager->currentVersion() + 1;
    const std::string artifact = "{\"version\":" + std::to_string(version)
                                 + R"(,"start_token_id":225,"end_token_id":2,"sep":"_",)"
                                   R"("prefix_dict":{"225":[10],"225_10":[20],"225_10_20":[30],"225_10_20_30":[2]},)"
                                   R"("sid_count":1,"prefix_count":4,"created_at_epoch_ms":1234})";

    const auto result = manager->updatePrefixDictFromJson(artifact);

    ASSERT_EQ(PrefixTreeUpdateCode::UPDATED, result.code) << result.message;
    const auto snapshot = manager->snapshot();
    ASSERT_NE(nullptr, snapshot);
    EXPECT_EQ(version, snapshot->version());
    EXPECT_EQ(4, snapshot->prefixCount());
    EXPECT_EQ((std::vector<int32_t>{30}), snapshot->getCandidateTokens("225_10_20"));
}

}  // namespace rtp_llm
